from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.fpem import confounder_regularization as cr
from models.our_model import StableST


def as_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def make_args(mode="both", **overrides):
    args = SimpleNamespace(
        confounder_dep_mode=mode,
        confounder_projection_ridge=1.0e-3,
        confounder_dep_detach_target=True,
        confounder_dep_warmup_epochs=0,
        confounder_dep_ramp_epochs=0,
        gci_weight=1.0,
        scd_weight=1.0,
        gci_edge_preserve_beta=0.1,
        confounder_kl_weight=1.0e-4,
        gci_graph_hops=1,
        gci_symmetrize_adj=True,
        gci_add_self_loops=True,
        confounder_virtual_batch_enabled=False,
        confounder_virtual_batch_size=8,
        dep_eps=1.0e-6,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class CountingExtractor(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C_param = torch.nn.Parameter(C.clone())
        self.calls = 0

    def forward(self, z_variant, training=True):
        self.calls += 1
        C = self.C_param[: z_variant.shape[0]]
        kl = C.pow(2).mean()
        return C, kl, {
            "fpem/C_mean": C.detach().mean(),
            "fpem/C_std": C.detach().std(unbiased=False),
            "fpem/C_norm": C.detach().norm(dim=-1).mean(),
        }


class CountingGraphLearner(cr.FunctionalGraphLearner):
    def __init__(self, input_dim, embed_dim):
        super().__init__(input_dim=input_dim, embed_dim=embed_dim)
        self.calls = 0
        self.adj_shapes = []

    def forward(self, Z, detach_params=False):
        self.calls += 1
        E, A = super().forward(Z, detach_params=detach_params)
        self.adj_shapes.append(tuple(A.shape))
        return E, A


def test_project_out_confounder(device):
    torch.manual_seed(1)
    B, T, N, D, dc = 5, 3, 4, 6, 2
    z = torch.randn(B, T, N, D, device=device, requires_grad=True)
    C = torch.randn(B, dc, device=device, requires_grad=True)
    Zc, Zp, logs = cr.project_out_confounder(z, C, ridge=1.0e-3)
    assert Zc.shape == z.shape
    assert Zp.shape == z.shape
    assert torch.allclose(Zc.reshape(B, -1).mean(dim=0), torch.zeros(T * N * D, device=device), atol=1.0e-5)
    loss = Zp.pow(2).mean()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert C.grad is not None and torch.isfinite(C.grad).all()
    source = open(cr.__file__, "r", encoding="utf-8").read()
    assert "torch.inverse" not in source
    return {
        "Z_centered_shape": list(Zc.shape),
        "Z_perp_shape": list(Zp.shape),
        "projection_trace": as_float(logs["fpem/confounder_projection_trace"]),
    }


def _check_similarity(A, expected):
    assert list(A.shape) == [expected, expected]
    assert torch.allclose(A, A.t(), atol=1.0e-5)
    diag = A.diag()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1.0e-5)


def test_sample_similarity_and_graph_learner(device):
    torch.manual_seed(2)
    B, T, N, D = 6, 2, 5, 4
    z = torch.randn(B, T, N, D, device=device)
    A_sample = cr.sample_similarity_matrix(z)
    _check_similarity(A_sample, B)
    graph = cr.FunctionalGraphLearner(input_dim=T * D, embed_dim=7).to(device)
    E, A_graph = graph(z)
    assert list(E.shape) == [B, N, 7]
    assert list(A_graph.shape) == [B, N, N]
    assert torch.isfinite(E).all() and torch.isfinite(A_graph).all()
    assert torch.allclose(A_graph.sum(dim=-1), torch.ones(B, N, device=device), atol=1.0e-6)
    z2 = torch.randn(B + 1, T, N, D, device=device)
    E2, A2 = graph(z2)
    assert list(E2.shape) == [B + 1, N, 7]
    assert list(A2.shape) == [B + 1, N, N]
    return {
        "A_sample_shape": list(A_sample.shape),
        "E_shape": list(E.shape),
        "A_graph_shape": list(A_graph.shape),
        "A_graph_mean": as_float(A_graph.mean()),
        "A_graph_std": as_float(A_graph.std(unbiased=False)),
    }


def test_centered_normalized_graph_alignment(device):
    torch.manual_seed(9)
    B, N = 2, 5
    teacher_logits = torch.randn(B, N, N, device=device, requires_grad=True)
    teacher = torch.softmax(torch.relu(teacher_logits), dim=-1)
    uniform_student = torch.full((B, N, N), 1.0 / float(N), device=device, requires_grad=True)
    loss_uniform, raw_uniform = cr._centered_normalized_graph_alignment_loss(
        uniform_student,
        teacher,
        num_nodes=N,
    )
    assert as_float(loss_uniform) > 0.5
    assert as_float(raw_uniform) > 0.0
    loss_uniform.backward()
    assert uniform_student.grad is not None and as_float(uniform_student.grad.abs().sum()) > 0.0
    assert teacher_logits.grad is None

    exact_student = teacher.detach().clone().requires_grad_(True)
    loss_exact, raw_exact = cr._centered_normalized_graph_alignment_loss(
        exact_student,
        teacher,
        num_nodes=N,
    )
    assert as_float(loss_exact) < 1.0e-8
    assert as_float(raw_exact) < 1.0e-8
    return {
        "uniform_student_normalized_loss": as_float(loss_uniform),
        "uniform_student_raw_mse": as_float(raw_uniform),
        "exact_student_normalized_loss": as_float(loss_exact),
        "teacher_has_grad": teacher_logits.grad is not None,
    }


def test_relative_loss_behavior(device):
    before = torch.tensor([1.0, 0.5], device=device, requires_grad=True)
    after_lower = torch.tensor([0.2, 0.4], device=device, requires_grad=True)
    loss_zero = torch.relu(after_lower - before.detach()).mean()
    assert as_float(loss_zero) == 0.0
    loss_zero.backward()
    assert before.grad is None

    before2 = torch.tensor([1.0, 0.5], device=device, requires_grad=True)
    after_higher = torch.tensor([1.2, 0.7], device=device, requires_grad=True)
    loss_positive = torch.relu(after_higher - before2.detach()).mean()
    assert as_float(loss_positive) > 0.0
    loss_positive.backward()
    assert before2.grad is None
    assert after_higher.grad is not None and as_float(after_higher.grad.abs().sum()) > 0.0
    return {
        "loss_after_lower": as_float(loss_zero),
        "loss_after_higher": as_float(loss_positive),
        "before_has_grad": before2.grad is not None,
        "after_grad_sum": as_float(after_higher.grad.abs().sum()),
    }


def test_conditional_masks(device):
    torch.manual_seed(3)
    B, T, N, D, dc = 5, 2, 4, 3, 2
    z = torch.randn(B, T, N, D, device=device)
    C = torch.randn(B, dc, device=device)
    Zc, Zp, _ = cr.project_out_confounder(z, C)

    # SCD: only diagonal excluded.
    scd_loss, scd_logs, Ab, Aa, valid = cr.sample_dependence_loss(Zc, Zp)
    offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
    manual = torch.relu(Aa[offdiag].pow(2) - Ab[offdiag].pow(2).detach()).mean()
    assert torch.allclose(scd_loss, manual, atol=1.0e-7)
    assert int(valid.sum().item()) == B * (B - 1)
    assert torch.allclose(scd_logs["relative_violation"], manual.detach(), atol=1.0e-7)

    # GCI: graph edges and diagonal excluded; only nonedges are penalized.
    adj = torch.zeros(N, N, device=device)
    adj[0, 1] = adj[1, 0] = 1.0
    support = cr.build_graph_support_mask(adj, N, device, symmetrize_adj=True, add_self_loops=True)
    nonedge = (~support) & (~torch.eye(N, dtype=torch.bool, device=device))
    graph = cr.FunctionalGraphLearner(input_dim=T * D, embed_dim=5).to(device)
    gci_loss, gci_logs = cr.graph_conditional_independence_loss_full(
        z, C, adj, graph_learner=graph, detach_target=False
    )
    _E_before, A_before = graph(Zc.detach())
    _E_after, A_after = graph(Zp, detach_params=True)
    nonedge_mask = nonedge.to(dtype=A_after.dtype).unsqueeze(0)
    mass_before = (A_before * nonedge_mask).sum(dim=-1)
    mass_after = (A_after * nonedge_mask).sum(dim=-1)
    drop = torch.relu(mass_after - mass_before.detach()).mean()
    edge = support & (~torch.eye(N, dtype=torch.bool, device=device))
    keep = (A_after[:, edge] - A_before[:, edge].detach()).pow(2).mean()
    manual_gci = drop + 0.1 * keep
    assert torch.allclose(gci_loss, manual_gci.to(dtype=gci_loss.dtype), atol=1.0e-6)
    assert as_float(gci_logs["fpem/gci_edge_count"]) == 2.0
    assert as_float(gci_logs["fpem/gci_nonedge_count"]) == float(nonedge.sum().item())
    return {
        "scd_valid_pairs": int(valid.sum().item()),
        "gci_nonedge_pairs": int(nonedge.sum().item()),
        "graph_align_target_available": as_float(gci_logs["fpem/gci_graph_align_target_available"]),
    }


def test_scd_updates_confounder_only_through_after(device):
    last_loss = None
    last_grad = 0.0
    for seed in range(20):
        torch.manual_seed(30 + seed)
        B, T, N, D, dc = 6, 2, 4, 3, 2
        z = torch.randn(B, T, N, D, device=device)
        C = torch.randn(B, dc, device=device, requires_grad=True)
        Z_before, Z_after, _ = cr.project_out_confounder(z.detach(), C)
        scd_loss, logs = cr._scd_from_projected(Z_before, Z_after)
        if as_float(scd_loss) <= 1.0e-10:
            continue
        scd_loss.backward()
        last_loss = as_float(scd_loss)
        last_grad = as_float(C.grad.detach().abs().sum())
        break
    assert last_loss is not None and last_loss > 0.0
    assert last_grad > 0.0
    assert as_float(logs["fpem/scd_relative_violation"]) == last_loss
    return {
        "scd_relative_loss": last_loss,
        "c_grad_from_scd": last_grad,
    }


def test_gci_gradient_isolation(device):
    torch.manual_seed(7)
    B, T, N, D, dc = 5, 2, 4, 3, 2
    z = torch.randn(B, T, N, D, device=device)
    C = torch.randn(B, dc, device=device, requires_grad=True)
    graph = cr.FunctionalGraphLearner(input_dim=T * D, embed_dim=5).to(device)
    adj = torch.zeros(N, N, device=device)
    adj[0, 1] = adj[1, 0] = 1.0
    target_logits = torch.randn(N, N, device=device, requires_grad=True)
    target = torch.softmax(torch.relu(target_logits), dim=-1)
    Z_before, Z_after, _ = cr.project_out_confounder(z.detach(), C)
    gci_loss, align_loss, logs = cr._gci_from_projected(
        Z_before,
        Z_after,
        adj,
        graph,
        graph_target_adj=target,
    )
    graph.zero_grad()
    if C.grad is not None:
        C.grad.zero_()
    gci_loss.backward(retain_graph=True)
    graph_grad_from_gci = sum(
        0.0 if p.grad is None else float(p.grad.detach().abs().sum().cpu().item())
        for p in graph.parameters()
    )
    c_grad_from_gci = float(C.grad.detach().abs().sum().cpu().item())
    assert graph_grad_from_gci == 0.0
    assert c_grad_from_gci > 0.0

    graph.zero_grad()
    C.grad.zero_()
    align_loss.backward()
    graph_grad_from_align = sum(
        0.0 if p.grad is None else float(p.grad.detach().abs().sum().cpu().item())
        for p in graph.parameters()
    )
    c_grad_from_align = float(C.grad.detach().abs().sum().cpu().item())
    target_grad_from_align = (
        0.0
        if target_logits.grad is None
        else float(target_logits.grad.detach().abs().sum().cpu().item())
    )
    assert graph_grad_from_align > 0.0
    assert c_grad_from_align == 0.0
    assert target_grad_from_align == 0.0
    assert as_float(logs["fpem/gci_graph_align_target_available"]) == 1.0
    assert as_float(logs["fpem/gci_target_adj_std"]) >= 0.0
    assert as_float(logs["fpem/gci_graph_align_normalized"]) == as_float(logs["fpem/loss_gci_graph_align"])
    assert as_float(logs["fpem/gci_graph_align_raw_mse"]) >= 0.0
    return {
        "graph_grad_from_gci": graph_grad_from_gci,
        "c_grad_from_gci": c_grad_from_gci,
        "graph_grad_from_align": graph_grad_from_align,
        "c_grad_from_align": c_grad_from_align,
        "target_grad_from_align": target_grad_from_align,
        "target_adj_std": as_float(logs["fpem/gci_target_adj_std"]),
        "graph_align_raw_mse": as_float(logs["fpem/gci_graph_align_raw_mse"]),
        "graph_align_normalized": as_float(logs["fpem/gci_graph_align_normalized"]),
    }


def test_pretrained_teacher_uses_encoder_inv(device):
    torch.manual_seed(8)
    N, D = 4, 3
    inv_emb = torch.randn(N, D, device=device)
    env_emb = torch.randn(N, D, device=device) + 5.0
    fake = SimpleNamespace(
        fpem_use_pretrained_inv_agcrn=True,
        encoder_inv=SimpleNamespace(node_embeddings=torch.nn.Parameter(inv_emb.clone())),
        encoder_env=SimpleNamespace(node_embeddings=torch.nn.Parameter(env_emb.clone())),
    )
    ref = torch.zeros(1, device=device)
    got = StableST._confounder_agcrn_adaptive_adj(fake, ref)
    expected_inv = torch.softmax(torch.relu(inv_emb.matmul(inv_emb.t())), dim=1).detach()
    expected_env = torch.softmax(torch.relu(env_emb.matmul(env_emb.t())), dim=1).detach()
    assert torch.allclose(got, expected_inv, atol=1.0e-6)
    assert not torch.allclose(got, expected_env, atol=1.0e-6)
    assert not got.requires_grad

    fake.fpem_use_pretrained_inv_agcrn = False
    got_env = StableST._confounder_agcrn_adaptive_adj(fake, ref)
    assert torch.allclose(got_env, expected_env, atol=1.0e-6)
    return {
        "pretrained_teacher_from_encoder_inv": True,
        "teacher_requires_grad": bool(got.requires_grad),
        "inv_teacher_std": as_float(got.std(unbiased=False)),
    }


def test_modes_and_both_single_projection(device):
    torch.manual_seed(4)
    B, T, N, D, dc = 4, 2, 3, 5, 2
    z = torch.randn(B, T, N, D, device=device)
    C = torch.randn(B, dc, device=device)
    adj = torch.zeros(N, N, device=device)
    extractor = CountingExtractor(C).to(device)

    none_total, none_logs = cr.confounder_dependence_terms(
        make_args("none"), extractor, adj, z, epoch=1, training=True, ref=z
    )
    assert as_float(none_total) == 0.0
    assert extractor.calls == 0
    assert as_float(none_logs["fpem/confounder_dep_enabled"]) == 0.0

    results = {}
    for mode in ["gci", "scd", "both"]:
        extractor = CountingExtractor(C).to(device)
        graph = CountingGraphLearner(input_dim=T * D, embed_dim=6).to(device)
        target = torch.softmax(torch.relu(torch.randn(N, N, device=device)), dim=-1)
        calls = {"project": 0}
        old_project = cr.project_out_confounder

        def counted_project(*args, **kwargs):
            calls["project"] += 1
            return old_project(*args, **kwargs)

        cr.project_out_confounder = counted_project
        try:
            total, logs = cr.confounder_dependence_terms(
                make_args(mode), extractor, adj, z, graph_learner=graph, graph_target_adj=target, epoch=1, training=True, ref=z
            )
        finally:
            cr.project_out_confounder = old_project
        assert extractor.calls == 1
        assert calls["project"] == 1
        assert torch.isfinite(total)
        assert as_float(logs["fpem/confounder_projection_calls"]) == 1.0
        expected_graph_calls = 2 if mode in {"gci", "both"} else 0
        assert graph.calls == expected_graph_calls
        results[mode] = {
            "total": as_float(total),
            "loss_gci": as_float(logs["fpem/loss_gci"]),
            "loss_scd": as_float(logs["fpem/loss_scd"]),
            "project_calls": calls["project"],
            "extractor_calls": extractor.calls,
            "graph_calls": graph.calls,
            "graph_embed_dim": as_float(logs["fpem/gci_graph_embed_dim"]),
            "graph_align_loss": as_float(logs["fpem/loss_gci_graph_align"]),
            "graph_align_target_available": as_float(logs["fpem/gci_graph_align_target_available"]),
        }
    assert results["gci"]["loss_scd"] == 0.0
    assert results["scd"]["loss_gci"] == 0.0
    assert results["both"]["loss_gci"] > 0.0 or results["both"]["loss_scd"] > 0.0
    return results


def test_virtual_batch_b1_empty_queue_skip(device):
    torch.manual_seed(41)
    B, T, N, D, dc = 1, 2, 4, 3, 2
    z = torch.randn(B, T, N, D, device=device)
    C = torch.randn(B, dc, device=device)
    extractor = CountingExtractor(C).to(device)
    graph = CountingGraphLearner(input_dim=T * D, embed_dim=5).to(device)
    queue = {"Z": [], "C": []}
    total, logs = cr.confounder_dependence_terms(
        make_args("gci", confounder_virtual_batch_enabled=True, confounder_virtual_batch_size=8),
        extractor,
        torch.zeros(N, N, device=device),
        z,
        graph_learner=graph,
        epoch=1,
        training=True,
        ref=z,
        virtual_queue=queue,
    )
    assert as_float(total) == 0.0
    assert extractor.calls == 1
    assert graph.calls == 0
    assert as_float(logs["fpem/physical_batch_size"]) == 1.0
    assert as_float(logs["fpem/confounder_virtual_batch_enabled"]) == 1.0
    assert as_float(logs["fpem/confounder_virtual_batch_actual_size"]) == 1.0
    assert as_float(logs["fpem/confounder_queue_size"]) == 0.0
    assert as_float(logs["fpem/confounder_projection_calls"]) == 0.0
    assert len(queue["Z"]) == 1 and len(queue["C"]) == 1
    assert not queue["Z"][0].requires_grad and not queue["C"][0].requires_grad
    return {
        "total": as_float(total),
        "queue_after": len(queue["Z"]),
        "projection_calls": as_float(logs["fpem/confounder_projection_calls"]),
    }


def test_virtual_batch_projection_current_grad_history_detached(device):
    torch.manual_seed(42)
    T, N, D, dc = 2, 4, 3, 2
    current_z = torch.randn(1, T, N, D, device=device)
    current_C = torch.randn(1, dc, device=device, requires_grad=True)
    hist_z = torch.randn(7, T, N, D, device=device)
    hist_C = torch.randn(7, dc, device=device)
    Z_virtual = torch.cat([current_z.detach(), hist_z.detach()], dim=0)
    C_virtual = torch.cat([current_C, hist_C.detach()], dim=0)
    Z_before, Z_after, logs = cr.project_out_confounder(Z_virtual, C_virtual)
    assert list(Z_before.shape) == [8, T, N, D]
    assert list(Z_after.shape) == [8, T, N, D]
    assert as_float(logs["fpem/confounder_centered_z_norm"]) > 0.0
    assert as_float(logs["fpem/confounder_centered_c_norm"]) > 0.0
    assert as_float(logs["fpem/z_before_after_diff_norm"]) > 0.0
    current_loss = Z_after[:1].pow(2).mean()
    current_loss.backward()
    assert current_C.grad is not None and as_float(current_C.grad.abs().sum()) > 0.0
    assert not hist_z.requires_grad and not hist_C.requires_grad
    return {
        "virtual_size": 8,
        "centered_z_norm": as_float(logs["fpem/confounder_centered_z_norm"]),
        "centered_c_norm": as_float(logs["fpem/confounder_centered_c_norm"]),
        "before_after_diff_norm": as_float(logs["fpem/z_before_after_diff_norm"]),
        "current_c_grad": as_float(current_C.grad.abs().sum()),
    }


def test_virtual_batch_b1_with_history_gci_current_only(device):
    last_payload = None
    for seed in range(20):
        torch.manual_seed(50 + seed)
        T, N, D, dc = 2, 4, 3, 2
        z = torch.randn(1, T, N, D, device=device)
        C = torch.randn(1, dc, device=device)
        extractor = CountingExtractor(C).to(device)
        graph = CountingGraphLearner(input_dim=T * D, embed_dim=5).to(device)
        queue = {
            "Z": [torch.randn(1, T, N, D) for _ in range(7)],
            "C": [torch.randn(1, dc) for _ in range(7)],
        }
        args = make_args(
            "gci",
            confounder_virtual_batch_enabled=True,
            confounder_virtual_batch_size=8,
            confounder_kl_weight=0.0,
            gci_graph_align_weight=0.0,
            gci_edge_preserve_beta=1.0,
        )
        adj = torch.zeros(N, N, device=device)
        adj[0, 1] = adj[1, 0] = 1.0
        total, logs = cr.confounder_dependence_terms(
            args,
            extractor,
            adj,
            z,
            graph_learner=graph,
            epoch=1,
            training=True,
            ref=z,
            virtual_queue=queue,
        )
        last_payload = (total, logs, extractor, graph, queue)
        if as_float(logs["fpem/loss_gci"]) > 1.0e-10:
            total.backward()
            break
    total, logs, extractor, graph, queue = last_payload
    c_grad = (
        0.0
        if extractor.C_param.grad is None
        else float(extractor.C_param.grad.detach().abs().sum().cpu().item())
    )
    assert as_float(logs["fpem/confounder_virtual_batch_actual_size"]) == 8.0
    assert as_float(logs["fpem/confounder_queue_size"]) == 7.0
    assert graph.calls == 2
    assert graph.adj_shapes and all(shape[0] == 1 and shape[1] == N and shape[2] == N for shape in graph.adj_shapes)
    assert as_float(logs["fpem/confounder_centered_z_norm"]) > 0.0
    assert as_float(logs["fpem/confounder_centered_c_norm"]) > 0.0
    assert as_float(logs["fpem/z_before_after_diff_norm"]) > 0.0
    assert c_grad > 0.0
    assert len(queue["Z"]) == 7 and len(queue["C"]) == 7
    assert all(not item.requires_grad for item in queue["Z"])
    assert all(not item.requires_grad for item in queue["C"])
    return {
        "actual_virtual_batch": as_float(logs["fpem/confounder_virtual_batch_actual_size"]),
        "queue_size_before": as_float(logs["fpem/confounder_queue_size"]),
        "graph_adj_shapes": [list(shape) for shape in graph.adj_shapes],
        "c_grad_from_virtual_gci": c_grad,
    }


def test_virtual_batch_b4_keeps_original_path(device):
    torch.manual_seed(43)
    B, T, N, D, dc = 4, 2, 4, 3, 2
    z = torch.randn(B, T, N, D, device=device)
    C = torch.randn(B, dc, device=device)
    extractor = CountingExtractor(C).to(device)
    queue = {
        "Z": [torch.randn(1, T, N, D) for _ in range(7)],
        "C": [torch.randn(1, dc) for _ in range(7)],
    }
    total, logs = cr.confounder_dependence_terms(
        make_args("scd", confounder_virtual_batch_enabled=True, confounder_virtual_batch_size=8),
        extractor,
        None,
        z,
        epoch=1,
        training=True,
        ref=z,
        virtual_queue=queue,
    )
    assert torch.isfinite(total)
    assert as_float(logs["fpem/physical_batch_size"]) == 4.0
    assert as_float(logs["fpem/confounder_virtual_batch_enabled"]) == 0.0
    assert as_float(logs["fpem/confounder_virtual_batch_actual_size"]) == 4.0
    assert as_float(logs["fpem/confounder_projection_calls"]) == 1.0
    assert len(queue["Z"]) == 7 and len(queue["C"]) == 7
    return {
        "physical_batch": as_float(logs["fpem/physical_batch_size"]),
        "virtual_enabled": as_float(logs["fpem/confounder_virtual_batch_enabled"]),
        "actual_size": as_float(logs["fpem/confounder_virtual_batch_actual_size"]),
        "queue_after": len(queue["Z"]),
    }


def test_virtual_batch_largest_mock_gci_current_graph_only(device):
    torch.manual_seed(44)
    T, N, D, dc = 1, 716, 2, 2
    z = torch.randn(1, T, N, D, device=device)
    C = torch.randn(1, dc, device=device)
    extractor = CountingExtractor(C).to(device)
    graph = CountingGraphLearner(input_dim=T * D, embed_dim=3).to(device)
    queue = {
        "Z": [torch.randn(1, T, N, D) for _ in range(7)],
        "C": [torch.randn(1, dc) for _ in range(7)],
    }
    total, logs = cr.confounder_dependence_terms(
        make_args(
            "gci",
            confounder_virtual_batch_enabled=True,
            confounder_virtual_batch_size=8,
            gci_graph_align_weight=0.0,
            confounder_kl_weight=0.0,
        ),
        extractor,
        torch.zeros(N, N, device=device),
        z,
        graph_learner=graph,
        epoch=1,
        training=True,
        ref=z,
        virtual_queue=queue,
    )
    assert torch.isfinite(total)
    assert as_float(logs["fpem/confounder_virtual_batch_actual_size"]) == 8.0
    assert graph.calls == 2
    assert graph.adj_shapes and all(shape == (1, N, N) for shape in graph.adj_shapes)
    assert all(shape != (8, N, N) for shape in graph.adj_shapes)
    return {
        "actual_virtual_batch": as_float(logs["fpem/confounder_virtual_batch_actual_size"]),
        "graph_adj_shapes": [list(shape) for shape in graph.adj_shapes],
        "nonedge_mass_after": as_float(logs["fpem/gci_nonedge_mass_after_C"]),
    }


def test_empty_valid_pairs(device):
    torch.manual_seed(5)
    z = torch.randn(1, 2, 1, 3, device=device)
    C = torch.randn(1, 2, device=device)
    Zc, Zp, _ = cr.project_out_confounder(z, C)
    loss_sample, logs_sample, *_ = cr.sample_dependence_loss(Zc, Zp)
    assert as_float(loss_sample) == 0.0
    loss_gci, logs_gci = cr.graph_conditional_independence_loss_full(
        z,
        C,
        torch.ones(1, 1, device=device),
        graph_learner=cr.FunctionalGraphLearner(input_dim=2 * 3, embed_dim=4).to(device),
        detach_target=False,
    )
    assert as_float(loss_gci) == 0.0
    return {
        "sample_valid_pairs": as_float(logs_sample["valid_pair_count"]),
        "gci_nonedge_pairs": as_float(logs_gci["fpem/gci_nonedge_count"]),
    }


def test_deprecated_alias_conflict(device):
    torch.manual_seed(6)
    z = torch.randn(3, 2, 2, 4, device=device)
    C = torch.randn(3, 2, device=device)
    extractor = CountingExtractor(C).to(device)
    args = make_args("both", gci_ridge=1.0e-3, scd_ridge=2.0e-3, confounder_projection_ridge=None)
    try:
        cr.confounder_dependence_terms(args, extractor, None, z, epoch=1, training=True, ref=z)
    except ValueError as exc:
        assert "deprecated aliases" in str(exc)
        return {"conflict_raises": True}
    raise AssertionError("conflicting deprecated ridge aliases did not raise")


def run(device):
    outputs = {
        "project_out_confounder": test_project_out_confounder(device),
        "sample_similarity_and_graph_learner": test_sample_similarity_and_graph_learner(device),
        "centered_normalized_graph_alignment": test_centered_normalized_graph_alignment(device),
        "relative_loss_behavior": test_relative_loss_behavior(device),
        "conditional_masks": test_conditional_masks(device),
        "scd_updates_confounder": test_scd_updates_confounder_only_through_after(device),
        "gci_gradient_isolation": test_gci_gradient_isolation(device),
        "pretrained_teacher_uses_encoder_inv": test_pretrained_teacher_uses_encoder_inv(device),
        "modes_and_both_single_projection": test_modes_and_both_single_projection(device),
        "virtual_batch_b1_empty_queue_skip": test_virtual_batch_b1_empty_queue_skip(device),
        "virtual_batch_projection_current_grad_history_detached": test_virtual_batch_projection_current_grad_history_detached(device),
        "virtual_batch_b1_with_history_gci_current_only": test_virtual_batch_b1_with_history_gci_current_only(device),
        "virtual_batch_b4_keeps_original_path": test_virtual_batch_b4_keeps_original_path(device),
        "virtual_batch_largest_mock_gci_current_graph_only": test_virtual_batch_largest_mock_gci_current_graph_only(device),
        "empty_valid_pairs": test_empty_valid_pairs(device),
        "deprecated_alias_conflict": test_deprecated_alias_conflict(device),
    }
    for name, payload in outputs.items():
        def finite_tree(obj):
            if isinstance(obj, dict):
                return all(finite_tree(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return all(finite_tree(v) for v in obj)
            if isinstance(obj, (float, int)):
                return math.isfinite(float(obj))
            return True
        assert finite_tree(payload), name
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    outputs = run(device)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
