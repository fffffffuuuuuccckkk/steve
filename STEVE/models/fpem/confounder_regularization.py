"""Latent-confounder residualization with SCD and GCI branches.

GCI and SCD share the same full ``Z_V [B,T,N,D]``, the same sample-specific
latent ``C [B,d_c]``, and the same one-time ``project_out_confounder()`` that
returns ``Z_before`` and ``Z_after``.  The branches then measure different
structures:

  * SCD: full-sample cosine similarity reduction.
  * GCI: learned sample-specific functional connectivity reduction on graph
    nonedges via a shared linear node projection.

There is no mean/max/adaptive representation pooling, no forecasting residual,
no spatial reconstruction, and no kernel-of-Gram / HSIC / alignment objective
in the active path.
"""

import inspect
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _zero(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(())


def _scalar(ref: torch.Tensor, value) -> torch.Tensor:
    return ref.new_tensor(float(value))


def _linear_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve ``a @ x = b`` without forming an explicit inverse."""

    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        return torch.linalg.solve(a, b)
    return torch.solve(b, a).solution


class BatchFirstMultiheadAttention(nn.Module):
    """Batch-first wrapper compatible with older PyTorch versions."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        init_params = inspect.signature(nn.MultiheadAttention.__init__).parameters
        self.supports_batch_first = "batch_first" in init_params
        kwargs = {"embed_dim": embed_dim, "num_heads": num_heads, "dropout": dropout}
        if self.supports_batch_first:
            kwargs["batch_first"] = True
        self.attn = nn.MultiheadAttention(**kwargs)

    def forward(self, query, key, value, need_weights=False):
        if self.supports_batch_first:
            q, k, v = query, key, value
        else:
            q = query.transpose(0, 1).contiguous()
            k = key.transpose(0, 1).contiguous()
            v = value.transpose(0, 1).contiguous()
        out, weights = self.attn(q, k, v, need_weights=need_weights)
        if not self.supports_batch_first:
            out = out.transpose(0, 1).contiguous()
        return out, weights


def normalize_confounder_dep_mode(mode: str) -> str:
    mode = str(mode or "none").strip().lower()
    if mode == "ctc":
        warnings.warn(
            "confounder_dep_mode=ctc is deprecated and is mapped to scd. "
            "The old CTC implementation has been removed; use "
            "confounder_dep_mode=scd explicitly.",
            DeprecationWarning,
        )
        return "scd"
    if mode not in {"none", "gci", "scd", "both"}:
        raise ValueError("confounder_dep_mode must be one of none, gci, scd, both")
    return mode


def reshape_variant_tokens(z_variant: torch.Tensor) -> torch.Tensor:
    """Reshape full variant representation into attention tokens.

    Expected STEVE/FPEM layout is ``[B,T,N,D]``. ``[B,N,D]`` is accepted only for
    legacy callers and is treated as a token sequence of length ``N``. This is a
    reshape only; no averaging or max pooling is performed.
    """

    if z_variant.dim() == 4:
        b, t, n, d = z_variant.shape
        return z_variant.reshape(b, t * n, d)
    if z_variant.dim() == 3:
        b, n, d = z_variant.shape
        return z_variant.reshape(b, n, d)
    raise ValueError(
        "variant representation must be [B,T,N,D] or [B,N,D], "
        f"got {tuple(z_variant.shape)}"
    )


class LatentConfounderExtractor(nn.Module):
    """Confounder-token cross-attention extractor.

    One learned query token attends to all ``Z_V`` tokens and produces one
    sample-specific latent confounder ``C``. This preserves the existing
    extractor architecture and does not inject C into the prediction path.
    """

    def __init__(
        self,
        hidden_dim: int,
        confounder_dim: int = 8,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
        variational: bool = True,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        confounder_dim = int(confounder_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"confounder_num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.confounder_dim = confounder_dim
        self.variational = bool(variational)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.confounder_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attention = BatchFirstMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(attention_dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        out_dim = confounder_dim * (2 if self.variational else 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_variant: torch.Tensor, training: bool = True):
        tokens = reshape_variant_tokens(z_variant)
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: extractor={self.hidden_dim}, "
                f"input={tokens.shape[-1]}"
            )
        tokens = self.token_norm(tokens)
        query = self.confounder_token.expand(tokens.shape[0], 1, self.hidden_dim)
        c_token, _attn = self.cross_attention(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        c_token = self.out_norm(c_token.squeeze(1))
        stats = self.proj(c_token)
        if self.variational:
            mu, logvar = stats.chunk(2, dim=-1)
            logvar = logvar.clamp(min=-10.0, max=10.0)
            if bool(training) and self.training:
                C = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
            else:
                C = mu
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        else:
            C = stats
            kl = stats.new_zeros(())
        logs = {
            "fpem/C_mean": C.detach().mean(),
            "fpem/C_std": C.detach().std(unbiased=False),
            "fpem/C_norm": C.detach().norm(dim=-1).mean(),
            "fpem/confounder_token_norm": c_token.detach().norm(dim=-1).mean(),
            "fpem/confounder_variational": C.new_tensor(float(self.variational)),
        }
        return C, kl, logs


def confounder_ramp_factor(epoch, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch is None:
        return 1.0
    try:
        ep = int(epoch)
    except Exception:
        return 1.0
    warmup = max(int(warmup_epochs), 0)
    ramp = max(int(ramp_epochs), 0)
    if ep <= warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return max(0.0, min(1.0, float(ep - warmup) / float(ramp)))


def build_graph_support_mask(
    adj: Optional[torch.Tensor],
    num_nodes: int,
    device,
    symmetrize_adj: bool = True,
    add_self_loops: bool = True,
    graph_hops: int = 1,
) -> torch.Tensor:
    if adj is None:
        support = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    else:
        if not torch.is_tensor(adj):
            adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
        else:
            adj = adj.to(device=device)
        if adj.dim() == 3:
            adj = adj.abs().sum(dim=0)
        if adj.shape[-2:] != (num_nodes, num_nodes):
            raise ValueError(
                f"adjacency shape {tuple(adj.shape)} incompatible with N={num_nodes}"
            )
        support = adj.float().abs() > 0
    if symmetrize_adj:
        support = support | support.transpose(0, 1)
    if add_self_loops:
        support = support | torch.eye(num_nodes, dtype=torch.bool, device=device)
    hops = max(int(graph_hops), 1)
    if hops > 1:
        reach = support.clone()
        frontier = support.clone()
        support_f = support.float()
        for _ in range(1, hops):
            frontier = frontier.float().matmul(support_f) > 0
            reach = reach | frontier
        support = reach
    return support


def _require_variant_4d(z_variant: torch.Tensor, name: str = "z_variant") -> None:
    if z_variant.dim() != 4:
        raise ValueError(f"{name} must be [B,T,N,D], got {tuple(z_variant.shape)}")


def project_out_confounder(
    z_variant: torch.Tensor,
    C: torch.Tensor,
    ridge: float = 1.0e-3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Project full sample vectors of ``Z_V`` away from the latent confounder C.

    Args:
        z_variant: full variant representation ``[B,T,N,D]``.
        C: sample-specific latent confounder ``[B,d_c]``.
        ridge: ridge coefficient for the batch-axis linear solve.

    Returns:
        ``Z_centered`` and ``Z_perp``, both reshaped back to ``[B,T,N,D]``.
        The before/after associations must be computed from these two tensors
        so the only difference is projection on C, not centering.
    """

    _require_variant_4d(z_variant)
    if C.dim() != 2 or C.shape[0] != z_variant.shape[0]:
        raise ValueError(
            f"C must be [B,d_c], got {tuple(C.shape)} for Z={tuple(z_variant.shape)}"
        )
    B = z_variant.shape[0]
    original_dtype = z_variant.dtype
    Zf = z_variant.reshape(B, -1).float()
    Cc = C.float() - C.float().mean(dim=0, keepdim=True)
    Zc = Zf - Zf.mean(dim=0, keepdim=True)
    dc = Cc.shape[1]
    eye = torch.eye(dc, dtype=Cc.dtype, device=Cc.device)
    c_cov = Cc.transpose(0, 1).matmul(Cc)
    ridge_matrix = c_cov + float(ridge) * eye
    beta = _linear_solve(ridge_matrix, Cc.transpose(0, 1).matmul(Zc))
    explained = Cc.matmul(beta)
    Z_perp_flat = Zc - explained

    trace_term = torch.trace(_linear_solve(ridge_matrix, c_cov)).detach()
    Z_centered = Zc.reshape_as(z_variant).to(dtype=original_dtype)
    Z_perp = Z_perp_flat.reshape_as(z_variant).to(dtype=original_dtype)
    logs = {
        "fpem/confounder_projection_trace": trace_term.to(device=z_variant.device),
        "fpem/confounder_projection_ridge": z_variant.new_tensor(float(ridge)),
        "fpem/confounder_projection_rank_dim": z_variant.new_tensor(float(dc)),
        "fpem/confounder_projection_calls": z_variant.new_tensor(1.0),
        "fpem/confounder_projection_beta_norm": beta.detach().float().norm().to(device=z_variant.device),
        "fpem/confounder_centered_c_norm": Cc.detach().float().norm(dim=-1).mean().to(device=z_variant.device),
        "fpem/confounder_centered_z_norm": Zc.detach().float().norm(dim=-1).mean().to(device=z_variant.device),
        "fpem/z_before_norm": Z_centered.detach().float().reshape(B, -1).norm(dim=-1).mean().to(device=z_variant.device),
        "fpem/z_after_norm": Z_perp.detach().float().reshape(B, -1).norm(dim=-1).mean().to(device=z_variant.device),
        "fpem/z_before_after_diff_norm": (Z_centered.detach().float() - Z_perp.detach().float()).reshape(B, -1).norm(dim=-1).mean().to(device=z_variant.device),
    }
    return Z_centered, Z_perp, logs


def _queue_list(queue: Optional[Dict[str, list]], key: str) -> list:
    if queue is None:
        return []
    values = queue.get(key)
    if values is None:
        values = []
        queue[key] = values
    return values


def _virtual_queue_size(queue: Optional[Dict[str, list]]) -> int:
    if queue is None:
        return 0
    return min(len(_queue_list(queue, "Z")), len(_queue_list(queue, "C")))


def _take_virtual_history(
    queue: Optional[Dict[str, list]],
    max_items: int,
    device,
    z_dtype: torch.dtype,
    c_dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Return detached history tensors for virtual statistical batching."""

    count = min(_virtual_queue_size(queue), max(int(max_items), 0))
    if count <= 0:
        return None, None, 0
    z_items = _queue_list(queue, "Z")[-count:]
    c_items = _queue_list(queue, "C")[-count:]
    z_hist = torch.cat([item.detach() for item in z_items], dim=0).to(device=device, dtype=z_dtype)
    c_hist = torch.cat([item.detach() for item in c_items], dim=0).to(device=device, dtype=c_dtype)
    z_hist = z_hist.detach()
    c_hist = c_hist.detach()
    return z_hist, c_hist, count


def _update_virtual_queue(
    queue: Optional[Dict[str, list]],
    z_current: torch.Tensor,
    C_current: torch.Tensor,
    target_size: int,
) -> None:
    """Append current detached representation/confounder to the FIFO CPU queue."""

    if queue is None:
        return
    max_history = max(int(target_size) - 1, 0)
    if max_history <= 0:
        _queue_list(queue, "Z").clear()
        _queue_list(queue, "C").clear()
        return
    z_values = _queue_list(queue, "Z")
    c_values = _queue_list(queue, "C")
    z_values.append(z_current.detach().to(device="cpu").clone())
    c_values.append(C_current.detach().to(device="cpu").clone())
    del z_values[:-max_history]
    del c_values[:-max_history]


def _virtual_batch_logs(
    ref: torch.Tensor,
    physical_batch_size: int,
    enabled: bool,
    target_size: int,
    actual_size: int,
    queue_size: int,
) -> Dict[str, torch.Tensor]:
    return {
        "fpem/physical_batch_size": _scalar(ref, physical_batch_size),
        "fpem/confounder_virtual_batch_enabled": _scalar(ref, enabled),
        "fpem/confounder_virtual_batch_target_size": _scalar(ref, target_size),
        "fpem/confounder_virtual_batch_actual_size": _scalar(ref, actual_size),
        "fpem/confounder_queue_size": _scalar(ref, queue_size),
    }


def sample_similarity_matrix(
    Z: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Full-sample cosine similarity ``[B,B]`` from ``Z [B,T,N,D]``."""

    _require_variant_4d(Z, name="Z")
    B, T, N, D = Z.shape
    U = Z.reshape(B, T * N * D).float()
    U = F.normalize(U, dim=-1, eps=float(eps))
    return U.matmul(U.transpose(0, 1)).clamp(min=-1.0, max=1.0)


def association_matrix(Z: torch.Tensor, axis: str = "sample", eps: float = 1.0e-6) -> torch.Tensor:
    """Backward-compatible wrapper for the SCD sample-space similarity.

    ``axis='node'`` is intentionally not supported here anymore. The active GCI
    path uses ``FunctionalGraphLearner`` instead of node cosine association.
    """

    if str(axis).strip().lower() != "sample":
        raise ValueError("association_matrix now supports only axis='sample'; GCI uses FunctionalGraphLearner")
    return sample_similarity_matrix(Z, eps=eps)


def _offdiag_mask(size: int, device) -> torch.Tensor:
    return ~torch.eye(size, dtype=torch.bool, device=device)


def sample_dependence_loss(
    Z_before: torch.Tensor,
    Z_after: torch.Tensor,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """SCD zero-margin relative reduction on off-diagonal sample associations."""

    S_before = sample_similarity_matrix(Z_before, eps=eps)
    S_after = sample_similarity_matrix(Z_after, eps=eps)
    valid = _offdiag_mask(S_after.shape[0], S_after.device)
    if bool(valid.any()):
        assoc_before = S_before[valid].pow(2).detach()
        assoc_after = S_after[valid].pow(2)
        relative_violation = F.relu(assoc_after - assoc_before)
        loss = relative_violation.mean()
        dep_before = assoc_before.mean()
        dep_after = assoc_after.mean()
        violation_mean = relative_violation.mean()
    else:
        loss = S_after.new_zeros(())
        dep_before = S_after.new_zeros(())
        dep_after = S_after.new_zeros(())
        violation_mean = S_after.new_zeros(())
    logs = {
        "dep_before": dep_before.detach(),
        "dep_after": dep_after.detach(),
        "dep_reduction": (dep_before - dep_after).detach(),
        "relative_violation": violation_mean.detach(),
        "valid_pair_count": S_after.new_tensor(float(valid.sum().item())),
        "association_dim": S_after.new_tensor(float(S_after.shape[0])),
    }
    return loss, logs, S_before, S_after, valid


def conditional_dependence_loss(
    Z_centered: torch.Tensor,
    Z_perp: torch.Tensor,
    axis: str = "sample",
    mask: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated compatibility wrapper for SCD sample dependence only."""

    if mask is not None:
        raise ValueError("conditional_dependence_loss no longer accepts masks; use graph_connectivity_loss for GCI")
    if str(axis).strip().lower() != "sample":
        raise ValueError("conditional_dependence_loss now supports only axis='sample'")
    return sample_dependence_loss(Z_centered, Z_perp, eps=eps)


class FunctionalGraphLearner(nn.Module):
    """Shared linear node projection plus AGCRN-style functional adjacency."""

    def __init__(self, input_dim: int, embed_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        if self.input_dim <= 0 or self.embed_dim <= 0:
            raise ValueError("FunctionalGraphLearner input_dim/embed_dim must be positive")
        self.node_proj = nn.Linear(self.input_dim, self.embed_dim)

    def forward(self, Z: torch.Tensor, detach_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        _require_variant_4d(Z, name="Z")
        B, T, N, D = Z.shape
        if T * D != self.input_dim:
            raise ValueError(
                f"FunctionalGraphLearner expected T*D={self.input_dim}, got {T * D}"
            )
        X = Z.permute(0, 2, 1, 3).contiguous().reshape(B, N, T * D)
        weight = self.node_proj.weight.detach() if detach_params else self.node_proj.weight
        bias = self.node_proj.bias.detach() if detach_params and self.node_proj.bias is not None else self.node_proj.bias
        E = F.linear(X.float(), weight, bias)
        score = F.relu(E.matmul(E.transpose(-1, -2)))
        A = F.softmax(score, dim=-1)
        return E, A


def _prepare_graph_alignment_target(
    graph_target_adj: Optional[torch.Tensor],
    batch_size: int,
    num_nodes: int,
    ref: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if graph_target_adj is None or not torch.is_tensor(graph_target_adj):
        return None, ref.new_zeros(())
    target = graph_target_adj.detach().to(device=ref.device, dtype=torch.float32)
    if target.dim() == 2:
        if target.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"graph target shape {tuple(target.shape)} incompatible with N={num_nodes}"
            )
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
    elif target.dim() == 3:
        if target.shape[1:] != (num_nodes, num_nodes):
            raise ValueError(
                f"graph target shape {tuple(target.shape)} incompatible with N={num_nodes}"
            )
        if target.shape[0] == 1:
            target = target.expand(batch_size, -1, -1)
        elif target.shape[0] != batch_size:
            raise ValueError(
                f"graph target batch {target.shape[0]} incompatible with B={batch_size}"
            )
    else:
        raise ValueError(f"graph target must be [N,N] or [B,N,N], got {tuple(target.shape)}")
    return target, ref.new_tensor(1.0)


def _centered_normalized_graph_alignment_loss(
    A_before: torch.Tensor,
    A_teacher: torch.Tensor,
    num_nodes: int,
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alignment on the teacher's contrast relative to a uniform row-softmax graph.

    The raw MSE is tiny for ``N=200`` when both graphs are close to uniform.
    The normalized objective compares ``A - 1/N`` and scales by the teacher
    contrast energy, so the graph Linear is trained to match the pretrained
    AGCRN's non-uniform structure instead of merely staying flat.
    """

    teacher = A_teacher.detach().float()
    student = A_before.float()
    raw_mse = F.mse_loss(student, teacher)
    uniform = 1.0 / float(num_nodes)
    student_centered = student - uniform
    teacher_centered = teacher - uniform
    numerator = (student_centered - teacher_centered).pow(2).mean()
    denominator = teacher_centered.pow(2).mean() + float(eps)
    normalized = numerator / denominator
    return normalized.to(dtype=A_before.dtype), raw_mse.to(dtype=A_before.dtype)


def _gci_from_projected(
    Z_before: torch.Tensor,
    Z_after: torch.Tensor,
    adj: Optional[torch.Tensor],
    graph_learner: FunctionalGraphLearner,
    graph_target_adj: Optional[torch.Tensor] = None,
    graph_hops: int = 1,
    symmetrize_adj: bool = True,
    add_self_loops: bool = True,
    align_weight: float = 0.1,
    edge_preserve_beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if graph_learner is None:
        raise ValueError("GCI requires a shared FunctionalGraphLearner")
    N = Z_before.shape[2]
    support = build_graph_support_mask(
        adj,
        N,
        Z_before.device,
        symmetrize_adj=symmetrize_adj,
        add_self_loops=add_self_loops,
        graph_hops=graph_hops,
    )
    offdiag = _offdiag_mask(N, Z_before.device)
    nonedge = (~support) & offdiag
    edge = support & offdiag

    # Alignment branch: train graph Linear only.  Z_before and AGCRN target are
    # detached so this loss cannot update encoder/C/AGCRN graph.
    E_before, A_before = graph_learner(Z_before.detach(), detach_params=False)
    target_adj, target_available = _prepare_graph_alignment_target(
        graph_target_adj,
        Z_before.shape[0],
        N,
        A_before,
    )
    if target_adj is not None:
        graph_align_loss, graph_align_raw_mse = _centered_normalized_graph_alignment_loss(
            A_before,
            target_adj,
            num_nodes=N,
        )
        diag = torch.eye(N, dtype=torch.bool, device=A_before.device)
        target_adj_detached = target_adj.detach().float()
        target_adj_mean = target_adj_detached.mean()
        target_adj_std = target_adj_detached.std(unbiased=False)
        target_edge_mean = (
            target_adj_detached[:, edge].mean() if bool(edge.any()) else A_before.new_zeros(())
        )
        target_nonedge_mean = (
            target_adj_detached[:, nonedge].mean() if bool(nonedge.any()) else A_before.new_zeros(())
        )
        target_diag_mean = target_adj_detached[:, diag].mean()
    else:
        graph_align_loss = A_before.new_zeros(())
        graph_align_raw_mse = A_before.new_zeros(())
        target_adj_mean = A_before.new_zeros(())
        target_adj_std = A_before.new_zeros(())
        target_edge_mean = A_before.new_zeros(())
        target_nonedge_mean = A_before.new_zeros(())
        target_diag_mean = A_before.new_zeros(())

    # GCI branch: use the same numerical Linear parameters but stop gradients to
    # them.  Do not detach Z_after, so GCI still updates C through projection.
    E_after, A_after = graph_learner(Z_after, detach_params=True)
    if bool(nonedge.any()):
        nonedge_mask = nonedge.to(dtype=A_after.dtype).unsqueeze(0)
        nonedge_mass_before_vec = (A_before * nonedge_mask).sum(dim=-1)
        nonedge_mass_after_vec = (A_after * nonedge_mask).sum(dim=-1)
        gci_relative_violation = F.relu(
            nonedge_mass_after_vec - nonedge_mass_before_vec.detach()
        )
        loss_gci_drop = gci_relative_violation.mean()
        nonedge_mass_before = nonedge_mass_before_vec.mean()
        nonedge_mass_after = nonedge_mass_after_vec.mean()
        nonedge_before = A_before[:, nonedge].mean()
        nonedge_after = A_after[:, nonedge].mean()
    else:
        nonedge_before = A_after.new_zeros(())
        nonedge_after = A_after.new_zeros(())
        nonedge_mass_before = A_after.new_zeros(())
        nonedge_mass_after = A_after.new_zeros(())
        gci_relative_violation = A_after.new_zeros(())
        loss_gci_drop = A_after.new_zeros(())
    edge_before = A_before[:, edge].mean() if bool(edge.any()) else A_after.new_zeros(())
    edge_after = A_after[:, edge].mean() if bool(edge.any()) else A_after.new_zeros(())
    if bool(edge.any()):
        loss_gci_keep = (A_after[:, edge] - A_before[:, edge].detach()).pow(2).mean()
    else:
        loss_gci_keep = A_after.new_zeros(())
    beta = float(edge_preserve_beta)
    loss_gci = loss_gci_drop + beta * loss_gci_keep
    logs = {
        "fpem/loss_gci": loss_gci.detach(),
        "fpem/loss_gci_drop": loss_gci_drop.detach(),
        "fpem/loss_gci_keep": loss_gci_keep.detach(),
        "fpem/loss_gci_graph_align": graph_align_loss.detach(),
        "fpem/gci_graph_align_raw_mse": graph_align_raw_mse.detach(),
        "fpem/gci_graph_align_normalized": graph_align_loss.detach(),
        "fpem/gci_graph_align_weight": A_after.new_tensor(float(align_weight)),
        "fpem/gci_edge_preserve_beta": A_after.new_tensor(beta),
        "fpem/gci_graph_align_target_available": target_available.detach(),
        "fpem/gci_target_adj_mean": target_adj_mean.detach().to(device=A_after.device),
        "fpem/gci_target_adj_std": target_adj_std.detach().to(device=A_after.device),
        "fpem/gci_target_edge_mean": target_edge_mean.detach().to(device=A_after.device),
        "fpem/gci_target_nonedge_mean": target_nonedge_mean.detach().to(device=A_after.device),
        "fpem/gci_target_diag_mean": target_diag_mean.detach().to(device=A_after.device),
        "fpem/gci_graph_align_updates_agcrn": A_after.new_zeros(()),
        "fpem/gci_updates_graph_linear": A_after.new_zeros(()),
        "fpem/gci_graph_construction_agcrn_softmax": A_after.new_tensor(1.0),
        "fpem/gci_nonedge_mass_before_C": nonedge_mass_before.detach(),
        "fpem/gci_nonedge_mass_after_C": nonedge_mass_after.detach(),
        "fpem/gci_nonedge_mass_reduction": (nonedge_mass_before - nonedge_mass_after).detach(),
        "fpem/gci_relative_violation": gci_relative_violation.detach().mean(),
        "fpem/gci_edge_before_C": edge_before.detach(),
        "fpem/gci_edge_after_C": edge_after.detach(),
        "fpem/gci_nonedge_conn_before_C": nonedge_before.detach(),
        "fpem/gci_nonedge_conn_after_C": nonedge_after.detach(),
        "fpem/gci_nonedge_conn_reduction": (nonedge_before - nonedge_after).detach(),
        "fpem/gci_edge_conn_before_C": edge_before.detach(),
        "fpem/gci_edge_conn_after_C": edge_after.detach(),
        "fpem/gci_edge_conn_reduction": (edge_before - edge_after).detach(),
        # Backward-compatible summary aliases.
        "fpem/gci_nonedge_dep_before_C": nonedge_before.detach(),
        "fpem/gci_nonedge_dep_after_C": nonedge_after.detach(),
        "fpem/gci_nonedge_dep_reduction": (nonedge_before - nonedge_after).detach(),
        "fpem/gci_edge_dep_before_C": edge_before.detach(),
        "fpem/gci_edge_dep_after_C": edge_after.detach(),
        "fpem/gci_nonedge_count": A_after.new_tensor(float(nonedge.sum().item())),
        "fpem/gci_edge_count": A_after.new_tensor(float(edge.sum().item())),
        "fpem/gci_node_count": A_after.new_tensor(float(N)),
        "fpem/gci_node_vector_dim": A_after.new_tensor(float(Z_before.shape[1] * Z_before.shape[3])),
        "fpem/gci_graph_embed_dim": A_after.new_tensor(float(graph_learner.embed_dim)),
        "fpem/gci_graph_embedding_norm_before": E_before.detach().float().norm(dim=-1).mean().to(device=A_after.device),
        "fpem/gci_graph_embedding_norm_after": E_after.detach().float().norm(dim=-1).mean().to(device=A_after.device),
        "fpem/gci_graph_embedding_std_before": E_before.detach().float().std(unbiased=False).to(device=A_after.device),
        "fpem/gci_graph_embedding_std_after": E_after.detach().float().std(unbiased=False).to(device=A_after.device),
        "fpem/gci_graph_adj_mean_before": A_before.detach().float().mean().to(device=A_after.device),
        "fpem/gci_graph_adj_mean_after": A_after.detach().float().mean().to(device=A_after.device),
        "fpem/gci_graph_adj_std_before": A_before.detach().float().std(unbiased=False).to(device=A_after.device),
        "fpem/gci_graph_adj_std_after": A_after.detach().float().std(unbiased=False).to(device=A_after.device),
    }
    return loss_gci.to(dtype=Z_before.dtype), graph_align_loss.to(dtype=Z_before.dtype), logs


def _scd_from_projected(
    Z_centered: torch.Tensor,
    Z_perp: torch.Tensor,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss, dep_logs, A_before, _A_after, _valid = sample_dependence_loss(
        Z_centered,
        Z_perp,
        eps=eps,
    )
    logs = {
        "fpem/loss_scd": loss.detach(),
        "fpem/scd_assoc_before_C": dep_logs["dep_before"],
        "fpem/scd_assoc_after_C": dep_logs["dep_after"],
        "fpem/scd_assoc_reduction": dep_logs["dep_reduction"],
        "fpem/scd_relative_violation": dep_logs["relative_violation"],
        "fpem/sample_assoc_before_C": dep_logs["dep_before"],
        "fpem/sample_assoc_after_C": dep_logs["dep_after"],
        "fpem/sample_assoc_reduction": dep_logs["dep_reduction"],
        # Backward-compatible aliases.
        "fpem/sample_dep_before_C": dep_logs["dep_before"],
        "fpem/sample_dep_after_C": dep_logs["dep_after"],
        "fpem/sample_dep_reduction": dep_logs["dep_reduction"],
        "fpem/scd_batch_size": A_before.new_tensor(float(Z_centered.shape[0])),
        "fpem/scd_sample_vector_dim": A_before.new_tensor(float(Z_centered.shape[1] * Z_centered.shape[2] * Z_centered.shape[3])),
        # Deprecated zero-valued diagnostics kept only for old summary readers.
        "fpem/sample_kernel_alignment": A_before.new_zeros(()),
        "fpem/scd_alignment_loss": A_before.new_zeros(()),
    }
    return loss.to(dtype=Z_centered.dtype), logs


def graph_conditional_independence_loss_full(
    z_variant: torch.Tensor,
    C: torch.Tensor,
    adj: Optional[torch.Tensor],
    graph_learner: Optional[FunctionalGraphLearner] = None,
    graph_target_adj: Optional[torch.Tensor] = None,
    ridge: float = 1.0e-3,
    detach_target: bool = True,
    graph_hops: int = 1,
    symmetrize_adj: bool = True,
    add_self_loops: bool = True,
    edge_preserve_beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Backward-compatible GCI wrapper using the shared projection path once."""

    Z_target = z_variant.detach() if bool(detach_target) else z_variant
    Z_centered, Z_perp, projection_logs = project_out_confounder(
        Z_target,
        C,
        ridge=ridge,
    )
    loss, align_loss, logs = _gci_from_projected(
        Z_centered,
        Z_perp,
        adj,
        graph_learner=graph_learner,
        graph_target_adj=graph_target_adj,
        graph_hops=graph_hops,
        symmetrize_adj=symmetrize_adj,
        add_self_loops=add_self_loops,
        edge_preserve_beta=edge_preserve_beta,
    )
    logs.update(projection_logs)
    logs["fpem/confounder_dep_detach_target"] = _scalar(z_variant, detach_target)
    logs["fpem/loss_gci_graph_align"] = align_loss.detach()
    return loss, logs


def sample_confounder_dependence_loss_full(
    z_variant: torch.Tensor,
    C: torch.Tensor,
    ridge: float = 1.0e-3,
    detach_target: bool = True,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Backward-compatible SCD wrapper using the same shared projection path."""

    Z_target = z_variant.detach() if bool(detach_target) else z_variant
    Z_centered, Z_perp, projection_logs = project_out_confounder(
        Z_target,
        C,
        ridge=ridge,
    )
    loss, logs = _scd_from_projected(Z_centered, Z_perp, eps=eps)
    logs.update(projection_logs)
    logs["fpem/confounder_dep_detach_target"] = _scalar(z_variant, detach_target)
    return loss, logs


def _optional_attr(args, name):
    return getattr(args, name, None) if hasattr(args, name) else None


def _resolve_deprecated_alias_pair(args, canonical_name, alias_a, alias_b, default, cast_fn):
    canonical = _optional_attr(args, canonical_name)
    a = _optional_attr(args, alias_a)
    b = _optional_attr(args, alias_b)
    alias_values = [v for v in (a, b) if v is not None]
    if alias_values:
        first = cast_fn(alias_values[0])
        for value in alias_values[1:]:
            if cast_fn(value) != first:
                raise ValueError(
                    f"deprecated aliases {alias_a}/{alias_b} conflict; use {canonical_name}"
                )
        if canonical is not None and cast_fn(canonical) != first:
            raise ValueError(
                f"{canonical_name} conflicts with deprecated aliases {alias_a}/{alias_b}"
            )
        warnings.warn(
            f"{alias_a}/{alias_b} are deprecated; use {canonical_name}",
            DeprecationWarning,
        )
        return first
    if canonical is None:
        return cast_fn(default)
    return cast_fn(canonical)


def resolve_confounder_projection_config(args) -> Tuple[float, bool]:
    ridge = _resolve_deprecated_alias_pair(
        args,
        "confounder_projection_ridge",
        "gci_ridge",
        "scd_ridge",
        1.0e-3,
        float,
    )
    detach = _resolve_deprecated_alias_pair(
        args,
        "confounder_dep_detach_target",
        "gci_detach_target",
        "scd_detach_target",
        True,
        _as_bool,
    )
    align_weight = _optional_attr(args, "scd_alignment_weight")
    if align_weight is not None and float(align_weight) != 0.0:
        warnings.warn(
            "scd_alignment_weight is deprecated and ignored; SCD no longer uses "
            "kernel alignment as an active objective.",
            DeprecationWarning,
        )
    return float(ridge), bool(detach)


def zero_confounder_dependence_logs(ref: torch.Tensor, mode: str = "none") -> Dict[str, torch.Tensor]:
    mode = normalize_confounder_dep_mode(mode)
    keys = [
        "fpem/confounder_dep_enabled",
        "fpem/confounder_dep_mode_gci",
        "fpem/confounder_dep_mode_scd",
        "fpem/confounder_dep_mode_both",
        "fpem/confounder_dep_total",
        "fpem/loss_gci",
        "fpem/loss_gci_drop",
        "fpem/loss_gci_keep",
        "fpem/loss_gci_graph_align",
        "fpem/loss_scd",
        "fpem/loss_conf_kl",
        "fpem/effective_gci_weight",
        "fpem/effective_gci_graph_align_weight",
        "fpem/effective_scd_weight",
        "fpem/effective_conf_kl_weight",
        "fpem/confounder_dep_ramp_factor",
        "fpem/C_mean",
        "fpem/C_std",
        "fpem/C_norm",
        "fpem/confounder_token_norm",
        "fpem/confounder_variational",
        "fpem/confounder_projection_trace",
        "fpem/confounder_projection_ridge",
        "fpem/confounder_projection_rank_dim",
        "fpem/confounder_projection_calls",
        "fpem/confounder_projection_beta_norm",
        "fpem/confounder_centered_c_norm",
        "fpem/confounder_centered_z_norm",
        "fpem/z_before_norm",
        "fpem/z_after_norm",
        "fpem/z_before_after_diff_norm",
        "fpem/confounder_dep_detach_target",
        "fpem/physical_batch_size",
        "fpem/confounder_virtual_batch_enabled",
        "fpem/confounder_virtual_batch_target_size",
        "fpem/confounder_virtual_batch_actual_size",
        "fpem/confounder_queue_size",
        "fpem/gci_nonedge_mass_before_C",
        "fpem/gci_nonedge_mass_after_C",
        "fpem/gci_nonedge_mass_reduction",
        "fpem/gci_relative_violation",
        "fpem/gci_edge_before_C",
        "fpem/gci_edge_after_C",
        "fpem/gci_nonedge_conn_before_C",
        "fpem/gci_nonedge_conn_after_C",
        "fpem/gci_nonedge_conn_reduction",
        "fpem/gci_edge_conn_before_C",
        "fpem/gci_edge_conn_after_C",
        "fpem/gci_edge_conn_reduction",
        "fpem/gci_nonedge_dep_before_C",
        "fpem/gci_nonedge_dep_after_C",
        "fpem/gci_nonedge_dep_reduction",
        "fpem/gci_edge_dep_before_C",
        "fpem/gci_edge_dep_after_C",
        "fpem/gci_nonedge_count",
        "fpem/gci_edge_count",
        "fpem/gci_node_count",
        "fpem/gci_node_vector_dim",
        "fpem/gci_graph_embed_dim",
        "fpem/gci_graph_embedding_norm_before",
        "fpem/gci_graph_embedding_norm_after",
        "fpem/gci_graph_embedding_std_before",
        "fpem/gci_graph_embedding_std_after",
        "fpem/gci_graph_adj_mean_before",
        "fpem/gci_graph_adj_mean_after",
        "fpem/gci_graph_adj_std_before",
        "fpem/gci_graph_adj_std_after",
        "fpem/gci_graph_align_weight",
        "fpem/gci_graph_align_raw_mse",
        "fpem/gci_graph_align_normalized",
        "fpem/gci_edge_preserve_beta",
        "fpem/gci_graph_align_target_available",
        "fpem/gci_target_adj_mean",
        "fpem/gci_target_adj_std",
        "fpem/gci_target_edge_mean",
        "fpem/gci_target_nonedge_mean",
        "fpem/gci_target_diag_mean",
        "fpem/gci_graph_align_updates_agcrn",
        "fpem/gci_updates_graph_linear",
        "fpem/gci_graph_construction_agcrn_softmax",
        "fpem/scd_assoc_before_C",
        "fpem/scd_assoc_after_C",
        "fpem/scd_assoc_reduction",
        "fpem/scd_relative_violation",
        "fpem/sample_assoc_before_C",
        "fpem/sample_assoc_after_C",
        "fpem/sample_assoc_reduction",
        "fpem/sample_dep_before_C",
        "fpem/sample_dep_after_C",
        "fpem/sample_dep_reduction",
        "fpem/scd_batch_size",
        "fpem/scd_sample_vector_dim",
        # Deprecated summary keys, always zero unless old checkpoints logged them.
        "fpem/sample_kernel_alignment",
        "fpem/scd_alignment_loss",
    ]
    logs = {key: _zero(ref) for key in keys}
    logs["fpem/confounder_dep_mode_gci"] = _scalar(ref, mode == "gci")
    logs["fpem/confounder_dep_mode_scd"] = _scalar(ref, mode == "scd")
    logs["fpem/confounder_dep_mode_both"] = _scalar(ref, mode == "both")
    return logs


def confounder_dependence_terms(
    args,
    extractor: Optional[LatentConfounderExtractor],
    adj: Optional[torch.Tensor],
    z_variant: Optional[torch.Tensor],
    graph_learner: Optional[FunctionalGraphLearner] = None,
    graph_target_adj: Optional[torch.Tensor] = None,
    epoch=None,
    training: bool = True,
    ref: Optional[torch.Tensor] = None,
    virtual_queue: Optional[Dict[str, list]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute optional strictly-dual GCI/SCD/KL regularization."""

    if ref is None:
        ref = z_variant
    if ref is None:
        raise ValueError("confounder_dependence_terms requires a reference tensor")
    mode = normalize_confounder_dep_mode(getattr(args, "confounder_dep_mode", "none"))
    if mode == "none" or extractor is None or z_variant is None:
        return _zero(ref), zero_confounder_dependence_logs(ref, mode=mode)

    C, kl_loss, c_logs = extractor(z_variant, training=training)
    projection_ridge, detach_target = resolve_confounder_projection_config(args)
    physical_batch_size = int(z_variant.shape[0])
    virtual_target_size = max(int(getattr(args, "confounder_virtual_batch_size", 8)), 2)
    virtual_requested = _as_bool(getattr(args, "confounder_virtual_batch_enabled", False))
    virtual_enabled = bool(training and physical_batch_size == 1 and virtual_requested)
    queue_size_before = _virtual_queue_size(virtual_queue)
    Z_target_current = z_variant.detach() if detach_target else z_variant
    Z_project = Z_target_current
    C_project = C
    virtual_actual_size = physical_batch_size
    if virtual_enabled:
        Z_hist, C_hist, history_count = _take_virtual_history(
            virtual_queue,
            virtual_target_size - 1,
            z_variant.device,
            Z_target_current.dtype,
            C.dtype,
        )
        if history_count > 0 and Z_hist is not None and C_hist is not None:
            Z_project = torch.cat([Z_target_current, Z_hist], dim=0)
            C_project = torch.cat([C, C_hist], dim=0)
        virtual_actual_size = int(Z_project.shape[0])

    ramp = confounder_ramp_factor(
        epoch,
        int(getattr(args, "confounder_dep_warmup_epochs", 5)),
        int(getattr(args, "confounder_dep_ramp_epochs", 10)),
    )
    eps = float(getattr(args, "dep_eps", 1.0e-6))
    gci_loss = _zero(ref)
    gci_graph_align_loss = _zero(ref)
    scd_loss = _zero(ref)
    logs = zero_confounder_dependence_logs(ref, mode=mode)
    logs.update(c_logs)
    logs.update(
        _virtual_batch_logs(
            ref,
            physical_batch_size=physical_batch_size,
            enabled=virtual_enabled,
            target_size=virtual_target_size,
            actual_size=virtual_actual_size,
            queue_size=queue_size_before,
        )
    )
    logs["fpem/confounder_dep_enabled"] = _scalar(ref, True)
    logs["fpem/confounder_dep_ramp_factor"] = _scalar(ref, ramp)
    logs["fpem/confounder_dep_detach_target"] = _scalar(ref, detach_target)
    effective_gci = float(getattr(args, "gci_weight", 1.0e-3)) * ramp
    effective_graph_align = float(getattr(args, "gci_graph_align_weight", 0.1))
    effective_scd = float(getattr(args, "scd_weight", 1.0e-3)) * ramp
    effective_kl = float(getattr(args, "confounder_kl_weight", 1.0e-4)) * ramp

    if virtual_enabled and virtual_actual_size < 2:
        _update_virtual_queue(virtual_queue, Z_target_current, C, virtual_target_size)
        total = _zero(ref)
        logs.update(
            {
                "fpem/effective_gci_weight": _scalar(ref, effective_gci),
                "fpem/effective_gci_graph_align_weight": _scalar(ref, effective_graph_align),
                "fpem/effective_scd_weight": _scalar(ref, effective_scd),
                "fpem/effective_conf_kl_weight": _scalar(ref, effective_kl),
                "fpem/confounder_dep_total": total.detach(),
            }
        )
        return total, logs

    Z_centered, Z_perp, projection_logs = project_out_confounder(
        Z_project,
        C_project,
        ridge=projection_ridge,
    )
    logs.update(projection_logs)
    if virtual_enabled:
        Z_centered_current = Z_centered[:1]
        Z_perp_current = Z_perp[:1]
    else:
        Z_centered_current = Z_centered
        Z_perp_current = Z_perp

    if mode in {"gci", "both"}:
        gci_loss, gci_graph_align_loss, gci_logs = _gci_from_projected(
            Z_centered_current,
            Z_perp_current,
            adj,
            graph_learner=graph_learner,
            graph_target_adj=graph_target_adj,
            graph_hops=int(getattr(args, "gci_graph_hops", 1)),
            symmetrize_adj=_as_bool(getattr(args, "gci_symmetrize_adj", True)),
            add_self_loops=_as_bool(getattr(args, "gci_add_self_loops", True)),
            align_weight=float(getattr(args, "gci_graph_align_weight", 0.1)),
            edge_preserve_beta=float(getattr(args, "gci_edge_preserve_beta", 0.1)),
        )
        logs.update(gci_logs)

    if mode in {"scd", "both"}:
        scd_loss, scd_logs = _scd_from_projected(Z_centered, Z_perp, eps=eps)
        logs.update(scd_logs)

    total = (
        effective_gci * gci_loss
        + effective_graph_align * gci_graph_align_loss
        + effective_scd * scd_loss
        + effective_kl * kl_loss
    )
    logs.update(
        {
            "fpem/loss_gci": gci_loss.detach(),
            "fpem/loss_gci_graph_align": gci_graph_align_loss.detach(),
            "fpem/loss_scd": scd_loss.detach(),
            "fpem/loss_conf_kl": kl_loss.detach(),
            "fpem/effective_gci_weight": _scalar(ref, effective_gci),
            "fpem/effective_gci_graph_align_weight": _scalar(ref, effective_graph_align),
            "fpem/effective_scd_weight": _scalar(ref, effective_scd),
            "fpem/effective_conf_kl_weight": _scalar(ref, effective_kl),
            "fpem/confounder_dep_total": total.detach(),
        }
    )
    if virtual_enabled:
        _update_virtual_queue(virtual_queue, Z_target_current, C, virtual_target_size)
    return total, logs
