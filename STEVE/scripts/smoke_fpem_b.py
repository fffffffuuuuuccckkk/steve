#!/usr/bin/env python
"""Lightweight FPEM-B smoke checks on one NYCTaxi_TDS batch.

This script intentionally avoids creating experiment directories.  It builds the
same model/data/scaler path as ``run_tds_nyctaxi.py``, runs a one-batch
forward/backward/optimizer step, and records whether the new hyper-prototype path
is actually connected to gradients and prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_tds_nyctaxi as runner  # noqa: E402


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def make_args(case: str, device: str, batch_size: int, ckpt_path: str) -> Any:
    base = [
        "run_tds_nyctaxi.py",
        "--config_filename", "configs/NYCTaxi.yaml",
        "--dataset", "NYCTaxi_TDS",
        "--data_dir", "data",
        "--graph_file", "data/NYCTaxi_TDS/adj_mx.npz",
        "--model", "steve",
        "--epochs", "20",
        "--batch_size", str(batch_size),
        "--test_batch_size", str(batch_size),
        "--device", device,
        "--result_root", "experiments/NYCTaxi_TDS",
        "--resume", "false",
        "--fpem_backbone", "agcrn",
        "--fpem_use_pretrained_inv_agcrn", "true",
        "--fpem_pretrained_inv_agcrn_path", ckpt_path,
        "--fpem_use_confounder_extractor", "false",
        "--fpem_use_env_mask", "false",
        "--fpem_lambda_mask_sparse", "0.0",
        "--fpem_lambda_mask_entropy", "0.0",
        "--fpem_use_env_route", "true",
        "--fpem_env_route_k", "3",
        "--fpem_env_route_warmup_epochs", "0",
        "--fpem_env_route_lambda_balance", "0.1",
        "--fpem_env_route_lambda_diverse", "0.02",
        "--fpem_env_route_lambda_proto_align", "0.01",
        "--fpem_env_route_lambda_global", "0.0",
        "--fpem_lambda_inv_pred", "0.0",
        "--fpem_env_use_exogenous", "true",
        "--fpem_use_env_supervision", "false",
        "--fpem_use_env_supcon", "false",
        "--fpem_use_inv_projector", "false",
        "--fpem_use_inv_env_adversarial", "false",
        "--fpem_use_cross_cov_sep", "false",
        "--fpem_use_club_mi", "false",
        "--fpem_lambda_club_mi", "0.0",
        "--fpem_use_env_fusion", "false",
        "--fpem_env_route_use_inv_fallback_expert", "false",
        "--fpem_use_env_prototype_router", "true",
        "--fpem_env_route_target_mode", "env_prototype",
        "--fpem_use_sinkhorn_route", "true",
        "--fpem_sinkhorn_iters", "3",
        "--fpem_sinkhorn_epsilon", "0.05",
        "--fpem_expert_uniform_warmup_epochs", "5",
        "--fpem_env_route_balance_warmup_epochs", "10",
        "--fpem_env_route_initial_temperature", "1.0",
        "--fpem_env_route_final_temperature", "0.3",
        "--fpem_hyper_alpha_mode", "sample_gate",
        "--fpem_use_future_mi", "false",
        "--fpem_lambda_future_mi", "0.0",
        "--fpem_future_mi_target_mode", "env_encoder",
        "--fpem_future_mi_warmup_epochs", "5",
        "--fpem_use_swap", "false",
        "--fpem_lambda_swap", "0.0",
    ]
    cases = {
        "concat_proto_reference": [
            "--fpem_env_route_head_mode", "concat_input",
            "--fpem_use_env_fusion", "true",
            "--fpem_use_future_mi", "true",
            "--fpem_lambda_future_mi", "0.02",
            "--fpem_use_swap", "true",
            "--fpem_lambda_swap", "0.01",
        ],
        "hyper_prediction_router_reference": [
            "--fpem_env_route_head_mode", "hyper_inv_film",
            "--fpem_use_env_prototype_router", "false",
            "--fpem_use_sinkhorn_route", "false",
            "--fpem_env_route_target_mode", "prediction_oracle",
        ],
        "hyper_proto_sinkhorn": [
            "--fpem_env_route_head_mode", "hyper_inv_film_proto",
        ],
        "hyper_proto_sinkhorn_future_swap": [
            "--fpem_env_route_head_mode", "hyper_inv_film_proto",
            "--fpem_use_future_mi", "true",
            "--fpem_lambda_future_mi", "0.02",
            "--fpem_use_swap", "true",
            "--fpem_lambda_swap", "0.01",
            "--fpem_swap_warmup_epochs", "0",
        ],
    }
    if case not in cases:
        raise ValueError(f"unsupported smoke case: {case}")
    old_argv = sys.argv
    try:
        sys.argv = base + cases[case]
        return runner.parse_args()
    finally:
        sys.argv = old_argv


def grad_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name.startswith(prefix) and param.grad is not None:
            total += float(param.grad.detach().float().norm().cpu().item())
    return total


def param_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    total = 0.0
    for name, param in module.named_parameters():
        old = before[name].to(device=param.device, dtype=param.dtype)
        total = max(total, float((param.detach() - old).abs().max().cpu().item()))
    return total


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in module.named_parameters()}


def run_case(case: str, cli: argparse.Namespace) -> dict[str, Any]:
    args = make_args(case, cli.device, cli.batch_size, cli.ckpt_path)
    runner.init_seed(2025)
    loaders, scaler, _counts = runner.build_tds_data(args)
    graph = runner.load_graph(args.graph_file, device=args.device)
    model, lr = runner.build_model(args, graph)
    model.train()
    batch = runner.to_device(next(iter(loaders["train"])), args.device)
    data, target, time_label, c = batch

    with torch.no_grad():
        initial = model.forward_output(data, exog=c, time_label=time_label, training=True, epoch=11)
        y_heads = initial.get("y_hyper_heads")
        y_inv = initial["y_inv"]
        if torch.is_tensor(y_heads) and y_heads.shape[1] == int(getattr(args, "fpem_env_route_k", 3)):
            near_identity = float((y_heads - y_inv.unsqueeze(1)).abs().mean().detach().cpu().item())
        else:
            near_identity = math.nan

    inv_before = snapshot(model.encoder_inv)
    proto_before = model.env_prototypes.detach().cpu().clone() if hasattr(model, "env_prototypes") else None

    main_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("mi_net.")
    ]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    optimizer.zero_grad(set_to_none=True)
    z, h = model(data, c=c, time_label=time_label)
    loss, _sep_loss, _lm = model.calculate_loss(
        z, h, target, c, time_label, scaler, None, p=11 / max(float(args.epochs), 1.0), training=True
    )
    loss.backward()

    outputs = getattr(model, "latest_fpem_outputs", {})
    q = outputs.get("env_route_q")
    q_proto = outputs.get("env_route_q_prototype")
    y_hyper = outputs.get("y_hyper_heads")
    pred = outputs.get("prediction")
    recon_error = math.nan
    if torch.is_tensor(q) and torch.is_tensor(y_hyper) and torch.is_tensor(pred) and q.shape[1] == y_hyper.shape[1]:
        recon = (q.view(q.shape[0], q.shape[1], 1, 1, 1) * y_hyper).sum(dim=1)
        recon_error = float((recon - pred).detach().abs().max().cpu().item())
    row_sums = []
    col_masses = []
    if torch.is_tensor(q):
        row_sums = [float(v) for v in q.detach().sum(dim=1).cpu().tolist()]
        col_masses = [float(v) for v in q.detach().sum(dim=0).cpu().tolist()]
    route_proto_delta = math.nan
    if torch.is_tensor(q) and torch.is_tensor(q_proto) and q.shape == q_proto.shape:
        route_proto_delta = float((q.detach() - q_proto.detach()).abs().max().cpu().item())
    latest_logs = getattr(model, "latest_fpem_logs", {}) or {}

    proto_grad = grad_norm(model, "env_prototypes")
    result = {
        "case": case,
        "loss": float(loss.detach().cpu().item()),
        "route_head_mode": getattr(args, "fpem_env_route_head_mode", None),
        "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "near_identity_mean_abs_y_head_minus_y_inv": near_identity,
        "hypernet_grad_norm": grad_norm(model, "hyper_inv_heads.hypernets"),
        "alpha_gate_grad_norm": grad_norm(model, "hyper_inv_heads.alpha_gates"),
        "encoder_env_grad_norm": grad_norm(model, "encoder_env"),
        "prototype_grad_norm": proto_grad,
        "future_mi_grad_norm": grad_norm(model, "future_env_mu") + grad_norm(model, "future_env_logvar"),
        "encoder_inv_requires_grad_any": any(p.requires_grad for p in model.encoder_inv.parameters()),
        "encoder_inv_grad_norm": grad_norm(model, "encoder_inv"),
        "sinkhorn_row_sum_min": min(row_sums) if row_sums else math.nan,
        "sinkhorn_row_sum_max": max(row_sums) if row_sums else math.nan,
        "sinkhorn_col_masses": col_masses,
        "y_final_reconstruction_max_abs": recon_error,
        "route_q_minus_prototype_q_max_abs": route_proto_delta,
        "primary_uses_env_fusion": float(outputs.get("primary_uses_env_fusion", torch.tensor(float("nan"))).detach().cpu().item()) if torch.is_tensor(outputs.get("primary_uses_env_fusion")) else math.nan,
        "fallback_q_abs_max": float(outputs.get("fallback_q", torch.tensor(float("nan"))).detach().abs().max().cpu().item()) if torch.is_tensor(outputs.get("fallback_q")) else math.nan,
        "loss_future_mi_log": float(latest_logs.get("fpem/future_mi_loss", math.nan)),
        "loss_swap_log": float(latest_logs.get("fpem/swap_loss", math.nan)),
        "swap_prediction_delta_log": float(latest_logs.get("fpem/swap_prediction_delta", math.nan)),
    }
    optimizer.step()
    result["encoder_inv_param_delta_after_step"] = param_delta(inv_before, model.encoder_inv)
    if proto_before is not None:
        result["prototype_param_delta_after_step"] = float(
            (model.env_prototypes.detach().cpu() - proto_before).abs().max().item()
        )
    else:
        result["prototype_param_delta_after_step"] = math.nan
    if hasattr(model, "clear_fpem_runtime_cache"):
        model.clear_fpem_runtime_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--ckpt_path",
        default=str(PROJECT_ROOT / "experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth"),
    )
    parser.add_argument(
        "--cases",
        default="concat_proto_reference,hyper_prediction_router_reference,hyper_proto_sinkhorn,hyper_proto_sinkhorn_future_swap",
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "experiments/NYCTaxi_TDS/fpem_b_smoke_results.json"))
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        cli.device = "cpu"
    results = [run_case(case.strip(), cli) for case in cli.cases.split(",") if case.strip()]
    out = Path(cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[smoke] wrote {out}")


if __name__ == "__main__":
    main()
