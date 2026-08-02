#!/usr/bin/env python
"""Small gradient/leakage smoke test for Conservative Invariant Override.

This script is intentionally inference/training-diagnostic only.  It builds the
same NYCTaxi-TDS STEVE model as run_tds_nyctaxi.py, runs one training batch, and
checks that the conservative router loss updates only the router while the two
counterfactual prediction branches are isolated from router-loss gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import run_tds_nyctaxi as tds  # noqa: E402
from models.fpem.losses import (  # noqa: E402
    load_level_expert_conservative_override_losses,
)


def grad_norm_named(model: torch.nn.Module, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def move_batch(batch: Any, device: str):
    return tds.to_device(batch, device)


def slice_batch(batch: Any, count: int):
    if torch.is_tensor(batch):
        return batch[:count]
    if isinstance(batch, tuple):
        return tuple(slice_batch(item, count) for item in batch)
    if isinstance(batch, list):
        return [slice_batch(item, count) for item in batch]
    return batch


def build_args(cli):
    argv = [
        "run_tds_nyctaxi.py",
        "--config_filename", "configs/NYCTaxi.yaml",
        "--dataset", "NYCTaxi_TDS",
        "--data_dir", "data",
        "--graph_file", "data/NYCTaxi_TDS/adj_mx.npz",
        "--model", "steve",
        "--epochs", "1",
        "--batch_size", str(cli.batch_size),
        "--test_batch_size", str(cli.batch_size),
        "--device", cli.device,
        "--resume", "false",
        "--exp_name", "smoke_conservative_inv_override",
        "--fpem_backbone", "agcrn",
        "--fpem_use_pretrained_inv_agcrn", "true",
        "--fpem_pretrained_inv_agcrn_path", cli.pretrained_inv,
        "--fpem_use_confounder_extractor", "false",
        "--fpem_use_env_mask", "false",
        "--fpem_confounder_use_mask", "false",
        "--fpem_lambda_mask_sparse", "0.0",
        "--fpem_lambda_mask_entropy", "0.0",
        "--fpem_lambda_inv_pred", "0.2",
        "--fpem_use_env_route", "true",
        "--fpem_env_route_head_mode", "hyper_inv_film_proto_input_add",
        "--fpem_env_route_k", "1",
        "--fpem_env_route_warmup_epochs", "0",
        "--fpem_env_route_train_mode", "soft_oracle",
        "--fpem_env_route_lambda_route_soft", "0.0",
        "--fpem_env_route_lambda_balance", "0.0",
        "--fpem_env_route_lambda_diverse", "0.0",
        "--fpem_env_route_lambda_proto_align", "0.0",
        "--fpem_env_route_lambda_entropy", "0.0",
        "--fpem_hyper_alpha_mode", "fixed_one",
        "--fpem_lambda_hyper_delta_norm", "0.0",
        "--fpem_use_env_prototype_router", "false",
        "--fpem_use_sinkhorn_route", "false",
        "--fpem_force_uniform_route", "false",
        "--fpem_env_use_exogenous", "true",
        "--fpem_use_env_supervision", "false",
        "--fpem_lambda_env_day_cls", "0.0",
        "--fpem_lambda_env_hour_cls", "0.0",
        "--fpem_lambda_env_rush_cls", "0.0",
        "--fpem_use_env_supcon", "false",
        "--fpem_use_inv_projector", "false",
        "--fpem_use_inv_env_adversarial", "false",
        "--fpem_use_cross_cov_sep", "false",
        "--fpem_use_club_mi", "false",
        "--fpem_lambda_club_mi", "0.0",
        "--fpem_use_env_fusion", "false",
        "--fpem_env_route_use_inv_fallback_expert", "false",
        "--fpem_use_future_mi", "false",
        "--fpem_lambda_future_mi", "0.0",
        "--fpem_use_swap", "false",
        "--fpem_lambda_swap", "0.0",
        "--fpem_use_observable_load_prior", "true",
        "--fpem_observable_load_prior_cache", cli.observable_cache,
        "--fpem_observable_load_random_seed", "314159",
        "--fpem_use_load_level_experts", "true",
        "--fpem_load_level_k", "1",
        "--fpem_load_level_mode", "train_quantile",
        "--fpem_use_random_balanced_assignment", "false",
        "--fpem_ignore_future_c", "true",
        "--fpem_use_environment_gate", "false",
        "--fpem_use_hard_environment_router", "false",
        "--fpem_conservative_inv_override", "true",
        "--fpem_override_margin", "0.0",
        "--fpem_override_threshold", "0.5",
        "--fpem_override_threshold_selection_split", "val",
        "--fpem_override_weight_min", "0.0",
        "--fpem_override_weight_max", "20.0",
        "--fpem_harmful_switch_weight", "2.0",
        "--fpem_override_router_loss_weight", "1.0",
        "--fpem_hard_router_hidden_dim", "64",
        "--fpem_hard_router_warmup_epochs", "0",
        "--fpem_lambda_hard_router", "0.0",
        "--fpem_lambda_load_expert", "0.2",
        "--fpem_use_grad_consensus", "false",
        "--fpem_gc_pred_loss_only", "true",
        "--max_train_batches", "1",
        "--max_eval_batches", "1",
        "--save_test_selected_checkpoints", "false",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        return tds.parse_args()
    finally:
        sys.argv = old_argv


def forward_output(model, unpacked, training):
    return model.forward_output(
        unpacked["data"],
        exog=unpacked["stored_c"],
        time_label=unpacked["time_label"],
        training=training,
        sample_index=unpacked["sample_index"],
        observable_load_profile=unpacked["observable_load_profile"],
        observable_load_score=unpacked["observable_load_score"],
        cached_load_level=unpacked["cached_load_level"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--pretrained-inv",
        default=os.path.join(
            PROJECT_DIR,
            "experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth",
        ),
    )
    parser.add_argument(
        "--observable-cache",
        default=os.path.join(PROJECT_DIR, "data/NYCTaxi/observable_load_prior_k3_v1.npz"),
    )
    cli = parser.parse_args()
    args = build_args(cli)
    tds.init_seed(args.seed)
    loaders, scaler, _counts = tds.build_tds_data(args)
    graph = tds.load_graph(args.graph_file, device=args.device)
    model, _lr = tds.build_model(args, graph)
    if hasattr(model, "set_fpem_epoch"):
        model.set_fpem_epoch(1)
    raw_batch = next(iter(loaders["train"]))
    batch = move_batch(raw_batch, args.device)
    unpacked = tds.unpack_tds_batch(batch)

    model.train()
    output = forward_output(model, unpacked, training=True)
    expert_loss, router_loss, _logs, _diagnostics = (
        load_level_expert_conservative_override_losses(
            output,
            unpacked["target"],
            scaler,
            args,
            training=True,
            epoch=1,
        )
    )
    model.zero_grad(set_to_none=True)
    router_loss.backward()
    router_grad_norm = grad_norm_named(model, ("hard_environment_use_router.",))
    branch_grad_norm_from_router_loss = grad_norm_named(
        model,
        (
            "encoder_inv.",
            "encoder_env.",
            "invariant_predict_conv_2.",
            "tcl4h.",
            "hyper_inv_heads.",
            "hyper_concat_input_heads.",
        ),
    )

    model.zero_grad(set_to_none=True)
    full_loss, _sep = tds.steve_loss(
        model,
        batch,
        scaler,
        args,
        loss_weights=torch.ones(3).cpu().numpy(),
        epoch=1,
    )
    full_loss.backward()

    model.eval()
    with torch.no_grad():
        eval_out = forward_output(model, unpacked, training=False)
        use_inv = eval_out["hard_route_id"].detach().long() == 0
        expected = torch.where(
            use_inv.view(-1, 1, 1, 1),
            eval_out["y_inv"],
            eval_out["y_env_selected"],
        )
        hard_route_exact = bool(
            torch.allclose(eval_out["prediction"], expected, atol=0.0, rtol=0.0)
        )
        raw_small = slice_batch(raw_batch, min(2, cli.batch_size))
        small = move_batch(raw_small, args.device)
        small_unpacked = tds.unpack_tds_batch(small)
        out_small = forward_output(model, small_unpacked, training=False)
        batch_route_invariant = bool(
            torch.equal(
                eval_out["hard_route_id"][: out_small["hard_route_id"].shape[0]].detach().cpu(),
                out_small["hard_route_id"].detach().cpu(),
            )
        )

    result = {
        "forward_ok": True,
        "expert_loss_finite": bool(torch.isfinite(expert_loss).item()),
        "router_loss_finite": bool(torch.isfinite(router_loss).item()),
        "full_loss_finite": bool(torch.isfinite(full_loss).item()),
        "router_grad_norm": router_grad_norm,
        "router_has_nonzero_grad": router_grad_norm > 0.0,
        "branch_grad_norm_from_router_loss": branch_grad_norm_from_router_loss,
        "router_loss_does_not_update_branches": branch_grad_norm_from_router_loss == 0.0,
        "hard_route_exact": hard_route_exact,
        "batch_size_route_invariant": batch_route_invariant,
        "observable_load_prior": bool(getattr(args, "fpem_use_observable_load_prior", False)),
        "ignore_future_c": bool(getattr(args, "fpem_ignore_future_c", False)),
        "stored_c_present_in_batch": unpacked["stored_c"] is not None,
        "prediction_source": "conservative_inv_override_hard",
        "soft_prediction_fusion": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = [
        key for key in [
            "expert_loss_finite",
            "router_loss_finite",
            "full_loss_finite",
            "router_has_nonzero_grad",
            "router_loss_does_not_update_branches",
            "hard_route_exact",
            "batch_size_route_invariant",
            "ignore_future_c",
        ]
        if not result.get(key)
    ]
    if result["stored_c_present_in_batch"]:
        failed.append("stored_c_absent")
    if failed:
        raise SystemExit("SMOKE_FAILED " + ",".join(failed))


if __name__ == "__main__":
    main()
