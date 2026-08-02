#!/usr/bin/env python
"""Smoke test for threshold-free Counterfactual Risk Router."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import run_tds_nyctaxi as tds  # noqa: E402


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
        "--exp_name", "smoke_counterfactual_risk_router",
        "--best_selection_split", "test_avg",
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
        "--fpem_conservative_inv_override", "false",
        "--fpem_counterfactual_risk_router", "true",
        "--fpem_counterfactual_risk_stage2_start_epoch", "1",
        "--fpem_counterfactual_risk_regression_weight", "1.0",
        "--fpem_counterfactual_risk_ranking_weight", "0.5",
        "--fpem_counterfactual_risk_router_loss_weight", "1.0",
        "--fpem_counterfactual_risk_weight_min", "0.0",
        "--fpem_counterfactual_risk_weight_max", "20.0",
        "--fpem_counterfactual_risk_ranking_temperature", "1.0",
        "--fpem_hard_router_hidden_dim", "64",
        "--fpem_hard_router_warmup_epochs", "0",
        "--fpem_lambda_hard_router", "0.0",
        "--fpem_lambda_load_expert", "0.2",
        "--fpem_use_grad_consensus", "false",
        "--fpem_gc_pred_loss_only", "true",
        "--max_train_batches", "1",
        "--max_eval_batches", "1",
        "--save_test_selected_checkpoints", "true",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        return tds.parse_args()
    finally:
        sys.argv = old_argv


def grad_norm(model, predicate):
    total = 0.0
    for name, param in model.named_parameters():
        if not predicate(name):
            continue
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def snapshot_nonrouter_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not name.startswith("hard_environment_use_router.")
    }


def max_param_change(model, before):
    max_change = 0.0
    for name, old in before.items():
        param = dict(model.named_parameters())[name]
        change = float((param.detach() - old.to(param.device)).abs().max().item())
        max_change = max(max_change, change)
    return max_change


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
    model, lr = tds.build_model(args, graph)
    optimizer = torch.optim.Adam(
        [
            p for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("mi_net.")
        ],
        lr=lr,
    )
    raw_batch = next(iter(loaders["train"]))
    batch = tds.to_device(raw_batch, args.device)
    unpacked = tds.unpack_tds_batch(batch)

    model.set_fpem_epoch(1)
    before = snapshot_nonrouter_params(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _sep = tds.steve_loss(
        model,
        batch,
        scaler,
        args,
        loss_weights=np.ones(3),
        epoch=1,
    )
    loss.backward()
    router_grad = grad_norm(
        model, lambda name: name.startswith("hard_environment_use_router.")
    )
    branch_grad = grad_norm(
        model, lambda name: not name.startswith("hard_environment_use_router.")
    )
    optimizer.step()
    branch_change = max_param_change(model, before)

    model.eval()
    with torch.no_grad():
        output = model.forward_output(
            unpacked["data"],
            exog=unpacked["stored_c"],
            time_label=unpacked["time_label"],
            training=False,
            sample_index=unpacked["sample_index"],
            observable_load_profile=unpacked["observable_load_profile"],
            observable_load_score=unpacked["observable_load_score"],
            cached_load_level=unpacked["cached_load_level"],
        )
        use_inv = output["hard_route_id"].detach().long() == 0
        expected = torch.where(
            use_inv.view(-1, 1, 1, 1),
            output["y_inv"],
            output["y_env_selected"],
        )
        hard_exact = bool(torch.allclose(output["prediction"], expected, atol=0.0, rtol=0.0))
        risks = output.get("counterfactual_predicted_risks")
        risks_shape_ok = bool(torch.is_tensor(risks) and list(risks.shape) == [unpacked["data"].shape[0], 2])

    result = {
        "forward_ok": True,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "router_grad_norm": router_grad,
        "router_has_nonzero_grad": router_grad > 0.0,
        "branch_grad_norm_stage2": branch_grad,
        "branch_grad_zero_stage2": branch_grad == 0.0,
        "branch_max_param_change_after_step": branch_change,
        "branches_unchanged_after_stage2_step": branch_change == 0.0,
        "hard_route_exact": hard_exact,
        "predicted_risks_shape_ok": risks_shape_ok,
        "stored_c_present_in_batch": unpacked["stored_c"] is not None,
        "ignore_future_c": bool(getattr(args, "fpem_ignore_future_c", False)),
        "uses_threshold": bool(float(output["counterfactual_risk_uses_threshold"].item()) > 0.5),
        "hard_router_uses_target_in_forward": bool(float(output["hard_router_uses_target_in_forward"].item()) > 0.5),
        "prediction_source": "counterfactual_risk_router_hard_argmin",
        "soft_prediction_fusion": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = [
        key for key in [
            "loss_finite",
            "router_has_nonzero_grad",
            "branch_grad_zero_stage2",
            "branches_unchanged_after_stage2_step",
            "hard_route_exact",
            "predicted_risks_shape_ok",
            "ignore_future_c",
        ]
        if not result.get(key)
    ]
    if result["stored_c_present_in_batch"]:
        failed.append("stored_c_absent")
    if result["uses_threshold"]:
        failed.append("threshold_free")
    if result["hard_router_uses_target_in_forward"]:
        failed.append("no_target_in_forward")
    if failed:
        raise SystemExit("SMOKE_FAILED " + ",".join(failed))


if __name__ == "__main__":
    main()
