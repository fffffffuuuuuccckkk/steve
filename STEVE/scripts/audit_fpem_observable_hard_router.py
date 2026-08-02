#!/usr/bin/env python3
"""Inference-only audit for cached observable experts and hard routing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lib.utils import load_graph  # noqa: E402
from run_tds_nyctaxi import (  # noqa: E402
    build_model,
    build_tds_data,
    load_checkpoint,
    to_device,
    unpack_tds_batch,
)


def masked_sample_error(prediction, target, yita):
    error = float(yita) * np.abs(prediction[..., 0] - target[..., 0])
    error += (1.0 - float(yita)) * np.abs(
        prediction[..., 1] - target[..., 1]
    )
    mask = target[..., 0] > 5.0
    flat_error = error.reshape(error.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    denominator = flat_mask.sum(axis=1)
    sample = np.where(
        denominator > 0,
        (flat_error * flat_mask).sum(axis=1)
        / np.maximum(denominator, 1),
        flat_error.mean(axis=1),
    )
    global_mae = float(
        (flat_error * flat_mask).sum()
        / max(float(flat_mask.sum()), 1.0)
    )
    return sample, global_mae


def metrics(prediction, target, yita):
    sample, global_mae = masked_sample_error(
        prediction, target, yita
    )
    return {
        "global_masked_mae": global_mae,
        "samplewise_mean_mae": float(sample.mean()),
    }, sample


def safe_div(numerator, denominator):
    return float(numerator) / max(float(denominator), 1.0)


def router_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    fn = int(np.sum(~predicted & target))
    tn = int(np.sum(~predicted & ~target))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def forward_from_batch(model, unpacked, exog=None):
    return model.forward_output(
        unpacked["data"],
        exog=exog,
        time_label=unpacked["time_label"],
        training=False,
        sample_index=unpacked["sample_index"],
        observable_load_profile=unpacked["observable_load_profile"],
        observable_load_score=unpacked["observable_load_score"],
        cached_load_level=unpacked["cached_load_level"],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=-1)
    return parser.parse_args()


def main():
    cli = parse_args()
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(cli.checkpoint, "cpu")
    config = dict(checkpoint.get("args", {}))
    config.update(
        {
            "device": cli.device,
            "test_batch_size": cli.batch_size,
            "batch_size": cli.batch_size,
            "resume": False,
            "fpem_observable_load_prior_cache": os.path.abspath(
                cli.cache
            ),
            "fpem_ignore_future_c": True,
        }
    )
    args = SimpleNamespace(**config)
    loaders, scaler, data_counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, _ = build_model(args, graph)
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    state = checkpoint.get("fpem_load_level_state")
    if state and hasattr(model, "load_load_level_state_from_checkpoint"):
        model.load_load_level_state_from_checkpoint(state)
    model.set_fpem_epoch(int(checkpoint.get("epoch", 0)))
    model.eval()

    parts = {
        key: []
        for key in (
            "target",
            "native",
            "invariant",
            "environment",
            "heads",
            "route",
            "level",
            "score",
        )
    }
    first_unpacked = None
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loaders["test_mixed"]):
            if cli.max_batches >= 0 and batch_index >= cli.max_batches:
                break
            unpacked = unpack_tds_batch(
                to_device(raw_batch, args.device)
            )
            if unpacked["stored_c"] is not None:
                raise AssertionError(
                    "observable inference batch unexpectedly contains stored c"
                )
            if first_unpacked is None:
                first_unpacked = unpacked
            output = forward_from_batch(model, unpacked)
            parts["target"].append(unpacked["target"].detach().cpu())
            parts["native"].append(output["y_final"].detach().cpu())
            parts["invariant"].append(output["y_inv"].detach().cpu())
            parts["environment"].append(
                output["y_env_selected"].detach().cpu()
            )
            parts["heads"].append(output["y_env_heads"].detach().cpu())
            parts["route"].append(output["hard_route_id"].detach().cpu())
            parts["level"].append(output["load_level"].detach().cpu())
            parts["score"].append(
                output["observable_load_score"].detach().cpu()
            )

    if first_unpacked is None:
        raise RuntimeError("audit received no test batches")
    target = scaler.inverse_transform(torch.cat(parts["target"])).numpy()
    native = scaler.inverse_transform(torch.cat(parts["native"])).numpy()
    invariant = scaler.inverse_transform(
        torch.cat(parts["invariant"])
    ).numpy()
    environment = scaler.inverse_transform(
        torch.cat(parts["environment"])
    ).numpy()
    heads = scaler.inverse_transform(torch.cat(parts["heads"])).numpy()
    route = torch.cat(parts["route"]).numpy().astype(np.int64)
    level = torch.cat(parts["level"]).numpy().astype(np.int64)
    score = torch.cat(parts["score"]).numpy()
    yita = float(getattr(args, "yita", 0.5))
    sample_count, expert_count = heads.shape[:2]

    predictions = {
        "native_hard_route": native,
        "invariant_only": invariant,
        "always_environment": environment,
    }
    for expert in range(expert_count):
        predictions[f"fixed_expert_{expert}"] = heads[:, expert]
    strategy = {}
    errors = {}
    for name, prediction in predictions.items():
        strategy[name], errors[name] = metrics(
            prediction, target, yita
        )

    inv_error = errors["invariant_only"]
    env_error = errors["always_environment"]
    head_error = np.stack(
        [
            masked_sample_error(heads[:, expert], target, yita)[0]
            for expert in range(expert_count)
        ],
        axis=1,
    )
    relative_margin = float(
        getattr(args, "fpem_hard_router_relative_margin", 0.01)
    )
    margin = relative_margin * np.maximum(inv_error, 0.0)
    oracle_use_environment = env_error + margin < inv_error
    oracle = np.where(
        oracle_use_environment.reshape(-1, 1, 1, 1),
        environment,
        invariant,
    )
    strategy["oracle_hard_route"], errors["oracle_hard_route"] = metrics(
        oracle, target, yita
    )

    route_report = router_metrics(route == 1, oracle_use_environment)
    route_report.update(
        {
            "invariant_route_count": int(np.sum(route == 0)),
            "environment_route_count": int(np.sum(route == 1)),
            "invariant_route_ratio": float(np.mean(route == 0)),
            "environment_route_ratio": float(np.mean(route == 1)),
            "oracle_environment_ratio": float(
                oracle_use_environment.mean()
            ),
            "regret": float(
                errors["native_hard_route"].mean()
                - errors["oracle_hard_route"].mean()
            ),
        }
    )
    invariant_mae = strategy["invariant_only"]["samplewise_mean_mae"]
    native_mae = strategy["native_hard_route"]["samplewise_mean_mae"]
    oracle_mae = strategy["oracle_hard_route"]["samplewise_mean_mae"]
    route_report["oracle_gap_closed"] = (
        (invariant_mae - native_mae)
        / max(invariant_mae - oracle_mae, 1e-12)
        if invariant_mae > oracle_mae
        else 0.0
    )

    cross_regime = np.full(
        (expert_count, expert_count), np.nan, dtype=np.float64
    )
    per_level = {}
    for load_level in range(expert_count):
        mask = level == load_level
        if not bool(mask.any()):
            continue
        per_level[str(load_level)] = {
            "count": int(mask.sum()),
            "native_hard_route_mae": float(
                errors["native_hard_route"][mask].mean()
            ),
            "invariant_mae": float(inv_error[mask].mean()),
            "selected_environment_mae": float(env_error[mask].mean()),
            "environment_route_ratio": float((route[mask] == 1).mean()),
            "oracle_environment_ratio": float(
                oracle_use_environment[mask].mean()
            ),
        }
        for expert in range(expert_count):
            cross_regime[load_level, expert] = float(
                head_error[mask, expert].mean()
            )

    pairwise_difference = np.zeros(
        (expert_count, expert_count), dtype=np.float64
    )
    pairwise_cosine = np.eye(expert_count, dtype=np.float64)
    flat_heads = heads.transpose(1, 0, 2, 3, 4).reshape(
        expert_count, -1
    )
    for left in range(expert_count):
        for right in range(expert_count):
            pairwise_difference[left, right] = float(
                np.mean(np.abs(flat_heads[left] - flat_heads[right]))
            )
            denom = np.linalg.norm(flat_heads[left]) * np.linalg.norm(
                flat_heads[right]
            )
            pairwise_cosine[left, right] = float(
                np.dot(flat_heads[left], flat_heads[right])
                / max(denom, 1e-12)
            )
    off_diagonal = ~np.eye(expert_count, dtype=bool)
    collapse = {
        "pairwise_prediction_mean_absolute_difference": pairwise_difference.tolist(),
        "pairwise_prediction_cosine": pairwise_cosine.tolist(),
        "mean_off_diagonal_absolute_difference": (
            float(pairwise_difference[off_diagonal].mean())
            if expert_count > 1
            else 0.0
        ),
        "mean_off_diagonal_cosine": (
            float(pairwise_cosine[off_diagonal].mean())
            if expert_count > 1
            else 1.0
        ),
        "collapsed_pair_count_at_mae_1e-6": (
            int(
                np.sum(
                    pairwise_difference[np.triu_indices(expert_count, 1)]
                    <= 1e-6
                )
            )
            if expert_count > 1
            else 0
        ),
    }

    # This diagnostic is intentionally performed after native predictions.
    # Stored c is loaded only here and is never part of the inference loader.
    with np.load(
        PROJECT / "data" / "NYCTaxi" / "test.npz",
        allow_pickle=False,
    ) as raw_test:
        stored_c = torch.from_numpy(
            raw_test["c"][: first_unpacked["data"].shape[0]].astype(
                np.float32
            )
        ).to(args.device)
    shuffled_c = stored_c.flip(0)
    zero_c = torch.zeros_like(stored_c)
    with torch.no_grad():
        baseline = forward_from_batch(model, first_unpacked, exog=None)
        shuffled = forward_from_batch(
            model, first_unpacked, exog=shuffled_c
        )
        zeroed = forward_from_batch(model, first_unpacked, exog=zero_c)

    c_invariance = {}
    for name, candidate in (
        ("shuffled_stored_c", shuffled),
        ("zeroed_stored_c", zeroed),
    ):
        prediction_equal = torch.equal(
            baseline["y_final"], candidate["y_final"]
        )
        logits_equal = torch.equal(
            baseline["hard_router_logits"],
            candidate["hard_router_logits"],
        )
        routes_equal = torch.equal(
            baseline["hard_route_id"], candidate["hard_route_id"]
        )
        c_invariance[name] = {
            "predictions_exactly_identical": bool(prediction_equal),
            "router_logits_exactly_identical": bool(logits_equal),
            "routes_exactly_identical": bool(routes_equal),
            "prediction_max_abs_difference": float(
                (
                    baseline["y_final"] - candidate["y_final"]
                ).abs().max().item()
            ),
            "router_logits_max_abs_difference": float(
                (
                    baseline["hard_router_logits"]
                    - candidate["hard_router_logits"]
                ).abs().max().item()
            ),
        }
        if not (prediction_equal and logits_equal and routes_equal):
            raise AssertionError(
                f"stored c affects observable hard routing: {name}"
            )

    summary = {
        "checkpoint": os.path.abspath(cli.checkpoint),
        "selected_by": "best validation checkpoint; test not used for selection",
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "cache": data_counts.get("observable_load_prior"),
        "strategies": strategy,
        "routes": route_report,
        "per_load_level": per_level,
        "expert_cross_regime_mae": cross_regime.tolist(),
        "expert_collapse_diagnostics": collapse,
        "stored_c_test_invariance": c_invariance,
        "forward_uses_target": False,
        "prediction_is_weighted_complete_fusion": False,
    }
    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output_dir / "summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["section", "name", "global_masked_mae", "samplewise_mean_mae", "value"]
        )
        for name, value in strategy.items():
            writer.writerow(
                [
                    "strategy",
                    name,
                    value["global_masked_mae"],
                    value["samplewise_mean_mae"],
                    "",
                ]
            )
        for name, value in route_report.items():
            writer.writerow(["router", name, "", "", value])
    with (output_dir / "expert_cross_regime_mae.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["load_level"]
            + [f"expert_{index}" for index in range(expert_count)]
        )
        for load_level in range(expert_count):
            writer.writerow(
                [load_level] + cross_regime[load_level].tolist()
            )
    with (output_dir / "samplewise_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "sample_index",
            "load_score",
            "load_level",
            "route_id",
            "oracle_route_id",
            "invariant_mae",
            "environment_mae",
            "native_mae",
            "oracle_mae",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(sample_count):
            writer.writerow(
                {
                    "sample_index": index,
                    "load_score": score[index],
                    "load_level": level[index],
                    "route_id": route[index],
                    "oracle_route_id": int(
                        oracle_use_environment[index]
                    ),
                    "invariant_mae": inv_error[index],
                    "environment_mae": env_error[index],
                    "native_mae": errors["native_hard_route"][index],
                    "oracle_mae": errors["oracle_hard_route"][index],
                }
            )
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "native_mae": native_mae,
                "oracle_mae": oracle_mae,
                "stored_c_invariant": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
