#!/usr/bin/env python3
"""Inference-only three-route counterfactual audit for observable-load FPEM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.utils.data import SequentialSampler  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lib.metrics import test_metrics  # noqa: E402
from lib.utils import load_graph  # noqa: E402
from run_tds_nyctaxi import (  # noqa: E402
    build_model,
    build_tds_data,
    load_checkpoint,
    to_device,
    unpack_tds_batch,
)

STRATEGIES = (
    ("all_environment", "All Environment"),
    ("all_invariant", "All Invariant"),
    ("selective_hard_routing", "Selective Hard Routing"),
)
TIE_TOLERANCE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-samples", type=int, default=546)
    return parser.parse_args()


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_state(before, after):
    if set(before) != set(after):
        return False, sorted(set(before).symmetric_difference(after))
    changed = [
        name
        for name in before
        if before[name].shape != after[name].shape
        or before[name].dtype != after[name].dtype
        or not torch.equal(before[name], after[name])
    ]
    return not changed, changed


def forward_once(model, unpacked):
    # Target is deliberately absent from this call.
    return model.forward_output(
        unpacked["data"],
        exog=None,
        time_label=unpacked["time_label"],
        training=False,
        sample_index=unpacked["sample_index"],
        observable_load_profile=unpacked["observable_load_profile"],
        observable_load_score=unpacked["observable_load_score"],
        cached_load_level=unpacked["cached_load_level"],
    )


def inverse_transform(scaler, tensor):
    return scaler.inverse_transform(tensor).detach().cpu().numpy()


def samplewise_masked_weighted_error(prediction, target, yita):
    """Match models.fpem.losses.head_prediction_losses exactly."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    mask = target[..., 0] > 5.0
    error = float(yita) * np.abs(
        prediction[..., 0] - target[..., 0]
    )
    error += (1.0 - float(yita)) * np.abs(
        prediction[..., 1] - target[..., 1]
    )
    flat_mask = mask.reshape(mask.shape[0], -1)
    flat_error = error.reshape(error.shape[0], -1)
    valid_counts = flat_mask.sum(axis=1)
    if np.any(valid_counts <= 0):
        invalid = np.flatnonzero(valid_counts <= 0).tolist()
        raise AssertionError(
            f"samples without target[...,0] > 5 mask entries: {invalid}"
        )
    sample_mae = (
        (flat_error * flat_mask).sum(axis=1) / valid_counts
    )
    global_mae = float(
        (flat_error * flat_mask).sum() / flat_mask.sum()
    )
    return sample_mae.astype(np.float64), global_mae, valid_counts


def strategy_metrics(prediction, target, yita):
    sample_mae, global_mae, valid_counts = (
        samplewise_masked_weighted_error(prediction, target, yita)
    )
    standard_mae, standard_mape = test_metrics(prediction, target)
    return (
        {
            "global_masked_weighted_mae": global_mae,
            "samplewise_mean_masked_weighted_mae": float(
                sample_mae.mean()
            ),
            "samplewise_median_masked_weighted_mae": float(
                np.median(sample_mae)
            ),
            "standard_test_mae": float(standard_mae),
            "standard_test_mape": float(standard_mape),
            "valid_sample_count": int((valid_counts > 0).sum()),
        },
        sample_mae,
    )


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def pairwise_wins(
    first_name, second_name, first_error, second_error
):
    difference = np.asarray(first_error) - np.asarray(second_error)
    ties = np.abs(difference) <= TIE_TOLERANCE
    first_wins = difference < -TIE_TOLERANCE
    second_wins = difference > TIE_TOLERANCE
    total = int(difference.size)
    counts = (
        int(first_wins.sum()),
        int(second_wins.sum()),
        int(ties.sum()),
    )
    if sum(counts) != total:
        raise AssertionError(
            f"pairwise counts do not sum to {total}: {counts}"
        )
    winner = np.full(total, "tie", dtype=object)
    winner[first_wins] = first_name
    winner[second_wins] = second_name
    return {
        "comparison": f"{first_name}_vs_{second_name}",
        "first_strategy": first_name,
        "second_strategy": second_name,
        "first_win_count": counts[0],
        "first_win_percentage": 100.0 * counts[0] / total,
        "second_win_count": counts[1],
        "second_win_percentage": 100.0 * counts[1] / total,
        "tie_count": counts[2],
        "tie_percentage": 100.0 * counts[2] / total,
        "sample_count": total,
        "mae_difference_definition": "first_minus_second",
        "mean_samplewise_mae_difference": float(difference.mean()),
        "median_samplewise_mae_difference": float(
            np.median(difference)
        ),
    }, winner


def binary_router_metrics(predicted, oracle):
    predicted = np.asarray(predicted, dtype=np.int64)
    oracle = np.asarray(oracle, dtype=np.int64)
    tp = int(np.sum((predicted == 1) & (oracle == 1)))
    fp = int(np.sum((predicted == 1) & (oracle == 0)))
    fn = int(np.sum((predicted == 0) & (oracle == 1)))
    tn = int(np.sum((predicted == 0) & (oracle == 0)))
    total = int(predicted.size)
    correct = tp + tn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "correct_route_count": correct,
        "incorrect_route_count": total - correct,
        "route_accuracy": safe_div(correct, total),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "predicted_environment_ratio": float(
            np.mean(predicted == 1)
        ),
        "oracle_environment_ratio": float(np.mean(oracle == 1)),
        "useful_environment_gains_captured": recall,
        "harmful_environment_corrections_avoided": specificity,
        "total_samples": total,
    }


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chart_canvas(width=1200, height=720):
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image), ImageFont.load_default(size=20)


def centered_text(draw, xy, text, font, fill="black"):
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def plot_pairwise(rows, path):
    image, draw, font = chart_canvas()
    title_font = ImageFont.load_default(size=28)
    small_font = ImageFont.load_default(size=16)
    centered_text(
        draw,
        (600, 42),
        "Pairwise sample-wise masked MAE winners",
        title_font,
    )
    left, right = 245, 1120
    bar_width = right - left
    colors = ("#4C78A8", "#F58518", "#BAB0AC")
    for index, row in enumerate(rows):
        y0 = 135 + index * 155
        y1 = y0 + 72
        label = (
            f"{row['first_strategy']}\nvs {row['second_strategy']}"
        )
        draw.multiline_text(
            (20, y0 + 10),
            label,
            font=small_font,
            fill="black",
            spacing=5,
        )
        percentages = (
            row["first_win_percentage"],
            row["second_win_percentage"],
            row["tie_percentage"],
        )
        labels = ("First", "Second", "Tie")
        cursor = left
        for percentage, label_text, color in zip(
            percentages, labels, colors
        ):
            width = int(round(bar_width * percentage / 100.0))
            end = min(cursor + width, right)
            draw.rectangle((cursor, y0, end, y1), fill=color)
            if width >= 70:
                centered_text(
                    draw,
                    ((cursor + end) / 2, (y0 + y1) / 2),
                    f"{label_text}\n{percentage:.1f}%",
                    small_font,
                    fill="white" if color != "#BAB0AC" else "black",
                )
            cursor = end
        draw.rectangle((left, y0, right, y1), outline="black", width=2)
    draw.text(
        (left, 640),
        "Tie tolerance: absolute sample-wise MAE difference <= 1e-6",
        font=small_font,
        fill="black",
    )
    image.save(path)


def plot_strategy_mae(summary_rows, path):
    image, draw, font = chart_canvas()
    title_font = ImageFont.load_default(size=28)
    small_font = ImageFont.load_default(size=16)
    centered_text(
        draw, (600, 42), "Three-route forecasting quality", title_font
    )
    metrics = (
        ("global_masked_weighted_mae", "Global weighted", "#4C78A8"),
        (
            "samplewise_mean_masked_weighted_mae",
            "Sample-wise mean",
            "#F58518",
        ),
        ("standard_test_mae", "Repository standard", "#54A24B"),
    )
    maximum = max(
        float(row[key])
        for row in summary_rows
        for key, _, _ in metrics
    )
    plot_left, plot_top, plot_right, plot_bottom = 105, 105, 1140, 585
    draw.line(
        (plot_left, plot_top, plot_left, plot_bottom),
        fill="black",
        width=2,
    )
    draw.line(
        (plot_left, plot_bottom, plot_right, plot_bottom),
        fill="black",
        width=2,
    )
    group_width = (plot_right - plot_left) / len(summary_rows)
    bar_width = 75
    for group, row in enumerate(summary_rows):
        center = plot_left + group_width * (group + 0.5)
        for metric_index, (key, _, color) in enumerate(metrics):
            value = float(row[key])
            height = int(
                (plot_bottom - plot_top - 35)
                * value
                / max(maximum, 1e-12)
            )
            x0 = int(
                center
                + (metric_index - 1) * (bar_width + 12)
                - bar_width / 2
            )
            y0 = plot_bottom - height
            draw.rectangle(
                (x0, y0, x0 + bar_width, plot_bottom),
                fill=color,
                outline="black",
            )
            centered_text(
                draw,
                (x0 + bar_width / 2, y0 - 14),
                f"{value:.3f}",
                small_font,
            )
        centered_text(
            draw,
            (center, plot_bottom + 42),
            row["display_name"],
            small_font,
        )
    legend_x = 125
    for _, label, color in metrics:
        draw.rectangle(
            (legend_x, 660, legend_x + 22, 682),
            fill=color,
            outline="black",
        )
        draw.text(
            (legend_x + 30, 659),
            label,
            font=small_font,
            fill="black",
        )
        legend_x += 320
    image.save(path)


def plot_confusion(primary, path):
    matrix = np.array(
        [
            [primary["true_negative"], primary["false_positive"]],
            [primary["false_negative"], primary["true_positive"]],
        ],
        dtype=np.int64,
    )
    image, draw, font = chart_canvas(760, 720)
    title_font = ImageFont.load_default(size=26)
    value_font = ImageFont.load_default(size=34)
    centered_text(
        draw,
        (380, 40),
        "Margin-aware router oracle confusion",
        title_font,
    )
    cell = 220
    left, top = 210, 120
    maximum = max(int(matrix.max()), 1)
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            intensity = int(245 - 170 * value / maximum)
            color = (intensity, intensity, 255)
            x0 = left + column * cell
            y0 = top + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                fill=color,
                outline="black",
                width=3,
            )
            centered_text(
                draw,
                (x0 + cell / 2, y0 + cell / 2),
                str(matrix[row, column]),
                value_font,
            )
    centered_text(
        draw, (left + cell / 2, 590), "Pred invariant", font
    )
    centered_text(
        draw, (left + 1.5 * cell, 590), "Pred environment", font
    )
    centered_text(
        draw, (105, top + cell / 2), "Oracle\ninvariant", font
    )
    centered_text(
        draw, (105, top + 1.5 * cell), "Oracle\nenvironment", font
    )
    image.save(path)


def main():
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"required checkpoint does not exist: {checkpoint_path}"
        )
    output_dir = (
        Path(cli.output_dir).resolve()
        if cli.output_dir
        else checkpoint_path.parent / "three_route_counterfactual_test_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(str(checkpoint_path), "cpu")
    checkpoint_args = dict(checkpoint.get("args", {}))
    config = dict(checkpoint_args)
    config.update(
        {
            "device": cli.device,
            "batch_size": cli.batch_size,
            "test_batch_size": cli.batch_size,
            "resume": False,
            "fpem_observable_load_prior_cache": str(
                Path(cli.cache).resolve()
            ),
            "fpem_ignore_future_c": True,
        }
    )
    args = SimpleNamespace(**config)
    if not bool(
        getattr(args, "fpem_use_observable_load_prior", False)
    ):
        raise AssertionError(
            "checkpoint is not an observable-load-prior model"
        )
    if not bool(
        getattr(args, "fpem_use_hard_environment_router", False)
    ):
        raise AssertionError(
            "checkpoint does not enable the hard environment router"
        )
    if not bool(getattr(args, "fpem_ignore_future_c", False)):
        raise AssertionError("fpem_ignore_future_c must be true")

    loaders, scaler, data_counts = build_tds_data(args)
    test_loader = loaders["test_mixed"]
    sampler_name = type(test_loader.sampler).__name__
    if not isinstance(test_loader.sampler, SequentialSampler):
        raise AssertionError(
            f"test loader is not sequential: {sampler_name}"
        )
    graph = load_graph(args.graph_file, device=args.device)
    model, _ = build_model(args, graph)
    load_result = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {load_result}")
    load_level_state = checkpoint.get("fpem_load_level_state")
    if load_level_state and hasattr(
        model, "load_load_level_state_from_checkpoint"
    ):
        model.load_load_level_state_from_checkpoint(load_level_state)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if hasattr(model, "set_fpem_epoch"):
        model.set_fpem_epoch(checkpoint_epoch)
    model.eval()

    state_before = snapshot_state(model)
    digest_before = state_digest(state_before)
    collected = {
        name: []
        for name in (
            "target_normalized",
            "all_environment_normalized",
            "all_invariant_normalized",
            "selective_normalized",
            "sample_index",
            "load_level",
            "predicted_route",
        )
    }
    forward_calls = 0
    selective_identity_max_abs_difference = 0.0
    native_y_final_exact = True
    cached_level_matches_output = True

    with torch.no_grad():
        for raw_batch in test_loader:
            unpacked = unpack_tds_batch(
                to_device(raw_batch, args.device)
            )
            if unpacked["stored_c"] is not None:
                raise AssertionError(
                    "stored future c is present in inference batch"
                )
            # Exactly one model forward for this batch.
            output = forward_once(model, unpacked)
            forward_calls += 1

            y_all_environment = output["y_env_selected"]
            y_all_invariant = output["y_inv"]
            y_selective = output["prediction"]
            predicted_route = output["hard_route_id"].detach().long()
            if not torch.all(
                (predicted_route == 0) | (predicted_route == 1)
            ):
                raise AssertionError(
                    "hard route contains values outside {0,1}"
                )
            route_view = predicted_route.view(
                -1, *([1] * (y_selective.ndim - 1))
            )
            expected_selective = torch.where(
                route_view == 0,
                y_all_invariant,
                y_all_environment,
            )
            maximum = float(
                (y_selective - expected_selective).abs().max().item()
            )
            selective_identity_max_abs_difference = max(
                selective_identity_max_abs_difference, maximum
            )
            if not torch.allclose(
                y_selective,
                expected_selective,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise AssertionError(
                    "native prediction is not the verified hard selection"
                )
            native_y_final_exact = native_y_final_exact and torch.equal(
                y_selective, output["y_final"]
            )
            cached_level_matches_output = (
                cached_level_matches_output
                and torch.equal(
                    output["load_level"].detach().long(),
                    unpacked["cached_load_level"].detach().long(),
                )
            )

            # Target is collected only after every prediction is complete.
            collected["target_normalized"].append(
                unpacked["target"].detach().cpu()
            )
            collected["all_environment_normalized"].append(
                y_all_environment.detach().cpu()
            )
            collected["all_invariant_normalized"].append(
                y_all_invariant.detach().cpu()
            )
            collected["selective_normalized"].append(
                y_selective.detach().cpu()
            )
            collected["sample_index"].append(
                unpacked["sample_index"].detach().cpu()
            )
            collected["load_level"].append(
                output["load_level"].detach().cpu()
            )
            collected["predicted_route"].append(
                predicted_route.detach().cpu()
            )

    normalized = {
        name: torch.cat(parts, dim=0)
        for name, parts in collected.items()
    }
    sample_index = normalized["sample_index"].numpy().astype(np.int64)
    load_level = normalized["load_level"].numpy().astype(np.int64)
    predicted_route = (
        normalized["predicted_route"].numpy().astype(np.int64)
    )
    sample_count = int(sample_index.size)
    if sample_count != cli.expected_samples:
        raise AssertionError(
            f"expected {cli.expected_samples} test samples, got {sample_count}"
        )
    if not np.array_equal(
        sample_index, np.arange(sample_count, dtype=np.int64)
    ):
        raise AssertionError(
            "test sample indexes are not deterministic 0..N-1 order"
        )
    expected_forward_calls = math.ceil(
        sample_count / float(cli.batch_size)
    )
    if forward_calls != expected_forward_calls:
        raise AssertionError(
            f"expected {expected_forward_calls} forwards, got {forward_calls}"
        )

    target = inverse_transform(
        scaler, normalized["target_normalized"]
    )
    predictions = {
        "all_environment": inverse_transform(
            scaler, normalized["all_environment_normalized"]
        ),
        "all_invariant": inverse_transform(
            scaler, normalized["all_invariant_normalized"]
        ),
        "selective_hard_routing": inverse_transform(
            scaler, normalized["selective_normalized"]
        ),
    }

    yita = float(getattr(args, "yita", 0.5))
    summary_rows = []
    sample_errors = {}
    for name, display_name in STRATEGIES:
        metrics, errors = strategy_metrics(
            predictions[name], target, yita
        )
        summary_rows.append(
            {
                "strategy": name,
                "display_name": display_name,
                **metrics,
            }
        )
        sample_errors[name] = errors

    pair_specs = (
        ("all_environment", "all_invariant"),
        ("all_environment", "selective_hard_routing"),
        ("all_invariant", "selective_hard_routing"),
    )
    pairwise_rows = []
    pairwise_winner = {}
    for first, second in pair_specs:
        row, winner = pairwise_wins(
            first,
            second,
            sample_errors[first],
            sample_errors[second],
        )
        pairwise_rows.append(row)
        pairwise_winner[(first, second)] = winner

    inv_error = sample_errors["all_invariant"]
    env_error = sample_errors["all_environment"]
    relative_margin = float(
        getattr(args, "fpem_hard_router_relative_margin", 0.01)
    )
    margin = relative_margin * inv_error
    oracle_route_margin = (
        env_error + margin < inv_error
    ).astype(np.int64)
    oracle_route_argmin = (env_error < inv_error).astype(np.int64)
    primary_router = binary_router_metrics(
        predicted_route, oracle_route_margin
    )
    no_margin_router = binary_router_metrics(
        predicted_route, oracle_route_argmin
    )
    router_report = {
        "route_convention": {
            "0": "invariant",
            "1": "environment",
        },
        "primary_oracle": "margin_aware_training_rule",
        "relative_margin": relative_margin,
        "margin_formula": "relative_margin * L_inv",
        "margin_aware": primary_router,
        "no_margin_argmin": {
            "correct_route_count": no_margin_router[
                "correct_route_count"
            ],
            "incorrect_route_count": no_margin_router[
                "incorrect_route_count"
            ],
            "route_accuracy": no_margin_router["route_accuracy"],
            "oracle_environment_ratio": no_margin_router[
                "oracle_environment_ratio"
            ],
            "total_samples": sample_count,
        },
    }

    # Determinism replay: data only, no second model forward.
    replay_indexes = []
    replay_targets = []
    replay_stored_c_absent = True
    for raw_batch in test_loader:
        unpacked = unpack_tds_batch(raw_batch)
        replay_stored_c_absent = (
            replay_stored_c_absent
            and unpacked["stored_c"] is None
        )
        replay_indexes.append(
            unpacked["sample_index"].detach().cpu()
        )
        replay_targets.append(unpacked["target"].detach().cpu())
    replay_indexes = torch.cat(replay_indexes)
    replay_targets = torch.cat(replay_targets)
    deterministic_indexes = torch.equal(
        replay_indexes, normalized["sample_index"]
    )
    deterministic_targets = torch.equal(
        replay_targets, normalized["target_normalized"]
    )
    if not (
        deterministic_indexes
        and deterministic_targets
        and replay_stored_c_absent
    ):
        raise AssertionError("test loader replay is not deterministic")

    state_after = snapshot_state(model)
    digest_after = state_digest(state_after)
    state_unchanged, changed_state_names = compare_state(
        state_before, state_after
    )
    if not state_unchanged:
        raise AssertionError(
            f"model parameters/buffers changed: {changed_state_names}"
        )

    sample_rows = []
    for row_index in range(sample_count):
        sample_rows.append(
            {
                "sample_index": int(sample_index[row_index]),
                "load_level": int(load_level[row_index]),
                "predicted_route": int(predicted_route[row_index]),
                "oracle_route_margin": int(
                    oracle_route_margin[row_index]
                ),
                "oracle_route_argmin": int(
                    oracle_route_argmin[row_index]
                ),
                "mae_all_environment": float(
                    env_error[row_index]
                ),
                "mae_all_invariant": float(
                    inv_error[row_index]
                ),
                "mae_selective": float(
                    sample_errors["selective_hard_routing"][row_index]
                ),
                "all_environment_vs_invariant_winner": pairwise_winner[
                    ("all_environment", "all_invariant")
                ][row_index],
                "all_environment_vs_selective_winner": pairwise_winner[
                    (
                        "all_environment",
                        "selective_hard_routing",
                    )
                ][row_index],
                "all_invariant_vs_selective_winner": pairwise_winner[
                    (
                        "all_invariant",
                        "selective_hard_routing",
                    )
                ][row_index],
            }
        )

    write_csv(
        output_dir / "samplewise_three_route_comparison.csv",
        list(sample_rows[0]),
        sample_rows,
    )
    write_csv(
        output_dir / "summary_three_strategies.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    write_csv(
        output_dir / "summary_pairwise_wins.csv",
        list(pairwise_rows[0]),
        pairwise_rows,
    )
    np.savez_compressed(
        output_dir / "predictions_three_routes.npz",
        target=target,
        y_all_environment=predictions["all_environment"],
        y_all_invariant=predictions["all_invariant"],
        y_selective=predictions["selective_hard_routing"],
        sample_index=sample_index,
        load_level=load_level,
        predicted_route=predicted_route,
        oracle_route_margin=oracle_route_margin,
        oracle_route_argmin=oracle_route_argmin,
    )
    with (
        output_dir / "summary_router_correctness.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(router_report, handle, indent=2, sort_keys=True)

    metadata = {
        "audit_type": "inference_only_three_route_counterfactual",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_args": json_safe(checkpoint_args),
        "cache": str(Path(cli.cache).resolve()),
        "cache_metadata": data_counts.get("observable_load_prior"),
        "output_dir": str(output_dir),
        "device": cli.device,
        "batch_size": cli.batch_size,
        "sample_count": sample_count,
        "yita": yita,
        "relative_margin": relative_margin,
        "tie_tolerance": TIE_TOLERANCE,
        "test_loader_sampler": sampler_name,
        "forward_calls": forward_calls,
        "one_forward_per_batch": (
            forward_calls == expected_forward_calls
        ),
        "validation": {
            "model_eval": not model.training,
            "torch_no_grad_used": True,
            "stored_future_c_absent": replay_stored_c_absent,
            "target_passed_to_forward": False,
            "route_convention_verified": True,
            "selective_matches_hard_where_atol_1e_5": True,
            "selective_identity_max_abs_difference": (
                selective_identity_max_abs_difference
            ),
            "prediction_equals_y_final_exactly": native_y_final_exact,
            "cached_load_level_matches_output": (
                cached_level_matches_output
            ),
            "identical_sample_order_and_targets": True,
            "test_loader_shuffle_false": True,
            "test_loader_replay_indexes_identical": (
                deterministic_indexes
            ),
            "test_loader_replay_targets_identical": (
                deterministic_targets
            ),
            "pairwise_counts_sum_to_total": all(
                row["first_win_count"]
                + row["second_win_count"]
                + row["tie_count"]
                == sample_count
                for row in pairwise_rows
            ),
            "model_parameters_and_buffers_unchanged": (
                state_unchanged
            ),
            "changed_state_names": changed_state_names,
            "state_digest_before": digest_before,
            "state_digest_after": digest_after,
        },
        "strategy_summary": summary_rows,
        "pairwise_summary": pairwise_rows,
        "router_summary": router_report,
        "metric_notes": {
            "masked_weighted": (
                "mask=target[...,0]>5; error=yita*abs(flow_error)"
                "+(1-yita)*abs(speed_error)"
            ),
            "standard": (
                "lib.metrics.test_metrics on inverse-transformed arrays"
            ),
        },
    }
    with (output_dir / "audit_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    plot_pairwise(
        pairwise_rows, output_dir / "pairwise_win_proportions.png"
    )
    plot_strategy_mae(
        summary_rows, output_dir / "three_strategy_mae.png"
    )
    plot_confusion(
        primary_router, output_dir / "router_oracle_confusion.png"
    )

    print(
        json.dumps(
            {
                "checkpoint_epoch": checkpoint_epoch,
                "sample_count": sample_count,
                "strategies": summary_rows,
                "pairwise": pairwise_rows,
                "router": router_report,
                "validation": metadata["validation"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
