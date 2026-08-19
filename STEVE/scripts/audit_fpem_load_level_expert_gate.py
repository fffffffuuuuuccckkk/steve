#!/usr/bin/env python
"""Inference-only audit for traffic-load-aligned experts and environment gating."""

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.utils import load_graph
from run_tds_nyctaxi import build_model, build_tds_data, load_checkpoint, to_device


def masked_sample_error(pred, target, yita):
    error = float(yita) * np.abs(pred[..., 0] - target[..., 0])
    error += (1.0 - float(yita)) * np.abs(pred[..., 1] - target[..., 1])
    mask = target[..., 0] > 5.0
    flat_error = error.reshape(error.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    numerator = (flat_error * flat_mask).sum(axis=1)
    denominator = flat_mask.sum(axis=1)
    fallback = flat_error.mean(axis=1)
    sample = np.where(denominator > 0, numerator / np.maximum(denominator, 1), fallback)
    global_mae = float((flat_error * flat_mask).sum() / max(float(flat_mask.sum()), 1.0))
    return sample, global_mae


def strategy_metrics(prediction, target, yita):
    sample_error, global_mae = masked_sample_error(prediction, target, yita)
    return {
        "global_masked_mae": global_mae,
        "samplewise_mean_mae": float(sample_error.mean()),
    }, sample_error


def safe_div(numerator, denominator):
    return float(numerator) / max(float(denominator), 1.0)


def save_matrix_plot(matrix, labels_x, labels_y, title, path, fmt=".3f"):
    width, height = 760, 620
    left, top, right, bottom = 150, 80, 30, 140
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 24), title, fill="black", font=font)
    values = np.asarray(matrix, dtype=np.float64)
    finite = values[np.isfinite(values)]
    minimum = float(finite.min()) if finite.size else 0.0
    maximum = float(finite.max()) if finite.size else 1.0
    cell_width = (width - left - right) / max(values.shape[1], 1)
    cell_height = (height - top - bottom) / max(values.shape[0], 1)
    for row in range(values.shape[0]):
        draw.text((8, top + row * cell_height + cell_height / 2), str(labels_y[row]), fill="black", font=font)
        for col in range(values.shape[1]):
            value = float(values[row, col])
            ratio = 0.0 if not np.isfinite(value) else (value - minimum) / max(maximum - minimum, 1e-12)
            color = (
                int(40 + 210 * ratio),
                int(30 + 180 * (1.0 - abs(ratio - 0.5) * 2.0)),
                int(160 - 120 * ratio),
            )
            x0 = left + col * cell_width
            y0 = top + row * cell_height
            draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_height), fill=color, outline="white")
            text = "NA" if not np.isfinite(value) else format(value, fmt)
            draw.text((x0 + 8, y0 + cell_height / 2), text, fill="white", font=font)
    for col, label in enumerate(labels_x):
        draw.text((left + col * cell_width + 4, height - bottom + 20), str(label), fill="black", font=font)
    image.save(path)


def save_gate_histogram(gate, level, expert_count, path):
    width, height = 760, 480
    left, top, right, bottom = 70, 50, 30, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bins = np.linspace(0.0, 1.0, 21)
    colors = [(45, 110, 220), (40, 170, 90), (220, 90, 55), (155, 80, 190)]
    histograms = []
    max_count = 1
    for load_level in range(expert_count):
        values = gate[level == load_level]
        histogram = np.histogram(values, bins=bins)[0] if values.size else np.zeros(20, dtype=np.int64)
        histograms.append(histogram)
        max_count = max(max_count, int(histogram.max()))
    plot_width = width - left - right
    plot_height = height - top - bottom
    draw.line((left, top, left, top + plot_height), fill="black", width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="black", width=2)
    bin_width = plot_width / 20.0
    group_width = bin_width / max(expert_count, 1)
    for load_level, histogram in enumerate(histograms):
        color = colors[load_level % len(colors)]
        for idx, count in enumerate(histogram):
            bar_height = plot_height * float(count) / float(max_count)
            x0 = left + idx * bin_width + load_level * group_width
            y0 = top + plot_height - bar_height
            draw.rectangle((x0, y0, x0 + group_width, top + plot_height), fill=color)
        draw.rectangle((left + 110 * load_level, 18, left + 110 * load_level + 12, 30), fill=color)
        draw.text((left + 110 * load_level + 16, 17), f"level {load_level}", fill="black", font=font)
    draw.text((left + plot_width / 2 - 60, height - 30), "environment-use gate", fill="black", font=font)
    image.save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=-1)
    return parser.parse_args()


def main():
    cli = parse_args()
    os.makedirs(cli.output_dir, exist_ok=True)
    checkpoint = load_checkpoint(cli.checkpoint, "cpu")
    config = dict(checkpoint.get("args", {}))
    config["device"] = cli.device
    config["test_batch_size"] = cli.batch_size
    config["batch_size"] = cli.batch_size
    config["resume"] = False
    args = SimpleNamespace(**config)
    loaders, scaler, data_counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, _lr = build_model(args, graph)
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print("CHECKPOINT_LOAD_WARNING", result)
    state = checkpoint.get("fpem_load_level_state")
    if state and hasattr(model, "load_load_level_state_from_checkpoint"):
        model.load_load_level_state_from_checkpoint(state)
    if hasattr(model, "set_fpem_epoch"):
        model.set_fpem_epoch(int(checkpoint.get("epoch", 0)))
    model.eval()

    target_parts = []
    inv_parts = []
    head_parts = []
    selected_parts = []
    final_parts = []
    gate_parts = []
    level_parts = []
    score_parts = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loaders["test_mixed"]):
            if cli.max_batches >= 0 and batch_index >= cli.max_batches:
                break
            batch = to_device(raw_batch, args.device)
            if len(batch) == 5:
                data, target, time_label, c, sample_index = batch
            else:
                data, target, time_label, c = batch
                sample_index = None
            # Target is deliberately not an argument to forward_output.
            output = model.forward_output(
                data,
                exog=c,
                time_label=time_label,
                training=False,
                sample_index=sample_index,
            )
            target_parts.append(target.detach().cpu())
            inv_parts.append(output["y_inv"].detach().cpu())
            head_parts.append(output["y_env_heads"].detach().cpu())
            selected_parts.append(output["y_env_selected"].detach().cpu())
            final_parts.append(output["y_final"].detach().cpu())
            gate_parts.append(output["env_use_gate"].detach().cpu())
            level_parts.append(output["load_level"].detach().cpu())
            score_parts.append(output["observable_load_score"].detach().cpu())

    target = scaler.inverse_transform(torch.cat(target_parts)).numpy()
    y_inv = scaler.inverse_transform(torch.cat(inv_parts)).numpy()
    y_heads_tensor = torch.cat(head_parts)
    y_heads = scaler.inverse_transform(y_heads_tensor).numpy()
    y_selected = scaler.inverse_transform(torch.cat(selected_parts)).numpy()
    y_final = scaler.inverse_transform(torch.cat(final_parts)).numpy()
    gate = torch.cat(gate_parts).numpy().reshape(-1)
    level = torch.cat(level_parts).numpy().astype(np.int64).reshape(-1)
    load_score = torch.cat(score_parts).numpy().reshape(-1)
    yita = float(getattr(args, "yita", 0.5))
    sample_count, expert_count = y_heads.shape[:2]

    strategies = {
        "native load-level gate": y_final,
        "always invariant": y_inv,
        "always environment": y_selected,
        "uniform environment experts": y_heads.mean(axis=1),
    }
    for expert in range(expert_count):
        strategies[f"fixed environment expert {expert}"] = y_heads[:, expert]

    strategy_summary = {}
    strategy_sample_errors = {}
    for name, prediction in strategies.items():
        strategy_summary[name], strategy_sample_errors[name] = strategy_metrics(
            prediction, target, yita
        )

    inv_error = strategy_sample_errors["always invariant"]
    selected_error = strategy_sample_errors["always environment"]
    head_error = np.stack(
        [masked_sample_error(y_heads[:, expert], target, yita)[0] for expert in range(expert_count)],
        axis=1,
    )
    prefer_environment = selected_error < inv_error
    oracle_load_uses_environment = prefer_environment
    oracle_load_prediction = np.where(
        oracle_load_uses_environment.reshape(-1, 1, 1, 1),
        y_selected,
        y_inv,
    )
    strategy_summary["oracle invariant-vs-load-expert gate"], strategy_sample_errors[
        "oracle invariant-vs-load-expert gate"
    ] = strategy_metrics(oracle_load_prediction, target, yita)

    all_candidate_error = np.concatenate([inv_error[:, None], head_error], axis=1)
    oracle_all_index = all_candidate_error.argmin(axis=1)
    all_candidates = np.concatenate([y_inv[:, None], y_heads], axis=1)
    oracle_all_prediction = all_candidates[np.arange(sample_count), oracle_all_index]
    strategy_summary["oracle among all candidates"], strategy_sample_errors[
        "oracle among all candidates"
    ] = strategy_metrics(oracle_all_prediction, target, yita)

    native_use_environment = gate >= 0.5
    true_positive = int(np.sum(native_use_environment & prefer_environment))
    false_positive = int(np.sum(native_use_environment & ~prefer_environment))
    false_negative = int(np.sum(~native_use_environment & prefer_environment))
    true_negative = int(np.sum(~native_use_environment & ~prefer_environment))
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    specificity = safe_div(true_negative, true_negative + false_positive)
    balanced_accuracy = 0.5 * (recall + specificity)
    native_mae = strategy_summary["native load-level gate"]["samplewise_mean_mae"]
    invariant_mae = strategy_summary["always invariant"]["samplewise_mean_mae"]
    oracle_load_mae = strategy_summary["oracle invariant-vs-load-expert gate"]["samplewise_mean_mae"]
    overall = {
        "invariant_preferred_ratio": float((~prefer_environment).mean()),
        "environment_preferred_ratio": float(prefer_environment.mean()),
        "native_gate_environment_use_ratio": float(native_use_environment.mean()),
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": f1,
        "gate_balanced_accuracy": balanced_accuracy,
        "harmful_environment_corrections_avoided": true_negative,
        "harmful_environment_corrections_avoided_ratio": safe_div(true_negative, np.sum(~prefer_environment)),
        "missed_environment_gains": false_negative,
        "missed_environment_gains_ratio": safe_div(false_negative, np.sum(prefer_environment)),
        "native_vs_oracle_regret": native_mae - oracle_load_mae,
        "oracle_gap_closed": (
            (invariant_mae - native_mae) / max(invariant_mae - oracle_load_mae, 1e-12)
            if invariant_mae > oracle_load_mae
            else 0.0
        ),
    }

    per_level = {}
    cross_mae = np.full((expert_count, expert_count), np.nan, dtype=np.float64)
    for load_level in range(expert_count):
        mask = level == load_level
        if not bool(mask.any()):
            continue
        per_level[str(load_level)] = {
            "count": int(mask.sum()),
            "invariant_mae": float(inv_error[mask].mean()),
            "corresponding_expert_mae": float(selected_error[mask].mean()),
            "final_gated_mae": float(strategy_sample_errors["native load-level gate"][mask].mean()),
            "environment_preferred_ratio": float(prefer_environment[mask].mean()),
            "mean_gate_value": float(gate[mask].mean()),
        }
        for expert in range(expert_count):
            cross_mae[load_level, expert] = float(head_error[mask, expert].mean())

    thresholds = (
        state.get("thresholds", [])
        if state
        else model.load_level_thresholds.detach().cpu().tolist()
    )
    threshold_summary = {
        "thresholds": thresholds,
        "mode": getattr(args, "fpem_load_level_mode", "train_quantile"),
        "fit_split": getattr(args, "fpem_load_level_threshold_fit_split", "train"),
        "source": getattr(args, "fpem_load_level_source", None),
        "uses_target_or_future_load": False,
        "checkpoint_thresholds_identical_to_model": bool(
            np.array_equal(
                np.asarray(thresholds, dtype=np.float32),
                model.load_level_thresholds.detach().cpu().numpy(),
            )
        ),
        "data_counts": data_counts,
    }
    with open(os.path.join(cli.output_dir, "load_level_thresholds.json"), "w", encoding="utf-8") as handle:
        json.dump(threshold_summary, handle, ensure_ascii=False, indent=2)

    summary = {
        "checkpoint": os.path.abspath(cli.checkpoint),
        "selected_by": "best validation checkpoint",
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "strategies": strategy_summary,
        "overall": overall,
        "per_load_level": per_level,
        "thresholds": threshold_summary,
    }
    with open(os.path.join(cli.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(cli.output_dir, "summary.tsv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["section", "name", "global_masked_mae", "samplewise_mean_mae", "value"])
        for name, metrics in strategy_summary.items():
            writer.writerow(["strategy", name, metrics["global_masked_mae"], metrics["samplewise_mean_mae"], ""])
        for name, value in overall.items():
            writer.writerow(["gate", name, "", "", value])

    sample_columns = [
        "sample_index",
        "load_score",
        "load_level",
        "gate_value",
        "environment_preferred",
        "native_use_environment",
        "invariant_mae",
        "corresponding_expert_mae",
        "native_gate_mae",
        "oracle_load_gate_mae",
        "oracle_all_mae",
    ]
    with open(os.path.join(cli.output_dir, "samplewise_results.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_columns)
        writer.writeheader()
        for index in range(sample_count):
            writer.writerow({
                "sample_index": index,
                "load_score": load_score[index],
                "load_level": level[index],
                "gate_value": gate[index],
                "environment_preferred": int(prefer_environment[index]),
                "native_use_environment": int(native_use_environment[index]),
                "invariant_mae": inv_error[index],
                "corresponding_expert_mae": selected_error[index],
                "native_gate_mae": strategy_sample_errors["native load-level gate"][index],
                "oracle_load_gate_mae": strategy_sample_errors["oracle invariant-vs-load-expert gate"][index],
                "oracle_all_mae": strategy_sample_errors["oracle among all candidates"][index],
            })

    with open(os.path.join(cli.output_dir, "load_level_expert_cross_mae.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["load_level"] + [f"expert_{idx}" for idx in range(expert_count)])
        for load_level in range(expert_count):
            writer.writerow([load_level] + cross_mae[load_level].tolist())

    confusion = np.asarray([[true_negative, false_positive], [false_negative, true_positive]], dtype=np.float64)
    save_matrix_plot(
        confusion,
        ["native invariant", "native environment"],
        ["environment harmful", "environment helpful"],
        "Environment gate confusion matrix",
        os.path.join(cli.output_dir, "gate_confusion_matrix.png"),
        fmt=".0f",
    )

    save_gate_histogram(
        gate,
        level,
        expert_count,
        os.path.join(cli.output_dir, "load_level_gate_distribution.png"),
    )

    delta = y_heads - y_inv[:, None]
    delta_flat = delta.transpose(1, 0, 2, 3, 4).reshape(expert_count, -1)
    delta_norm = np.linalg.norm(delta_flat, axis=1, keepdims=True)
    delta_cosine = (delta_flat @ delta_flat.T) / np.maximum(delta_norm @ delta_norm.T, 1e-12)
    save_matrix_plot(
        delta_cosine,
        [f"expert {idx}" for idx in range(expert_count)],
        [f"expert {idx}" for idx in range(expert_count)],
        "Expert correction delta cosine",
        os.path.join(cli.output_dir, "expert_delta_similarity.png"),
    )
    save_matrix_plot(
        cross_mae,
        [f"expert {idx}" for idx in range(expert_count)],
        [f"load {idx}" for idx in range(expert_count)],
        "Load-level expert cross MAE",
        os.path.join(cli.output_dir, "load_level_expert_cross_mae.png"),
    )
    print(json.dumps({"summary": os.path.join(cli.output_dir, "summary.json"), **overall}, sort_keys=True))


if __name__ == "__main__":
    main()
