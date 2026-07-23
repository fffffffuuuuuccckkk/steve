#!/usr/bin/env python3
"""Masked sample-wise MAE comparison for three original STEVE variants."""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(project_root(), path)


def load_result(exp_dir):
    path = os.path.join(exp_dir, "sample_predictions.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    required = ["prediction", "target", "sample_index"]
    for key in required:
        if key not in data:
            raise KeyError("{} missing {}".format(path, key))
    pred = data["prediction"]
    target = data["target"]
    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    if target.ndim == 3:
        target = target[:, None, :, :]
    return {
        "prediction": pred.astype(np.float64),
        "target": target.astype(np.float64),
        "sample_index": data["sample_index"].astype(np.int64),
    }


def sample_unmasked_mae(pred, target):
    return np.abs(pred - target).mean(axis=tuple(range(1, pred.ndim)))


def sample_masked_mae(pred, target, mask_value):
    err = np.abs(pred - target)
    mask = target > float(mask_value)
    axes = tuple(range(1, pred.ndim))
    count = mask.sum(axis=axes)
    summed = np.where(mask, err, 0.0).sum(axis=axes)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    global_mae = float(err[mask].mean()) if bool(mask.any()) else float("nan")
    return out, count, global_mae


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def simple_canvas(width=1000, height=620):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None, None
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def axes(draw, width, height, margin=70):
    left, top, right, bottom = margin, margin, width - margin, height - margin
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    return left, top, right, bottom


def save_sorted_curves(values_by_model, path):
    image, draw = simple_canvas()
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = axes(draw, width, height)
    finite = np.concatenate([v[np.isfinite(v)] for v in values_by_model.values()])
    if finite.size == 0:
        return False
    ymin, ymax = float(finite.min()), float(finite.max())
    if abs(ymax - ymin) < 1e-12:
        ymin -= 1.0
        ymax += 1.0
    colors = {
        "full": (45, 105, 190),
        "inv_disentangle": (50, 155, 95),
        "inv_no_env": (215, 95, 85),
    }
    for model_name, values in values_by_model.items():
        values = np.sort(values[np.isfinite(values)])
        if values.size == 0:
            continue
        pts = []
        denom = max(values.size - 1, 1)
        for i, value in enumerate(values):
            x = left + i / denom * (right - left)
            y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
            pts.append((x, y))
        draw.line(pts, fill=colors.get(model_name, (80, 80, 80)), width=2)
    legend_x = left
    for j, model_name in enumerate(values_by_model):
        color = colors.get(model_name, (80, 80, 80))
        y = 18 + 20 * j
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=color)
        draw.text((legend_x + 20, y), model_name, fill="black")
    draw.text((left, bottom + 18), "sorted masked sample-wise MAE; lower is better", fill="black")
    image.save(path)
    return True


def save_best_bar(counts, path):
    image, draw = simple_canvas(900, 560)
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = axes(draw, width, height)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    max_count = max(max(values), 1)
    gap = 36
    bar_w = (right - left - gap * (len(labels) + 1)) / max(len(labels), 1)
    colors = [(45, 105, 190), (50, 155, 95), (215, 95, 85), (150, 150, 150)]
    for i, (label, value) in enumerate(zip(labels, values)):
        x0 = left + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - (bottom - top) * float(value) / max_count
        draw.rectangle((x0, y0, x1, bottom), fill=colors[i % len(colors)], outline="black")
        draw.text((x0, y0 - 22), str(int(value)), fill="black")
        draw.text((x0, bottom + 12), label, fill="black")
    draw.text((left, 22), "per-sample best model count, masked MAE", fill="black")
    image.save(path)
    return True


def choose_best(row_values, names, threshold):
    values = np.asarray(row_values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return "invalid", float("nan")
    best = float(values.min())
    winners = [names[i] for i, value in enumerate(values) if value <= best + threshold]
    sorted_values = np.sort(values)
    margin = float(sorted_values[1] - sorted_values[0]) if len(sorted_values) > 1 else 0.0
    if len(winners) == 1:
        return winners[0], margin
    return "tie:" + "+".join(winners), margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", default="experiments/NYCTaxi_TDS/steve_full_reproduce_seed2024")
    parser.add_argument("--inv_disentangle_dir", default="experiments/NYCTaxi_TDS/steve_inv_only_with_disentangle_seed2024")
    parser.add_argument("--inv_no_env_dir", default="experiments/NYCTaxi_TDS/steve_inv_only_no_env_seed2024")
    parser.add_argument("--output_dir", default="experiments/NYCTaxi_TDS/steve_three_model_masked_samplewise_comparison_seed2024")
    parser.add_argument("--mask_value", type=float, default=5.0)
    parser.add_argument("--tie_threshold", type=float, default=1e-6)
    args = parser.parse_args()

    dirs = OrderedDict(
        [
            ("full", resolve(args.full_dir)),
            ("inv_disentangle", resolve(args.inv_disentangle_dir)),
            ("inv_no_env", resolve(args.inv_no_env_dir)),
        ]
    )
    outputs = OrderedDict((name, load_result(path)) for name, path in dirs.items())
    names = list(outputs.keys())
    ref = outputs[names[0]]
    for name, item in outputs.items():
        assert np.array_equal(ref["sample_index"], item["sample_index"]), "sample_index mismatch: {}".format(name)
        assert ref["prediction"].shape == item["prediction"].shape, "prediction shape mismatch: {}".format(name)
        assert ref["target"].shape == item["target"].shape, "target shape mismatch: {}".format(name)
        assert np.allclose(ref["target"], item["target"]), "target mismatch: {}".format(name)

    sample_index = ref["sample_index"]
    target = ref["target"]
    masked = OrderedDict()
    unmasked = OrderedDict()
    valid_count = None
    global_masked = OrderedDict()
    for name, item in outputs.items():
        unmasked[name] = sample_unmasked_mae(item["prediction"], target)
        masked[name], count, global_mae = sample_masked_mae(item["prediction"], target, args.mask_value)
        global_masked[name] = global_mae
        valid_count = count if valid_count is None else valid_count

    masked_matrix = np.stack([masked[name] for name in names], axis=1)
    valid = np.all(np.isfinite(masked_matrix), axis=1)
    best_rows = []
    best_counts = OrderedDict((name, 0) for name in names)
    best_counts["tie"] = 0
    best_counts["invalid"] = 0
    for i, idx in enumerate(sample_index):
        best_model, margin = choose_best(masked_matrix[i], names, args.tie_threshold)
        if best_model in best_counts:
            best_counts[best_model] += 1
        elif best_model.startswith("tie:"):
            best_counts["tie"] += 1
        else:
            best_counts["invalid"] += 1
        row = OrderedDict()
        row["sample_index"] = int(idx)
        row["valid_count_masked"] = int(valid_count[i])
        for name in names:
            row["mae_masked_{}".format(name)] = "" if not np.isfinite(masked[name][i]) else float(masked[name][i])
            row["mae_unmasked_{}".format(name)] = float(unmasked[name][i])
        row["gain_full_vs_inv_disentangle"] = "" if not valid[i] else float(masked["inv_disentangle"][i] - masked["full"][i])
        row["gain_full_vs_inv_no_env"] = "" if not valid[i] else float(masked["inv_no_env"][i] - masked["full"][i])
        row["gain_inv_disentangle_vs_inv_no_env"] = "" if not valid[i] else float(masked["inv_no_env"][i] - masked["inv_disentangle"][i])
        row["best_model_masked"] = best_model
        row["best_margin_masked"] = "" if not np.isfinite(margin) else margin
        best_rows.append(row)

    summary_rows = []
    for name in names:
        row = OrderedDict()
        row["model"] = name
        row["exp_dir"] = dirs[name]
        row["num_samples"] = int(sample_index.shape[0])
        row["valid_samples"] = int(valid.sum())
        row["masked_sample_mae_mean"] = float(np.nanmean(masked[name]))
        row["masked_sample_mae_median"] = float(np.nanmedian(masked[name]))
        row["masked_sample_mae_q25"] = float(np.nanpercentile(masked[name], 25))
        row["masked_sample_mae_q75"] = float(np.nanpercentile(masked[name], 75))
        row["global_masked_mae"] = float(global_masked[name])
        row["unmasked_sample_mae_mean"] = float(np.mean(unmasked[name]))
        row["best_count"] = int(best_counts[name])
        row["best_prop"] = float(best_counts[name] / max(int(valid.sum()), 1))
        summary_rows.append(row)

    pair_rows = []
    pair_defs = [
        ("full", "inv_disentangle"),
        ("full", "inv_no_env"),
        ("inv_disentangle", "inv_no_env"),
    ]
    for a, b in pair_defs:
        diff = masked[b] - masked[a]
        row = OrderedDict()
        row["comparison"] = "{}_vs_{}".format(a, b)
        row["positive_means"] = "{} better".format(a)
        row["mean_gain"] = float(np.nanmean(diff))
        row["median_gain"] = float(np.nanmedian(diff))
        row["a_better_count"] = int(np.nansum(diff > args.tie_threshold))
        row["b_better_count"] = int(np.nansum(diff < -args.tie_threshold))
        row["tie_count"] = int(np.nansum(np.abs(diff) <= args.tie_threshold))
        row["valid_samples"] = int(np.isfinite(diff).sum())
        pair_rows.append(row)

    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "samplewise_masked_mae_comparison.csv"), best_rows, list(best_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_by_model.csv"), summary_rows, list(summary_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_pairwise.csv"), pair_rows, list(pair_rows[0].keys()))
    np.savez(
        os.path.join(output_dir, "samplewise_masked_mae_comparison.npz"),
        sample_index=sample_index,
        model_names=np.asarray(names),
        masked_mae=masked_matrix,
        unmasked_mae=np.stack([unmasked[name] for name in names], axis=1),
        valid_count_masked=valid_count,
        mask_value=np.asarray(float(args.mask_value)),
    )
    save_sorted_curves(masked, os.path.join(output_dir, "masked_mae_sorted_curves.png"))
    save_best_bar(best_counts, os.path.join(output_dir, "best_model_counts.png"))

    print("[steve-three-masked] output_dir={}".format(output_dir))
    for row in summary_rows:
        print("[steve-three-masked] model={} global_masked_mae={:.6f} sample_mean={:.6f} best_count={}".format(
            row["model"], row["global_masked_mae"], row["masked_sample_mae_mean"], row["best_count"]
        ))
    for row in pair_rows:
        print("[steve-three-masked] {} mean_gain={:.6f} a_better={} b_better={} tie={}".format(
            row["comparison"], row["mean_gain"], row["a_better_count"], row["b_better_count"], row["tie_count"]
        ))


if __name__ == "__main__":
    main()
