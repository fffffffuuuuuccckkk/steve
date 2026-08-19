#!/usr/bin/env python3
"""Compare sample-wise utility of original STEVE environment-aware prediction."""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np


def parse_seeds(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def split_dirs(text):
    if text is None or str(text).strip() == "":
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def resolve_project_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path, project):
    if os.path.isabs(path):
        return path
    return os.path.join(project, path)


def load_sample_dir(path):
    npz_path = os.path.join(path, "sample_predictions.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    required = ["prediction", "target", "sample_mae", "sample_mse", "sample_index"]
    for key in required:
        if key not in data:
            raise KeyError("{} missing key {}".format(npz_path, key))
    return {key: data[key] for key in required}


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def simple_png_canvas(width=900, height=560):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None, None
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def draw_axes(draw, margin, width, height):
    left, top, right, bottom = margin, margin, width - margin, height - margin
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    return left, top, right, bottom


def save_hist(values, path):
    image, draw = simple_png_canvas()
    if image is None:
        return False
    values = np.asarray(values, dtype=float)
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, 60, width, height)
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    counts, edges = np.histogram(values, bins=40, range=(vmin, vmax))
    max_count = max(int(counts.max()), 1)
    bar_w = (right - left) / len(counts)
    for i, count in enumerate(counts):
        x0 = left + i * bar_w
        x1 = left + (i + 1) * bar_w - 1
        y0 = bottom - (bottom - top) * float(count) / max_count
        draw.rectangle((x0, y0, x1, bottom), fill=(85, 130, 210), outline=(60, 90, 160))
    if vmin <= 0 <= vmax:
        xz = left + (0 - vmin) / (vmax - vmin) * (right - left)
        draw.line((xz, top, xz, bottom), fill="red", width=2)
    draw.text((left, 18), "mean_delta histogram (InvOnly MAE - Full MAE)", fill="black")
    draw.text((left, bottom + 15), "zero line in red; positive means Full better", fill="black")
    image.save(path)
    return True


def save_sorted_curve(values, path):
    image, draw = simple_png_canvas()
    if image is None:
        return False
    values = np.sort(np.asarray(values, dtype=float))
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, 60, width, height)
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    if vmin <= 0 <= vmax:
        yz = bottom - (0 - vmin) / (vmax - vmin) * (bottom - top)
        draw.line((left, yz, right, yz), fill="red", width=2)
    pts = []
    denom = max(len(values) - 1, 1)
    for i, value in enumerate(values):
        x = left + i / denom * (right - left)
        y = bottom - (value - vmin) / (vmax - vmin) * (bottom - top)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(45, 105, 190), width=2)
    draw.text((left, 18), "sorted mean_delta curve", fill="black")
    draw.text((left, bottom + 15), "below zero: InvOnly better; above zero: Full better", fill="black")
    image.save(path)
    return True


def save_consistency_bar(labels, counts, path):
    image, draw = simple_png_canvas()
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, 60, width, height)
    max_count = max(max(counts), 1)
    gap = 40
    bar_w = (right - left - gap * (len(counts) + 1)) / max(len(counts), 1)
    colors = [(65, 150, 90), (215, 95, 85), (150, 150, 150)]
    for i, (label, count) in enumerate(zip(labels, counts)):
        x0 = left + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - (bottom - top) * float(count) / max_count
        draw.rectangle((x0, y0, x1, bottom), fill=colors[i % len(colors)], outline="black")
        draw.text((x0, y0 - 22), str(int(count)), fill="black")
        draw.text((x0, bottom + 12), label, fill="black")
    draw.text((left, 18), "seed consistency categories", fill="black")
    image.save(path)
    return True


def save_scatter(x_values, y_values, path):
    image, draw = simple_png_canvas()
    if image is None:
        return False
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, 60, width, height)
    combined_min = float(min(x_values.min(), y_values.min()))
    combined_max = float(max(x_values.max(), y_values.max()))
    if abs(combined_max - combined_min) < 1e-12:
        combined_min -= 1.0
        combined_max += 1.0
    draw.line((left, bottom, right, top), fill="red", width=2)
    n = len(x_values)
    if n > 8000:
        rng = np.random.RandomState(0)
        keep = rng.choice(n, size=8000, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
    for x_val, y_val in zip(x_values, y_values):
        x = left + (x_val - combined_min) / (combined_max - combined_min) * (right - left)
        y = bottom - (y_val - combined_min) / (combined_max - combined_min) * (bottom - top)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(45, 105, 190))
    draw.text((left, 18), "Full vs InvOnly sample MAE", fill="black")
    draw.text((left, bottom + 15), "x=InvOnly, y=Full; below red y=x means Full better", fill="black")
    image.save(path)
    return True


def summarize_seed(seed, full, inv, tie_threshold, relative_threshold):
    full_idx = full["sample_index"]
    inv_idx = inv["sample_index"]
    assert np.array_equal(full_idx, inv_idx), "sample_index mismatch for seed {}".format(seed)
    assert full["prediction"].shape == inv["prediction"].shape, "prediction shape mismatch"
    assert full["target"].shape == inv["target"].shape, "target shape mismatch"
    assert np.allclose(full["target"], inv["target"]), "target mismatch for seed {}".format(seed)

    full_mae = full["sample_mae"].astype(float)
    inv_mae = inv["sample_mae"].astype(float)
    delta = inv_mae - full_mae
    full_better = delta > tie_threshold
    inv_better = delta < -tie_threshold
    tie = ~(full_better | inv_better)
    eps = 1e-12
    relative_gain = delta / np.maximum(inv_mae, eps)
    clear_full = relative_gain > relative_threshold
    clear_inv = relative_gain < -relative_threshold
    clear_small = ~(clear_full | clear_inv)
    row = OrderedDict()
    row["seed"] = seed
    row["num_samples"] = int(delta.shape[0])
    row["full_better_count"] = int(full_better.sum())
    row["full_better_prop"] = float(full_better.mean())
    row["inv_better_count"] = int(inv_better.sum())
    row["inv_better_prop"] = float(inv_better.mean())
    row["tie_count"] = int(tie.sum())
    row["tie_prop"] = float(tie.mean())
    row["mean_delta"] = float(delta.mean())
    row["median_delta"] = float(np.median(delta))
    row["std_delta"] = float(delta.std())
    row["q25_delta"] = float(np.percentile(delta, 25))
    row["q75_delta"] = float(np.percentile(delta, 75))
    row["full_mae"] = float(full_mae.mean())
    row["inv_only_mae"] = float(inv_mae.mean())
    row["oracle_mae"] = float(np.minimum(full_mae, inv_mae).mean())
    row["relative_full_clear_count"] = int(clear_full.sum())
    row["relative_full_clear_prop"] = float(clear_full.mean())
    row["relative_inv_clear_count"] = int(clear_inv.sum())
    row["relative_inv_clear_prop"] = float(clear_inv.mean())
    row["relative_small_count"] = int(clear_small.sum())
    row["relative_small_prop"] = float(clear_small.mean())
    return row, delta, relative_gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--full_dirs", default=None)
    parser.add_argument("--inv_dirs", default=None)
    parser.add_argument(
        "--full_template",
        default="experiments/NYCTaxi_TDS/steve_env_value_full_seed{seed}",
    )
    parser.add_argument(
        "--inv_template",
        default="experiments/NYCTaxi_TDS/steve_env_value_inv_only_seed{seed}",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/NYCTaxi_TDS/steve_samplewise_environment_analysis",
    )
    parser.add_argument("--tie_threshold", type=float, default=1e-6)
    parser.add_argument("--relative_threshold", type=float, default=0.01)
    args = parser.parse_args()

    project = resolve_project_path()
    seeds = parse_seeds(args.seeds)
    full_dirs = split_dirs(args.full_dirs)
    inv_dirs = split_dirs(args.inv_dirs)
    if full_dirs is not None and len(full_dirs) != len(seeds):
        raise ValueError("full_dirs length must match seeds length")
    if inv_dirs is not None and len(inv_dirs) != len(seeds):
        raise ValueError("inv_dirs length must match seeds length")

    output_dir = resolve_path(args.output_dir, project)
    os.makedirs(output_dir, exist_ok=True)

    seed_rows = []
    seed_full_mae = []
    seed_inv_mae = []
    seed_deltas = []
    seed_relative_gains = []
    sample_index = None
    target_ref = None

    for pos, seed in enumerate(seeds):
        full_dir = full_dirs[pos] if full_dirs is not None else args.full_template.format(seed=seed)
        inv_dir = inv_dirs[pos] if inv_dirs is not None else args.inv_template.format(seed=seed)
        full_dir = resolve_path(full_dir, project)
        inv_dir = resolve_path(inv_dir, project)
        full = load_sample_dir(full_dir)
        inv = load_sample_dir(inv_dir)
        row, delta, rel_gain = summarize_seed(seed, full, inv, args.tie_threshold, args.relative_threshold)
        row["full_dir"] = full_dir
        row["inv_dir"] = inv_dir
        seed_rows.append(row)
        seed_full_mae.append(full["sample_mae"].astype(float))
        seed_inv_mae.append(inv["sample_mae"].astype(float))
        seed_deltas.append(delta)
        seed_relative_gains.append(rel_gain)
        if sample_index is None:
            sample_index = full["sample_index"]
            target_ref = full["target"]
        else:
            assert np.array_equal(sample_index, full["sample_index"]), (
                "sample_index mismatch across seeds at seed {}".format(seed)
            )
            assert np.allclose(target_ref, full["target"]), (
                "target mismatch across seeds at seed {}".format(seed)
            )

    full_mae_arr = np.stack(seed_full_mae, axis=1)
    inv_mae_arr = np.stack(seed_inv_mae, axis=1)
    delta_arr = np.stack(seed_deltas, axis=1)
    rel_gain_arr = np.stack(seed_relative_gains, axis=1)
    signs = np.zeros_like(delta_arr, dtype=np.int8)
    signs[delta_arr > args.tie_threshold] = 1
    signs[delta_arr < -args.tie_threshold] = -1

    full_all = np.all(signs > 0, axis=1)
    inv_all = np.all(signs < 0, axis=1)
    mixed = ~(full_all | inv_all)
    full_at_least_two = (signs > 0).sum(axis=1) >= 2
    inv_at_least_two = (signs < 0).sum(axis=1) >= 2
    mean_delta = delta_arr.mean(axis=1)
    mean_full = full_mae_arr.mean(axis=1)
    mean_inv = inv_mae_arr.mean(axis=1)

    comparison_rows = []
    for i, idx in enumerate(sample_index):
        row = OrderedDict()
        row["sample_index"] = int(idx)
        for pos, seed in enumerate(seeds):
            row["full_mae_seed{}".format(seed)] = float(full_mae_arr[i, pos])
            row["inv_mae_seed{}".format(seed)] = float(inv_mae_arr[i, pos])
            row["delta_seed{}".format(seed)] = float(delta_arr[i, pos])
        row["mean_full_mae"] = float(mean_full[i])
        row["mean_inv_mae"] = float(mean_inv[i])
        row["mean_delta"] = float(mean_delta[i])
        row["full_win_count"] = int((signs[i] > 0).sum())
        row["inv_win_count"] = int((signs[i] < 0).sum())
        if full_all[i]:
            label = "Full consistently better"
        elif inv_all[i]:
            label = "InvOnly consistently better"
        else:
            label = "Mixed"
        row["consistency_label"] = label
        comparison_rows.append(row)

    cross = OrderedDict()
    n = int(sample_index.shape[0])
    cross["num_samples"] = n
    cross["full_consistently_better_count"] = int(full_all.sum())
    cross["full_consistently_better_prop"] = float(full_all.mean())
    cross["inv_consistently_better_count"] = int(inv_all.sum())
    cross["inv_consistently_better_prop"] = float(inv_all.mean())
    cross["mixed_count"] = int(mixed.sum())
    cross["mixed_prop"] = float(mixed.mean())
    cross["full_at_least_two_count"] = int(full_at_least_two.sum())
    cross["full_at_least_two_prop"] = float(full_at_least_two.mean())
    cross["inv_at_least_two_count"] = int(inv_at_least_two.sum())
    cross["inv_at_least_two_prop"] = float(inv_at_least_two.mean())
    cross["mean_of_mean_delta"] = float(mean_delta.mean())
    cross["median_of_mean_delta"] = float(np.median(mean_delta))
    cross["tie_threshold"] = float(args.tie_threshold)
    cross["relative_threshold"] = float(args.relative_threshold)

    write_csv(os.path.join(output_dir, "summary_by_seed.csv"), seed_rows, list(seed_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_cross_seed.csv"), [cross], list(cross.keys()))
    write_csv(
        os.path.join(output_dir, "samplewise_comparison.csv"),
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    np.savez(
        os.path.join(output_dir, "samplewise_comparison.npz"),
        sample_index=sample_index,
        seeds=np.asarray(seeds, dtype=np.int64),
        full_mae=full_mae_arr,
        inv_mae=inv_mae_arr,
        delta=delta_arr,
        relative_gain=rel_gain_arr,
        mean_full_mae=mean_full,
        mean_inv_mae=mean_inv,
        mean_delta=mean_delta,
        consistency_signs=signs,
    )

    save_hist(mean_delta, os.path.join(output_dir, "delta_histogram.png"))
    save_sorted_curve(mean_delta, os.path.join(output_dir, "delta_sorted_curve.png"))
    save_consistency_bar(
        ["Full all", "Inv all", "Mixed"],
        [int(full_all.sum()), int(inv_all.sum()), int(mixed.sum())],
        os.path.join(output_dir, "seed_consistency_bar.png"),
    )
    save_scatter(mean_inv, mean_full, os.path.join(output_dir, "full_vs_inv_scatter.png"))

    print("[analyze-steve-env-value] wrote outputs to {}".format(output_dir))
    print("[analyze-steve-env-value] cross-seed summary: {}".format(dict(cross)))


if __name__ == "__main__":
    main()
