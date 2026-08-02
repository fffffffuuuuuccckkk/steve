#!/usr/bin/env python3
"""Sample-wise environment correction analysis for one original STEVE checkpoint.

For each test sample from a trained STEVE-Full checkpoint, export

    y_base      = Y_h
    y_env_delta = C_weight * Y_c
    y_full      = Y_h + C_weight * Y_c

and compute

    delta_i = MAE_i(y_base) - MAE_i(y_full)

Positive delta means the learned environment correction improves that sample;
negative delta means the same checkpoint would have been better with the base
invariant path only for that sample.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import yaml

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from lib.dataloader import get_dataloader  # noqa: E402
from lib.utils import init_seed, load_graph  # noqa: E402
from models.steve_original import OriginalStableST  # noqa: E402


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT, path)


def load_checkpoint_with_retry(path, map_location, attempts=5, delay=1.0):
    last_exc = None
    for _ in range(max(1, attempts)):
        try:
            return torch.load(path, map_location=map_location)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def sample_mean(array):
    return array.mean(axis=tuple(range(1, array.ndim)))


def sample_masked_mean(array, mask):
    axes = tuple(range(1, array.ndim))
    mask = mask.astype(bool)
    count = mask.sum(axis=axes)
    summed = np.where(mask, array, 0.0).sum(axis=axes)
    out = np.full_like(summed, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    return out, count


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def simple_canvas(width=900, height=560):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None, None
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def draw_axes(draw, width, height, margin=60):
    left, top, right, bottom = margin, margin, width - margin, height - margin
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    return left, top, right, bottom


def save_delta_histogram(delta, path):
    image, draw = simple_canvas()
    if image is None:
        return False
    delta = np.asarray(delta, dtype=float)
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    vmin, vmax = float(delta.min()), float(delta.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    counts, edges = np.histogram(delta, bins=50, range=(vmin, vmax))
    max_count = max(int(counts.max()), 1)
    bar_w = (right - left) / len(counts)
    for i, count in enumerate(counts):
        x0 = left + i * bar_w
        x1 = left + (i + 1) * bar_w - 1
        y0 = bottom - (bottom - top) * float(count) / max_count
        draw.rectangle((x0, y0, x1, bottom), fill=(80, 135, 210), outline=(55, 95, 160))
    if vmin <= 0 <= vmax:
        xz = left + (0.0 - vmin) / (vmax - vmin) * (right - left)
        draw.line((xz, top, xz, bottom), fill="red", width=2)
    draw.text((left, 20), "delta_i = MAE(base Y_h) - MAE(full Y_h + env correction)", fill="black")
    draw.text((left, bottom + 15), "positive: environment correction helps; negative: hurts", fill="black")
    image.save(path)
    return True


def save_sorted_delta(delta, path):
    image, draw = simple_canvas()
    if image is None:
        return False
    values = np.sort(np.asarray(delta, dtype=float))
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    if vmin <= 0 <= vmax:
        yz = bottom - (0.0 - vmin) / (vmax - vmin) * (bottom - top)
        draw.line((left, yz, right, yz), fill="red", width=2)
    pts = []
    denom = max(len(values) - 1, 1)
    for i, value in enumerate(values):
        x = left + i / denom * (right - left)
        y = bottom - (value - vmin) / (vmax - vmin) * (bottom - top)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(45, 105, 190), width=2)
    draw.text((left, 20), "sorted sample-wise environment correction utility", fill="black")
    draw.text((left, bottom + 15), "below zero hurts, above zero helps", fill="black")
    image.save(path)
    return True


def save_base_vs_full_scatter(base_mae, full_mae, path):
    image, draw = simple_canvas()
    if image is None:
        return False
    x_values = np.asarray(base_mae, dtype=float)
    y_values = np.asarray(full_mae, dtype=float)
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    vmin = float(min(x_values.min(), y_values.min()))
    vmax = float(max(x_values.max(), y_values.max()))
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    draw.line((left, bottom, right, top), fill="red", width=2)
    n = len(x_values)
    if n > 8000:
        rng = np.random.RandomState(0)
        keep = rng.choice(n, size=8000, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
    for x_val, y_val in zip(x_values, y_values):
        x = left + (x_val - vmin) / (vmax - vmin) * (right - left)
        y = bottom - (y_val - vmin) / (vmax - vmin) * (bottom - top)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(45, 105, 190))
    draw.text((left, 20), "Base Y_h MAE vs Full STEVE MAE", fill="black")
    draw.text((left, bottom + 15), "x=base, y=full; below y=x means env correction helps", fill="black")
    image.save(path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--checkpoint_path", default="experiments/NYCTaxi_TDS/steve_full_reproduce_seed2024/best_model.pth")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="NYCTaxi_TDS")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--graph_file", default="data/NYCTaxi_TDS/adj_mx.npz")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--tie_threshold", type=float, default=1e-6)
    parser.add_argument("--metric_mask_value", type=float, default=5.0)
    parser.add_argument("--strict_load", type=parse_bool, default=False)
    args_cli = parser.parse_args()

    config_path = resolve_path(args_cli.config_filename)
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["seed"] = args_cli.seed
    cfg["device"] = args_cli.device
    cfg["dataset"] = args_cli.dataset
    cfg["data_dir"] = resolve_path(args_cli.data_dir)
    cfg["graph_file"] = resolve_path(args_cli.graph_file)
    cfg["model_impl"] = "steve_original"
    cfg["steve_prediction_mode"] = "full"
    cfg["ablation"] = "all"
    if args_cli.batch_size is not None:
        cfg["batch_size"] = args_cli.batch_size
    if args_cli.test_batch_size is not None:
        cfg["test_batch_size"] = args_cli.test_batch_size

    model_args = argparse.Namespace(**cfg)
    init_seed(model_args.seed)
    graph = load_graph(model_args.graph_file, device=model_args.device)
    dataloader = get_dataloader(
        data_dir=model_args.data_dir,
        dataset=model_args.dataset,
        batch_size=model_args.batch_size,
        test_batch_size=model_args.test_batch_size,
        device=model_args.device,
    )
    loader = dataloader[args_cli.split]
    scaler = dataloader["scaler"]

    model = OriginalStableST(
        args=model_args,
        adj=graph,
        in_channels=model_args.d_input,
        embed_size=model_args.d_model,
        T_dim=model_args.input_length,
        output_T_dim=1,
        output_dim=model_args.d_output,
        device=model_args.device,
    ).to(model_args.device)

    ckpt_path = resolve_path(args_cli.checkpoint_path)
    ckpt = load_checkpoint_with_retry(ckpt_path, map_location=torch.device(model_args.device))
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    load_result = model.load_state_dict(state, strict=bool(args_cli.strict_load))
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    print("[steve-env-correction] checkpoint={}".format(ckpt_path))
    print("[steve-env-correction] missing_keys={} unexpected_keys={}".format(missing, unexpected))
    print("[steve-env-correction] split={} max_batches={}".format(args_cli.split, args_cli.max_batches))
    print("[steve-env-correction] prediction: y_full = y_base + y_env_delta")

    model.eval()
    y_base_all = []
    y_delta_all = []
    y_full_all = []
    y_context_all = []
    target_all = []
    c_weight_all = []
    sample_index_all = []
    offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args_cli.max_batches >= 0 and batch_idx >= args_cli.max_batches:
                break
            if len(batch) == 4:
                data, target, _time_label, _c = batch
            else:
                data, target, _c = batch
            H_inv, Z_var = model(data, graph)
            out = model.predict_decomposition(H_inv, Z_var)
            y_base = scaler.inverse_transform(out["y_base"])
            y_full = scaler.inverse_transform(out["y_full"])
            target_real = scaler.inverse_transform(target[:, : out["y_full"].shape[1]])
            # Because the scaler is affine, the correction in real scale is the
            # difference between real-scale full and real-scale base.
            y_delta = y_full - y_base

            y_base_all.append(y_base.detach().cpu().numpy())
            y_delta_all.append(y_delta.detach().cpu().numpy())
            y_full_all.append(y_full.detach().cpu().numpy())
            y_context_all.append(out["y_context"].detach().cpu().numpy())
            target_all.append(target_real.detach().cpu().numpy())
            c_weight_all.append(out["c_weight"].detach().cpu().numpy())
            bsz = int(data.shape[0])
            sample_index_all.append(np.arange(offset, offset + bsz, dtype=np.int64))
            offset += bsz

    if not y_full_all:
        raise RuntimeError("no batches were processed")

    y_base = np.concatenate(y_base_all, axis=0)
    y_env_delta = np.concatenate(y_delta_all, axis=0)
    y_full = np.concatenate(y_full_all, axis=0)
    y_context_norm = np.concatenate(y_context_all, axis=0)
    target = np.concatenate(target_all, axis=0)
    c_weight = np.concatenate(c_weight_all, axis=0)
    sample_index = np.concatenate(sample_index_all, axis=0)

    assert y_base.shape == y_full.shape == target.shape
    assert y_env_delta.shape == y_full.shape
    assert np.allclose(y_full, y_base + y_env_delta, atol=1e-5)
    assert len(np.unique(sample_index)) == len(sample_index)

    abs_base_error = np.abs(y_base - target)
    abs_full_error = np.abs(y_full - target)
    base_mae = sample_mean(abs_base_error)
    full_mae = sample_mean(abs_full_error)
    sample_delta = base_mae - full_mae
    metric_mask = target > float(args_cli.metric_mask_value)
    base_mae_masked, metric_count = sample_masked_mean(abs_base_error, metric_mask)
    full_mae_masked, _ = sample_masked_mean(abs_full_error, metric_mask)
    sample_delta_masked = base_mae_masked - full_mae_masked
    valid_masked = metric_count > 0
    base_global_masked_mae = float(abs_base_error[metric_mask].mean()) if bool(metric_mask.any()) else float("nan")
    full_global_masked_mae = float(abs_full_error[metric_mask].mean()) if bool(metric_mask.any()) else float("nan")
    env_delta_abs_mean = sample_mean(np.abs(y_env_delta))
    env_delta_signed_mean = sample_mean(y_env_delta)

    positive = sample_delta > args_cli.tie_threshold
    negative = sample_delta < -args_cli.tie_threshold
    neutral = ~(positive | negative)
    positive_masked = (sample_delta_masked > args_cli.tie_threshold) & valid_masked
    negative_masked = (sample_delta_masked < -args_cli.tie_threshold) & valid_masked
    neutral_masked = valid_masked & ~(positive_masked | negative_masked)
    summary = OrderedDict()
    summary["checkpoint_path"] = ckpt_path
    summary["split"] = args_cli.split
    summary["num_samples"] = int(sample_delta.shape[0])
    summary["base_mae_mean"] = float(base_mae.mean())
    summary["full_mae_mean"] = float(full_mae.mean())
    summary["mean_delta_base_minus_full"] = float(sample_delta.mean())
    summary["median_delta"] = float(np.median(sample_delta))
    summary["std_delta"] = float(sample_delta.std())
    summary["q25_delta"] = float(np.percentile(sample_delta, 25))
    summary["q75_delta"] = float(np.percentile(sample_delta, 75))
    summary["env_helps_count"] = int(positive.sum())
    summary["env_helps_prop"] = float(positive.mean())
    summary["env_hurts_count"] = int(negative.sum())
    summary["env_hurts_prop"] = float(negative.mean())
    summary["env_neutral_count"] = int(neutral.sum())
    summary["env_neutral_prop"] = float(neutral.mean())
    summary["metric_mask_value"] = float(args_cli.metric_mask_value)
    summary["masked_valid_samples"] = int(valid_masked.sum())
    summary["base_global_masked_mae"] = base_global_masked_mae
    summary["full_global_masked_mae"] = full_global_masked_mae
    summary["global_delta_masked_base_minus_full"] = base_global_masked_mae - full_global_masked_mae
    summary["base_mae_masked_mean"] = float(np.nanmean(base_mae_masked))
    summary["full_mae_masked_mean"] = float(np.nanmean(full_mae_masked))
    summary["mean_delta_masked_base_minus_full"] = float(np.nanmean(sample_delta_masked))
    summary["median_delta_masked"] = float(np.nanmedian(sample_delta_masked))
    summary["std_delta_masked"] = float(np.nanstd(sample_delta_masked))
    summary["q25_delta_masked"] = float(np.nanpercentile(sample_delta_masked, 25))
    summary["q75_delta_masked"] = float(np.nanpercentile(sample_delta_masked, 75))
    summary["env_helps_masked_count"] = int(positive_masked.sum())
    summary["env_helps_masked_prop"] = float(positive_masked.sum() / max(int(valid_masked.sum()), 1))
    summary["env_hurts_masked_count"] = int(negative_masked.sum())
    summary["env_hurts_masked_prop"] = float(negative_masked.sum() / max(int(valid_masked.sum()), 1))
    summary["env_neutral_masked_count"] = int(neutral_masked.sum())
    summary["env_neutral_masked_prop"] = float(neutral_masked.sum() / max(int(valid_masked.sum()), 1))
    summary["tie_threshold"] = float(args_cli.tie_threshold)
    summary["env_delta_abs_mean"] = float(env_delta_abs_mean.mean())

    output_dir = args_cli.output_dir
    if output_dir is None:
        exp_dir = os.path.dirname(ckpt_path)
        output_dir = os.path.join(exp_dir, "environment_correction_analysis")
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "sample_environment_correction.npz"),
        y_base=y_base,
        y_env_delta=y_env_delta,
        y_full=y_full,
        y_context_norm=y_context_norm,
        target=target,
        c_weight=c_weight,
        sample_index=sample_index,
        base_mae=base_mae,
        full_mae=full_mae,
        delta=sample_delta,
        base_mae_masked=base_mae_masked,
        full_mae_masked=full_mae_masked,
        delta_masked=sample_delta_masked,
        metric_valid_count=metric_count,
        metric_mask_value=np.asarray(float(args_cli.metric_mask_value)),
        env_delta_abs_mean=env_delta_abs_mean,
        env_delta_signed_mean=env_delta_signed_mean,
    )

    rows = []
    for i, idx in enumerate(sample_index):
        rows.append(
            OrderedDict(
                sample_index=int(idx),
                base_mae=float(base_mae[i]),
                full_mae=float(full_mae[i]),
                delta_base_minus_full=float(sample_delta[i]),
                base_mae_masked="" if np.isnan(base_mae_masked[i]) else float(base_mae_masked[i]),
                full_mae_masked="" if np.isnan(full_mae_masked[i]) else float(full_mae_masked[i]),
                delta_masked_base_minus_full="" if np.isnan(sample_delta_masked[i]) else float(sample_delta_masked[i]),
                metric_valid_count=int(metric_count[i]),
                env_delta_abs_mean=float(env_delta_abs_mean[i]),
                env_delta_signed_mean=float(env_delta_signed_mean[i]),
                label="env_helps" if positive[i] else ("env_hurts" if negative[i] else "env_neutral"),
                label_masked=(
                    "metric_no_valid_target"
                    if not valid_masked[i]
                    else ("env_helps" if positive_masked[i] else ("env_hurts" if negative_masked[i] else "env_neutral"))
                ),
            )
        )
    write_csv(os.path.join(output_dir, "sample_environment_correction.csv"), rows, list(rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_environment_correction.csv"), [summary], list(summary.keys()))
    with open(os.path.join(output_dir, "summary_environment_correction.json"), "w") as f:
        json.dump(summary, f, indent=2)

    save_delta_histogram(sample_delta, os.path.join(output_dir, "delta_histogram.png"))
    save_sorted_delta(sample_delta, os.path.join(output_dir, "delta_sorted_curve.png"))
    save_base_vs_full_scatter(base_mae, full_mae, os.path.join(output_dir, "base_vs_full_scatter.png"))
    if bool(valid_masked.any()):
        save_delta_histogram(sample_delta_masked[valid_masked], os.path.join(output_dir, "delta_masked_histogram.png"))
        save_sorted_delta(sample_delta_masked[valid_masked], os.path.join(output_dir, "delta_masked_sorted_curve.png"))
        save_base_vs_full_scatter(
            base_mae_masked[valid_masked],
            full_mae_masked[valid_masked],
            os.path.join(output_dir, "base_vs_full_masked_scatter.png"),
        )

    print("[steve-env-correction] output_dir={}".format(output_dir))
    print("[steve-env-correction] summary={}".format(dict(summary)))


if __name__ == "__main__":
    main()
