#!/usr/bin/env python3
"""Inference-only sample-wise environment-routing audit for FPEM checkpoints."""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from json import JSONDecoder
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import run_tds_nyctaxi as tds  # noqa: E402


def resolve(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


def parse_args_json_from_log(exp_dir):
    log_path = os.path.join(exp_dir, "tds_run.log")
    if not os.path.isfile(log_path):
        raise FileNotFoundError("missing tds_run.log in exp_dir: {}".format(exp_dir))
    text = open(log_path, "r", encoding="utf-8").read()
    obj, _ = JSONDecoder().raw_decode(text)
    if "args" not in obj:
        raise KeyError("tds_run.log JSON has no args")
    return obj["args"]


def load_summary(exp_dir):
    path = os.path.join(exp_dir, "summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compare_state(before, after):
    changed = []
    for key, old in before.items():
        new = after.get(key)
        if new is None:
            changed.append(key)
            continue
        if old.shape != new.shape or not torch.equal(old, new.detach().cpu()):
            changed.append(key)
    extra = [key for key in after.keys() if key not in before]
    changed.extend(extra)
    return changed


def masked_sample_mae(pred, target, mask_threshold):
    err = np.abs(pred - target)
    mask = target > float(mask_threshold)
    axes = tuple(range(1, pred.ndim))
    count = mask.sum(axis=axes)
    summed = np.where(mask, err, 0.0).sum(axis=axes)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    global_mae = float(err[mask].mean()) if bool(mask.any()) else float("nan")
    return out, count, global_mae


def masked_global_mae(pred, target, mask_threshold):
    err = np.abs(pred - target)
    mask = target > float(mask_threshold)
    return float(err[mask].mean()) if bool(mask.any()) else float("nan")


def entropy(q):
    q = np.clip(q.astype(np.float64), 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    ra = rankdata_average(a[valid])
    rb = rankdata_average(b[valid])
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def simple_canvas(width=1100, height=680):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None, None
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def draw_axes(draw, width, height, margin=80):
    left, top, right, bottom = margin, margin, width - margin, height - margin
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    return left, top, right, bottom


def save_sorted_gain(gain, native_group_ok, path):
    image, draw = simple_canvas(1200, 720)
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    order = np.argsort(gain)
    values = gain[order]
    ok = native_group_ok[order]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    ymin, ymax = float(finite.min()), float(finite.max())
    ymin = min(ymin, -0.02)
    ymax = max(ymax, 0.02)
    if abs(ymax - ymin) < 1e-12:
        ymax += 1.0
    # threshold lines: -1%, 0, +1%
    for val, color in [(-0.01, (220, 120, 0)), (0.0, (200, 0, 0)), (0.01, (0, 150, 0))]:
        y = bottom - (val - ymin) / (ymax - ymin) * (bottom - top)
        draw.line((left, y, right, y), fill=color, width=2)
    n = len(values)
    denom = max(n - 1, 1)
    for i, val in enumerate(values):
        x = left + i / denom * (right - left)
        y = bottom - (val - ymin) / (ymax - ymin) * (bottom - top)
        color = (45, 105, 190) if ok[i] else (220, 55, 55)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.text((left, 24), "Sorted environment gain g_i=(MAE_inv-best_env)/MAE_inv", fill="black")
    draw.text((left, bottom + 16), "green/orange/red horizontal lines: +1%, -1%, 0; red dots = native group mismatch/proxy", fill="black")
    image.save(path)
    return True


def save_hist(values, path, title):
    image, draw = simple_canvas(1000, 620)
    if image is None:
        return False
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 1.0
        vmax += 1.0
    counts, _ = np.histogram(values, bins=50, range=(vmin, vmax))
    max_count = max(int(counts.max()), 1)
    bar_w = (right - left) / len(counts)
    for i, count in enumerate(counts):
        x0 = left + i * bar_w
        x1 = left + (i + 1) * bar_w - 1
        y0 = bottom - (bottom - top) * float(count) / max_count
        draw.rectangle((x0, y0, x1, bottom), fill=(80, 135, 210), outline=(55, 95, 160))
    draw.text((left, 24), title, fill="black")
    image.save(path)
    return True


def save_scatter(xv, yv, path, title, xlabel):
    image, draw = simple_canvas(1000, 620)
    if image is None:
        return False
    x = np.asarray(xv, dtype=np.float64)
    y = np.asarray(yv, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    lo, hi = min(xmin, ymin), max(xmax, ymax)
    if abs(hi - lo) < 1e-12:
        hi += 1.0
    draw.line((left, bottom, right, top), fill="red", width=2)
    n = x.size
    if n > 8000:
        rng = np.random.RandomState(0)
        keep = rng.choice(n, 8000, replace=False)
        x = x[keep]
        y = y[keep]
    for a, b in zip(x, y):
        px = left + (a - lo) / (hi - lo) * (right - left)
        py = bottom - (b - lo) / (hi - lo) * (bottom - top)
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(45, 105, 190))
    draw.text((left, 24), title, fill="black")
    draw.text((left, bottom + 16), xlabel + " vs native MAE; red line y=x", fill="black")
    image.save(path)
    return True


def save_bar(labels, values, path, title):
    image, draw = simple_canvas(1100, 620)
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    maxv = max(max(values), 1)
    gap = 30
    bar_w = (right - left - gap * (len(values) + 1)) / max(len(values), 1)
    colors = [(45, 105, 190), (50, 155, 95), (215, 95, 85), (150, 120, 220), (150, 150, 150)]
    for i, (label, val) in enumerate(zip(labels, values)):
        x0 = left + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - (bottom - top) * float(val) / maxv
        draw.rectangle((x0, y0, x1, bottom), fill=colors[i % len(colors)], outline="black")
        draw.text((x0, y0 - 22), str(int(val)), fill="black")
        draw.text((x0, bottom + 12), str(label), fill="black")
    draw.text((left, 24), title, fill="black")
    image.save(path)
    return True


def save_matrix(matrix, row_labels, col_labels, path, title):
    image, draw = simple_canvas(900, 760)
    if image is None:
        return False
    left, top = 170, 90
    cell = 80
    maxv = max(float(np.max(matrix)), 1.0)
    draw.text((left, 30), title, fill="black")
    for i, row in enumerate(row_labels):
        draw.text((20, top + i * cell + 28), str(row), fill="black")
    for j, col in enumerate(col_labels):
        draw.text((left + j * cell + 5, top - 30), str(col), fill="black")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = float(matrix[i, j])
            shade = int(255 - 180 * val / maxv)
            color = (shade, shade, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="black")
            draw.text((x0 + 20, y0 + 28), str(int(val)), fill="black")
    image.save(path)
    return True


def save_groupwise_table(rows, path):
    labels = [r["group"] for r in rows]
    metrics = ["invariant", "best_env", "best_fixed", "native", "oracle"]
    image, draw = simple_canvas(1200, 650)
    if image is None:
        return False
    width, height = image.size
    left, top, right, bottom = draw_axes(draw, width, height)
    all_values = [safe_float(r[m]) for r in rows for m in metrics if np.isfinite(safe_float(r[m]))]
    ymin, ymax = min(all_values), max(all_values)
    if abs(ymax - ymin) < 1e-12:
        ymax += 1.0
    colors = [(45, 105, 190), (50, 155, 95), (215, 95, 85), (150, 120, 220), (40, 40, 40)]
    group_w = (right - left) / max(len(labels), 1)
    bar_w = group_w / (len(metrics) + 1)
    for gi, row in enumerate(rows):
        gx = left + gi * group_w
        draw.text((gx + 5, bottom + 14), labels[gi][:22], fill="black")
        for mi, m in enumerate(metrics):
            val = safe_float(row[m])
            if not np.isfinite(val):
                continue
            x0 = gx + (mi + 0.5) * bar_w
            y0 = bottom - (val - ymin) / (ymax - ymin) * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_w * 0.75, bottom), fill=colors[mi], outline="black")
    for mi, m in enumerate(metrics):
        draw.rectangle((left + mi * 140, 24, left + mi * 140 + 16, 40), fill=colors[mi])
        draw.text((left + mi * 140 + 20, 22), m, fill="black")
    image.save(path)
    return True


def load_model_from_exp(exp_dir, checkpoint, device=None):
    exp_dir = resolve(exp_dir)
    checkpoint = resolve(checkpoint)
    args_dict = parse_args_json_from_log(exp_dir)
    if device is not None:
        args_dict["device"] = device
    if not torch.cuda.is_available() or str(args_dict.get("device", "cpu")).startswith("cuda") is False:
        if not torch.cuda.is_available():
            args_dict["device"] = "cpu"
    args_dict["log_dir"] = exp_dir
    args = SimpleNamespace(**args_dict)
    loaders, scaler, counts = tds.build_tds_data(args)
    graph = tds.load_graph(args.graph_file, args.device)
    model, _lr = tds.build_model(args, graph)
    ckpt = tds.load_checkpoint(checkpoint, args.device)
    result = model.load_state_dict(ckpt["model"], strict=False)
    if "fpem_progressive_gmm_state" in ckpt and hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
        model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
    model.eval()
    return args, loaders, scaler, graph, model, ckpt, result, counts


def gather_expert_prediction(y_experts, route_idx):
    b = y_experts.shape[0]
    index = route_idx.view(b, 1, 1, 1, 1).expand(-1, 1, *y_experts.shape[2:])
    return y_experts.gather(1, index).squeeze(1)


def predict_all_routes(model, batch, args, scaler):
    if len(batch) == 5:
        data, target, time_label, c, sample_index = batch
    else:
        raise ValueError("expected test batch with sample_index")
    output = model.forward_output(data, exog=c, time_label=time_label, training=False, sample_index=sample_index)
    y_native = output["prediction"]
    y_invariant = output.get("y_inv")
    if y_invariant is None:
        y_invariant = output.get("y_global")
    y_experts = output.get("y_route_heads")
    if y_experts is None:
        y_experts = output.get("y_hyper_heads")
    if y_experts is None:
        y_route = output.get("y_route")
        if y_route is None:
            raise RuntimeError("checkpoint output does not expose y_route_heads/y_hyper_heads/y_route")
        y_experts = y_route.unsqueeze(1)
    router_probs = output.get("env_route_q")
    if router_probs is None:
        router_probs = output.get("route_q")
    if router_probs is None:
        router_probs = y_experts.new_full((y_experts.shape[0], y_experts.shape[1]), 1.0 / y_experts.shape[1])
    # If probabilities include a fallback expert but y_experts do not, remove fallback column.
    if router_probs.shape[1] != y_experts.shape[1]:
        if router_probs.shape[1] == y_experts.shape[1] + 1:
            router_probs_for_env = router_probs[:, -y_experts.shape[1]:]
            sums = router_probs_for_env.sum(dim=1, keepdim=True).clamp_min(1e-8)
            router_probs_for_env = router_probs_for_env / sums
            router_probs = router_probs_for_env
        else:
            raise AssertionError("router_probs shape does not match y_experts")
    selected = router_probs.argmax(dim=1).long()
    y_native_argmax = gather_expert_prediction(y_experts, selected)
    y_uniform = y_experts.mean(dim=1)
    correction_norm = (y_native - y_invariant).detach().abs().mean(dim=tuple(range(1, y_native.ndim)))
    max_prob = router_probs.max(dim=1).values
    q = router_probs.clamp_min(1e-12)
    routing_entropy = (-(q * q.log()).sum(dim=1)).detach()
    # There is a real identity/no-env score only if the model exposes fallback
    # probability as a selectable prediction candidate. Current progressive GMM
    # checkpoints set this to false.
    has_identity = bool(getattr(model, "fpem_env_route_use_inv_fallback_expert", False))
    env_use_score = None
    if has_identity and output.get("fallback_q") is not None:
        env_use_score = 1.0 - output["fallback_q"].detach()
    return {
        "output": output,
        "target": target,
        "sample_index": sample_index,
        "y_native": scaler.inverse_transform(y_native).detach(),
        "y_invariant": scaler.inverse_transform(y_invariant).detach(),
        "y_native_argmax": scaler.inverse_transform(y_native_argmax).detach(),
        "y_uniform": scaler.inverse_transform(y_uniform).detach(),
        "y_experts": scaler.inverse_transform(y_experts).detach(),
        "router_probs": router_probs.detach(),
        "selected_route": selected.detach(),
        "env_use_score": env_use_score,
        "correction_norm": correction_norm.detach(),
        "routing_entropy": routing_entropy,
        "max_router_prob": max_prob.detach(),
        "progressive_cluster_id": output.get("progressive_cluster_id"),
        "top_level_prediction_source": output.get("top_level_prediction_source", ""),
        "route_head_mode": output.get("route_head_mode", getattr(model, "fpem_env_route_head_mode", "")),
        "env_route_proto_mode": output.get("env_route_proto_mode", ""),
    }


def collect_audit_outputs(model, loader, args, scaler, max_batches=None, validate_first_batch=True):
    arrays = {
        "target": [],
        "sample_index": [],
        "y_native": [],
        "y_invariant": [],
        "y_native_argmax": [],
        "y_uniform": [],
        "y_experts": [],
        "router_probs": [],
        "selected_route": [],
        "correction_norm": [],
        "routing_entropy": [],
        "max_router_prob": [],
        "env_use_score": [],
    }
    top_level_sources = Counter()
    route_head_modes = Counter()
    proto_modes = Counter()
    first_batch_native_checked = False
    with torch.no_grad():
        for batch_idx, raw in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = tds.to_device(raw, args.device)
            pred = predict_all_routes(model, batch, args, scaler)
            if validate_first_batch and not first_batch_native_checked:
                native = pred["y_native"]
                original_pred, _target = tds.predict_batch(model, batch, args)
                original_pred = scaler.inverse_transform(original_pred).detach()
                assert torch.allclose(native, original_pred, atol=1e-5, rtol=1e-5), (
                    "y_native_from_audit does not match original forward"
                )
                first_batch_native_checked = True
            target_real = scaler.inverse_transform(pred["target"][:, : pred["y_native"].shape[1]]).detach()
            arrays["target"].append(to_numpy(target_real))
            for key in [
                "sample_index",
                "y_native",
                "y_invariant",
                "y_native_argmax",
                "y_uniform",
                "y_experts",
                "router_probs",
                "selected_route",
                "correction_norm",
                "routing_entropy",
                "max_router_prob",
            ]:
                arrays[key].append(to_numpy(pred[key]))
            if pred["env_use_score"] is None:
                arrays["env_use_score"].append(np.full(pred["selected_route"].shape[0], np.nan, dtype=np.float64))
            else:
                arrays["env_use_score"].append(to_numpy(pred["env_use_score"]))
            top_level_sources.update([str(pred["top_level_prediction_source"])])
            route_head_modes.update([str(pred["route_head_mode"])])
            proto_modes.update([str(pred["env_route_proto_mode"])])
    out = {}
    for key, chunks in arrays.items():
        if not chunks:
            raise RuntimeError("no audit batches were processed")
        out[key] = np.concatenate(chunks, axis=0)
    out["top_level_sources"] = dict(top_level_sources)
    out["route_head_modes"] = dict(route_head_modes)
    out["proto_modes"] = dict(proto_modes)
    return out


def validate_batch_size_independence(args, exp_dir, checkpoint, output_dir, n_samples=32, atol=1e-3, rtol=1e-5):
    # Rebuild twice with different test batch sizes and compare first n_samples by sample_index.
    results = []
    for bsz in [1, min(16, max(1, n_samples))]:
        args_dict = parse_args_json_from_log(exp_dir)
        args_dict["device"] = args.device
        args_dict["test_batch_size"] = bsz
        args_dict["log_dir"] = exp_dir
        tmp_args = SimpleNamespace(**args_dict)
        loaders, scaler, _counts = tds.build_tds_data(tmp_args)
        graph = tds.load_graph(tmp_args.graph_file, tmp_args.device)
        model, _ = tds.build_model(tmp_args, graph)
        ckpt = tds.load_checkpoint(checkpoint, tmp_args.device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "fpem_progressive_gmm_state" in ckpt and hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
            model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
        model.eval()
        collected = collect_audit_outputs(model, loaders["test_mixed"], tmp_args, scaler, max_batches=None, validate_first_batch=False)
        keep = collected["sample_index"] < n_samples
        results.append((bsz, collected["sample_index"][keep], collected["y_native"][keep]))
    idx0, y0 = results[0][1], results[0][2]
    idx1, y1 = results[1][1], results[1][2]
    ok = np.array_equal(idx0, idx1) and np.allclose(y0, y1, atol=atol, rtol=rtol)
    report = {
        "batch_size_a": int(results[0][0]),
        "batch_size_b": int(results[1][0]),
        "n_compared": int(len(idx0)),
        "sample_index_equal": bool(np.array_equal(idx0, idx1)),
        "native_prediction_allclose": bool(np.allclose(y0, y1, atol=atol, rtol=rtol)) if len(idx0) else False,
        "max_abs_diff": float(np.max(np.abs(y0 - y1))) if len(idx0) else float("nan"),
        "ok": bool(ok),
    }
    with open(os.path.join(output_dir, "batch_size_invariance_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if not ok:
        raise AssertionError("batch-size independence check failed: {}".format(report))
    return report


def group_label(gain, threshold):
    if gain > threshold:
        return "environment_preferred"
    if gain < -threshold:
        return "invariant_preferred"
    return "neutral"


def mean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def load_steve_groups(path, sample_index, fpem_target=None, steve_target_npz=None):
    if path is None or not os.path.isfile(path):
        return None, "missing STEVE comparison csv"
    target_status = "target_not_checked"
    if fpem_target is not None and steve_target_npz and os.path.isfile(steve_target_npz):
        try:
            with np.load(steve_target_npz) as data:
                steve_target = None
                for key in ["target", "y_true", "Y", "y"]:
                    if key in data:
                        steve_target = np.asarray(data[key])
                        break
            if steve_target is None:
                target_status = "STEVE target npz has no target/y_true/Y/y key"
            else:
                fpem_t = np.asarray(fpem_target)
                if steve_target.shape != fpem_t.shape:
                    target_status = "target_shape_mismatch:{}!={}".format(tuple(steve_target.shape), tuple(fpem_t.shape))
                else:
                    max_abs = float(np.nanmax(np.abs(steve_target.astype(np.float64) - fpem_t.astype(np.float64))))
                    target_status = "target_aligned_max_abs_diff={:.6g}".format(max_abs) if max_abs <= 1e-4 else "target_value_mismatch_max_abs_diff={:.6g}".format(max_abs)
        except Exception as exc:
            target_status = "target_check_failed:{}".format(type(exc).__name__)
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["sample_index"])
            rows[idx] = row
    if any(int(idx) not in rows for idx in sample_index):
        return None, "sample_index missing in STEVE csv; {}".format(target_status)
    labels = []
    for idx in sample_index:
        row = rows[int(idx)]
        full = float(row.get("mae_masked_full", row.get("full_mae_masked", "nan")))
        inv = float(row.get("mae_masked_inv_disentangle", row.get("inv_disentangle_mae_masked", "nan")))
        rel = (inv - full) / max(inv, 1e-12)
        if rel > 0.01:
            labels.append("steve_environment_preferred")
        elif rel < -0.01:
            labels.append("steve_invariant_preferred")
        else:
            labels.append("steve_ambiguous")
    return np.asarray(labels), "ok; {}".format(target_status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="NYCTaxi_TDS")
    parser.add_argument("--mask_threshold", type=float, default=5.0)
    parser.add_argument("--relative_threshold", type=float, default=0.01)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--random_repeats", type=int, default=20)
    parser.add_argument(
        "--steve_comparison_csv",
        default="experiments/NYCTaxi_TDS/steve_three_model_masked_samplewise_comparison_seed2024/samplewise_masked_mae_comparison.csv",
    )
    parser.add_argument(
        "--steve_target_npz",
        default="experiments/NYCTaxi_TDS/steve_full_reproduce_seed2024/sample_predictions.npz",
    )
    args_cli = parser.parse_args()

    exp_dir = resolve(args_cli.exp_dir)
    checkpoint = resolve(args_cli.checkpoint)
    output_dir = resolve(args_cli.output_dir) if args_cli.output_dir else os.path.join(exp_dir, "samplewise_environment_routing_audit")
    os.makedirs(output_dir, exist_ok=True)

    args, loaders, scaler, graph, model, ckpt, load_result, counts = load_model_from_exp(exp_dir, checkpoint, args_cli.device)
    args.dataset = args_cli.dataset or args.dataset
    summary = load_summary(exp_dir)
    print("[fpem-audit] exp_dir={}".format(exp_dir))
    print("[fpem-audit] checkpoint={}".format(checkpoint))
    print("[fpem-audit] output_dir={}".format(output_dir))
    print("[fpem-audit] load missing={} unexpected={}".format(
        list(getattr(load_result, "missing_keys", [])), list(getattr(load_result, "unexpected_keys", []))
    ))

    arch = OrderedDict()
    arch["fpem_backbone"] = str(getattr(model, "fpem_backbone", "unknown"))
    arch["fpem_env_route_train_mode"] = str(getattr(model, "fpem_env_route_train_mode", "unknown"))
    arch["fpem_env_route_head_mode"] = str(getattr(model, "fpem_env_route_head_mode", "unknown"))
    arch["fpem_env_route_k"] = int(getattr(model, "fpem_env_route_k", 0))
    arch["fpem_env_route_use_inv_fallback_expert"] = bool(getattr(model, "fpem_env_route_use_inv_fallback_expert", False))
    arch["fpem_use_env_fusion"] = bool(getattr(model, "fpem_use_env_fusion", False))
    arch["fpem_force_uniform_route"] = bool(getattr(model, "fpem_force_uniform_route", False))
    arch["has_explicit_identity_route"] = bool(arch["fpem_env_route_use_inv_fallback_expert"])
    arch["can_explicitly_choose_no_environment"] = bool(arch["has_explicit_identity_route"])
    arch["environment_use_score_available"] = bool(arch["has_explicit_identity_route"])
    arch["native_prediction_source_in_summary"] = summary.get("top_level_prediction_source")
    with open(os.path.join(output_dir, "architecture_inspection.json"), "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)

    before_state = clone_state(model)
    max_batches = None if args_cli.max_batches < 0 else args_cli.max_batches
    collected = collect_audit_outputs(model, loaders["test_mixed"], args, scaler, max_batches=max_batches, validate_first_batch=True)
    after_state = clone_state(model)
    changed_state = compare_state(before_state, after_state)
    if changed_state:
        with open(os.path.join(output_dir, "state_change_error.json"), "w", encoding="utf-8") as f:
            json.dump({"changed_state_keys": changed_state}, f, indent=2)
        raise AssertionError("model parameters/buffers changed during audit: {}".format(changed_state[:20]))

    sample_index = collected["sample_index"].astype(np.int64)
    assert len(np.unique(sample_index)) == len(sample_index), "sample_index is not unique"
    target = collected["target"]
    y_native = collected["y_native"]
    y_inv = collected["y_invariant"]
    y_argmax = collected["y_native_argmax"]
    y_uniform = collected["y_uniform"]
    y_experts = collected["y_experts"]
    q = collected["router_probs"]
    selected_route = collected["selected_route"].astype(np.int64)
    assert target.shape == y_native.shape == y_inv.shape == y_argmax.shape == y_uniform.shape
    assert y_experts.shape[0] == target.shape[0]
    n, k = y_experts.shape[0], y_experts.shape[1]

    native_mae, valid_count, native_global = masked_sample_mae(y_native, target, args_cli.mask_threshold)
    inv_mae, _, inv_global = masked_sample_mae(y_inv, target, args_cli.mask_threshold)
    argmax_mae, _, argmax_global = masked_sample_mae(y_argmax, target, args_cli.mask_threshold)
    uniform_mae, _, uniform_global = masked_sample_mae(y_uniform, target, args_cli.mask_threshold)
    expert_maes = []
    expert_globals = []
    for expert_idx in range(k):
        mae, _, g = masked_sample_mae(y_experts[:, expert_idx], target, args_cli.mask_threshold)
        expert_maes.append(mae)
        expert_globals.append(g)
    expert_maes = np.stack(expert_maes, axis=1)

    candidate_names = ["invariant"] + ["expert_{}".format(i) for i in range(k)]
    candidate_maes = np.concatenate([inv_mae[:, None], expert_maes], axis=1)
    oracle_idx = np.nanargmin(candidate_maes, axis=1)
    oracle_mae = candidate_maes[np.arange(n), oracle_idx]
    oracle_route = np.asarray([candidate_names[i] for i in oracle_idx])
    best_env_mae = np.nanmin(expert_maes, axis=1)
    best_env_route_idx = np.nanargmin(expert_maes, axis=1)
    env_gain = (inv_mae - best_env_mae) / np.maximum(inv_mae, 1e-12)
    oracle_group = np.asarray([group_label(x, args_cli.relative_threshold) for x in env_gain])
    route_regret = native_mae - oracle_mae
    relative_route_regret = route_regret / np.maximum(oracle_mae, 1e-12)
    argmax_route_regret = argmax_mae - oracle_mae

    native_env_group_proxy = np.asarray(["environment_preferred"] * n)
    if arch["can_explicitly_choose_no_environment"]:
        native_env_group_proxy = np.where(selected_route > 0, "environment_preferred", "invariant_preferred")
    native_group_ok = (
        (oracle_group == "neutral")
        | ((oracle_group == "environment_preferred") & (native_env_group_proxy == "environment_preferred"))
        | ((oracle_group == "invariant_preferred") & (native_env_group_proxy == "invariant_preferred"))
    )

    # Fixed route and random routing.
    fixed_globals = {"invariant": inv_global}
    fixed_sample_means = {"invariant": float(np.nanmean(inv_mae))}
    for i in range(k):
        fixed_globals["expert_{}".format(i)] = expert_globals[i]
        fixed_sample_means["expert_{}".format(i)] = float(np.nanmean(expert_maes[:, i]))
    best_fixed_name = min(fixed_globals, key=lambda name: fixed_globals[name])
    best_fixed_global = fixed_globals[best_fixed_name]
    best_fixed_sample = fixed_sample_means[best_fixed_name]

    random_globals = []
    random_sample_means = []
    rng = np.random.RandomState(20240723)
    for _ in range(max(1, args_cli.random_repeats)):
        choices = rng.randint(0, k, size=n)
        y_rand = y_experts[np.arange(n), choices]
        mae, _, g = masked_sample_mae(y_rand, target, args_cli.mask_threshold)
        random_globals.append(g)
        random_sample_means.append(float(np.nanmean(mae)))

    oracle_global_pred = np.empty_like(y_native)
    for i in range(n):
        if oracle_idx[i] == 0:
            oracle_global_pred[i] = y_inv[i]
        else:
            oracle_global_pred[i] = y_experts[i, oracle_idx[i] - 1]
    oracle_global = masked_global_mae(oracle_global_pred, target, args_cli.mask_threshold)
    oracle_sample_mean = float(np.nanmean(oracle_mae))

    best_fixed_for_gap = min(expert_globals)
    best_fixed_env_name = "expert_{}".format(int(np.argmin(expert_globals)))
    oracle_gap = best_fixed_for_gap - oracle_global
    gap_closed = (best_fixed_for_gap - native_global) / oracle_gap if abs(oracle_gap) > 1e-12 else float("nan")

    # Harm and missed-gain metrics.
    inv_pref = oracle_group == "invariant_preferred"
    env_pref = oracle_group == "environment_preferred"
    neutral = oracle_group == "neutral"
    harm = ((native_mae - inv_mae) / np.maximum(inv_mae, 1e-12) > args_cli.relative_threshold) & inv_pref
    harm_rate = float(harm.sum() / max(inv_pref.sum(), 1))
    available_gain = inv_mae - best_env_mae
    missed = (native_mae > best_env_mae + 0.5 * available_gain) & env_pref
    missed_gain_rate = float(missed.sum() / max(env_pref.sum(), 1))
    recovered_gain = (inv_mae - native_mae) / np.maximum(available_gain, 1e-12)
    recovered_gain_env = recovered_gain[env_pref & np.isfinite(recovered_gain)]

    # Expert selection on samples where an environment expert is oracle-optimal.
    env_oracle = oracle_idx > 0
    oracle_expert = oracle_idx[env_oracle] - 1
    native_sel_env = selected_route[env_oracle]
    exact_expert_acc = float((native_sel_env == oracle_expert).mean()) if oracle_expert.size else float("nan")
    top2_acc = float("nan")
    tol_acc = float("nan")
    if oracle_expert.size:
        expert_errs_env = expert_maes[env_oracle]
        top2 = np.argsort(expert_errs_env, axis=1)[:, : min(2, k)]
        top2_acc = float(np.mean([native_sel_env[i] in top2[i] for i in range(len(native_sel_env))]))
        selected_env_mae = expert_errs_env[np.arange(len(native_sel_env)), native_sel_env.clip(0, k - 1)]
        oracle_env_mae = expert_errs_env[np.arange(len(oracle_expert)), oracle_expert]
        tol_acc = float(np.mean(selected_env_mae <= oracle_env_mae * (1.0 + args_cli.relative_threshold)))
    expert_regret = np.full(n, np.nan)
    selected_env_mae_all = expert_maes[np.arange(n), selected_route.clip(0, k - 1)]
    expert_regret = selected_env_mae_all - best_env_mae

    correction_norm = collected["correction_norm"].astype(np.float64)
    routing_entropy = collected["routing_entropy"].astype(np.float64)
    max_router_prob = collected["max_router_prob"].astype(np.float64)
    env_use_score = collected["env_use_score"].astype(np.float64)
    corr_gain_spearman = spearmanr(correction_norm, env_gain)
    entropy_gain_spearman = spearmanr(routing_entropy, env_gain)
    maxprob_gain_spearman = spearmanr(max_router_prob, env_gain)

    # Calibration/proxy bins.
    proxy_score = env_use_score if np.isfinite(env_use_score).any() else correction_norm
    proxy_name = "env_use_score" if np.isfinite(env_use_score).any() else "correction_norm_proxy"
    order = np.argsort(proxy_score)
    bins = np.array_split(order, 10)
    calib_rows = []
    for b, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        calib_rows.append(OrderedDict(
            bin=b,
            proxy_name=proxy_name,
            avg_proxy=float(np.nanmean(proxy_score[idx])),
            env_preferred_prop=float(np.mean(oracle_group[idx] == "environment_preferred")),
            avg_true_env_gain=float(np.nanmean(env_gain[idx])),
            avg_route_regret=float(np.nanmean(route_regret[idx])),
            count=int(len(idx)),
        ))

    # Group-wise table.
    group_rows = []
    group_order = ["environment_preferred", "invariant_preferred", "neutral"]
    best_fixed_mae_by_sample = candidate_maes[:, candidate_names.index(best_fixed_name)]
    best_env_fixed_by_sample = expert_maes[:, int(np.argmin(expert_globals))]
    for gname in group_order:
        mask = oracle_group == gname
        group_rows.append(OrderedDict(
            group=gname,
            count=int(mask.sum()),
            invariant=mean_or_nan(inv_mae[mask]),
            best_env=mean_or_nan(best_env_mae[mask]),
            best_fixed=mean_or_nan(best_fixed_mae_by_sample[mask]),
            best_fixed_env=mean_or_nan(best_env_fixed_by_sample[mask]),
            native=mean_or_nan(native_mae[mask]),
            oracle=mean_or_nan(oracle_mae[mask]),
        ))

    # External STEVE consistency.
    steve_labels, steve_status = load_steve_groups(
        resolve(args_cli.steve_comparison_csv),
        sample_index,
        fpem_target=target,
        steve_target_npz=resolve(args_cli.steve_target_npz),
    )
    external_rows = []
    if steve_labels is not None:
        for label in ["steve_environment_preferred", "steve_invariant_preferred", "steve_ambiguous"]:
            mask = steve_labels == label
            dist = Counter(selected_route[mask].tolist())
            external_rows.append(OrderedDict(
                steve_group=label,
                count=int(mask.sum()),
                native_mae=mean_or_nan(native_mae[mask]),
                invariant_mae=mean_or_nan(inv_mae[mask]),
                best_env_mae=mean_or_nan(best_env_mae[mask]),
                env_use_score=mean_or_nan(env_use_score[mask]),
                correction_norm=mean_or_nan(correction_norm[mask]),
                route_regret=mean_or_nan(route_regret[mask]),
                selected_expert_distribution=json.dumps({int(k): int(v) for k, v in dist.items()}, sort_keys=True),
            ))

    # Save sample-wise rows.
    sample_rows = []
    for i in range(n):
        row = OrderedDict()
        row["sample_index"] = int(sample_index[i])
        row["native_mae"] = float(native_mae[i])
        row["invariant_mae"] = float(inv_mae[i])
        for e in range(k):
            row["expert_{}_mae".format(e)] = float(expert_maes[i, e])
        row["native_argmax_mae"] = float(argmax_mae[i])
        row["uniform_mae"] = float(uniform_mae[i])
        row["oracle_mae"] = float(oracle_mae[i])
        row["oracle_route"] = oracle_route[i]
        row["native_route"] = int(selected_route[i])
        row["native_argmax_route"] = int(selected_route[i])
        row["best_env_route"] = int(best_env_route_idx[i])
        row["environment_gain"] = float(env_gain[i])
        row["oracle_environment_group"] = oracle_group[i]
        row["route_regret"] = float(route_regret[i])
        row["relative_route_regret"] = float(relative_route_regret[i])
        row["argmax_route_regret"] = float(argmax_route_regret[i])
        row["env_use_score"] = "" if not np.isfinite(env_use_score[i]) else float(env_use_score[i])
        row["correction_norm"] = float(correction_norm[i])
        row["routing_entropy"] = float(routing_entropy[i])
        row["max_router_prob"] = float(max_router_prob[i])
        row["valid_count"] = int(valid_count[i])
        row["expert_selection_regret"] = float(expert_regret[i])
        sample_rows.append(row)

    write_csv(os.path.join(output_dir, "samplewise_route_audit.csv"), sample_rows, list(sample_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_oracle_groups.csv"), group_rows, list(group_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_calibration_proxy.csv"), calib_rows, list(calib_rows[0].keys()))
    if external_rows:
        write_csv(os.path.join(output_dir, "summary_steve_external_consistency.csv"), external_rows, list(external_rows[0].keys()))

    # Expert-selection summary.
    oracle_dist = Counter((oracle_idx[oracle_idx > 0] - 1).tolist())
    native_dist = Counter(selected_route.tolist())
    expert_rows = []
    for e in range(k):
        expert_rows.append(OrderedDict(
            expert=e,
            oracle_count=int(oracle_dist.get(e, 0)),
            native_selected_count=int(native_dist.get(e, 0)),
            fixed_global_mae=float(expert_globals[e]),
            fixed_sample_mae_mean=float(np.nanmean(expert_maes[:, e])),
        ))
    expert_summary = OrderedDict(
        env_oracle_samples=int(env_oracle.sum()),
        exact_expert_selection_accuracy=exact_expert_acc,
        top2_expert_selection_accuracy=top2_acc,
        tolerance_1pct_expert_accuracy=tol_acc,
        mean_expert_selection_regret=float(np.nanmean(expert_regret)),
        median_expert_selection_regret=float(np.nanmedian(expert_regret)),
    )
    write_csv(os.path.join(output_dir, "expert_usage_distribution.csv"), expert_rows, list(expert_rows[0].keys()))
    write_csv(os.path.join(output_dir, "summary_expert_selection.csv"), [expert_summary], list(expert_summary.keys()))

    # Overall summaries.
    reported_test_mixed = safe_float(summary.get("test_mixed_mae"))
    reported_test_avg = safe_float(summary.get("test_avg_mae"))
    native_matches_reported_mixed = bool(np.isfinite(reported_test_mixed) and abs(native_global - reported_test_mixed) <= 1e-4)
    if np.isfinite(reported_test_mixed) and not native_matches_reported_mixed:
        raise AssertionError(
            "native global MAE mismatch: reconstructed={} reported_test_mixed={}".format(native_global, reported_test_mixed)
        )
    overall_rows = []
    for name, global_mae, sample_mean in [
        ("native_fpem", native_global, float(np.nanmean(native_mae))),
        ("invariant_only", inv_global, float(np.nanmean(inv_mae))),
        ("native_argmax", argmax_global, float(np.nanmean(argmax_mae))),
        ("uniform_expert_fusion", uniform_global, float(np.nanmean(uniform_mae))),
        ("oracle", oracle_global, oracle_sample_mean),
    ]:
        overall_rows.append(OrderedDict(candidate=name, global_masked_mae=global_mae, samplewise_masked_mae_mean=sample_mean))
    for e in range(k):
        overall_rows.append(OrderedDict(
            candidate="fixed_expert_{}".format(e),
            global_masked_mae=float(expert_globals[e]),
            samplewise_masked_mae_mean=float(np.nanmean(expert_maes[:, e])),
        ))
    overall_rows.append(OrderedDict(
        candidate="random_routing_mean",
        global_masked_mae=float(np.mean(random_globals)),
        samplewise_masked_mae_mean=float(np.mean(random_sample_means)),
    ))
    overall_rows.append(OrderedDict(
        candidate="random_routing_std",
        global_masked_mae=float(np.std(random_globals)),
        samplewise_masked_mae_mean=float(np.std(random_sample_means)),
    ))
    write_csv(os.path.join(output_dir, "summary_overall.csv"), overall_rows, list(overall_rows[0].keys()))

    route_quality = OrderedDict()
    route_quality["exp_dir"] = exp_dir
    route_quality["checkpoint"] = checkpoint
    route_quality["checkpoint_epoch"] = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    route_quality["reported_test_mixed_mae"] = reported_test_mixed
    route_quality["reported_test_avg_mae"] = reported_test_avg
    route_quality["reconstructed_native_global_masked_mae"] = native_global
    route_quality["native_matches_reported_test_mixed"] = native_matches_reported_mixed
    route_quality["native_samplewise_masked_mae_mean"] = float(np.nanmean(native_mae))
    route_quality["invariant_global_masked_mae"] = inv_global
    route_quality["invariant_samplewise_masked_mae_mean"] = float(np.nanmean(inv_mae))
    route_quality["expert_global_masked_mae"] = {str(i): float(expert_globals[i]) for i in range(k)}
    route_quality["best_fixed_route_global"] = best_fixed_name
    route_quality["best_fixed_route_global_mae"] = best_fixed_global
    route_quality["best_fixed_env_route_global"] = best_fixed_env_name
    route_quality["best_fixed_env_global_mae"] = best_fixed_for_gap
    route_quality["uniform_global_masked_mae"] = uniform_global
    route_quality["random_global_masked_mae_mean"] = float(np.mean(random_globals))
    route_quality["random_global_masked_mae_std"] = float(np.std(random_globals))
    route_quality["oracle_global_masked_mae"] = oracle_global
    route_quality["oracle_gap_best_fixed_env_minus_oracle"] = oracle_gap
    route_quality["gap_closed_by_fpem_vs_best_fixed_env"] = gap_closed
    route_quality["mean_route_regret"] = float(np.nanmean(route_regret))
    route_quality["median_route_regret"] = float(np.nanmedian(route_regret))
    route_quality["p90_route_regret"] = float(np.nanpercentile(route_regret, 90))
    route_quality["max_route_regret"] = float(np.nanmax(route_regret))
    route_quality["environment_preferred_count"] = int(env_pref.sum())
    route_quality["environment_preferred_prop"] = float(env_pref.mean())
    route_quality["invariant_preferred_count"] = int(inv_pref.sum())
    route_quality["invariant_preferred_prop"] = float(inv_pref.mean())
    route_quality["neutral_count"] = int(neutral.sum())
    route_quality["neutral_prop"] = float(neutral.mean())
    route_quality["harmful_correction_rate"] = harm_rate
    route_quality["missed_environment_gain_rate"] = missed_gain_rate
    route_quality["avg_available_env_gain_recovered_by_fpem"] = float(np.nanmean(recovered_gain_env)) if recovered_gain_env.size else float("nan")
    route_quality["exact_expert_selection_accuracy"] = exact_expert_acc
    route_quality["top2_expert_selection_accuracy"] = top2_acc
    route_quality["tolerance_1pct_expert_accuracy"] = tol_acc
    route_quality["correction_norm_spearman_with_true_env_gain"] = corr_gain_spearman
    route_quality["routing_entropy_spearman_with_true_env_gain"] = entropy_gain_spearman
    route_quality["max_router_prob_spearman_with_true_env_gain"] = maxprob_gain_spearman
    route_quality["has_explicit_identity_route"] = arch["has_explicit_identity_route"]
    route_quality["can_explicitly_choose_no_environment"] = arch["can_explicitly_choose_no_environment"]
    route_quality["binary_environment_use_metrics_available"] = bool(arch["can_explicitly_choose_no_environment"])
    route_quality["external_steve_consistency_status"] = steve_status
    route_quality["top_level_sources_observed"] = collected["top_level_sources"]
    route_quality["route_head_modes_observed"] = collected["route_head_modes"]
    route_quality["proto_modes_observed"] = collected["proto_modes"]
    route_quality["state_changed_during_audit"] = False
    route_quality["test_loader_shuffle"] = False
    route_quality["conclusion_note"] = (
        "This checkpoint has no explicit identity/no-environment route, so exact binary "
        "environment-use accuracy is not measurable; the audit evaluates expert selection "
        "and native correction quality against analytical invariant/expert candidates."
    )
    route_quality.update({"architecture": arch})
    route_quality.update({"expert_selection_summary": expert_summary})
    with open(os.path.join(output_dir, "summary_route_quality.json"), "w", encoding="utf-8") as f:
        json.dump(route_quality, f, indent=2)

    # Save full predictions.
    pred_kwargs = {
        "sample_index": sample_index,
        "target": target,
        "y_native": y_native,
        "y_invariant": y_inv,
        "y_native_argmax": y_argmax,
        "y_uniform": y_uniform,
        "y_experts": y_experts,
        "router_probs": q,
        "selected_route": selected_route,
        "env_use_score": env_use_score,
        "correction_norm": correction_norm,
        "routing_entropy": routing_entropy,
        "max_router_prob": max_router_prob,
        "native_mae": native_mae,
        "invariant_mae": inv_mae,
        "expert_maes": expert_maes,
        "oracle_mae": oracle_mae,
        "oracle_idx": oracle_idx,
        "best_env_mae": best_env_mae,
        "environment_gain": env_gain,
        "route_regret": route_regret,
        "valid_count": valid_count,
    }
    np.savez(os.path.join(output_dir, "predictions_all_routes.npz"), **pred_kwargs)

    # Figures.
    save_sorted_gain(env_gain, native_group_ok, os.path.join(output_dir, "environment_gain_sorted.png"))
    labels = candidate_names
    matrix = np.zeros((len(candidate_names), k), dtype=np.int64)
    for oi, nr in zip(oracle_idx, selected_route):
        matrix[int(oi), int(nr)] += 1
    save_matrix(matrix, labels, ["native_{}".format(i) for i in range(k)], os.path.join(output_dir, "oracle_vs_native_route_confusion.png"), "Oracle route vs native selected expert")
    save_hist(route_regret, os.path.join(output_dir, "route_regret_histogram.png"), "Route regret: native MAE - oracle MAE")
    save_scatter(oracle_mae, native_mae, os.path.join(output_dir, "native_vs_oracle_scatter.png"), "Native vs Oracle sample MAE", "oracle MAE")
    save_bar(candidate_names, [int((oracle_idx == i).sum()) for i in range(len(candidate_names))], os.path.join(output_dir, "oracle_route_distribution.png"), "Oracle route distribution")
    save_groupwise_table(group_rows, os.path.join(output_dir, "groupwise_counterfactual_mae.png"))
    save_bar([str(r["bin"]) for r in calib_rows], [int(round(1000 * r["env_preferred_prop"])) for r in calib_rows], os.path.join(output_dir, "environment_use_calibration.png"), "Proxy calibration: env-preferred proportion x1000")
    save_bar(["oracle_{}".format(i) for i in range(k)] + ["native_{}".format(i) for i in range(k)],
             [int(oracle_dist.get(i, 0)) for i in range(k)] + [int(native_dist.get(i, 0)) for i in range(k)],
             os.path.join(output_dir, "expert_usage_vs_oracle.png"), "Expert usage vs oracle")

    batch_check = validate_batch_size_independence(args, exp_dir, checkpoint, output_dir)
    route_quality["batch_size_invariance_check"] = batch_check
    with open(os.path.join(output_dir, "summary_route_quality.json"), "w", encoding="utf-8") as f:
        json.dump(route_quality, f, indent=2)

    print("[fpem-audit] native_global_masked_mae={:.6f}".format(native_global))
    print("[fpem-audit] invariant_global_masked_mae={:.6f}".format(inv_global))
    print("[fpem-audit] expert_globals={}".format({i: expert_globals[i] for i in range(k)}))
    print("[fpem-audit] best_fixed_env={} {:.6f}".format(best_fixed_env_name, best_fixed_for_gap))
    print("[fpem-audit] uniform_global_masked_mae={:.6f}".format(uniform_global))
    print("[fpem-audit] oracle_global_masked_mae={:.6f}".format(oracle_global))
    print("[fpem-audit] gap_closed={:.6f}".format(gap_closed))
    print("[fpem-audit] groups env={} inv={} neutral={}".format(int(env_pref.sum()), int(inv_pref.sum()), int(neutral.sum())))
    print("[fpem-audit] harmful_rate={:.6f} missed_gain_rate={:.6f}".format(harm_rate, missed_gain_rate))
    print("[fpem-audit] exact_expert_acc={:.6f} tol_acc={:.6f}".format(exact_expert_acc, tol_acc))
    print("[fpem-audit] correction_gain_spearman={:.6f}".format(corr_gain_spearman))
    print("[fpem-audit] output_dir={}".format(output_dir))


if __name__ == "__main__":
    main()
