#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Case-study pipeline for Progressive-GMM K=3/common=0.20 FPEM experiments.

Inference/analysis only.  This script restores a completed checkpoint, reuses
the authoritative online-routing evaluator utilities for cluster/expert routing
logic, and regenerates all tables/figures from the Progressive-GMM checkpoint.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False
    from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run_tds_nyctaxi import build_model, build_tds_data, load_checkpoint, load_graph, to_device  # noqa: E402
from tools.evaluate_online_expert_routing import (  # noqa: E402
    abs_sum_count,
    apply_cluster_mapping,
    compute_validation_cross_mae,
    effective_expert_number,
    hungarian_mapping,
    independent_mapping,
    namespace_from_checkpoint,
    one_hot,
    per_sample_expert_mae,
    predict_from_probs,
    route_usage,
    safe_float,
    split_mae_mape,
    usage_entropy,
)


RUN_PREFIX = "fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_progressive_gmm_0720"
CASE_NAME = "add_progressive_gmm_kmax3_common020"
DEFAULT_RESULT_ROOT = PROJECT / "experiments" / "NYCTaxi_TDS"
DEFAULT_ROUTE_EVAL_ROOT = DEFAULT_RESULT_ROOT / (RUN_PREFIX + "_online_route_eval")
MASK_VALUE = 5.0
RUSH_HOURS = set([7, 8, 9, 17, 18, 19])


def _font(size=12):
    if HAS_MPL:
        return None
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _norm_color(value, vmin, vmax):
    try:
        x = (float(value) - float(vmin)) / max(float(vmax) - float(vmin), 1e-12)
    except Exception:
        x = 0.0
    x = max(0.0, min(1.0, x))
    # Blue -> cyan -> yellow -> red, readable enough for diagnostics.
    if x < 0.5:
        t = x / 0.5
        r = int(40 * (1 - t) + 80 * t)
        g = int(80 * (1 - t) + 200 * t)
        b = int(180 * (1 - t) + 180 * t)
    else:
        t = (x - 0.5) / 0.5
        r = int(80 * (1 - t) + 220 * t)
        g = int(200 * (1 - t) + 60 * t)
        b = int(180 * (1 - t) + 40 * t)
    return (r, g, b)


def _palette(idx):
    colors = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ]
    return colors[int(idx) % len(colors)]


def _save_pil_heatmap(matrix, path, title, xlabels=None, ylabels=None, annotations=None):
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = matrix.shape
    cell_w, cell_h = 90, 60
    left, top, right, bottom = 100, 70, 30, 70
    img = Image.new("RGB", (left + cols * cell_w + right, top + rows * cell_h + bottom), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_text, f_small = _font(16), _font(11), _font(9)
    draw.text((12, 12), str(title), fill="black", font=f_title)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    for i in range(rows):
        draw.text((8, top + i * cell_h + cell_h // 2 - 8), (ylabels or ["Cluster {}".format(x) for x in range(rows)])[i], fill="black", font=f_text)
        for j in range(cols):
            x0, y0 = left + j * cell_w, top + i * cell_h
            color = _norm_color(matrix[i, j], vmin, vmax) if np.isfinite(matrix[i, j]) else (220, 220, 220)
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color, outline="white")
            txt = "{:.3f}".format(matrix[i, j]) if np.isfinite(matrix[i, j]) else "NA"
            if annotations and (i, j) in annotations:
                txt += "\n" + annotations[(i, j)]
            draw.multiline_text((x0 + 8, y0 + 14), txt, fill="black", font=f_small, spacing=2)
    for j in range(cols):
        draw.text((left + j * cell_w + 8, top + rows * cell_h + 8), (xlabels or ["Expert {}".format(x) for x in range(cols)])[j], fill="black", font=f_small)
    mkdir(Path(path).parent)
    img.save(str(path))


def _save_pil_bar(labels, values, path, title, ylabel="value", colors=None):
    labels = [str(x) for x in labels]
    values = np.asarray(values, dtype=np.float64)
    w, h = max(720, 90 * len(labels)), 480
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_text, f_small = _font(16), _font(11), _font(9)
    draw.text((15, 12), str(title), fill="black", font=f_title)
    left, top, bottom, right = 70, 70, 120, 25
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = float(np.nanmax(values)) if values.size else 1.0
    vmin = min(0.0, float(np.nanmin(values)) if values.size else 0.0)
    draw.line([left, top, left, top + plot_h, left + plot_w, top + plot_h], fill="black")
    draw.text((8, top + plot_h // 2), str(ylabel), fill="black", font=f_small)
    bw = max(8, int(plot_w / max(len(labels), 1) * 0.65))
    for i, v in enumerate(values):
        x = left + int((i + 0.5) * plot_w / max(len(labels), 1))
        y = top + plot_h - int((float(v) - vmin) / max(vmax - vmin, 1e-12) * plot_h)
        color = colors[i] if colors and i < len(colors) else _palette(i)
        if isinstance(color, str) and color.startswith("#") and len(color) == 7:
            color = tuple(int(color[j:j + 2], 16) for j in (1, 3, 5))
        draw.rectangle([x - bw // 2, y, x + bw // 2, top + plot_h], fill=tuple(color), outline="black")
        draw.text((x - bw, y - 14), "{:.3g}".format(v), fill="black", font=f_small)
        draw.text((x - bw, top + plot_h + 8), labels[i][:18], fill="black", font=f_small)
    mkdir(Path(path).parent)
    img.save(str(path))


def _save_pil_scatter(coords, color_values, path, title, discrete=True):
    coords = np.asarray(coords, dtype=np.float64)
    w, h = 640, 540
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_small = _font(16), _font(9)
    draw.text((15, 12), str(title), fill="black", font=f_title)
    left, top, right, bottom = 55, 60, 25, 45
    plot_w, plot_h = w - left - right, h - top - bottom
    if coords.shape[0] == 0:
        img.save(str(path)); return
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="black")
    vals = np.asarray(color_values)
    for i, xy in enumerate(coords):
        x = left + int((xy[0] - xmin) / max(xmax - xmin, 1e-12) * plot_w)
        y = top + plot_h - int((xy[1] - ymin) / max(ymax - ymin, 1e-12) * plot_h)
        if discrete:
            unique = sorted(np.unique(vals).tolist())
            color = _palette(unique.index(vals[i]) if vals[i] in unique else 0)
        else:
            color = _norm_color(float(vals[i]), float(np.nanmin(vals)), float(np.nanmax(vals)))
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    if discrete:
        for n, v in enumerate(sorted(np.unique(vals).tolist())[:12]):
            draw.rectangle([left + 8 + n * 48, h - 25, left + 20 + n * 48, h - 13], fill=_palette(n))
            draw.text((left + 23 + n * 48, h - 27), str(v)[:6], fill="black", font=f_small)
    mkdir(Path(path).parent)
    img.save(str(path))


def _save_pil_lines(series, path, title, labels=None):
    w, h = 860, 500
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_small = _font(15), _font(9)
    draw.text((15, 12), str(title), fill="black", font=f_title)
    left, top, right, bottom = 55, 70, 25, 60
    plot_w, plot_h = w - left - right, h - top - bottom
    all_y = np.concatenate([np.asarray(y, dtype=np.float64).reshape(-1) for _x, y in series if len(y)])
    ymin, ymax = float(all_y.min()), float(all_y.max())
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="black")
    for sidx, (x, y) in enumerate(series):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size < 1:
            continue
        xmin, xmax = float(x.min()), float(x.max())
        pts = []
        for xx, yy in zip(x, y):
            px = left + int((xx - xmin) / max(xmax - xmin, 1e-12) * plot_w)
            py = top + plot_h - int((yy - ymin) / max(ymax - ymin, 1e-12) * plot_h)
            pts.append((px, py))
        color = _palette(sidx)
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        for px, py in pts:
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
        if labels:
            draw.text((left + 8 + (sidx % 4) * 190, h - 48 + (sidx // 4) * 13), str(labels[sidx])[:28], fill=color, font=f_small)
    mkdir(Path(path).parent)
    img.save(str(path))


def _save_pil_hist_groups(groups, path, title):
    labels = [g[0] for g in groups]
    means = [float(np.mean(g[1])) if len(g[1]) else 0.0 for g in groups]
    _save_pil_bar(labels, means, path, title, ylabel="mean per-sample |delta|")


def log(msg):
    print(msg, flush=True)


def truthy(value):
    return str(value).strip().lower() in set(["1", "true", "yes", "y", "on"])


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    mkdir(Path(path).parent)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=safe_float)


def write_csv(path, rows, preferred=None):
    mkdir(Path(path).parent)
    keys = []
    for k in preferred or []:
        if k not in keys:
            keys.append(k)
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for k in keys:
                v = row.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            writer.writerow(out)


def write_tsv_matrix(path, matrix, row_name="cluster"):
    mkdir(Path(path).parent)
    matrix = np.asarray(matrix, dtype=np.float64)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row_name] + ["expert_{}".format(i) for i in range(matrix.shape[1])])
        for i in range(matrix.shape[0]):
            writer.writerow([i] + [float(matrix[i, j]) for j in range(matrix.shape[1])])


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def labels_from_time_label(time_label):
    arr = np.asarray(time_label, dtype=np.int64).reshape(-1)
    hour = np.mod(arr, 24)
    workday = arr < 24
    rush = np.asarray([(int(h) in RUSH_HOURS) for h in hour], dtype=np.bool_)
    return {
        "hour": hour.astype(np.int64),
        "day": np.where(workday, 1, 0).astype(np.int64),
        "workday": workday.astype(np.int64),
        "workday_or_holiday": np.where(workday, "workday", "holiday"),
        "rush_hour": rush.astype(np.int64),
        "rush_or_nonrush": np.where(rush, "rush", "non_rush"),
    }


def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(np.clip(x, -50.0, 50.0))
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def entropy_np(prob):
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0)
    return (-(p * np.log(p)).sum(axis=1)).astype(np.float32)


def per_sample_mae(pred_raw, target_raw):
    return per_sample_expert_mae(np.asarray(pred_raw)[:, None, ...], np.asarray(target_raw))[:, 0]


def vector_features(nodes):
    # nodes: [N, num_nodes, hidden]
    nodes = np.asarray(nodes, dtype=np.float32)
    return np.concatenate([nodes.mean(axis=1), nodes.std(axis=1)], axis=1).astype(np.float32)


def maybe_limit(data, max_samples):
    if max_samples is None or int(max_samples) <= 0:
        return data
    n = min(int(max_samples), int(data["targets_raw"].shape[0]))
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.shape[0] == data["targets_raw"].shape[0]:
            out[k] = v[:n]
        else:
            out[k] = v
    return out


def load_mapping(route_eval_dir, k):
    mapping_path = Path(route_eval_dir) / "cluster_to_expert_mapping.json"
    if not mapping_path.exists():
        raise RuntimeError("missing cluster_to_expert_mapping.json: {}".format(mapping_path))
    payload = load_json(mapping_path)
    def parse_map(name, default=None):
        src = payload.get(name, default or {})
        return {int(a): int(b) for a, b in src.items()}
    identity = parse_map("identity_mapping", {str(i): i for i in range(k)})
    hungarian = parse_map("hungarian_mapping")
    independent = parse_map("independent_mapping")
    cross = np.asarray(payload.get("validation_cross_mae"), dtype=np.float64)
    if cross.shape != (k, k):
        cross_file = Path(route_eval_dir) / "validation_cross_mae.npy"
        if cross_file.exists():
            cross = np.load(str(cross_file))
    if cross.shape != (k, k):
        raise RuntimeError("invalid validation_cross_mae shape: {}".format(cross.shape))
    return payload, identity, hungarian, independent, cross


def load_route_results(route_eval_dir):
    path = Path(route_eval_dir) / "online_route_results.json"
    if not path.exists():
        raise RuntimeError("missing online_route_results.json: {}".format(path))
    data = load_json(path)
    rows = data.get("test_results", [])
    best_fixed = data.get("validation_best_fixed_expert_id")
    if best_fixed is None:
        for row in rows:
            if row.get("routing_method") == "best_fixed":
                best_fixed = row.get("best_fixed_expert_id")
                break
    if best_fixed is None:
        raise RuntimeError("could not determine validation-selected best_fixed expert from {}".format(path))
    return data, int(best_fixed)


def build_runtime(checkpoint, output_dir, device):
    ckpt = load_checkpoint(str(checkpoint), device)
    args_cli = argparse.Namespace(
        device=device,
        batch_size=int((ckpt.get("args") or {}).get("test_batch_size", 16)),
        output_dir=str(output_dir),
    )
    args = namespace_from_checkpoint(ckpt.get("args", {}), args_cli)
    loaders, scaler, counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, _lr = build_model(args, graph)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    if "fpem_progressive_gmm_state" not in ckpt:
        raise RuntimeError("checkpoint has no fpem_progressive_gmm_state; refusing to initialize a new GMM")
    if hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
        model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
    else:
        raise RuntimeError("model does not expose load_progressive_gmm_state_from_checkpoint")
    model.eval()
    active_k = int(model.progressive_active_cluster_count.detach().cpu().item())
    if active_k != 3:
        raise RuntimeError("expected active_k == 3 for main Progressive GMM experiments, got {}".format(active_k))
    return ckpt, args, loaders, scaler, counts, graph, model, load_result, active_k


def collect_split(model, loader, scaler, args, split_name, max_samples=-1):
    model.eval()
    chunks = defaultdict(list)
    total = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, args.device)
            if len(batch) == 5:
                x, y, time_label, c, sample_index = batch
            elif len(batch) == 4:
                x, y, time_label, c = batch
                sample_index = torch.arange(total, total + x.shape[0], device=x.device, dtype=torch.long)
            else:
                raise RuntimeError("case study expects x,y,time_label,c[,sample_index] batches")
            out = model.forward_output(x, exog=c, time_label=time_label, training=False, sample_index=sample_index)
            heads = out.get("y_hyper_heads", out.get("y_route_heads"))
            if heads is None:
                raise RuntimeError("model output does not expose expert predictions")
            teacher = out.get("progressive_teacher_embedding")
            if teacher is None or not torch.is_tensor(teacher):
                raise RuntimeError("progressive teacher embedding is unavailable; GMM state may not be restored")
            logits = model._progressive_gmm_logits(teacher).detach().float()
            post = torch.softmax(logits, dim=1)
            cluster = logits.argmax(dim=1).long()
            route_cluster = out.get("progressive_cluster_id")
            if route_cluster is not None and torch.is_tensor(route_cluster):
                if bool((route_cluster.detach().long().cpu() != cluster.detach().cpu()).any()):
                    warnings.warn("{}: recomputed cluster IDs differ from forward output; using recomputed teacher GMM IDs".format(split_name))
            chunks["x"].append(x.detach().cpu().float().numpy())
            chunks["target"].append(y.detach().cpu().float().numpy())
            chunks["time_label"].append(time_label.detach().cpu().long().numpy().reshape(-1))
            chunks["c"].append(c.detach().cpu().float().numpy())
            chunks["sample_index"].append(sample_index.detach().cpu().long().numpy().reshape(-1))
            chunks["y_heads"].append(heads.detach().cpu().float().numpy())
            chunks["y_inv"].append(out["y_inv"].detach().cpu().float().numpy())
            chunks["z_inv"].append(out["Z_inv"].detach().cpu().float().numpy())
            chunks["e_env"].append(out["E_useful"].detach().cpu().float().numpy())
            chunks["teacher_embedding"].append(teacher.detach().cpu().float().numpy())
            chunks["gmm_logits"].append(logits.cpu().numpy())
            chunks["gmm_posterior"].append(post.cpu().numpy())
            chunks["gmm_cluster_id"].append(cluster.cpu().numpy())
            total += int(x.shape[0])
            if max_samples is not None and int(max_samples) > 0 and total >= int(max_samples):
                break
    if not chunks:
        raise RuntimeError("no samples collected for split={}".format(split_name))
    data = {}
    for k, vals in chunks.items():
        data[k] = np.concatenate(vals, axis=0)
    data["split"] = split_name
    data["x_raw"] = scaler.inverse_transform(torch.from_numpy(data["x"]).float()).detach().cpu().numpy().astype(np.float32)
    data["targets_raw"] = scaler.inverse_transform(torch.from_numpy(data["target"]).float()).detach().cpu().numpy().astype(np.float32)
    data["y_heads_raw"] = scaler.inverse_transform(torch.from_numpy(data["y_heads"]).float()).detach().cpu().numpy().astype(np.float32)
    data["y_inv_raw"] = scaler.inverse_transform(torch.from_numpy(data["y_inv"]).float()).detach().cpu().numpy().astype(np.float32)
    data["per_sample_expert_mae"] = per_sample_expert_mae(data["y_heads_raw"], data["targets_raw"])
    data["oracle_expert_id"] = data["per_sample_expert_mae"].argmin(axis=1).astype(np.int64)
    # Alias required by tools.evaluate_online_expert_routing.compute_validation_cross_mae.
    data["cluster_id"] = data["gmm_cluster_id"]
    return maybe_limit(data, max_samples)


def add_predictions(data, identity, hungarian, independent, best_fixed):
    k = int(data["y_heads_raw"].shape[1])
    cluster = data["gmm_cluster_id"].astype(np.int64)
    data["selected_expert_identity"] = apply_cluster_mapping(cluster, identity, k)
    data["selected_expert_hungarian"] = apply_cluster_mapping(cluster, hungarian, k)
    data["selected_expert_independent"] = apply_cluster_mapping(cluster, independent, k)
    data["best_fixed_expert_id"] = np.full(cluster.shape, int(best_fixed), dtype=np.int64)
    data["identity_prediction"] = predict_from_probs(data["y_heads_raw"], one_hot(data["selected_expert_identity"], k))
    data["gmm_hungarian_prediction"] = predict_from_probs(data["y_heads_raw"], one_hot(data["selected_expert_hungarian"], k))
    data["independent_prediction"] = predict_from_probs(data["y_heads_raw"], one_hot(data["selected_expert_independent"], k))
    data["best_fixed_prediction"] = predict_from_probs(data["y_heads_raw"], one_hot(data["best_fixed_expert_id"], k))
    data["uniform_prediction"] = data["y_heads_raw"].mean(axis=1).astype(np.float32)
    data["oracle_prediction"] = predict_from_probs(data["y_heads_raw"], one_hot(data["oracle_expert_id"], k))
    data["invariant_mae"] = per_sample_mae(data["y_inv_raw"], data["targets_raw"])
    data["gmm_hungarian_mae"] = per_sample_mae(data["gmm_hungarian_prediction"], data["targets_raw"])
    data["uniform_mae"] = per_sample_mae(data["uniform_prediction"], data["targets_raw"])
    data["best_fixed_mae"] = per_sample_mae(data["best_fixed_prediction"], data["targets_raw"])
    data["oracle_mae"] = per_sample_mae(data["oracle_prediction"], data["targets_raw"])
    data["identity_mae"] = per_sample_mae(data["identity_prediction"], data["targets_raw"])
    data["independent_mae"] = per_sample_mae(data["independent_prediction"], data["targets_raw"])
    data["routing_gain_vs_uniform"] = data["uniform_mae"] - data["gmm_hungarian_mae"]
    data["routing_gain_vs_best_fixed"] = data["best_fixed_mae"] - data["gmm_hungarian_mae"]
    data["oracle_gap"] = data["gmm_hungarian_mae"] - data["oracle_mae"]
    data["gmm_confidence"] = data["gmm_posterior"].max(axis=1)
    data["gmm_entropy"] = entropy_np(data["gmm_posterior"])
    return data


def per_sample_rows(data):
    labels = labels_from_time_label(data["time_label"])
    rows = []
    k = int(data["y_heads_raw"].shape[1])
    for i in range(data["targets_raw"].shape[0]):
        row = OrderedDict()
        row["split"] = data["split"]
        row["sample_index"] = int(data["sample_index"][i])
        row["temporal_index"] = int(i)
        row["hour"] = int(labels["hour"][i])
        row["day"] = int(labels["day"][i])
        row["workday_or_holiday"] = str(labels["workday_or_holiday"][i])
        row["rush_hour"] = int(labels["rush_hour"][i])
        row["gmm_cluster_id"] = int(data["gmm_cluster_id"][i])
        for j in range(k):
            row["gmm_posterior_{}".format(j)] = float(data["gmm_posterior"][i, j])
        row["gmm_confidence"] = float(data["gmm_confidence"][i])
        row["gmm_entropy"] = float(data["gmm_entropy"][i])
        row["selected_expert_hungarian"] = int(data["selected_expert_hungarian"][i])
        row["selected_expert_identity"] = int(data["selected_expert_identity"][i])
        row["selected_expert_independent"] = int(data["selected_expert_independent"][i])
        row["oracle_expert_id"] = int(data["oracle_expert_id"][i])
        row["best_fixed_expert_id"] = int(data["best_fixed_expert_id"][i])
        row["input_mean"] = float(data["x_raw"][i].mean())
        row["input_std"] = float(data["x_raw"][i].std())
        row["input_min"] = float(data["x_raw"][i].min())
        row["input_max"] = float(data["x_raw"][i].max())
        row["target_mean"] = float(data["targets_raw"][i].mean())
        row["external_load_mean"] = float(data["c"][i].mean())
        row["invariant_mae"] = float(data["invariant_mae"][i])
        for j in range(k):
            row["expert_{}_mae".format(j)] = float(data["per_sample_expert_mae"][i, j])
        row["gmm_hungarian_mae"] = float(data["gmm_hungarian_mae"][i])
        row["uniform_mae"] = float(data["uniform_mae"][i])
        row["best_fixed_mae"] = float(data["best_fixed_mae"][i])
        row["oracle_mae"] = float(data["oracle_mae"][i])
        row["routing_gain_vs_uniform"] = float(data["routing_gain_vs_uniform"][i])
        row["routing_gain_vs_best_fixed"] = float(data["routing_gain_vs_best_fixed"][i])
        row["oracle_gap"] = float(data["oracle_gap"][i])
        rows.append(row)
    return rows


def plot_heatmap(matrix, path, title, xlabels=None, ylabels=None, annotations=None, cmap="viridis"):
    if not HAS_MPL:
        _save_pil_heatmap(matrix, path, title, xlabels=xlabels, ylabels=ylabels, annotations=annotations)
        return
    matrix = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("GMM cluster")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(xlabels or ["Expert {}".format(i) for i in range(matrix.shape[1])])
    ax.set_yticklabels(ylabels or ["Cluster {}".format(i) for i in range(matrix.shape[0])])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = "{:.3f}".format(matrix[i, j]) if np.isfinite(matrix[i, j]) else "NA"
            if annotations and (i, j) in annotations:
                txt += "\n" + annotations[(i, j)]
            ax.text(j, i, txt, ha="center", va="center", color="white" if np.nanmean(matrix) < matrix[i, j] else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    mkdir(Path(path).parent)
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def plot_bar(labels, values, path, title, ylabel="MAE", colors=None, rotate=True):
    if not HAS_MPL:
        _save_pil_bar(labels, values, path, title, ylabel=ylabel, colors=colors)
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35 if rotate else 0, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    mkdir(Path(path).parent)
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def cluster_profiles(data, k):
    labels = labels_from_time_label(data["time_label"])
    rows = []
    n = max(int(data["targets_raw"].shape[0]), 1)
    for c in range(k):
        mask = data["gmm_cluster_id"] == c
        count = int(mask.sum())
        row = OrderedDict()
        row["split"] = data["split"]
        row["cluster"] = c
        row["sample_count"] = count
        row["sample_ratio"] = float(count / float(n))
        if count:
            row["workday_ratio"] = float(labels["workday"][mask].mean())
            row["holiday_ratio"] = float(1.0 - row["workday_ratio"])
            row["rush_ratio"] = float(labels["rush_hour"][mask].mean())
            row["mean_external_load"] = float(data["c"][mask].mean())
            row["mean_historical_flow"] = float(data["x_raw"][mask].mean())
            row["std_historical_flow"] = float(data["x_raw"][mask].std())
            row["mean_target_flow"] = float(data["targets_raw"][mask].mean())
            row["invariant_mae"] = float(np.nanmean(data["invariant_mae"][mask]))
            row["gmm_routed_mae"] = float(np.nanmean(data["gmm_hungarian_mae"][mask]))
            row["uniform_mae"] = float(np.nanmean(data["uniform_mae"][mask]))
            row["mean_correction_magnitude"] = float(np.abs(data["gmm_hungarian_prediction"][mask] - data["y_inv_raw"][mask]).mean())
            row["mean_gmm_confidence"] = float(data["gmm_confidence"][mask].mean())
            row["mean_gmm_entropy"] = float(data["gmm_entropy"][mask].mean())
            row["avg_consecutive_cluster_duration"] = mean_run_length(data["gmm_cluster_id"][mask]) if count else 0.0
        else:
            for key in [
                "workday_ratio", "holiday_ratio", "rush_ratio", "mean_external_load",
                "mean_historical_flow", "std_historical_flow", "mean_target_flow",
                "invariant_mae", "gmm_routed_mae", "uniform_mae", "mean_correction_magnitude",
                "mean_gmm_confidence", "mean_gmm_entropy", "avg_consecutive_cluster_duration",
            ]:
                row[key] = None
        rows.append(row)
    return rows


def categorical_distribution(data, by, values, value_name):
    rows = []
    clusters = sorted(np.unique(data["gmm_cluster_id"]).astype(int).tolist())
    for c in clusters:
        mask_c = data["gmm_cluster_id"] == c
        denom = max(int(mask_c.sum()), 1)
        for v in sorted(np.unique(values).tolist()):
            count = int(((values == v) & mask_c).sum())
            rows.append({"cluster": c, value_name: v, "count": count, "ratio_within_cluster": count / float(denom)})
    return rows


def mean_run_length(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return 0.0
    runs = []
    start = 0
    for i in range(1, labels.size):
        if labels[i] != labels[i - 1]:
            runs.append(i - start)
            start = i
    runs.append(labels.size - start)
    return float(np.mean(runs))


def make_cluster_plots(data, k, out_dir):
    labels = labels_from_time_label(data["time_label"])
    profiles = cluster_profiles(data, k)
    write_csv(out_dir / "cluster_profile.csv", profiles)
    hour_rows = categorical_distribution(data, "cluster", labels["hour"], "hour")
    work_rows = categorical_distribution(data, "cluster", labels["workday_or_holiday"], "workday_or_holiday")
    rush_rows = categorical_distribution(data, "cluster", labels["rush_or_nonrush"], "rush_or_nonrush")
    write_csv(out_dir / "cluster_hour_distribution.csv", hour_rows)
    write_csv(out_dir / "cluster_workday_distribution.csv", work_rows)
    write_csv(out_dir / "cluster_rush_distribution.csv", rush_rows)
    numeric_keys = [
        "sample_ratio", "workday_ratio", "rush_ratio", "mean_external_load", "mean_historical_flow",
        "std_historical_flow", "mean_target_flow", "invariant_mae", "gmm_routed_mae",
        "uniform_mae", "mean_correction_magnitude", "mean_gmm_confidence", "mean_gmm_entropy",
    ]
    mat = np.asarray([[float(r[k]) if r[k] is not None else np.nan for k in numeric_keys] for r in profiles], dtype=np.float64)
    z = mat.copy()
    for j in range(z.shape[1]):
        col = z[:, j]
        finite = np.isfinite(col)
        if finite.any():
            z[finite, j] = (col[finite] - col[finite].mean()) / max(col[finite].std(), 1e-8)
            z[~finite, j] = 0.0
    plot_heatmap(z, out_dir / "cluster_profile_heatmap.png", "{} cluster profile (standardized)".format(data["split"]), xlabels=numeric_keys)
    sizes = [profiles[i]["sample_count"] for i in range(k)]
    plot_bar(["Cluster {}".format(i) for i in range(k)], sizes, out_dir / "cluster_size.png", "Cluster size", ylabel="samples", rotate=False)

    hour_mat = np.zeros((k, 24), dtype=np.float64)
    for row in hour_rows:
        hour_mat[int(row["cluster"]), int(row["hour"])] = float(row["ratio_within_cluster"])
    plot_heatmap(hour_mat, out_dir / "cluster_by_hour.png", "Cluster by hour", xlabels=[str(i) for i in range(24)], cmap="magma")
    work_mat = np.zeros((k, 2), dtype=np.float64)
    work_labels = ["holiday", "workday"]
    for row in work_rows:
        col = 1 if row["workday_or_holiday"] == "workday" else 0
        work_mat[int(row["cluster"]), col] = float(row["ratio_within_cluster"])
    plot_heatmap(work_mat, out_dir / "cluster_by_workday_holiday.png", "Cluster by workday/holiday", xlabels=work_labels, cmap="Blues")
    rush_mat = np.zeros((k, 2), dtype=np.float64)
    for row in rush_rows:
        col = 1 if row["rush_or_nonrush"] == "rush" else 0
        rush_mat[int(row["cluster"]), col] = float(row["ratio_within_cluster"])
    plot_heatmap(rush_mat, out_dir / "cluster_by_rush_hour.png", "Cluster by rush hour", xlabels=["non_rush", "rush"], cmap="Greens")
    return profiles


def compute_embedding(features, method, seed):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), "none"
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=int(seed), n_neighbors=min(15, max(2, features.shape[0] - 1)))
            return reducer.fit_transform(features).astype(np.float32), "umap"
        except Exception as exc:
            warnings.warn("UMAP unavailable or failed ({}); falling back to PCA".format(exc))
            method = "pca"
    # PCA via SVD, no sklearn dependency.
    x = features - features.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32), "pca"
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    coords = np.dot(x, vt[:2].T)
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords[:, :2].astype(np.float32), "pca"


def scatter_plot(coords, color_values, path, title, discrete=True, cmap="tab10"):
    if not HAS_MPL:
        _save_pil_scatter(coords, color_values, path, title, discrete=discrete)
        return
    coords = np.asarray(coords)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    if discrete:
        vals = np.asarray(color_values)
        unique = sorted(np.unique(vals).tolist())
        for v in unique:
            mask = vals == v
            ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.8, label=str(v))
        ax.legend(markerscale=1.5, fontsize=8, loc="best")
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, s=12, alpha=0.8, cmap=cmap)
        fig.colorbar(sc, ax=ax, shrink=0.85)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    mkdir(Path(path).parent)
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def make_embeddings(data, k, out_dir, method, seed):
    labels = labels_from_time_label(data["time_label"])
    e_feat = vector_features(data["e_env"])
    z_feat = vector_features(data["z_inv"])
    e_coord, used_e = compute_embedding(e_feat, method, seed)
    z_coord, used_z = compute_embedding(z_feat, method, seed)
    rows = []
    for i in range(data["targets_raw"].shape[0]):
        rows.append({
            "split": data["split"],
            "sample_index": int(data["sample_index"][i]),
            "e_dim1": float(e_coord[i, 0]),
            "e_dim2": float(e_coord[i, 1]),
            "z_dim1": float(z_coord[i, 0]),
            "z_dim2": float(z_coord[i, 1]),
            "gmm_cluster_id": int(data["gmm_cluster_id"][i]),
            "selected_expert_hungarian": int(data["selected_expert_hungarian"][i]),
            "hour": int(labels["hour"][i]),
            "workday_or_holiday": str(labels["workday_or_holiday"][i]),
            "gmm_confidence": float(data["gmm_confidence"][i]),
            "embedding_method_e": used_e,
            "embedding_method_z": used_z,
        })
    write_csv(out_dir / "embedding_coordinates.csv", rows)
    scatter_plot(e_coord, data["gmm_cluster_id"], out_dir / "umap_e_env_by_gmm_cluster.png", "E_env by GMM cluster", True)
    scatter_plot(e_coord, data["selected_expert_hungarian"], out_dir / "umap_e_env_by_hungarian_expert.png", "E_env by Hungarian expert", True)
    scatter_plot(e_coord, labels["hour"], out_dir / "umap_e_env_by_hour.png", "E_env by hour", True)
    scatter_plot(e_coord, labels["workday_or_holiday"], out_dir / "umap_e_env_by_workday_holiday.png", "E_env by workday/holiday", True)
    scatter_plot(e_coord, data["gmm_confidence"], out_dir / "umap_e_env_by_gmm_confidence.png", "E_env by GMM confidence", False, cmap="viridis")
    scatter_plot(z_coord, data["gmm_cluster_id"], out_dir / "umap_z_inv_by_gmm_cluster.png", "Z_inv by GMM cluster", True)
    scatter_plot(z_coord, labels["hour"], out_dir / "umap_z_inv_by_hour.png", "Z_inv by hour", True)
    scatter_plot(z_coord, labels["workday_or_holiday"], out_dir / "umap_z_inv_by_workday_holiday.png", "Z_inv by workday/holiday", True)
    return e_feat, z_feat


def make_probe_results(e_feat, z_feat, data, out_dir, seed):
    rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        write_csv(out_dir / "linear_probe_results.csv", [{"status": "skipped", "reason": "sklearn unavailable: {}".format(exc)}])
        return rows
    labels = labels_from_time_label(data["time_label"])
    targets = OrderedDict([
        ("gmm_cluster", data["gmm_cluster_id"].astype(int)),
        ("workday", labels["workday"].astype(int)),
        ("rush_hour", labels["rush_hour"].astype(int)),
    ])
    feats = OrderedDict([("E_env", e_feat), ("Z_inv", z_feat)])
    for tname, y in targets.items():
        if len(np.unique(y)) < 2:
            continue
        for fname, x in feats.items():
            try:
                stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
                xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=int(seed), stratify=stratify)
                clf = LogisticRegression(max_iter=500, multi_class="auto")
                clf.fit(xtr, ytr)
                pred = clf.predict(xte)
                rows.append({
                    "feature": fname,
                    "target": tname,
                    "accuracy": float(accuracy_score(yte, pred)),
                    "f1_macro": float(f1_score(yte, pred, average="macro")),
                    "n_train": int(len(ytr)),
                    "n_test": int(len(yte)),
                })
            except Exception as exc:
                rows.append({"feature": fname, "target": tname, "status": "failed", "reason": str(exc)})
    write_csv(out_dir / "linear_probe_results.csv", rows)
    return rows


def cluster_expert_mae_csv(data, k, path):
    cross = compute_validation_cross_mae(data, k)
    rows = []
    for c in range(k):
        for e in range(k):
            rows.append({"split": data["split"], "cluster": c, "expert": e, "mae": float(cross[c, e])})
    write_csv(path, rows)
    return cross


def test_mapping_generalization(test_cross, identity, hungarian, independent, out_path):
    rows = []
    k = int(test_cross.shape[0])
    for c in range(k):
        best_test = int(np.nanargmin(test_cross[c]))
        selected = int(hungarian.get(c, c))
        rows.append({
            "cluster": c,
            "hungarian_selected_expert": selected,
            "identity_expert": int(identity.get(c, c)),
            "independent_validation_best_expert": int(independent.get(c, c)),
            "best_test_expert_diagnostic_only": best_test,
            "selected_expert_test_mae": float(test_cross[c, selected]),
            "best_test_expert_mae_diagnostic_only": float(test_cross[c, best_test]),
            "mae_gap_selected_minus_best_test": float(test_cross[c, selected] - test_cross[c, best_test]),
            "validation_mapping_generalized_to_test": bool(selected == best_test),
        })
    write_csv(out_path, rows)
    return rows


def make_routing_comparison(route_eval, out_dir):
    rows = route_eval.get("test_results", [])
    if not rows:
        return []
    methods = [r["routing_method"] for r in rows]
    values = [float(r["test_avg_mae"]) for r in rows]
    colors = []
    for r in rows:
        if r.get("uses_test_target"):
            colors.append("#d62728")
        elif int(r.get("active_experts_per_sample", 1)) > 1:
            colors.append("#2ca02c")
        elif r.get("deployable"):
            colors.append("#1f77b4")
        else:
            colors.append("#7f7f7f")
    plot_bar(methods, values, out_dir / "routing_method_comparison.png", "Routing baseline comparison", colors=colors)
    by = {r["routing_method"]: r for r in rows}
    primary = by.get("gmm_hard_val_hungarian")
    gain_rows = []
    if primary:
        gmm = float(primary["test_avg_mae"])
        for base in ["best_fixed", "random_uniform_top1", "random_prior_top1", "shuffled_gmm_route", "gmm_hard_identity", "uniform_all_experts", "oracle_top1"]:
            if base in by:
                b = float(by[base]["test_avg_mae"])
                gain_rows.append({
                    "comparison": "gmm_hard_val_hungarian_vs_{}".format(base),
                    "baseline_mae": b,
                    "gmm_hungarian_mae": gmm,
                    "gain_positive_means_gmm_better": b - gmm,
                    "oracle_gap_positive_means_oracle_better": gmm - b if base == "oracle_top1" else None,
                })
    write_csv(out_dir / "routing_gain_summary.csv", gain_rows)
    return gain_rows


def plot_typical_sample(data, idx, cluster, kind, path):
    k = int(data["y_heads_raw"].shape[1])
    hist = data["x_raw"][idx, ..., 0].mean(axis=1)
    target = data["targets_raw"][idx, ..., 0].mean(axis=1)
    inv = data["y_inv_raw"][idx, ..., 0].mean(axis=1)
    experts = [data["y_heads_raw"][idx, e, ..., 0].mean(axis=1) for e in range(k)]
    gmm = data["gmm_hungarian_prediction"][idx, ..., 0].mean(axis=1)
    uniform = data["uniform_prediction"][idx, ..., 0].mean(axis=1)
    oracle = data["oracle_prediction"][idx, ..., 0].mean(axis=1)
    t_hist = np.arange(hist.shape[0])
    t_future = np.arange(hist.shape[0], hist.shape[0] + target.shape[0])
    labels = labels_from_time_label(data["time_label"])
    title = (
        "Cluster {} {} sample={} hour={} {} {} | selected={} oracle={} conf={:.3f}; "
        "MAE gmm={:.3f} uniform={:.3f} best_fixed={:.3f} oracle={:.3f}"
    ).format(
        cluster, kind, int(data["sample_index"][idx]), int(labels["hour"][idx]),
        labels["workday_or_holiday"][idx], labels["rush_or_nonrush"][idx],
        int(data["selected_expert_hungarian"][idx]), int(data["oracle_expert_id"][idx]),
        float(data["gmm_confidence"][idx]),
        float(data["gmm_hungarian_mae"][idx]), float(data["uniform_mae"][idx]),
        float(data["best_fixed_mae"][idx]), float(data["oracle_mae"][idx]),
    )
    if not HAS_MPL:
        series = [(t_hist, hist), (t_future, target), (t_future, inv)]
        names = ["history input", "target", "frozen invariant"]
        for e in range(k):
            series.append((t_future, experts[e]))
            names.append("expert {}".format(e))
        series.extend([(t_future, gmm), (t_future, uniform), (t_future, oracle)])
        names.extend(["GMM Hungarian", "uniform", "oracle"])
        _save_pil_lines(series, path, title, labels=names)
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t_hist, hist, color="black", label="history input", linewidth=2)
    ax.plot(t_future, target, "o-", color="black", label="target")
    ax.plot(t_future, inv, "o--", label="frozen invariant")
    for e in range(k):
        ax.plot(t_future, experts[e], "o--", label="expert {}".format(e), alpha=0.75)
    ax.plot(t_future, gmm, "s-", label="GMM Hungarian", linewidth=2)
    ax.plot(t_future, uniform, "s--", label="uniform")
    ax.plot(t_future, oracle, "*-", label="oracle")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("relative time")
    ax.set_ylabel("mean flow channel 0")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    mkdir(Path(path).parent)
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def make_typical_samples(data, k, out_dir):
    rows = []
    for c in range(k):
        idxs = np.where(data["gmm_cluster_id"] == c)[0]
        if idxs.size == 0:
            continue
        gain = data["routing_gain_vs_uniform"][idxs]
        spread = data["per_sample_expert_mae"][idxs].max(axis=1) - data["per_sample_expert_mae"][idxs].min(axis=1)
        rules = [
            ("positive", idxs[int(np.nanargmax(gain))], "largest positive gain versus uniform"),
            ("neutral", idxs[int(np.nanargmin(np.abs(spread)))], "smallest expert-MAE spread"),
            ("failure", idxs[int(np.nanargmin(gain))], "largest negative gain versus uniform"),
        ]
        for kind, idx, rule in rules:
            fname = "cluster_{}_{}.png".format(c, kind)
            plot_typical_sample(data, int(idx), c, kind, out_dir / fname)
            rows.append({
                "cluster": c,
                "kind": kind,
                "selection_rule": rule,
                "sample_index": int(data["sample_index"][idx]),
                "temporal_index": int(idx),
                "selected_expert_hungarian": int(data["selected_expert_hungarian"][idx]),
                "oracle_expert_id": int(data["oracle_expert_id"][idx]),
                "gmm_confidence": float(data["gmm_confidence"][idx]),
                "gmm_entropy": float(data["gmm_entropy"][idx]),
                "gmm_hungarian_mae": float(data["gmm_hungarian_mae"][idx]),
                "uniform_mae": float(data["uniform_mae"][idx]),
                "best_fixed_mae": float(data["best_fixed_mae"][idx]),
                "oracle_mae": float(data["oracle_mae"][idx]),
                "routing_gain_vs_uniform": float(data["routing_gain_vs_uniform"][idx]),
                "oracle_gap": float(data["oracle_gap"][idx]),
                "figure": str(out_dir / fname),
            })
    write_csv(out_dir / "selected_case_metadata.csv", rows)
    return rows


def make_correction_analysis(data, k, out_dir):
    inv = data["y_inv_raw"]
    deltas = data["y_heads_raw"] - inv[:, None, ...]
    selected_delta = data["gmm_hungarian_prediction"] - inv
    rows = []
    for c in range(k):
        mask = data["gmm_cluster_id"] == c
        rows.append({
            "cluster": c,
            "sample_count": int(mask.sum()),
            "selected_correction_abs_mean": float(np.abs(selected_delta[mask]).mean()) if mask.any() else None,
            "invariant_mae_minus_selected_expert_mae": float(np.nanmean(data["invariant_mae"][mask] - data["gmm_hungarian_mae"][mask])) if mask.any() else None,
        })
    write_csv(out_dir / "cluster_correction_summary.csv", rows)
    plot_bar(["Cluster {}".format(r["cluster"]) for r in rows], [r["selected_correction_abs_mean"] or 0.0 for r in rows], out_dir / "cluster_correction_magnitude.png", "Selected correction magnitude", ylabel="mean |delta|", rotate=False)
    # Horizon x cluster. Current TDS target horizon is usually 1; keep generic.
    h = selected_delta.shape[1]
    mat = np.zeros((k, h), dtype=np.float64)
    for c in range(k):
        mask = data["gmm_cluster_id"] == c
        if mask.any():
            mat[c] = np.abs(selected_delta[mask]).mean(axis=(0, 2, 3))
    plot_heatmap(mat, out_dir / "cluster_horizon_correction_heatmap.png", "Cluster x horizon correction magnitude", xlabels=["h{}".format(i) for i in range(h)], cmap="magma")
    node_mag = np.abs(selected_delta).mean(axis=(0, 1, 3))
    top = np.argsort(-node_mag)[: min(30, node_mag.shape[0])]
    write_csv(out_dir / "top_nodes_by_correction.csv", [{"rank": i + 1, "node": int(n), "mean_abs_correction": float(node_mag[n])} for i, n in enumerate(top)])
    if not HAS_MPL:
        groups = []
        for c in range(k):
            mask = data["gmm_cluster_id"] == c
            vals = np.abs(selected_delta[mask]).reshape(max(int(mask.sum()), 1), -1).mean(axis=1) if mask.any() else np.asarray([])
            groups.append(("Cluster {}".format(c), vals))
        _save_pil_hist_groups(groups, out_dir / "selected_expert_correction_distribution.png", "Selected expert correction distribution")
        np.savez_compressed(str(out_dir / "correction_arrays.npz"), delta_expert=deltas, delta_selected=selected_delta)
        return rows
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in range(k):
        mask = data["gmm_cluster_id"] == c
        if mask.any():
            ax.hist(np.abs(selected_delta[mask]).reshape(mask.sum(), -1).mean(axis=1), bins=25, alpha=0.45, label="Cluster {}".format(c))
    ax.set_title("Selected expert correction distribution")
    ax.set_xlabel("per-sample mean |delta|")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_dir / "selected_expert_correction_distribution.png"), dpi=180)
    plt.close(fig)
    np.savez_compressed(str(out_dir / "correction_arrays.npz"), delta_expert=deltas, delta_selected=selected_delta)
    return rows


def make_confidence_analysis(val_data, test_data, out_dir):
    q = np.quantile(val_data["gmm_confidence"], [1.0 / 3.0, 2.0 / 3.0])
    bins = np.digitize(test_data["gmm_confidence"], q, right=False)
    names = ["low", "medium", "high"]
    rows = []
    for b, name in enumerate(names):
        mask = bins == b
        rows.append({
            "confidence_group": name,
            "lower_bound": float("-inf") if b == 0 else float(q[b - 1]),
            "upper_bound": float("inf") if b == 2 else float(q[b]),
            "sample_count": int(mask.sum()),
            "gmm_routing_mae": float(np.nanmean(test_data["gmm_hungarian_mae"][mask])) if mask.any() else None,
            "uniform_mae": float(np.nanmean(test_data["uniform_mae"][mask])) if mask.any() else None,
            "best_fixed_mae": float(np.nanmean(test_data["best_fixed_mae"][mask])) if mask.any() else None,
            "oracle_mae": float(np.nanmean(test_data["oracle_mae"][mask])) if mask.any() else None,
            "routing_gain_vs_uniform": float(np.nanmean(test_data["routing_gain_vs_uniform"][mask])) if mask.any() else None,
            "routing_gain_vs_best_fixed": float(np.nanmean(test_data["routing_gain_vs_best_fixed"][mask])) if mask.any() else None,
        })
    write_csv(out_dir / "confidence_bin_summary.csv", rows)
    for xkey, fname, xlabel in [
        ("gmm_confidence", "confidence_vs_gain.png", "GMM confidence"),
        ("gmm_entropy", "entropy_vs_gain.png", "GMM entropy"),
    ]:
        if not HAS_MPL:
            coords = np.stack([test_data[xkey], test_data["routing_gain_vs_uniform"]], axis=1)
            _save_pil_scatter(coords, test_data["routing_gain_vs_uniform"], out_dir / fname, "{} vs routing gain".format(xlabel), discrete=False)
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        ax.scatter(test_data[xkey], test_data["routing_gain_vs_uniform"], s=14, alpha=0.65)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("routing gain vs uniform (positive=GMM better)")
        ax.set_title("{} vs routing gain".format(xlabel))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(str(out_dir / fname), dpi=180)
        plt.close(fig)
    return rows


def make_temporal_analysis(data, k, out_dir):
    idx = np.asarray(data["sample_index"], dtype=np.int64)
    valid_order = bool(idx.size <= 1 or np.all(np.diff(idx) >= 0))
    payload = {"temporal_order_verified": valid_order}
    if not valid_order:
        write_json(out_dir / "temporal_skipped.json", payload)
        return payload
    cluster = data["gmm_cluster_id"].astype(np.int64)
    mat = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(cluster[:-1], cluster[1:]):
        mat[int(a), int(b)] += 1
    mat_prob = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    rows = []
    for i in range(k):
        for j in range(k):
            rows.append({"from_cluster": i, "to_cluster": j, "count": int(mat[i, j]), "probability": float(mat_prob[i, j])})
    write_csv(out_dir / "cluster_transition_matrix.csv", rows)
    plot_heatmap(mat_prob, out_dir / "cluster_transition_heatmap.png", "Cluster transition probability", cmap="Blues")
    if not HAS_MPL:
        _save_pil_lines([(np.arange(cluster.size), cluster)], out_dir / "cluster_timeline.png", "Cluster timeline", labels=["cluster"])
        dur_rows = []
        start = 0
        for i in range(1, cluster.size + 1):
            if i == cluster.size or cluster[i] != cluster[start]:
                dur_rows.append({"cluster": int(cluster[start]), "run_length": int(i - start)})
                start = i
        write_csv(out_dir / "cluster_duration_statistics.csv", dur_rows)
        payload.update({
            "self_transition_probability_mean": float(np.diag(mat_prob).mean()),
            "mean_run_length": float(np.mean([r["run_length"] for r in dur_rows])) if dur_rows else 0.0,
        })
        write_json(out_dir / "temporal_summary.json", payload)
        return payload
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(np.arange(cluster.size), cluster, linewidth=1)
    ax.set_title("Cluster timeline")
    ax.set_xlabel("sample order")
    ax.set_ylabel("cluster")
    ax.set_yticks(np.arange(k))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_dir / "cluster_timeline.png"), dpi=180)
    plt.close(fig)
    # Duration statistics.
    dur_rows = []
    start = 0
    for i in range(1, cluster.size + 1):
        if i == cluster.size or cluster[i] != cluster[start]:
            dur_rows.append({"cluster": int(cluster[start]), "run_length": int(i - start)})
            start = i
    write_csv(out_dir / "cluster_duration_statistics.csv", dur_rows)
    payload.update({
        "self_transition_probability_mean": float(np.diag(mat_prob).mean()),
        "mean_run_length": float(np.mean([r["run_length"] for r in dur_rows])) if dur_rows else 0.0,
    })
    write_json(out_dir / "temporal_summary.json", payload)
    return payload


def cross_mae_interpretation(cross, identity, hungarian, independent):
    k = int(cross.shape[0])
    identity_cost = sum(float(cross[c, identity.get(c, c)]) for c in range(k))
    hungarian_cost = sum(float(cross[c, hungarian.get(c, c)]) for c in range(k))
    independent_cost = sum(float(cross[c, independent.get(c, c)]) for c in range(k))
    row_best = {c: int(np.nanargmin(cross[c])) for c in range(k)}
    clear = []
    for c in range(k):
        vals = np.sort(cross[c])
        clear.append(float(vals[1] - vals[0]) if len(vals) > 1 else 0.0)
    return {
        "identity_mapping_is_same_as_hungarian": identity == hungarian,
        "identity_total_validation_mae": identity_cost,
        "hungarian_total_validation_mae": hungarian_cost,
        "independent_total_validation_mae": independent_cost,
        "identity_minus_hungarian_validation_mae": identity_cost - hungarian_cost,
        "row_best_expert": row_best,
        "independent_and_hungarian_agree": independent == hungarian,
        "cluster_preference_margin_second_minus_best": clear,
    }


def generate_readme(path, meta, files, interpretations):
    lines = []
    lines.append("# Progressive GMM case study")
    lines.append("")
    lines.append("- Checkpoint: `{}`".format(meta.get("checkpoint")))
    lines.append("- Experiment: `{}`".format(meta.get("exp_dir")))
    lines.append("- Seed: `{}`".format(meta.get("seed")))
    lines.append("- Git commit: `{}`".format(meta.get("git_commit")))
    lines.append("- Dataset: `{}`".format(meta.get("dataset")))
    lines.append("- Active GMM K: `{}`".format(meta.get("active_k")))
    lines.append("- Progressive common-loss weight: `{}`".format(meta.get("fpem_env_progressive_lambda_common")))
    lines.append("- Pretrained invariant checkpoint: `{}`".format(meta.get("pretrained_invariant_checkpoint")))
    lines.append("- Validation Hungarian mapping: `{}`".format(meta.get("hungarian_mapping")))
    lines.append("- Best-fixed expert from validation: `{}`".format(meta.get("best_fixed_expert_id")))
    lines.append("")
    lines.append("## Routing summary")
    for row in meta.get("routing_summary", []):
        lines.append("- `{}`: test_avg_mae={}".format(row.get("routing_method"), row.get("test_avg_mae")))
    lines.append("")
    lines.append("## Cross-MAE interpretation")
    lines.append("- Identity same as Hungarian: `{}`".format(interpretations.get("identity_mapping_is_same_as_hungarian")))
    lines.append("- Identity minus Hungarian validation MAE sum: `{}`".format(interpretations.get("identity_minus_hungarian_validation_mae")))
    lines.append("- Independent and Hungarian agree: `{}`".format(interpretations.get("independent_and_hungarian_agree")))
    lines.append("- Per-cluster second-best margin: `{}`".format(interpretations.get("cluster_preference_margin_second_minus_best")))
    lines.append("")
    lines.append("## Generated files")
    for f in files:
        lines.append("- `{}`".format(f))
    lines.append("")
    lines.append("## Required limitations")
    limitations = [
        "GMM environment discovery does not use target values.",
        "Cluster-to-expert mapping uses validation prediction errors.",
        "Test labels are used only for final evaluation and oracle diagnostics.",
        "Oracle is not deployable.",
        "Three seeds share the same seed-2024 invariant backbone.",
        "Two-dimensional UMAP/PCA plots do not prove strict disentanglement.",
        "Seed 2025 may exhibit limited expert differentiation.",
        "Case examples are selected using deterministic rules.",
    ]
    for item in limitations:
        lines.append("- " + item)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_one(cli):
    exp_dir = Path(cli.exp_dir).resolve()
    checkpoint = Path(cli.checkpoint).resolve() if cli.checkpoint else exp_dir / "best_val_model.pth"
    if not checkpoint.exists() or checkpoint.name != "best_val_model.pth":
        raise RuntimeError("best checkpoint missing; no fallback is allowed: {}".format(checkpoint))
    route_eval_dir = Path(cli.route_eval_dir).resolve()
    output_dir = Path(cli.output_dir).resolve() if cli.output_dir else exp_dir / "case_study" / "progressive_gmm_case_outputs"
    mkdir(output_dir)
    dirs = {
        "mappings": output_dir / "mappings",
        "cluster_profiles": output_dir / "cluster_profiles",
        "embeddings": output_dir / "embeddings",
        "routing": output_dir / "routing",
        "corrections": output_dir / "corrections",
        "typical_samples": output_dir / "typical_samples",
        "temporal": output_dir / "temporal",
    }
    for d in dirs.values():
        mkdir(d)

    ckpt, args, loaders, scaler, counts, _graph, model, load_result, active_k = build_runtime(checkpoint, output_dir, cli.device)
    k = active_k
    mapping_payload, identity, hungarian, independent, validation_cross = load_mapping(route_eval_dir, k)
    route_eval, best_fixed = load_route_results(route_eval_dir)

    val = collect_split(model, loaders["val"], scaler, args, "val", max_samples=cli.max_samples)
    test = collect_split(model, loaders["test_mixed"], scaler, args, "test", max_samples=cli.max_samples)
    val = add_predictions(val, identity, hungarian, independent, best_fixed)
    test = add_predictions(test, identity, hungarian, independent, best_fixed)

    # Strict oracle hard-route check, mirroring evaluator acceptance.
    for hard_name in ["identity_mae", "gmm_hungarian_mae", "independent_mae", "best_fixed_mae"]:
        if float(np.nanmean(test["oracle_mae"])) > float(np.nanmean(test[hard_name])) + 1e-5:
            raise RuntimeError("oracle is worse than hard route {} on test; invalid per-sample oracle".format(hard_name))

    write_csv(output_dir / "per_sample_metrics.csv", per_sample_rows(val) + per_sample_rows(test))
    np.savez_compressed(
        str(output_dir / "arrays.npz"),
        val_targets=val["targets_raw"],
        val_expert_predictions=val["y_heads_raw"],
        val_invariant_prediction=val["y_inv_raw"],
        val_environment_feature=val["e_env"],
        val_invariant_feature=val["z_inv"],
        val_gmm_posterior=val["gmm_posterior"],
        val_gmm_cluster_id=val["gmm_cluster_id"],
        test_inputs=test["x_raw"],
        test_targets=test["targets_raw"],
        test_expert_predictions=test["y_heads_raw"],
        test_invariant_prediction=test["y_inv_raw"],
        test_uniform_prediction=test["uniform_prediction"],
        test_gmm_hungarian_prediction=test["gmm_hungarian_prediction"],
        test_oracle_prediction=test["oracle_prediction"],
        test_environment_feature=test["e_env"],
        test_invariant_feature=test["z_inv"],
        test_gmm_posterior=test["gmm_posterior"],
        test_gmm_cluster_id=test["gmm_cluster_id"],
    )

    annotations = {}
    for c, e in identity.items():
        annotations[(c, e)] = (annotations.get((c, e), "") + "I").strip()
    for c, e in hungarian.items():
        annotations[(c, e)] = (annotations.get((c, e), "") + " H").strip()
    for c, e in independent.items():
        annotations[(c, e)] = (annotations.get((c, e), "") + " B").strip()
    plot_heatmap(validation_cross, dirs["mappings"] / "validation_cross_mae_heatmap.png", "Validation cluster x expert MAE", annotations=annotations)
    write_tsv_matrix(dirs["mappings"] / "validation_cross_mae.tsv", validation_cross)
    write_json(dirs["mappings"] / "cluster_to_expert_mapping.json", mapping_payload)
    cross_interp = cross_mae_interpretation(validation_cross, identity, hungarian, independent)
    write_json(dirs["mappings"] / "validation_cross_mae_interpretation.json", cross_interp)

    profiles = make_cluster_plots(test, k, dirs["cluster_profiles"])
    e_feat, z_feat = make_embeddings(test, k, dirs["embeddings"], cli.embedding_method, cli.random_seed)
    make_probe_results(e_feat, z_feat, test, dirs["embeddings"], cli.random_seed)

    val_cross = cluster_expert_mae_csv(val, k, output_dir / "validation_cluster_expert_mae.csv")
    test_cross = cluster_expert_mae_csv(test, k, output_dir / "test_cluster_expert_mae.csv")
    plot_heatmap(test_cross, output_dir / "test_cluster_expert_mae_heatmap.png", "Test cluster x expert MAE", annotations=annotations)
    test_mapping_generalization(test_cross, identity, hungarian, independent, output_dir / "test_mapping_generalization.csv")

    routing_gain_rows = make_routing_comparison(route_eval, dirs["routing"])
    typical_rows = make_typical_samples(test, k, dirs["typical_samples"])
    correction_rows = make_correction_analysis(test, k, dirs["corrections"])
    confidence_rows = make_confidence_analysis(val, test, output_dir)
    temporal_summary = make_temporal_analysis(test, k, dirs["temporal"])

    metadata = OrderedDict()
    metadata["checkpoint"] = str(checkpoint)
    metadata["exp_dir"] = str(exp_dir)
    metadata["route_eval_dir"] = str(route_eval_dir)
    metadata["output_dir"] = str(output_dir)
    metadata["seed"] = int(cli.seed if cli.seed is not None else getattr(args, "seed", 0))
    metadata["git_commit"] = git_commit()
    metadata["dataset"] = str(getattr(args, "dataset", "NYCTaxi_TDS"))
    metadata["counts"] = counts
    metadata["active_k"] = active_k
    metadata["fpem_env_progressive_lambda_common"] = float(getattr(args, "fpem_env_progressive_lambda_common", 0.2))
    metadata["pretrained_invariant_checkpoint"] = str(getattr(args, "fpem_pretrained_inv_agcrn_path", ""))
    metadata["three_seed_shared_pretrained_invariant_backbone"] = "pure_agcrn_seed2024/best_val_model.pth" in metadata["pretrained_invariant_checkpoint"]
    metadata["identity_mapping"] = identity
    metadata["hungarian_mapping"] = hungarian
    metadata["independent_mapping"] = independent
    metadata["best_fixed_expert_id"] = int(best_fixed)
    metadata["routing_summary"] = route_eval.get("test_results", [])
    metadata["load_missing_keys"] = list(getattr(load_result, "missing_keys", []) or [])
    metadata["load_unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []) or [])
    metadata["progressive_gmm_state_restored"] = {
        "means_buffer_present": bool(hasattr(model, "progressive_gmm_mu")),
        "variances_buffer_present": bool(hasattr(model, "progressive_gmm_var")),
        "priors_buffer_present": bool(hasattr(model, "progressive_gmm_log_prior")),
        "feature_mean_present": bool(hasattr(model, "progressive_feature_mean")),
        "feature_std_present": bool(hasattr(model, "progressive_feature_std")),
        "ema_teacher_encoder_present": bool(hasattr(model, "encoder_env_teacher")),
        "component_alignment_mapping": [int(hungarian.get(i, i)) for i in range(k)],
    }
    metadata["leakage_protocol"] = {
        "gmm_fit_source": "training",
        "mapping_source": "validation",
        "best_fixed_source": "validation",
        "test_target_used_by": "evaluation_and_oracle_only",
        "no_checkpoint_fallback": True,
    }
    metadata["summary_tables"] = {
        "cluster_profiles": profiles,
        "routing_gains": routing_gain_rows,
        "typical_samples": typical_rows,
        "corrections": correction_rows,
        "confidence_bins": confidence_rows,
        "temporal": temporal_summary,
    }
    write_json(output_dir / "metadata.json", metadata)
    files = []
    for root, _dirs, names in os.walk(str(output_dir)):
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name), str(output_dir)))
    generate_readme(output_dir / "README.md", metadata, sorted(files), cross_interp)
    log(json.dumps({
        "status": "ok",
        "seed": metadata["seed"],
        "output_dir": str(output_dir),
        "active_k": active_k,
        "hungarian_mapping": hungarian,
        "best_fixed_expert_id": best_fixed,
        "num_files": len(files),
    }, ensure_ascii=False, indent=2, default=safe_float))
    return metadata


def default_exp_dir(seed):
    return DEFAULT_RESULT_ROOT / "{}_{}_seed{}".format(RUN_PREFIX, CASE_NAME, seed)


def default_route_eval_dir(seed):
    return DEFAULT_ROUTE_EVAL_ROOT / "{}_{}_seed{}".format(RUN_PREFIX, CASE_NAME, seed)


def summarize_all(seed_outputs, summary_dir):
    mkdir(summary_dir)
    summary_rows = []
    profile_rows = []
    gain_rows = []
    cross_rows = []
    for seed, out_dir in seed_outputs:
        meta_path = Path(out_dir) / "metadata.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        route_by = {r["routing_method"]: r for r in meta.get("routing_summary", [])}
        gmm = route_by.get("gmm_hard_val_hungarian", {})
        uniform = route_by.get("uniform_all_experts", {})
        shuffled = route_by.get("shuffled_gmm_route", {})
        random_u = route_by.get("random_uniform_top1", {})
        oracle = route_by.get("oracle_top1", {})
        summary_rows.append({
            "seed": seed,
            "output_dir": str(out_dir),
            "hungarian_mapping": meta.get("hungarian_mapping"),
            "best_fixed_expert_id": meta.get("best_fixed_expert_id"),
            "gmm_hungarian_test_avg_mae": gmm.get("test_avg_mae"),
            "uniform_test_avg_mae": uniform.get("test_avg_mae"),
            "shuffled_test_avg_mae": shuffled.get("test_avg_mae"),
            "random_uniform_test_avg_mae": random_u.get("test_avg_mae"),
            "oracle_test_avg_mae": oracle.get("test_avg_mae"),
            "gain_vs_uniform": (uniform.get("test_avg_mae") - gmm.get("test_avg_mae")) if uniform.get("test_avg_mae") is not None and gmm.get("test_avg_mae") is not None else None,
            "gain_vs_shuffled": (shuffled.get("test_avg_mae") - gmm.get("test_avg_mae")) if shuffled.get("test_avg_mae") is not None and gmm.get("test_avg_mae") is not None else None,
            "oracle_gap": (gmm.get("test_avg_mae") - oracle.get("test_avg_mae")) if oracle.get("test_avg_mae") is not None and gmm.get("test_avg_mae") is not None else None,
            "effective_expert_number": gmm.get("effective_expert_number"),
        })
        for row in meta.get("summary_tables", {}).get("cluster_profiles", []):
            rr = OrderedDict(row)
            rr["seed"] = seed
            profile_rows.append(rr)
        for row in meta.get("summary_tables", {}).get("routing_gains", []):
            rr = OrderedDict(row)
            rr["seed"] = seed
            gain_rows.append(rr)
        cross_path = Path(out_dir) / "mappings" / "cluster_to_expert_mapping.json"
        if cross_path.exists():
            mp = load_json(cross_path)
            cross = np.asarray(mp.get("validation_cross_mae"), dtype=np.float64)
            for c in range(cross.shape[0]):
                for e in range(cross.shape[1]):
                    cross_rows.append({"seed": seed, "cluster": c, "expert": e, "validation_cross_mae": float(cross[c, e])})
    write_csv(Path(summary_dir) / "three_seed_case_study_summary.csv", summary_rows)
    write_csv(Path(summary_dir) / "three_seed_cluster_profiles.csv", profile_rows)
    write_csv(Path(summary_dir) / "three_seed_routing_gains.csv", gain_rows)
    write_csv(Path(summary_dir) / "three_seed_cross_mae_summary.csv", cross_rows)
    # Combined validation cross-MAE figure if available.
    if cross_rows:
        seeds = sorted(set([r["seed"] for r in cross_rows]))
        if not HAS_MPL:
            mats = []
            for seed in seeds:
                mat = np.zeros((3, 3), dtype=np.float64)
                for r in cross_rows:
                    if r["seed"] == seed:
                        mat[int(r["cluster"]), int(r["expert"])] = float(r["validation_cross_mae"])
                tmp = Path(summary_dir) / ("_tmp_seed_{}.png".format(seed))
                _save_pil_heatmap(mat, tmp, "seed {}".format(seed))
                mats.append(Image.open(str(tmp)).copy())
                try:
                    tmp.unlink()
                except Exception:
                    pass
            total_w = sum(img.size[0] for img in mats)
            max_h = max(img.size[1] for img in mats)
            canvas = Image.new("RGB", (total_w, max_h), "white")
            x = 0
            for img in mats:
                canvas.paste(img, (x, 0))
                x += img.size[0]
            canvas.save(str(Path(summary_dir) / "validation_cross_mae_all_seeds.png"))
        else:
            fig, axes = plt.subplots(1, len(seeds), figsize=(5.2 * len(seeds), 4.2), squeeze=False)
            for ax, seed in zip(axes[0], seeds):
                mat = np.zeros((3, 3), dtype=np.float64)
                for r in cross_rows:
                    if r["seed"] == seed:
                        mat[int(r["cluster"]), int(r["expert"])] = float(r["validation_cross_mae"])
                im = ax.imshow(mat, cmap="viridis", aspect="auto")
                ax.set_title("seed {}".format(seed))
                ax.set_xlabel("expert")
                ax.set_ylabel("cluster")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, "{:.3f}".format(mat[i, j]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
            fig.savefig(str(Path(summary_dir) / "validation_cross_mae_all_seeds.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)
    readme = [
        "# Three-seed Progressive GMM case-study summary",
        "",
        "Cluster IDs are seed-local and are not compared directly.  This summary keeps per-seed cluster IDs and reports seed-level routing/profile statistics.",
        "",
        "Generated files:",
        "- `three_seed_case_study_summary.csv`",
        "- `three_seed_cluster_profiles.csv`",
        "- `three_seed_routing_gains.csv`",
        "- `three_seed_cross_mae_summary.csv`",
        "- `validation_cross_mae_all_seeds.png`",
    ]
    Path(summary_dir, "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return summary_rows


def run_all(cli):
    seeds = [int(x) for x in str(cli.seeds).split(",") if str(x).strip()]
    outputs = []
    for seed in seeds:
        exp = default_exp_dir(seed)
        route = default_route_eval_dir(seed)
        out = exp / "case_study" / "progressive_gmm_case_outputs"
        if getattr(cli, "summary_only", False):
            outputs.append((seed, out))
            continue
        sub = argparse.Namespace(**vars(cli))
        sub.exp_dir = str(exp)
        sub.checkpoint = str(exp / "best_val_model.pth")
        sub.route_eval_dir = str(route)
        sub.output_dir = str(out)
        sub.seed = seed
        analyze_one(sub)
        outputs.append((seed, out))
    summary_dir = DEFAULT_RESULT_ROOT / (RUN_PREFIX + "_case_study_summary")
    summarize_all(outputs, summary_dir)
    log("[SUMMARY] {}".format(summary_dir))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_dir")
    p.add_argument("--checkpoint")
    p.add_argument("--route_eval_dir")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--embedding_method", choices=["pca", "umap"], default="pca")
    p.add_argument("--random_seed", type=int, default=20260721)
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--summary_only", action="store_true", help="with --run_all, only summarize existing case-study outputs")
    p.add_argument("--seeds", default="2024,2025,2026")
    return p.parse_args()


def main():
    cli = parse_args()
    np.random.seed(int(cli.random_seed))
    torch.manual_seed(int(cli.random_seed))
    if cli.run_all:
        run_all(cli)
        return
    if not cli.exp_dir:
        raise SystemExit("--exp_dir is required unless --run_all")
    if not cli.route_eval_dir:
        raise SystemExit("--route_eval_dir is required unless --run_all")
    analyze_one(cli)


if __name__ == "__main__":
    main()
