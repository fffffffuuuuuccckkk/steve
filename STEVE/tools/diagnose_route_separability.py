#!/usr/bin/env python
"""Route separability diagnostics for trained NYCTaxi-TDS FPEM checkpoints.

This script is inference-only: it loads a checkpoint, runs the selected splits,
exports observable route features, and compares simple non-neural assignments
against the model's route/expert ids.  It intentionally does not add any loss or
training-side behavior.
"""

from __future__ import annotations

import argparse
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

from run_tds_nyctaxi import build_model, build_tds_data, load_checkpoint, load_graph, to_device  # noqa: E402


def str2list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def labels_from_time_label(time_label):
    arr = np.asarray(time_label, dtype=np.int64).reshape(-1)
    day = (arr < 24).astype(np.int64)
    hour = arr % 24
    hour_bin = np.zeros_like(hour)
    hour_bin[(hour >= 6) & (hour <= 9)] = 1
    hour_bin[(hour >= 10) & (hour <= 15)] = 2
    hour_bin[hour >= 16] = 3
    rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(np.int64)
    return day, hour, hour_bin, rush


def as_namespace(args_dict, cli):
    cfg = dict(args_dict)
    cfg["device"] = cli.device
    cfg["resume"] = False
    cfg["max_train_batches"] = None
    cfg["max_eval_batches"] = None
    cfg["log_dir"] = str(Path(cli.output_dir).resolve())
    cfg.setdefault("dataset", "NYCTaxi_TDS")
    cfg.setdefault("data_dir", str(PROJECT / "data"))
    cfg.setdefault("graph_file", str(PROJECT / "data" / "NYCTaxi_TDS" / "adj_mx.npz"))
    cfg.setdefault("data_seed", cfg.get("seed", 2024))
    if cfg.get("fpem_pretrained_inv_agcrn_path"):
        p = Path(cfg["fpem_pretrained_inv_agcrn_path"])
        if not p.is_absolute():
            cfg["fpem_pretrained_inv_agcrn_path"] = str((PROJECT / p).resolve())
    for key in ["data_dir", "graph_file"]:
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str((PROJECT / p).resolve())
    return SimpleNamespace(**cfg)


def collect_split(model, loader, args, split, max_batches):
    rows = []
    feats = []
    q_list = []
    y_true = []
    y_pred = []
    time_labels = []
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if max_batches is not None and max_batches >= 0 and batch_idx >= max_batches:
                break
            batch = to_device(raw_batch, args.device)
            x, y, time_label, c = batch
            out = model.forward_output(x, exog=c, time_label=time_label, training=False)
            route_features = out.get("route_features")
            if route_features is None:
                latest = getattr(model, "latest_fpem_outputs", {}) or {}
                route_features = latest.get("route_features")
            if route_features is None:
                raise RuntimeError("model did not expose route_features; checkpoint/code is too old")
            q = out.get("env_route_q")
            if q is None:
                raise RuntimeError("model did not expose env_route_q")
            feats.append(route_features.detach().cpu().float().numpy())
            q_list.append(q.detach().cpu().float().numpy())
            y_true.append(y.detach().cpu().float().numpy())
            y_pred.append(out["prediction"].detach().cpu().float().numpy())
            time_labels.append(time_label.detach().cpu().long().numpy())
    if not feats:
        raise RuntimeError(f"no batches collected for split={split}")
    feats = np.concatenate(feats, axis=0)
    q = np.concatenate(q_list, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    time_label = np.concatenate(time_labels, axis=0).reshape(-1)
    expert_id = q.argmax(axis=1).astype(np.int64)
    day, hour, hour_bin, rush = labels_from_time_label(time_label)
    return {
        "split": split,
        "route_features": feats,
        "route_q": q,
        "expert_id": expert_id,
        "time_label": time_label,
        "day": day,
        "hour": hour,
        "hour_bin": hour_bin,
        "rush": rush,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def centroid_predict(x_train, y_train, x_test, k):
    centroids = []
    for idx in range(k):
        mask = y_train == idx
        centroids.append(x_train[mask].mean(axis=0) if mask.any() else x_train.mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    dist = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
    return dist.argmin(axis=1), dist


def gaussian_predict(x_train, y_train, x_test, k, min_var=1e-4):
    mu = []
    var = []
    priors = []
    for idx in range(k):
        mask = y_train == idx
        part = x_train[mask] if mask.any() else x_train
        mu.append(part.mean(axis=0))
        var.append(np.maximum(part.var(axis=0), min_var))
        priors.append(float(max(mask.sum(), 1)))
    mu = np.stack(mu, axis=0)
    var = np.stack(var, axis=0)
    priors = np.asarray(priors, dtype=np.float64)
    cost = ((x_test[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :]).mean(axis=-1)
    score = -cost + np.log(priors / priors.sum())[None, :]
    return score.argmax(axis=1), cost


def knn_predict(x_train, y_train, x_test, k_neighbors=5, chunk=256):
    preds = []
    k_neighbors = max(1, int(k_neighbors))
    for start in range(0, x_test.shape[0], chunk):
        part = x_test[start : start + chunk]
        dist = ((part[:, None, :] - x_train[None, :, :]) ** 2).mean(axis=-1)
        nn_idx = np.argsort(dist, axis=1)[:, :k_neighbors]
        votes = y_train[nn_idx]
        out = []
        for row in votes:
            counts = np.bincount(row, minlength=int(y_train.max()) + 1)
            out.append(int(counts.argmax()))
        preds.append(np.asarray(out, dtype=np.int64))
    return np.concatenate(preds, axis=0)


def accuracy(pred, target):
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return float((pred == target).mean()) if target.size else float("nan")


def write_tsv(path, rows):
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--splits", default="train,val,test_mixed,test_workday,test_holiday")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--knn_k", type=int, default=5)
    cli = parser.parse_args()

    ckpt_path = Path(cli.ckpt_path).resolve()
    out_dir = Path(cli.output_dir or ckpt_path.parent / "route_diagnosis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(str(ckpt_path), cli.device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    args = as_namespace(ckpt_args, cli)
    loaders, _scaler, counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, _lr = build_model(args, graph)
    result = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    split_map = {"test": "test_mixed"}
    collected = {}
    max_batches = None if int(cli.max_batches) < 0 else int(cli.max_batches)
    for split in str2list(cli.splits):
        split = split_map.get(split, split)
        if split not in loaders:
            continue
        data = collect_split(model, loaders[split], args, split, max_batches)
        collected[split] = data
        np.savez_compressed(out_dir / f"route_features_{split}.npz", **data)

    if "train" not in collected:
        raise RuntimeError("diagnosis needs train split as reference")
    train = collected["train"]
    x_train = train["route_features"].astype(np.float64)
    x_train = (x_train - x_train.mean(axis=0, keepdims=True)) / np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    y_train = train["expert_id"]
    num_experts = int(train["route_q"].shape[1])

    rows = []
    for split, data in collected.items():
        x = data["route_features"].astype(np.float64)
        x = (x - train["route_features"].mean(axis=0, keepdims=True)) / np.maximum(
            train["route_features"].std(axis=0, keepdims=True), 1e-6
        )
        y = data["expert_id"]
        pred_centroid, _ = centroid_predict(x_train, y_train, x, num_experts)
        pred_gaussian, _ = gaussian_predict(x_train, y_train, x, num_experts)
        pred_knn = knn_predict(x_train, y_train, x, k_neighbors=cli.knn_k)
        rows.extend(
            [
                {"split": split, "method": "centroid", "expert_acc": accuracy(pred_centroid, y), "n": int(y.size)},
                {"split": split, "method": "gaussian", "expert_acc": accuracy(pred_gaussian, y), "n": int(y.size)},
                {"split": split, "method": f"knn_{cli.knn_k}", "expert_acc": accuracy(pred_knn, y), "n": int(y.size)},
            ]
        )

    summary = {
        "ckpt_path": str(ckpt_path),
        "output_dir": str(out_dir),
        "counts": counts,
        "missing_keys": list(getattr(result, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(result, "unexpected_keys", []) or []),
        "rows": rows,
    }
    with open(out_dir / "route_separability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_tsv(out_dir / "route_separability_summary.tsv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
