#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified inference-only routing evaluation for Progressive-GMM FPEM experts.

This script evaluates multiple expert-routing baselines on one fixed checkpoint.
It never retrains the model, never refits GMMs from validation/test data, and
uses test targets only for the explicit non-deployable oracle diagnostic.
"""

from __future__ import print_function

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from lib.metrics import test_metrics  # noqa: E402
from run_tds_nyctaxi import build_model, build_tds_data, load_checkpoint, load_graph, to_device  # noqa: E402


STANDARD_METHODS = [
    "best_fixed",
    "random_uniform_top1",
    "random_prior_top1",
    "shuffled_gmm_route",
    "gmm_hard_identity",
    "gmm_hard_val_hungarian",
    "gmm_hard_val_independent",
    "uniform_all_experts",
    "oracle_top1",
]


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return safe_float(value.detach().cpu().numpy())
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return value


def json_default(value):
    if isinstance(value, OrderedDict):
        return dict(value)
    return safe_float(value)


def str_to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def to_abs_project_path(path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT / p).resolve())


def namespace_from_checkpoint(ckpt_args, cli):
    cfg = dict(ckpt_args or {})
    cfg["device"] = cli.device
    cfg["batch_size"] = int(cli.batch_size)
    cfg["test_batch_size"] = int(cli.batch_size)
    cfg["resume"] = False
    cfg["max_train_batches"] = None
    cfg["max_eval_batches"] = None
    cfg["log_dir"] = str(Path(cli.output_dir).resolve())
    cfg.setdefault("dataset", "NYCTaxi_TDS")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("graph_file", "data/NYCTaxi_TDS/adj_mx.npz")
    cfg.setdefault("data_seed", cfg.get("seed", 2024))
    if cfg.get("data_seed") is None:
        cfg["data_seed"] = cfg.get("seed", 2024)
    cfg["data_dir"] = to_abs_project_path(cfg["data_dir"])
    cfg["graph_file"] = to_abs_project_path(cfg["graph_file"])
    if cfg.get("fpem_pretrained_inv_agcrn_path"):
        cfg["fpem_pretrained_inv_agcrn_path"] = to_abs_project_path(cfg["fpem_pretrained_inv_agcrn_path"])
    return argparse.Namespace(**cfg)


def inverse_np(scaler, array):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    with torch.no_grad():
        out = scaler.inverse_transform(tensor).detach().cpu().numpy()
    return out.astype(np.float32)


def write_tsv(path, rows, preferred_keys=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for key in preferred_keys or []:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in keys})


def write_matrix_tsv(path, matrix, row_name="cluster"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float64)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row_name] + ["expert_{}".format(i) for i in range(matrix.shape[1])])
        for i in range(matrix.shape[0]):
            writer.writerow([i] + [matrix[i, j] for j in range(matrix.shape[1])])


def one_hot(indices, k):
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(k)), dtype=np.float32)
    if idx.size:
        out[np.arange(idx.shape[0]), np.clip(idx, 0, int(k) - 1)] = 1.0
    return out


def predict_from_probs(y_heads_raw, route_probs):
    weights = np.asarray(route_probs, dtype=np.float32)
    while weights.ndim < y_heads_raw.ndim:
        weights = weights[..., None]
    return (weights * y_heads_raw).sum(axis=1).astype(np.float32)


def labels_from_time_label(time_label):
    arr = np.asarray(time_label, dtype=np.int64).reshape(-1)
    workday = arr < 24
    hour = arr % 24
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    return {
        "workday": workday,
        "holiday": ~workday,
        "rush": rush,
        "hour": hour,
    }


def split_mae_mape(pred_raw, target_raw):
    mae, mape = test_metrics(np.asarray(pred_raw, dtype=np.float32), np.asarray(target_raw, dtype=np.float32))
    return float(mae), float(mape)


def abs_sum_count(pred_raw, target_raw, mask_value=5.0):
    pred = np.asarray(pred_raw, dtype=np.float64)
    target = np.asarray(target_raw, dtype=np.float64)
    err = np.abs(pred - target)
    mask = target > float(mask_value)
    if bool(mask.any()):
        return float(err[mask].sum(dtype=np.float64)), int(mask.sum())
    return float(err.sum(dtype=np.float64)), int(err.size)


def masked_mae_from_sum(sum_abs, count):
    return float(sum_abs) / max(float(count), 1.0)


def per_sample_expert_mae(y_heads_raw, target_raw, mask_value=5.0):
    # y_heads_raw: [N,K,...], target_raw: [N,...]
    err = np.abs(np.asarray(y_heads_raw, dtype=np.float64) - np.asarray(target_raw, dtype=np.float64)[:, None, ...])
    target = np.asarray(target_raw, dtype=np.float64)[:, None, ...]
    mask = target > float(mask_value)
    reduce_dims = tuple(range(2, err.ndim))
    if not reduce_dims:
        return err.astype(np.float32)
    masked_sum = (err * mask).sum(axis=reduce_dims)
    denom = mask.sum(axis=reduce_dims)
    fallback = err.mean(axis=reduce_dims)
    out = np.where(denom > 0, masked_sum / np.maximum(denom, 1.0), fallback)
    return out.astype(np.float32)


def route_usage(route_probs, k):
    q = np.asarray(route_probs, dtype=np.float64)
    if q.shape[0] == 0:
        return np.zeros((int(k),), dtype=np.float32)
    return q.mean(axis=0).astype(np.float32)


def usage_entropy(usage):
    p = np.asarray(usage, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()) if nz.size else 0.0


def per_sample_entropy(route_probs):
    q = np.clip(np.asarray(route_probs, dtype=np.float64), 1e-12, 1.0)
    return float((-(q * np.log(q)).sum(axis=1)).mean()) if q.shape[0] else 0.0


def effective_expert_number(usage):
    return float(math.exp(usage_entropy(usage)))


def evaluate_prediction_on_mixed(split_data, pred_raw):
    target = split_data["targets_raw"]
    labels = labels_from_time_label(split_data["time_labels"])
    mixed_mae, mixed_mape = split_mae_mape(pred_raw, target)
    work_mae = None
    holiday_mae = None
    if bool(labels["workday"].any()):
        work_mae, _ = split_mae_mape(pred_raw[labels["workday"]], target[labels["workday"]])
    if bool(labels["holiday"].any()):
        holiday_mae, _ = split_mae_mape(pred_raw[labels["holiday"]], target[labels["holiday"]])
    valid_group = [x for x in [work_mae, holiday_mae] if x is not None]
    avg_mae = float(np.mean(valid_group)) if valid_group else mixed_mae
    return mixed_mae, work_mae, holiday_mae, avg_mae, mixed_mape


def make_result_row(
    method,
    split_data,
    route_probs,
    pred_raw,
    seed,
    checkpoint,
    best_epoch,
    deployable,
    uses_test_target,
    active_experts_per_sample,
    oracle_mixed_mae,
    mapping_source="not_used",
    gmm_fit_source="not_used",
    best_fixed_source="not_used",
    random_trials=None,
    random_stats=None,
    extra=None,
):
    k = int(route_probs.shape[1])
    mixed_mae, work_mae, holiday_mae, avg_mae, mixed_mape = evaluate_prediction_on_mixed(split_data, pred_raw)
    usage = route_usage(route_probs, k)
    row = OrderedDict()
    row["seed"] = int(seed)
    row["checkpoint"] = str(checkpoint)
    row["best_epoch"] = int(best_epoch) if best_epoch is not None else None
    row["routing_method"] = str(method)
    row["deployable"] = bool(deployable)
    row["uses_test_target"] = bool(uses_test_target)
    row["active_experts_per_sample"] = int(active_experts_per_sample)
    row["test_mixed_mae"] = mixed_mae
    row["test_workday_mae"] = work_mae
    row["test_holiday_mae"] = holiday_mae
    row["test_avg_mae"] = avg_mae
    row["test_mape"] = mixed_mape
    row["oracle_hard_mae"] = float(oracle_mixed_mae) if oracle_mixed_mae is not None else None
    row["router_regret"] = None if oracle_mixed_mae is None else mixed_mae - float(oracle_mixed_mae)
    row["routing_usage_per_expert"] = [float(x) for x in usage.tolist()]
    row["effective_expert_number"] = effective_expert_number(usage)
    row["routing_entropy"] = per_sample_entropy(route_probs)
    row["random_trials"] = random_trials or []
    stats = random_stats or {}
    row["random_mae_mean"] = stats.get("mean")
    row["random_mae_std"] = stats.get("std")
    row["random_mae_min"] = stats.get("min")
    row["random_mae_max"] = stats.get("max")
    row["mapping_source"] = mapping_source
    row["gmm_fit_source"] = gmm_fit_source
    row["best_fixed_source"] = best_fixed_source
    row["test_target_used_by"] = "oracle_only"
    if extra:
        row.update(extra)
    return row


def get_batch_items(raw_batch):
    if len(raw_batch) == 5:
        x, y, time_label, c, sample_index = raw_batch
    elif len(raw_batch) == 4:
        x, y, time_label, c = raw_batch
        sample_index = None
    elif len(raw_batch) == 3:
        x, y, c = raw_batch
        time_label = None
        sample_index = None
    else:
        raise RuntimeError("unsupported batch tuple length: {}".format(len(raw_batch)))
    return x, y, time_label, c, sample_index


def collect_split(model, loader, scaler, args, split_name, max_batches=-1):
    model.eval()
    heads_list = []
    target_list = []
    cluster_list = []
    route_q_list = []
    time_list = []
    sample_index_list = []
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if int(max_batches) >= 0 and batch_idx >= int(max_batches):
                break
            batch = to_device(raw_batch, args.device)
            x, y, time_label, c, sample_index = get_batch_items(batch)
            out = model.forward_output(
                x,
                exog=c,
                time_label=time_label,
                training=False,
                sample_index=sample_index,
            )
            heads = out.get("y_hyper_heads", None)
            if heads is None:
                heads = out.get("y_route_heads", None)
            if heads is None:
                raise RuntimeError("model output does not expose y_hyper_heads/y_route_heads")
            clusters = out.get("progressive_cluster_id", None)
            if clusters is None:
                clusters = out.get("env_route_selected_head", None)
            if clusters is None:
                clusters = heads.new_zeros(heads.shape[0], dtype=torch.long)
            q = out.get("env_route_q", None)
            if q is None:
                q = heads.new_full((heads.shape[0], heads.shape[1]), 1.0 / float(heads.shape[1]))

            heads_list.append(heads.detach().cpu().float().numpy())
            target_list.append(y.detach().cpu().float().numpy())
            cluster_list.append(clusters.detach().cpu().long().numpy().reshape(-1))
            route_q_list.append(q.detach().cpu().float().numpy())
            if time_label is None:
                time_list.append(np.full((heads.shape[0],), -1, dtype=np.int64))
            else:
                time_list.append(time_label.detach().cpu().long().numpy().reshape(-1))
            if sample_index is None:
                start = sum(x.shape[0] for x in sample_index_list) if sample_index_list else 0
                sample_index_list.append(np.arange(start, start + heads.shape[0], dtype=np.int64))
            else:
                sample_index_list.append(sample_index.detach().cpu().long().numpy().reshape(-1))

    if not heads_list:
        raise RuntimeError("no samples collected for split={}".format(split_name))
    data = {
        "split": split_name,
        "y_heads": np.concatenate(heads_list, axis=0).astype(np.float32),
        "targets": np.concatenate(target_list, axis=0).astype(np.float32),
        "cluster_id": np.concatenate(cluster_list, axis=0).astype(np.int64),
        "model_route_q": np.concatenate(route_q_list, axis=0).astype(np.float32),
        "time_labels": np.concatenate(time_list, axis=0).astype(np.int64),
        "sample_indices": np.concatenate(sample_index_list, axis=0).astype(np.int64),
    }
    data["y_heads_raw"] = inverse_np(scaler, data["y_heads"])
    data["targets_raw"] = inverse_np(scaler, data["targets"])
    data["per_sample_expert_mae"] = per_sample_expert_mae(data["y_heads_raw"], data["targets_raw"])
    data["oracle_idx"] = data["per_sample_expert_mae"].argmin(axis=1).astype(np.int64)
    return data


def compute_validation_cross_mae(val_data, k):
    clusters = np.asarray(val_data["cluster_id"], dtype=np.int64)
    heads = val_data["y_heads_raw"]
    target = val_data["targets_raw"]
    sums = np.zeros((int(k), int(k)), dtype=np.float64)
    counts = np.zeros((int(k), int(k)), dtype=np.float64)
    for cluster_id in range(int(k)):
        mask = clusters == cluster_id
        if not bool(mask.any()):
            continue
        target_c = target[mask]
        for expert_id in range(int(k)):
            s, c = abs_sum_count(heads[mask, expert_id], target_c)
            sums[cluster_id, expert_id] += s
            counts[cluster_id, expert_id] += c
    cross = np.full((int(k), int(k)), np.inf, dtype=np.float64)
    valid = counts > 0
    cross[valid] = sums[valid] / np.maximum(counts[valid], 1.0)
    return cross


def hungarian_mapping(cross_mae):
    cross = np.asarray(cross_mae, dtype=np.float64)
    finite = np.isfinite(cross)
    if not bool(finite.any()):
        raise RuntimeError("validation cross-MAE matrix has no finite entries")
    large = float(np.nanmax(cross[finite]) + 1.0e6)
    cost = np.where(finite, cross, large)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        # Fallback keeps the script testable on minimal environments; production
        # runs should use scipy as requested.
        best_perm = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(cost.shape[1]), cost.shape[0]):
            value = float(sum(cost[i, perm[i]] for i in range(cost.shape[0])))
            if value < best_cost:
                best_cost = value
                best_perm = perm
        rows = np.arange(cost.shape[0], dtype=np.int64)
        cols = np.asarray(best_perm, dtype=np.int64)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def independent_mapping(cross_mae):
    cross = np.asarray(cross_mae, dtype=np.float64)
    finite = np.isfinite(cross)
    if bool(finite.any()):
        large = float(np.nanmax(cross[finite]) + 1.0e6)
    else:
        large = 1.0e6
    cost = np.where(finite, cross, large)
    return {int(i): int(cost[i].argmin()) for i in range(cost.shape[0])}


def apply_cluster_mapping(cluster_ids, mapping, k):
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    out = np.zeros_like(clusters)
    for i, c in enumerate(clusters):
        out[i] = int(mapping.get(int(c), int(c) % int(k)))
    return np.clip(out, 0, int(k) - 1).astype(np.int64)


def fixed_route_probs(split_data, expert_id, k):
    return one_hot(np.full((split_data["y_heads_raw"].shape[0],), int(expert_id), dtype=np.int64), k)


def route_by_mapping(split_data, mapping, k):
    idx = apply_cluster_mapping(split_data["cluster_id"], mapping, k)
    return one_hot(idx, k)


def model_training_cluster_prior(model, k):
    if hasattr(model, "progressive_gmm_log_prior"):
        prior = torch.exp(model.progressive_gmm_log_prior.detach().cpu().float()).numpy()
        prior = prior[: int(k)].astype(np.float64)
    else:
        prior = np.ones((int(k),), dtype=np.float64)
    if not np.isfinite(prior).all() or float(prior.sum()) <= 0:
        prior = np.ones((int(k),), dtype=np.float64)
    prior = prior / max(float(prior.sum()), 1e-12)
    return prior.astype(np.float32)


def cluster_prior_to_expert_prior(cluster_prior, mapping, k):
    out = np.zeros((int(k),), dtype=np.float64)
    for c, p in enumerate(np.asarray(cluster_prior, dtype=np.float64)[: int(k)]):
        out[int(mapping.get(int(c), int(c) % int(k)))] += float(p)
    if float(out.sum()) <= 0:
        out[:] = 1.0 / float(k)
    else:
        out = out / float(out.sum())
    return out.astype(np.float32)


def random_route_probs(n, k, seed, prior=None):
    rng = np.random.default_rng(int(seed))
    if prior is None:
        prior = np.ones((int(k),), dtype=np.float64) / float(k)
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / max(float(prior.sum()), 1e-12)
    idx = rng.choice(int(k), size=int(n), replace=True, p=prior)
    return one_hot(idx, k)


def shuffled_route_probs(original_idx, k, seed):
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(np.asarray(original_idx, dtype=np.int64).reshape(-1))
    return one_hot(shuffled, k)


def random_stats_from_trials(trials, key="test_avg_mae"):
    values = np.asarray([float(t[key]) for t in trials if t.get(key) is not None], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def trial_record(trial_seed, split_data, route_probs, pred_raw, method):
    mixed_mae, work_mae, holiday_mae, avg_mae, mixed_mape = evaluate_prediction_on_mixed(split_data, pred_raw)
    usage = route_usage(route_probs, route_probs.shape[1])
    return OrderedDict([
        ("trial_seed", int(trial_seed)),
        ("routing_method", method),
        ("test_mixed_mae", mixed_mae),
        ("test_workday_mae", work_mae),
        ("test_holiday_mae", holiday_mae),
        ("test_avg_mae", avg_mae),
        ("test_mape", mixed_mape),
        ("routing_usage_per_expert", [float(x) for x in usage.tolist()]),
        ("effective_expert_number", effective_expert_number(usage)),
        ("routing_entropy", per_sample_entropy(route_probs)),
    ])


def evaluate_random_method(method, split_data, k, seed_base, num_trials, prior, oracle_mixed_mae, row_kwargs):
    trials = []
    pred_sum = None
    q_sum = None
    for t in range(int(num_trials)):
        trial_seed = int(seed_base) + t
        if method == "shuffled_gmm_route":
            q = shuffled_route_probs(row_kwargs["original_gmm_idx"], k, trial_seed)
        else:
            q = random_route_probs(split_data["y_heads_raw"].shape[0], k, trial_seed, prior=prior)
        pred = predict_from_probs(split_data["y_heads_raw"], q)
        trials.append(trial_record(trial_seed, split_data, q, pred, method))
        pred_sum = pred if pred_sum is None else pred_sum + pred
        q_sum = q if q_sum is None else q_sum + q
    stats = random_stats_from_trials(trials, key="test_avg_mae")
    mixed_stats = random_stats_from_trials(trials, key="test_mixed_mae")
    q_mean = q_sum / max(float(num_trials), 1.0)
    pred_mean = pred_sum / max(float(num_trials), 1.0)
    # The aggregate row uses the mean prediction/mean route only for usage-like
    # fields; MAE summary columns are overwritten by trial statistics.
    row = make_result_row(
        method,
        split_data,
        q_mean,
        pred_mean,
        random_trials=trials,
        random_stats=stats,
        oracle_mixed_mae=oracle_mixed_mae,
        **{k: v for k, v in row_kwargs.items() if k != "original_gmm_idx"}
    )
    row["test_mixed_mae"] = mixed_stats["mean"]
    row["test_avg_mae"] = stats["mean"]
    row["test_workday_mae"] = float(np.mean([float(t["test_workday_mae"]) for t in trials if t["test_workday_mae"] is not None])) if trials else None
    row["test_holiday_mae"] = float(np.mean([float(t["test_holiday_mae"]) for t in trials if t["test_holiday_mae"] is not None])) if trials else None
    row["test_mape"] = float(np.mean([float(t["test_mape"]) for t in trials])) if trials else None
    row["router_regret"] = None if oracle_mixed_mae is None else row["test_mixed_mae"] - float(oracle_mixed_mae)
    return row


def validate_oracle_bound(oracle_row, rows):
    oracle_mae = float(oracle_row["test_mixed_mae"])
    hard_methods = {
        "best_fixed",
        "gmm_hard_identity",
        "gmm_hard_val_hungarian",
        "gmm_hard_val_independent",
    }
    for row in rows:
        method = row.get("routing_method")
        if method in hard_methods:
            mae = float(row["test_mixed_mae"])
            if oracle_mae > mae + 1e-5:
                raise RuntimeError(
                    "Invalid oracle result: oracle hard MAE is larger than "
                    "the evaluated hard-routing MAE. method={} oracle={} routed={}".format(
                        method, oracle_mae, mae
                    )
                )
        if method in {"random_uniform_top1", "random_prior_top1", "shuffled_gmm_route"}:
            for trial in row.get("random_trials", []) or []:
                mae = float(trial["test_mixed_mae"])
                if oracle_mae > mae + 1e-5:
                    raise RuntimeError(
                        "Invalid oracle result: oracle hard MAE is larger than "
                        "random/shuffled trial MAE. method={} seed={} oracle={} routed={}".format(
                            method, trial.get("trial_seed"), oracle_mae, mae
                        )
                    )


def run_unit_tests():
    target = np.array([[[[0.0]]], [[[10.0]]], [[[20.0]]], [[[30.0]]]], dtype=np.float32)
    heads = np.stack(
        [
            target + np.array([[[[3.0]]], [[[9.0]]], [[[9.0]]], [[[9.0]]]], dtype=np.float32),
            target + np.array([[[[9.0]]], [[[2.0]]], [[[9.0]]], [[[9.0]]]], dtype=np.float32),
            target + np.array([[[[9.0]]], [[[9.0]]], [[[1.0]]], [[[4.0]]]], dtype=np.float32),
        ],
        axis=1,
    )
    mae = per_sample_expert_mae(heads, target, mask_value=-1.0)
    oracle_idx = mae.argmin(axis=1)
    assert oracle_idx.tolist() == [0, 1, 2, 2], oracle_idx
    oracle_q = one_hot(oracle_idx, 3)
    oracle_pred = predict_from_probs(heads, oracle_q)
    oracle_sum, oracle_count = abs_sum_count(oracle_pred, target, mask_value=-1.0)
    oracle_mae = masked_mae_from_sum(oracle_sum, oracle_count)
    for expert in range(3):
        q = one_hot(np.full((4,), expert, dtype=np.int64), 3)
        pred = predict_from_probs(heads, q)
        s, c = abs_sum_count(pred, target, mask_value=-1.0)
        assert oracle_mae <= masked_mae_from_sum(s, c) + 1e-8

    cross = np.array([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    assert hungarian_mapping(cross) == {0: 2, 1: 1, 2: 0}

    q1 = random_route_probs(1000, 3, seed=77, prior=[0.2, 0.3, 0.5])
    q2 = random_route_probs(1000, 3, seed=77, prior=[0.2, 0.3, 0.5])
    q3 = random_route_probs(1000, 3, seed=78, prior=[0.2, 0.3, 0.5])
    assert np.array_equal(q1, q2)
    assert not np.array_equal(q1, q3)
    freq = q1.mean(axis=0)
    assert np.all(np.abs(freq - np.array([0.2, 0.3, 0.5])) < 0.06), freq

    original = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    shuffled = shuffled_route_probs(original, 3, seed=99).argmax(axis=1)
    assert np.bincount(original, minlength=3).tolist() == np.bincount(shuffled, minlength=3).tolist()
    print("UNIT_TESTS_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--num_random_trials", type=int, default=20)
    parser.add_argument("--random_seed_base", type=int, default=20260721)
    parser.add_argument("--run_unit_tests", action="store_true")
    args_cli = parser.parse_args()

    if args_cli.run_unit_tests:
        run_unit_tests()
        return
    if not args_cli.checkpoint_path:
        raise SystemExit("--checkpoint_path is required unless --run_unit_tests is used")
    if not args_cli.output_dir:
        raise SystemExit("--output_dir is required unless --run_unit_tests is used")
    if int(args_cli.num_random_trials) < 20:
        raise SystemExit("--num_random_trials must be at least 20")

    start_time = time.time()
    out_dir = Path(args_cli.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args_cli.checkpoint_path).resolve()
    if ckpt_path.name != "best_val_model.pth":
        raise RuntimeError("evaluation must use best_val_model.pth; got {}".format(ckpt_path))
    if not ckpt_path.exists():
        raise RuntimeError("checkpoint not found: {}".format(ckpt_path))

    ckpt = load_checkpoint(str(ckpt_path), args_cli.device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    args = namespace_from_checkpoint(ckpt_args, args_cli)
    seed = int(getattr(args, "seed", 0))
    best_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1

    loaders, scaler, counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, _lr = build_model(args, graph)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    if "fpem_progressive_gmm_state" in ckpt and hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
        model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
    model.eval()

    print(json.dumps({
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "seed": seed,
        "best_epoch": best_epoch,
        "fpem_env_route_train_mode": str(getattr(args, "fpem_env_route_train_mode", "")),
        "fpem_use_pretrained_inv_agcrn": bool(str_to_bool(getattr(args, "fpem_use_pretrained_inv_agcrn", False))),
        "fpem_pretrained_inv_agcrn_path": str(getattr(args, "fpem_pretrained_inv_agcrn_path", "")),
    }, ensure_ascii=False, indent=2))

    val_data = collect_split(model, loaders["val"], scaler, args, "val", max_batches=args_cli.max_batches)
    test_data = collect_split(model, loaders["test_mixed"], scaler, args, "test_mixed", max_batches=args_cli.max_batches)
    k = int(test_data["y_heads_raw"].shape[1])
    if k != 3:
        print("[WARN] expected K=3 for main Progressive GMM evaluation, got K={}".format(k), file=sys.stderr)

    cross_mae = compute_validation_cross_mae(val_data, k)
    identity = {int(i): int(i) for i in range(k)}
    hungarian = hungarian_mapping(cross_mae)
    independent = independent_mapping(cross_mae)
    np.save(str(out_dir / "validation_cross_mae.npy"), cross_mae)
    write_matrix_tsv(str(out_dir / "validation_cross_mae.tsv"), cross_mae)

    cluster_prior = model_training_cluster_prior(model, k)
    expert_prior = cluster_prior_to_expert_prior(cluster_prior, hungarian, k)

    mapping_payload = OrderedDict()
    mapping_payload["checkpoint"] = str(ckpt_path)
    mapping_payload["seed"] = seed
    mapping_payload["k"] = k
    mapping_payload["mapping_source"] = "validation"
    mapping_payload["validation_cross_mae"] = cross_mae.tolist()
    mapping_payload["identity_mapping"] = {str(k0): int(v) for k0, v in identity.items()}
    mapping_payload["hungarian_mapping"] = {str(k0): int(v) for k0, v in hungarian.items()}
    mapping_payload["independent_mapping"] = {str(k0): int(v) for k0, v in independent.items()}
    mapping_payload["training_cluster_prior"] = [float(x) for x in cluster_prior.tolist()]
    mapping_payload["training_prior_converted_to_expert_prior_by_hungarian_mapping"] = [
        float(x) for x in expert_prior.tolist()
    ]
    mapping_payload["gmm_fit_source"] = "training"
    with open(str(out_dir / "cluster_to_expert_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, ensure_ascii=False, indent=2, default=json_default)

    # Validation-only fixed expert selection.
    val_expert_maes = []
    for expert_id in range(k):
        pred = val_data["y_heads_raw"][:, expert_id]
        mae, _mape = split_mae_mape(pred, val_data["targets_raw"])
        val_expert_maes.append(float(mae))
    best_fixed_expert = int(np.argmin(np.asarray(val_expert_maes, dtype=np.float64)))

    # Oracle diagnostic first, so every hard method can be checked against it.
    oracle_q = one_hot(test_data["oracle_idx"], k)
    oracle_pred = predict_from_probs(test_data["y_heads_raw"], oracle_q)
    oracle_row = make_result_row(
        "oracle_top1",
        test_data,
        oracle_q,
        oracle_pred,
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=False,
        uses_test_target=True,
        active_experts_per_sample=1,
        oracle_mixed_mae=None,
        mapping_source="not_used",
        gmm_fit_source="not_used",
        best_fixed_source="not_used",
        extra={"diagnostic_only": True},
    )
    oracle_row["oracle_hard_mae"] = oracle_row["test_mixed_mae"]
    oracle_row["router_regret"] = 0.0
    oracle_mixed_mae = float(oracle_row["test_mixed_mae"])

    rows = []

    fixed_q = fixed_route_probs(test_data, best_fixed_expert, k)
    rows.append(make_result_row(
        "best_fixed",
        test_data,
        fixed_q,
        predict_from_probs(test_data["y_heads_raw"], fixed_q),
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=1,
        oracle_mixed_mae=oracle_mixed_mae,
        mapping_source="not_used",
        gmm_fit_source="not_used",
        best_fixed_source="validation",
        extra={
            "best_fixed_expert_id": best_fixed_expert,
            "validation_expert_mae": val_expert_maes,
        },
    ))

    base_kwargs = dict(
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=1,
        mapping_source="not_used",
        gmm_fit_source="not_used",
        best_fixed_source="not_used",
    )
    rows.append(evaluate_random_method(
        "random_uniform_top1",
        test_data,
        k,
        seed_base=int(args_cli.random_seed_base),
        num_trials=int(args_cli.num_random_trials),
        prior=np.ones((k,), dtype=np.float32) / float(k),
        oracle_mixed_mae=oracle_mixed_mae,
        row_kwargs=dict(base_kwargs),
    ))

    prior_kwargs = dict(base_kwargs)
    prior_kwargs["mapping_source"] = "validation"
    prior_kwargs["gmm_fit_source"] = "training"
    prior_kwargs["extra"] = {
        "prior_source": "training",
        "cluster_prior": [float(x) for x in cluster_prior.tolist()],
        "expert_prior": [float(x) for x in expert_prior.tolist()],
    }
    rows.append(evaluate_random_method(
        "random_prior_top1",
        test_data,
        k,
        seed_base=int(args_cli.random_seed_base) + 100000,
        num_trials=int(args_cli.num_random_trials),
        prior=expert_prior,
        oracle_mixed_mae=oracle_mixed_mae,
        row_kwargs=prior_kwargs,
    ))

    identity_q = route_by_mapping(test_data, identity, k)
    rows.append(make_result_row(
        "gmm_hard_identity",
        test_data,
        identity_q,
        predict_from_probs(test_data["y_heads_raw"], identity_q),
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=1,
        oracle_mixed_mae=oracle_mixed_mae,
        mapping_source="identity",
        gmm_fit_source="training",
        best_fixed_source="not_used",
        extra={"cluster_to_expert": identity},
    ))

    hungarian_q = route_by_mapping(test_data, hungarian, k)
    hungarian_idx = hungarian_q.argmax(axis=1).astype(np.int64)
    rows.append(make_result_row(
        "gmm_hard_val_hungarian",
        test_data,
        hungarian_q,
        predict_from_probs(test_data["y_heads_raw"], hungarian_q),
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=1,
        oracle_mixed_mae=oracle_mixed_mae,
        mapping_source="validation",
        gmm_fit_source="training",
        best_fixed_source="not_used",
        extra={"cluster_to_expert": hungarian, "primary_gmm_hard_result": True},
    ))

    independent_q = route_by_mapping(test_data, independent, k)
    rows.append(make_result_row(
        "gmm_hard_val_independent",
        test_data,
        independent_q,
        predict_from_probs(test_data["y_heads_raw"], independent_q),
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=1,
        oracle_mixed_mae=oracle_mixed_mae,
        mapping_source="validation",
        gmm_fit_source="training",
        best_fixed_source="not_used",
        extra={"cluster_to_expert": independent, "diagnostic_only": True},
    ))

    shuffle_kwargs = dict(base_kwargs)
    shuffle_kwargs["deployable"] = False
    shuffle_kwargs["mapping_source"] = "validation"
    shuffle_kwargs["gmm_fit_source"] = "training"
    shuffle_kwargs["original_gmm_idx"] = hungarian_idx
    shuffle_kwargs["extra"] = {"shuffle_preserves_exact_gmm_usage_counts": True}
    rows.append(evaluate_random_method(
        "shuffled_gmm_route",
        test_data,
        k,
        seed_base=int(args_cli.random_seed_base) + 200000,
        num_trials=int(args_cli.num_random_trials),
        prior=None,
        oracle_mixed_mae=oracle_mixed_mae,
        row_kwargs=shuffle_kwargs,
    ))

    uniform_q = np.ones((test_data["y_heads_raw"].shape[0], k), dtype=np.float32) / float(k)
    rows.append(make_result_row(
        "uniform_all_experts",
        test_data,
        uniform_q,
        predict_from_probs(test_data["y_heads_raw"], uniform_q),
        seed=seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        deployable=True,
        uses_test_target=False,
        active_experts_per_sample=k,
        oracle_mixed_mae=oracle_mixed_mae,
        mapping_source="not_used",
        gmm_fit_source="not_used",
        best_fixed_source="not_used",
    ))

    rows.append(oracle_row)
    rows_by_method = {row["routing_method"]: row for row in rows}
    rows = [rows_by_method[m] for m in STANDARD_METHODS if m in rows_by_method]
    validate_oracle_bound(oracle_row, rows)

    preferred = [
        "seed",
        "checkpoint",
        "best_epoch",
        "routing_method",
        "deployable",
        "uses_test_target",
        "active_experts_per_sample",
        "test_mixed_mae",
        "test_workday_mae",
        "test_holiday_mae",
        "test_avg_mae",
        "test_mape",
        "oracle_hard_mae",
        "router_regret",
        "routing_usage_per_expert",
        "effective_expert_number",
        "routing_entropy",
        "random_mae_mean",
        "random_mae_std",
        "random_mae_min",
        "random_mae_max",
        "mapping_source",
        "gmm_fit_source",
        "best_fixed_source",
        "test_target_used_by",
    ]
    write_tsv(str(out_dir / "online_route_results.tsv"), rows, preferred_keys=preferred)

    np.savez_compressed(
        str(out_dir / "online_route_predictions.npz"),
        y_true=test_data["targets_raw"],
        y_heads=test_data["y_heads_raw"],
        time_labels=test_data["time_labels"],
        sample_indices=test_data["sample_indices"],
        test_cluster_id=test_data["cluster_id"],
        oracle_idx=test_data["oracle_idx"],
        gmm_hard_val_hungarian_idx=hungarian_idx,
    )

    results = OrderedDict()
    results["checkpoint"] = str(ckpt_path)
    results["output_dir"] = str(out_dir)
    results["seed"] = seed
    results["best_epoch"] = best_epoch
    results["k"] = k
    results["counts"] = counts
    results["checkpoint_args_subset"] = {
        "run_name": getattr(args, "run_name", None),
        "fpem_env_route_train_mode": getattr(args, "fpem_env_route_train_mode", None),
        "fpem_env_max_clusters": getattr(args, "fpem_env_max_clusters", None),
        "fpem_env_progressive_lambda_common": getattr(args, "fpem_env_progressive_lambda_common", None),
        "fpem_use_pretrained_inv_agcrn": getattr(args, "fpem_use_pretrained_inv_agcrn", None),
        "fpem_pretrained_inv_agcrn_path": getattr(args, "fpem_pretrained_inv_agcrn_path", None),
    }
    results["pretrained_invariant_protocol"] = {
        "fpem_use_pretrained_inv_agcrn": bool(str_to_bool(getattr(args, "fpem_use_pretrained_inv_agcrn", False))),
        "fpem_pretrained_inv_agcrn_path": str(getattr(args, "fpem_pretrained_inv_agcrn_path", "")),
        "note": "Protocol is read from checkpoint args and not changed by this evaluator.",
    }
    results["load_missing_keys"] = list(getattr(load_result, "missing_keys", []) or [])
    results["load_unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []) or [])
    results["mapping"] = mapping_payload
    results["validation_best_fixed_expert_id"] = best_fixed_expert
    results["validation_expert_mae"] = val_expert_maes
    results["test_results"] = rows
    results["standard_methods"] = STANDARD_METHODS
    results["leakage_protocol"] = {
        "mapping_source": "validation",
        "best_fixed_source": "validation",
        "gmm_fit_source": "training",
        "test_target_used_by": "oracle_only",
        "test_targets_used_for_mapping": False,
        "test_targets_used_for_best_fixed": False,
        "test_targets_used_for_gmm_fit": False,
        "test_targets_used_for_router_training": False,
        "test_targets_used_for_hyperparameter_selection": False,
    }
    results["output_files"] = [
        "online_route_results.json",
        "online_route_results.tsv",
        "validation_cross_mae.npy",
        "validation_cross_mae.tsv",
        "cluster_to_expert_mapping.json",
        "online_route_predictions.npz",
    ]
    results["elapsed_seconds"] = time.time() - start_time

    with open(str(out_dir / "online_route_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    print(json.dumps({
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "seed": seed,
        "best_epoch": best_epoch,
        "methods": [row["routing_method"] for row in rows],
        "primary_gmm_hard_test_avg_mae": rows_by_method["gmm_hard_val_hungarian"]["test_avg_mae"],
        "oracle_top1_test_avg_mae": rows_by_method["oracle_top1"]["test_avg_mae"],
        "leakage_protocol": results["leakage_protocol"],
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
