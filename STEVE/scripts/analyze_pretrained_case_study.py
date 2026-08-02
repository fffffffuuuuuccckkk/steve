#!/usr/bin/env python3
"""Case study / mechanism analysis for pretrained-frozen invariant AGCRN FPEM.

This script is intentionally inference-only.  It reuses run_tds_nyctaxi.py for
model construction, TDS data/scaler preparation, and checkpoint loading, then
exports representation probes, route/expert diagnostics, route interventions,
and environment-swap counterfactuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.metrics import test_metrics  # noqa: E402
from lib.utils import init_seed, load_graph  # noqa: E402
from run_tds_nyctaxi import (  # noqa: E402
    TDSDataset,
    build_model,
    build_tds_data,
    load_checkpoint,
    load_npz,
    standard_transform_float32,
)


DEFAULT_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "NYCTaxi_TDS"
    / "fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025"
    / "best_val_model.pth"
)

RUSH_HOURS = {7, 8, 9, 17, 18, 19}
MASK_VALUE = 5.0


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_project_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT))
    parser.add_argument("--exp_dir", default=None)
    parser.add_argument("--split", default="test", choices=["test", "test_mixed", "test_workday", "test_holiday", "val", "train"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=2025)
    parser.add_argument("--viz_method", default="auto", choices=["auto", "umap", "tsne", "pca"])
    return parser.parse_args()


def load_yaml_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        config_path = str(PROJECT_ROOT / "configs" / "NYCTaxi.yaml")
    path = Path(resolve_project_path(config_path) or config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader) or {}
    return data if isinstance(data, dict) else {}


def make_runtime_args(ckpt: dict[str, Any], cli: argparse.Namespace, exp_dir: Path) -> SimpleNamespace:
    ckpt_args = dict(ckpt.get("args") or {})
    cfg = load_yaml_defaults(ckpt_args.get("config_filename"))
    cfg.update(ckpt_args)
    cfg.setdefault("model", "steve")
    cfg.setdefault("dataset", "NYCTaxi_TDS")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("graph_file", "data/NYCTaxi_TDS/adj_mx.npz")
    cfg.setdefault("result_root", str(PROJECT_ROOT / "experiments" / "NYCTaxi_TDS"))
    cfg.setdefault("run_name", exp_dir.name)
    cfg.setdefault("log_dir", str(exp_dir))
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("test_batch_size", 16)
    cfg.setdefault("train_work_per_holiday", 2.5)
    cfg.setdefault("seed", 2025)
    cfg.setdefault("num_nodes", 200)
    cfg.setdefault("d_input", 2)
    cfg.setdefault("d_output", 2)
    cfg.setdefault("d_model", 64)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("yita", 0.5)
    cfg.setdefault("lr_init", 0.001)
    cfg.setdefault("agcrn_embed_dim", 10)
    cfg.setdefault("agcrn_num_layers", 2)
    cfg.setdefault("agcrn_cheb_k", 2)
    cfg.setdefault("agcrn_rnn_units", 64)
    cfg.setdefault("fpem_backbone", "agcrn")

    cfg["device"] = cli.device if torch.cuda.is_available() or not str(cli.device).startswith("cuda") else "cpu"
    cfg["data_dir"] = resolve_project_path(cfg["data_dir"])
    cfg["graph_file"] = resolve_project_path(cfg["graph_file"])
    if cfg.get("fpem_pretrained_inv_agcrn_path"):
        cfg["fpem_pretrained_inv_agcrn_path"] = resolve_project_path(cfg["fpem_pretrained_inv_agcrn_path"])
    return SimpleNamespace(**cfg)


def split_pack(args: SimpleNamespace, split: str, scaler: Any) -> dict[str, np.ndarray]:
    dataset_root = Path(args.data_dir) / args.dataset
    if not dataset_root.is_dir() and args.dataset == "NYCTaxi_TDS":
        fallback = Path(args.data_dir) / "NYCTaxi"
        if fallback.is_dir():
            dataset_root = fallback

    if split in {"test", "test_mixed", "test_workday", "test_holiday"}:
        pack = load_npz(str(dataset_root / "test.npz"))
    elif split == "val":
        pack = load_npz(str(dataset_root / "val.npz"))
    elif split == "train":
        pack = load_npz(str(dataset_root / "train.npz"))
    else:
        raise ValueError(f"unsupported split: {split}")

    if split == "test_workday":
        idx = np.where(pack["time_label"] < 24)[0]
        pack = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == pack["x"].shape[0] else v) for k, v in pack.items()}
    elif split == "test_holiday":
        idx = np.where(pack["time_label"] >= 24)[0]
        pack = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == pack["x"].shape[0] else v) for k, v in pack.items()}

    pack["x"] = standard_transform_float32(pack["x"], scaler)
    pack["y"] = standard_transform_float32(pack["y"], scaler)
    pack["c"] = np.ascontiguousarray(pack["c"], dtype=np.float32)
    pack["time_label"] = np.ascontiguousarray(pack["time_label"], dtype=np.int64)
    return pack


def make_analysis_loader(args: SimpleNamespace, split: str, scaler: Any, num_workers: int = 0) -> torch.utils.data.DataLoader:
    pack = split_pack(args, split, scaler)
    dataset = TDSDataset(pack["x"], pack["y"], pack["time_label"], pack["c"], include_time=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.test_batch_size),
        shuffle=False,
        drop_last=False,
        pin_memory=str(args.device).startswith("cuda"),
        num_workers=num_workers,
    )


def masked_mae_np(pred: np.ndarray, true: np.ndarray, mask_value: float = MASK_VALUE) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    mask = true > mask_value
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def per_sample_masked_mae(pred: np.ndarray, true: np.ndarray, mask_value: float = MASK_VALUE) -> np.ndarray:
    pred = np.asarray(pred)
    true = np.asarray(true)
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_true = true.reshape(true.shape[0], -1)
    mask = flat_true > mask_value
    out = np.full(flat_true.shape[0], np.nan, dtype=np.float64)
    for i in range(flat_true.shape[0]):
        if np.any(mask[i]):
            out[i] = np.mean(np.abs(flat_pred[i, mask[i]] - flat_true[i, mask[i]]))
    return out


def inverse_np(scaler: Any, tensor: torch.Tensor) -> np.ndarray:
    return scaler.inverse_transform(tensor.detach().cpu()).numpy()


def tensor_np(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return np.asarray(arr, dtype=dtype)


def labels_from_time(time_label: np.ndarray) -> dict[str, np.ndarray]:
    if time_label.size == 0:
        warnings.warn("time_label is empty; environment labels are unavailable")
        n = 0
        return {
            "workday": np.empty(n, dtype=np.int64),
            "holiday": np.empty(n, dtype=np.int64),
            "hour": np.empty(n, dtype=np.int64),
            "hour_bin": np.empty(n, dtype=np.int64),
            "rush_hour": np.empty(n, dtype=np.int64),
        }
    workday = (time_label < 24).astype(np.int64)
    hour = (time_label % 24).astype(np.int64)
    rush = np.isin(hour, sorted(RUSH_HOURS)).astype(np.int64)
    return {
        "workday": workday,
        "holiday": 1 - workday,
        "hour": hour,
        "hour_bin": hour,
        "rush_hour": rush,
    }


def route_override_prediction(model: torch.nn.Module, output: dict[str, torch.Tensor], q_override: torch.Tensor) -> torch.Tensor | None:
    if "y_route_heads" not in output or "y_inv" not in output:
        return None
    y_heads = output["y_route_heads"]
    if y_heads.dim() != 5:
        return None
    q = q_override.to(device=y_heads.device, dtype=y_heads.dtype)
    y_route = (q.view(q.shape[0], q.shape[1], 1, 1, 1) * y_heads).sum(dim=1)
    y_inv = output["y_inv"]
    route_active = float(output.get("primary_uses_route", y_inv.new_tensor(1.0)).detach().cpu().item()) > 0.0
    fusion_active = float(output.get("primary_uses_env_fusion", y_inv.new_tensor(0.0)).detach().cpu().item()) > 0.0
    if fusion_active and hasattr(model, "fusion"):
        y_final, _alpha, _logs = model.fusion(y_inv, y_route, output["Z_inv"], output["E_useful"], q)
        return y_final
    if route_active:
        return y_route
    return y_inv


@dataclass
class Extracted:
    arrays: dict[str, np.ndarray]
    per_sample: pd.DataFrame
    route_preds: dict[str, np.ndarray]
    route_modes: list[str]
    eval_metrics: dict[str, float]


def extract_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    scaler: Any,
    args: SimpleNamespace,
    max_batches: int,
    random_state: int,
) -> Extracted:
    model.eval()
    rng = torch.Generator(device=args.device if str(args.device).startswith("cuda") else "cpu")
    rng.manual_seed(int(random_state))

    lists: dict[str, list[np.ndarray]] = {
        "z_inv_raw": [],
        "z_inv": [],
        "e_env": [],
        "route_weight": [],
        "expert_id": [],
        "y_true": [],
        "y_inv": [],
        "y_final": [],
        "delta_env": [],
        "time_label": [],
        "c": [],
        "hyper_reconstruction_error": [],
        "route_proto_max_abs_delta": [],
        "fusion_alpha_abs_mean": [],
        "fallback_q_abs_mean": [],
    }
    optional_lists: dict[str, list[np.ndarray]] = {
        "y_expert": [],
        "hyper_alpha": [],
        "route_weight_prototype": [],
        "route_weight_prediction": [],
        "hyper_gamma_norm_per_head": [],
        "hyper_beta_norm_per_head": [],
    }
    route_preds: dict[str, list[np.ndarray]] = {}

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            data, target, time_label, c = raw_batch
            data = data.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            c = c.to(args.device, non_blocking=True)

            output = model.forward_output(data, exog=c, time_label=time_label, training=False, epoch=None)
            route_q = output.get("env_route_q", output.get("route_q"))
            if route_q is None:
                raise RuntimeError("model output does not contain env_route_q/route_q")

            y_true = inverse_np(scaler, target)
            y_inv = inverse_np(scaler, output["y_inv"])
            y_final = inverse_np(scaler, output["prediction"])

            lists["z_inv_raw"].append(tensor_np(output.get("Z_inv_raw", output["Z_inv"])))
            lists["z_inv"].append(tensor_np(output["Z_inv"]))
            lists["e_env"].append(tensor_np(output.get("C_cur", output["E_useful"])))
            lists["route_weight"].append(tensor_np(route_q))
            lists["expert_id"].append(tensor_np(route_q.argmax(dim=-1), dtype=np.int64))
            lists["y_true"].append(y_true)
            lists["y_inv"].append(y_inv)
            lists["y_final"].append(y_final)
            lists["delta_env"].append(y_final - y_inv)
            lists["time_label"].append(time_label.numpy().astype(np.int64))
            lists["c"].append(c.detach().cpu().numpy().astype(np.float32))
            if "y_route_heads" in output:
                optional_lists["y_expert"].append(inverse_np(scaler, output["y_route_heads"]))
                if route_q.shape[1] == output["y_route_heads"].shape[1]:
                    recon = (
                        route_q.view(route_q.shape[0], route_q.shape[1], 1, 1, 1)
                        * output["y_route_heads"]
                    ).sum(dim=1)
                    recon_err = (recon - output["prediction"]).detach().abs().reshape(route_q.shape[0], -1).max(dim=1).values
                    lists["hyper_reconstruction_error"].append(tensor_np(recon_err))
                else:
                    lists["hyper_reconstruction_error"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))
            else:
                lists["hyper_reconstruction_error"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))
            proto_q = output.get("env_route_q_prototype")
            if torch.is_tensor(proto_q):
                optional_lists["route_weight_prototype"].append(tensor_np(proto_q))
                if route_q.shape == proto_q.shape:
                    proto_delta = (route_q - proto_q).detach().abs().reshape(route_q.shape[0], -1).max(dim=1).values
                    lists["route_proto_max_abs_delta"].append(tensor_np(proto_delta))
                else:
                    lists["route_proto_max_abs_delta"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))
            else:
                lists["route_proto_max_abs_delta"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))
            pred_q = output.get("env_route_q_prediction")
            if torch.is_tensor(pred_q):
                optional_lists["route_weight_prediction"].append(tensor_np(pred_q))
            hyper_alpha = output.get("hyper_alpha")
            if torch.is_tensor(hyper_alpha):
                optional_lists["hyper_alpha"].append(tensor_np(hyper_alpha))
            gamma_norm = output.get("hyper_gamma_norm_per_head")
            if torch.is_tensor(gamma_norm):
                optional_lists["hyper_gamma_norm_per_head"].append(tensor_np(gamma_norm[None, :]))
            beta_norm = output.get("hyper_beta_norm_per_head")
            if torch.is_tensor(beta_norm):
                optional_lists["hyper_beta_norm_per_head"].append(tensor_np(beta_norm[None, :]))
            fusion_alpha = output.get("fusion_alpha")
            if torch.is_tensor(fusion_alpha):
                fusion_mean = fusion_alpha.detach().abs().reshape(fusion_alpha.shape[0], -1).mean(dim=1)
                lists["fusion_alpha_abs_mean"].append(tensor_np(fusion_mean))
            else:
                lists["fusion_alpha_abs_mean"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))
            fallback_q = output.get("fallback_q")
            if torch.is_tensor(fallback_q):
                lists["fallback_q_abs_mean"].append(tensor_np(fallback_q.detach().abs().reshape(route_q.shape[0], -1).mean(dim=1)))
            else:
                lists["fallback_q_abs_mean"].append(np.full((route_q.shape[0],), np.nan, dtype=np.float32))

            # Route interventions.
            k = int(route_q.shape[1])
            modes: dict[str, torch.Tensor] = {
                "learned": route_q,
                "uniform": torch.full_like(route_q, 1.0 / max(k, 1)),
            }
            if route_q.shape[0] > 1:
                perm = torch.randperm(route_q.shape[0], generator=rng, device=route_q.device)
                modes["shuffled"] = route_q.index_select(0, perm)
            else:
                modes["shuffled"] = route_q
            for idx in range(k):
                q_remove = route_q.clone()
                q_remove[:, idx] = 0.0
                denom = q_remove.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                q_remove = q_remove / denom
                modes[f"remove_{idx}"] = q_remove

            for mode_name, q in modes.items():
                if mode_name == "learned":
                    pred = output["prediction"]
                else:
                    pred = route_override_prediction(model, output, q)
                    if pred is None:
                        warnings.warn(f"route override {mode_name} unavailable; skipping")
                        continue
                route_preds.setdefault(mode_name, []).append(inverse_np(scaler, pred))

    arrays = {key: np.concatenate(value, axis=0) for key, value in lists.items()}
    if optional_lists["y_expert"]:
        arrays["y_expert"] = np.concatenate(optional_lists["y_expert"], axis=0)
    for key in [
        "hyper_alpha",
        "route_weight_prototype",
        "route_weight_prediction",
        "hyper_gamma_norm_per_head",
        "hyper_beta_norm_per_head",
    ]:
        if optional_lists[key]:
            arrays[key] = np.concatenate(optional_lists[key], axis=0)
    if hasattr(model, "env_prototypes"):
        arrays["env_prototypes"] = tensor_np(model.env_prototypes)
    arrays["z_inv_raw_pool"] = arrays["z_inv_raw"].mean(axis=1)
    arrays["z_inv_pool"] = arrays["z_inv"].mean(axis=1)
    arrays["e_env_pool"] = arrays["e_env"].mean(axis=1)
    labels = labels_from_time(arrays["time_label"])
    arrays.update(labels)
    route_pred_arrays = {key: np.concatenate(value, axis=0) for key, value in route_preds.items()}

    y_true = arrays["y_true"]
    y_final = arrays["y_final"]
    mae, mape = test_metrics(y_final, y_true)
    eval_metrics = {"test_mae": float(mae), "test_mape": float(mape)}

    per_sample = pd.DataFrame({
        "sample_index": np.arange(arrays["y_true"].shape[0]),
        "time_label": arrays["time_label"],
        "workday": arrays["workday"],
        "holiday": arrays["holiday"],
        "hour": arrays["hour"],
        "hour_bin": arrays["hour_bin"],
        "rush_hour": arrays["rush_hour"],
        "expert_id": arrays["expert_id"],
        "mae_inv": per_sample_masked_mae(arrays["y_inv"], y_true),
        "mae_final": per_sample_masked_mae(y_final, y_true),
        "mean_abs_delta_env": np.mean(np.abs(arrays["delta_env"].reshape(arrays["delta_env"].shape[0], -1)), axis=1),
    })
    for k in range(arrays["route_weight"].shape[1]):
        per_sample[f"route_weight_{k}"] = arrays["route_weight"][:, k]
    if arrays["c"].ndim >= 2:
        flat_c = arrays["c"].reshape(arrays["c"].shape[0], -1)
        for j in range(min(flat_c.shape[1], 8)):
            per_sample[f"c_{j}"] = flat_c[:, j]

    return Extracted(
        arrays=arrays,
        per_sample=per_sample,
        route_preds=route_pred_arrays,
        route_modes=list(route_pred_arrays),
        eval_metrics=eval_metrics,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def probe_classification(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as SKStandardScaler
    from sklearn.svm import LinearSVC

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    feature_sets = {
        "z_inv_raw": arrays.get("z_inv_raw_pool", arrays["z_inv_pool"]),
        "h_inv": arrays["z_inv_pool"],
        "e_env": arrays["e_env_pool"],
        "h_inv_plus_e_env": np.concatenate([arrays["z_inv_pool"], arrays["e_env_pool"]], axis=1),
    }
    targets = {
        "workday": arrays["workday"],
        "rush_hour": arrays["rush_hour"],
    }
    rows: list[dict[str, Any]] = []
    for target_name, y in targets.items():
        valid = np.isfinite(y)
        y = y[valid].astype(int)
        if np.unique(y).size < 2 or y.size < 8:
            warnings.warn(f"skip classification probe for {target_name}: insufficient labels")
            continue
        stratify = y if min(np.bincount(y)) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(y.size), test_size=0.3, random_state=0, stratify=stratify
        )
        classifiers: dict[str, Any] = {
            "logistic_regression": make_pipeline(
                SKStandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            )
        }
        if target_name == "workday":
            classifiers["linear_svm"] = make_pipeline(
                SKStandardScaler(),
                CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000), cv=3),
            )
        for feature_name, x_all in feature_sets.items():
            x = x_all[valid]
            for clf_name, clf in classifiers.items():
                clf.fit(x[train_idx], y[train_idx])
                pred = clf.predict(x[test_idx])
                if hasattr(clf, "predict_proba"):
                    score = clf.predict_proba(x[test_idx])[:, 1]
                elif hasattr(clf, "decision_function"):
                    score = clf.decision_function(x[test_idx])
                else:
                    score = pred
                try:
                    auc = roc_auc_score(y[test_idx], score)
                except ValueError:
                    auc = float("nan")
                rows.append({
                    "target": target_name,
                    "feature_set": feature_name,
                    "classifier": clf_name,
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "accuracy": accuracy_score(y[test_idx], pred),
                    "auc": auc,
                    "f1": f1_score(y[test_idx], pred, zero_division=0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def probe_residual(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as SKStandardScaler

    y = (arrays["y_true"] - arrays["y_inv"]).reshape(arrays["y_true"].shape[0], -1)
    feature_sets = {
        "z_inv_raw": arrays.get("z_inv_raw_pool", arrays["z_inv_pool"]),
        "h_inv": arrays["z_inv_pool"],
        "e_env": arrays["e_env_pool"],
        "h_inv_plus_e_env": np.concatenate([arrays["z_inv_pool"], arrays["e_env_pool"]], axis=1),
    }
    train_idx, test_idx = train_test_split(np.arange(y.shape[0]), test_size=0.3, random_state=0)
    regressors = {
        "ridge": make_pipeline(SKStandardScaler(), Ridge(alpha=1.0)),
        "linear_regression": make_pipeline(SKStandardScaler(), LinearRegression()),
    }
    rows: list[dict[str, Any]] = []
    for feature_name, x in feature_sets.items():
        for reg_name, reg in regressors.items():
            reg.fit(x[train_idx], y[train_idx])
            pred = reg.predict(x[test_idx])
            rows.append({
                "feature_set": feature_name,
                "regressor": reg_name,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "residual_mae": mean_absolute_error(y[test_idx], pred),
                "r2": r2_score(y[test_idx], pred, multioutput="uniform_average"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def normalize_crosstab(labels: np.ndarray, experts: np.ndarray, k: int) -> pd.DataFrame:
    df = pd.DataFrame({"label": labels, "expert": experts})
    table = pd.crosstab(df["label"], df["expert"], normalize="index")
    for idx in range(k):
        if idx not in table.columns:
            table[idx] = 0.0
    table = table[[idx for idx in range(k)]]
    table.columns = [f"expert_{idx}" for idx in range(k)]
    table = table.reset_index().rename(columns={"label": "category"})
    return table


def expert_crosstabs(arrays: dict[str, np.ndarray], out_csv: Path) -> dict[str, pd.DataFrame]:
    experts = arrays["expert_id"].astype(int)
    k = arrays["route_weight"].shape[1]
    labels = {
        "workday_holiday": np.where(arrays["workday"] == 1, "workday", "holiday"),
        "hour": arrays["hour"].astype(int),
        "rush_hour": np.where(arrays["rush_hour"] == 1, "rush_hour", "non_rush_hour"),
    }
    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for name, label in labels.items():
        table = normalize_crosstab(label, experts, k)
        tables[name] = table
        for row in table.to_dict("records"):
            rows.append({"category_type": name, **row})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return tables


def route_usage_summary(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    q = arrays["route_weight"].astype(float)
    experts = arrays["expert_id"].astype(int)
    k = q.shape[1]
    q_mean = q.mean(axis=0)
    hard_counts = np.bincount(experts, minlength=k).astype(float)
    hard_ratio = hard_counts / max(float(experts.size), 1.0)
    mean_entropy = -float(np.mean(np.sum(q * np.log(np.clip(q, 1e-12, 1.0)), axis=1)))
    dist_entropy = -float(np.sum(q_mean * np.log(np.clip(q_mean, 1e-12, 1.0))))
    effective_experts = float(np.exp(dist_entropy))
    rows: list[dict[str, Any]] = []
    for expert in range(k):
        rows.append({
            "scope": "expert",
            "expert": expert,
            "soft_usage_mean": float(q_mean[expert]),
            "hard_count": int(hard_counts[expert]),
            "hard_usage_ratio": float(hard_ratio[expert]),
            "route_entropy_mean": mean_entropy,
            "route_mean_distribution_entropy": dist_entropy,
            "effective_expert_number": effective_experts,
            "max_expert_usage_ratio": float(q_mean.max()),
            "min_expert_usage_ratio": float(q_mean.min()),
        })
    rows.append({
        "scope": "overall",
        "expert": "all",
        "soft_usage_mean": float(q_mean.sum()),
        "hard_count": int(hard_counts.sum()),
        "hard_usage_ratio": float(hard_ratio.sum()),
        "route_entropy_mean": mean_entropy,
        "route_mean_distribution_entropy": dist_entropy,
        "effective_expert_number": effective_experts,
        "max_expert_usage_ratio": float(q_mean.max()),
        "min_expert_usage_ratio": float(q_mean.min()),
    })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def hyper_head_statistics(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    k = int(arrays["route_weight"].shape[1])
    hyper_alpha = arrays.get("hyper_alpha")
    gamma_norm = arrays.get("hyper_gamma_norm_per_head")
    beta_norm = arrays.get("hyper_beta_norm_per_head")
    for expert in range(k):
        rows.append({
            "expert": expert,
            "route_soft_usage": float(arrays["route_weight"][:, expert].mean()),
            "route_hard_count": int(np.sum(arrays["expert_id"] == expert)),
            "hyper_alpha_mean": float(np.mean(hyper_alpha[:, expert])) if hyper_alpha is not None and hyper_alpha.shape[1] > expert else float("nan"),
            "hyper_gamma_norm_mean": float(np.mean(gamma_norm[:, expert])) if gamma_norm is not None and gamma_norm.shape[1] > expert else float("nan"),
            "hyper_beta_norm_mean": float(np.mean(beta_norm[:, expert])) if beta_norm is not None and beta_norm.shape[1] > expert else float("nan"),
            "head_reconstruction_max_abs": float(np.nanmax(arrays["hyper_reconstruction_error"])),
            "route_proto_max_abs": float(np.nanmax(arrays["route_proto_max_abs_delta"])),
            "fusion_alpha_abs_mean": float(np.nanmean(arrays["fusion_alpha_abs_mean"])),
            "fallback_q_abs_mean": float(np.nanmean(arrays["fallback_q_abs_mean"])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def hyper_head_per_environment_mae(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    y_expert = arrays.get("y_expert")
    y_true = arrays["y_true"]
    k = int(arrays["route_weight"].shape[1])
    groups = {
        "overall": np.ones(y_true.shape[0], dtype=bool),
        "workday": arrays["workday"] == 1,
        "holiday": arrays["workday"] == 0,
        "rush_hour": arrays["rush_hour"] == 1,
        "non_rush_hour": arrays["rush_hour"] == 0,
    }
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        for expert in range(k):
            row = {
                "group": group_name,
                "expert": expert,
                "num_samples": int(mask.sum()),
                "soft_usage_on_group": float(arrays["route_weight"][mask, expert].mean()) if mask.any() else float("nan"),
                "hard_count_on_group": int(np.sum((arrays["expert_id"] == expert) & mask)),
            }
            if y_expert is not None and y_expert.shape[1] > expert and mask.any():
                row["expert_head_mae"] = masked_mae_np(y_expert[mask, expert], y_true[mask])
            else:
                row["expert_head_mae"] = float("nan")
                row["note"] = "y_hyper_heads/y_route_heads unavailable"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def hyper_route_usage_summary(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    q = arrays["route_weight"].astype(float)
    q_proto = arrays.get("route_weight_prototype")
    q_pred = arrays.get("route_weight_prediction")
    rows: list[dict[str, Any]] = []
    rows.append({
        "metric": "max_abs_y_final_minus_sum_q_heads",
        "value": float(np.nanmax(arrays["hyper_reconstruction_error"])),
    })
    rows.append({
        "metric": "max_abs_route_q_minus_prototype_q",
        "value": float(np.nanmax(arrays["route_proto_max_abs_delta"])),
    })
    rows.append({
        "metric": "fusion_alpha_abs_mean",
        "value": float(np.nanmean(arrays["fusion_alpha_abs_mean"])),
    })
    rows.append({
        "metric": "fallback_q_abs_mean",
        "value": float(np.nanmean(arrays["fallback_q_abs_mean"])),
    })
    if q_proto is not None and q_proto.shape == q.shape:
        rows.append({
            "metric": "mean_abs_route_q_minus_prototype_q",
            "value": float(np.mean(np.abs(q - q_proto))),
        })
    if q_pred is not None and q_pred.shape == q.shape:
        rows.append({
            "metric": "mean_abs_route_q_minus_prediction_q",
            "value": float(np.mean(np.abs(q - q_pred))),
        })
    for expert in range(q.shape[1]):
        rows.append({
            "metric": f"expert_{expert}_soft_usage",
            "value": float(q[:, expert].mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def hyper_swap_analysis(swap_df: pd.DataFrame, arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    rows = swap_df.to_dict("records") if not swap_df.empty else []
    rows.append({
        "swap_type": "diagnostic",
        "mean_abs_pred_swap_minus_original": float("nan"),
        "delta_mae": float("nan"),
        "max_abs_y_final_minus_sum_q_heads": float(np.nanmax(arrays["hyper_reconstruction_error"])),
        "max_abs_route_q_minus_prototype_q": float(np.nanmax(arrays["route_proto_max_abs_delta"])),
        "fusion_alpha_abs_mean": float(np.nanmean(arrays["fusion_alpha_abs_mean"])),
        "fallback_q_abs_mean": float(np.nanmean(arrays["fallback_q_abs_mean"])),
    })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def pairwise_cosine_rows(x: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    x = x.astype(float)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sim = x @ x.T
    rows: list[dict[str, Any]] = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            rows.append({
                "matrix": prefix,
                "row": int(i),
                "col": int(j),
                "cosine": float(sim[i, j]),
            })
    return rows


def prototype_similarity(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "env_prototypes" in arrays:
        rows.extend(pairwise_cosine_rows(arrays["env_prototypes"], "env_prototypes"))
    if "y_expert" in arrays:
        # Per-expert prediction signatures: average predicted trajectory over samples.
        sig = arrays["y_expert"].reshape(arrays["y_expert"].shape[0], arrays["y_expert"].shape[1], -1).mean(axis=0)
        rows.extend(pairwise_cosine_rows(sig, "expert_prediction_signature"))
    if not rows:
        rows.append({
            "matrix": "unavailable",
            "row": "",
            "col": "",
            "cosine": float("nan"),
            "note": "env_prototypes and y_expert were not available in extracted outputs",
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def prototype_env_distribution(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    experts = arrays["expert_id"].astype(int)
    k = arrays["route_weight"].shape[1]
    labels = {
        "workday_holiday": np.where(arrays["workday"] == 1, "workday", "holiday"),
        "hour": arrays["hour"].astype(int),
        "rush_hour": np.where(arrays["rush_hour"] == 1, "rush_hour", "non_rush_hour"),
    }
    rows: list[dict[str, Any]] = []
    for name, label in labels.items():
        table = pd.DataFrame({"label": label, "expert": experts})
        counts = pd.crosstab(table["label"], table["expert"])
        for expert in range(k):
            if expert not in counts.columns:
                counts[expert] = 0
        counts = counts[[expert for expert in range(k)]]
        probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        for category in counts.index:
            row: dict[str, Any] = {
                "category_type": name,
                "category": category,
                "num_samples": int(counts.loc[category].sum()),
            }
            for expert in range(k):
                row[f"expert_{expert}_count"] = int(counts.loc[category, expert])
                row[f"expert_{expert}_prob"] = float(probs.loc[category, expert])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def expert_per_env_mae(arrays: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = arrays["y_true"]
    y_final = arrays["y_final"]
    y_expert = arrays.get("y_expert")
    experts = arrays["expert_id"].astype(int)
    k = arrays["route_weight"].shape[1]
    env_groups = {
        "workday_holiday": {
            "workday": arrays["workday"] == 1,
            "holiday": arrays["workday"] == 0,
        },
        "rush_hour": {
            "rush_hour": arrays["rush_hour"] == 1,
            "non_rush_hour": arrays["rush_hour"] == 0,
        },
    }
    for group_type, groups in env_groups.items():
        for group_name, group_mask in groups.items():
            for expert in range(k):
                assigned = group_mask & (experts == expert)
                row: dict[str, Any] = {
                    "group_type": group_type,
                    "group": group_name,
                    "expert": expert,
                    "num_group_samples": int(group_mask.sum()),
                    "num_assigned_samples": int(assigned.sum()),
                    "assigned_final_mae": masked_mae_np(y_final[assigned], y_true[assigned]) if assigned.any() else float("nan"),
                    "group_final_mae": masked_mae_np(y_final[group_mask], y_true[group_mask]) if group_mask.any() else float("nan"),
                }
                if y_expert is not None:
                    row["expert_head_mae_on_group"] = masked_mae_np(
                        y_expert[group_mask, expert], y_true[group_mask]
                    ) if group_mask.any() else float("nan")
                else:
                    row["expert_head_mae_on_group"] = float("nan")
                    row["note"] = "y_route_heads unavailable; only assignment final MAE is available"
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def route_intervention_mae(arrays: dict[str, np.ndarray], route_preds: dict[str, np.ndarray], out_csv: Path) -> pd.DataFrame:
    y_true = arrays["y_true"]
    groups = {
        "overall": np.ones(y_true.shape[0], dtype=bool),
        "workday": arrays["workday"] == 1,
        "holiday": arrays["workday"] == 0,
        "rush_hour": arrays["rush_hour"] == 1,
        "non_rush_hour": arrays["rush_hour"] == 0,
    }
    rows: list[dict[str, Any]] = []
    for mode, pred in route_preds.items():
        for group, mask in groups.items():
            rows.append({
                "route_mode": mode,
                "group": group,
                "num_samples": int(mask.sum()),
                "mae": masked_mae_np(pred[mask], y_true[mask]) if mask.any() else float("nan"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def swapped_indices(workday: np.ndarray, cross: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = workday.shape[0]
    out = np.arange(n)
    for i in range(n):
        if cross:
            candidates = np.where(workday != workday[i])[0]
        else:
            candidates = np.where(workday == workday[i])[0]
            candidates = candidates[candidates != i]
        if candidates.size > 0:
            out[i] = int(rng.choice(candidates))
    return out


def env_swap_analysis(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    scaler: Any,
    args: SimpleNamespace,
    out_csv: Path,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    z_inv = arrays["z_inv"]
    e_env = arrays["e_env"]
    y_true = arrays["y_true"]
    y_original = arrays["y_final"]
    workday = arrays["workday"]
    swap_rows: list[dict[str, Any]] = []
    change_values: dict[str, np.ndarray] = {}

    for swap_type, cross in [("same_env", False), ("cross_env", True)]:
        idx = swapped_indices(workday, cross=cross, seed=seed + (17 if cross else 0))
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, z_inv.shape[0], batch_size):
                end = min(start + batch_size, z_inv.shape[0])
                z = torch.from_numpy(z_inv[start:end]).to(args.device)
                e = torch.from_numpy(e_env[idx[start:end]]).to(args.device)
                out = model._predict_from_nodes(z, e, training=False, epoch=None)
                preds.append(inverse_np(scaler, out["prediction"]))
        pred_swap = np.concatenate(preds, axis=0)
        per_change = np.mean(np.abs((pred_swap - y_original).reshape(pred_swap.shape[0], -1)), axis=1)
        change_values[swap_type] = per_change
        swap_rows.append({
            "swap_type": swap_type,
            "num_samples": int(z_inv.shape[0]),
            "mean_abs_pred_swap_minus_original": float(np.mean(per_change)),
            "median_abs_pred_swap_minus_original": float(np.median(per_change)),
            "mae_original": masked_mae_np(y_original, y_true),
            "mae_swapped": masked_mae_np(pred_swap, y_true),
            "delta_mae": masked_mae_np(pred_swap, y_true) - masked_mae_np(y_original, y_true),
        })
    df = pd.DataFrame(swap_rows)
    df.to_csv(out_csv, index=False)
    return df, change_values


def try_load_font(size: int = 14) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def color_for_value(value: Any, palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    try:
        idx = int(value)
    except Exception:
        idx = abs(hash(str(value)))
    return palette[idx % len(palette)]


PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def draw_heatmap(table: pd.DataFrame, path: Path, title: str) -> None:
    expert_cols = [c for c in table.columns if str(c).startswith("expert_")]
    labels = [str(x) for x in table["category"].tolist()]
    data = table[expert_cols].to_numpy(dtype=float)
    cell_w, cell_h = 90, 38
    left, top = 150, 70
    width = left + cell_w * len(expert_cols) + 30
    height = top + cell_h * len(labels) + 60
    img = Image.new("RGB", (max(width, 420), max(height, 220)), "white")
    draw = ImageDraw.Draw(img)
    font = try_load_font(13)
    title_font = try_load_font(16)
    draw.text((20, 20), title, fill=(0, 0, 0), font=title_font)
    for j, col in enumerate(expert_cols):
        draw.text((left + j * cell_w + 5, top - 25), col, fill=(0, 0, 0), font=font)
    for i, label in enumerate(labels):
        draw.text((15, top + i * cell_h + 10), label, fill=(0, 0, 0), font=font)
        for j in range(len(expert_cols)):
            value = float(data[i, j])
            intensity = int(255 - 180 * min(max(value, 0.0), 1.0))
            color = (intensity, intensity, 255)
            x0, y0 = left + j * cell_w, top + i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], fill=color, outline=(220, 220, 220))
            draw.text((x0 + 8, y0 + 10), f"{value:.2f}", fill=(0, 0, 0), font=font)
    img.save(path)


def draw_boxplot(values: dict[str, np.ndarray], path: Path, title: str) -> None:
    width, height = 720, 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = try_load_font(13)
    title_font = try_load_font(16)
    draw.text((20, 15), title, fill=(0, 0, 0), font=title_font)
    all_vals = np.concatenate([v[np.isfinite(v)] for v in values.values() if np.isfinite(v).any()])
    if all_vals.size == 0:
        draw.text((20, 80), "No finite values", fill=(0, 0, 0), font=font)
        img.save(path)
        return
    y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
    if y_max <= y_min:
        y_max = y_min + 1.0
    plot_left, plot_top, plot_right, plot_bottom = 80, 70, width - 30, height - 80
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(0, 0, 0))

    def y_coord(v: float) -> float:
        return plot_bottom - (v - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    names = list(values)
    for i, name in enumerate(names):
        vals = values[name]
        vals = vals[np.isfinite(vals)]
        x = plot_left + (i + 0.5) * (plot_right - plot_left) / max(len(names), 1)
        if vals.size == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = np.percentile(vals, [5, 95])
        draw.line([x, y_coord(lo), x, y_coord(hi)], fill=(0, 0, 0), width=2)
        draw.rectangle([x - 45, y_coord(q3), x + 45, y_coord(q1)], outline=(31, 119, 180), fill=(200, 225, 245))
        draw.line([x - 45, y_coord(med), x + 45, y_coord(med)], fill=(214, 39, 40), width=2)
        draw.text((x - 55, plot_bottom + 15), name, fill=(0, 0, 0), font=font)
    draw.text((10, plot_top), f"{y_max:.2f}", fill=(0, 0, 0), font=font)
    draw.text((10, plot_bottom - 10), f"{y_min:.2f}", fill=(0, 0, 0), font=font)
    img.save(path)


def reduce_2d(x: np.ndarray, method: str = "auto", random_state: int = 0) -> tuple[np.ndarray, str]:
    if method in {"auto", "umap"}:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(x), "umap"
        except Exception:
            if method == "umap":
                warnings.warn("UMAP unavailable; falling back to PCA")
    if method in {"auto", "tsne"}:
        try:
            from sklearn.manifold import TSNE

            if x.shape[0] <= 2000 and method == "tsne":
                return TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto").fit_transform(x), "tsne"
        except Exception:
            if method == "tsne":
                warnings.warn("t-SNE unavailable; falling back to PCA")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as SKStandardScaler

    x_scaled = SKStandardScaler().fit_transform(x)
    return PCA(n_components=2, random_state=random_state).fit_transform(x_scaled), "pca"


def draw_scatter(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    width, height = 720, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = try_load_font(13)
    title_font = try_load_font(16)
    draw.text((20, 15), title, fill=(0, 0, 0), font=title_font)
    x = points[:, 0]
    y = points[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    left, top, right, bottom = 60, 60, width - 160, height - 50
    draw.rectangle([left, top, right, bottom], outline=(0, 0, 0))
    for xi, yi, label in zip(x, y, labels):
        px = left + (float(xi) - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (float(yi) - y_min) / (y_max - y_min) * (bottom - top)
        color = color_for_value(label, PALETTE)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color, outline=color)
    unique = list(dict.fromkeys([str(v) for v in labels[:2000]]))
    for i, label in enumerate(unique[:12]):
        color = color_for_value(label, PALETTE)
        y0 = top + i * 22
        draw.rectangle([right + 20, y0, right + 35, y0 + 15], fill=color)
        draw.text((right + 42, y0), label, fill=(0, 0, 0), font=font)
    img.save(path)


def make_visualizations(arrays: dict[str, np.ndarray], out_dir: Path, method: str, random_state: int) -> dict[str, str]:
    files: dict[str, str] = {}
    z_points, z_method = reduce_2d(arrays["z_inv_pool"], method, random_state)
    e_points, e_method = reduce_2d(arrays["e_env_pool"], method, random_state)
    env_label = np.where(arrays["workday"] == 1, 1, 0)

    paths = {
        "umap_z_inv_by_env.png": (z_points, env_label, f"z_inv by env ({z_method})"),
        "umap_e_env_by_env.png": (e_points, env_label, f"e_env by env ({e_method})"),
        "umap_e_env_by_expert.png": (e_points, arrays["expert_id"], f"e_env by expert ({e_method})"),
        "umap_z_inv_by_hour.png": (z_points, arrays["hour"], f"z_inv by hour ({z_method})"),
        "umap_e_env_by_hour.png": (e_points, arrays["hour"], f"e_env by hour ({e_method})"),
    }
    for name, (points, labels, title) in paths.items():
        path = out_dir / name
        draw_scatter(points, labels, path, title)
        files[name] = str(path)
    return files


def write_readme(
    path: Path,
    ckpt_path: Path,
    args: SimpleNamespace,
    extracted: Extracted,
    probe_env: pd.DataFrame,
    probe_resid: pd.DataFrame,
    route_df: pd.DataFrame,
    swap_df: pd.DataFrame,
    files: dict[str, Path | str],
) -> None:
    lines = [
        "# Pretrained frozen invariant AGCRN case study",
        "",
        f"- checkpoint: `{ckpt_path}`",
        f"- run_name: `{getattr(args, 'run_name', 'NA')}`",
        f"- split: `{files.get('split', 'test')}`",
        f"- samples: `{extracted.arrays['y_true'].shape[0]}`",
        f"- route K: `{extracted.arrays['route_weight'].shape[1]}`",
        f"- fpem_use_pretrained_inv_agcrn: `{getattr(args, 'fpem_use_pretrained_inv_agcrn', 'NA')}`",
        f"- frozen params: `{files.get('frozen_params', 'NA')}`",
        "",
        "## Evaluation",
        "",
        f"- learned-route test MAE: `{extracted.eval_metrics.get('test_mae', float('nan')):.6f}`",
        f"- learned-route test MAPE: `{extracted.eval_metrics.get('test_mape', float('nan')):.6f}`",
        "",
        "## Key probe numbers",
        "",
    ]
    if not probe_env.empty:
        best_env = probe_env.sort_values("auc", ascending=False).head(5)
        lines.append("Top environment classification probes by AUC:")
        for row in best_env.to_dict("records"):
            lines.append(
                f"- {row['target']} / {row['feature_set']} / {row['classifier']}: "
                f"acc={float(row['accuracy']):.4f}, auc={float(row['auc']):.4f}, f1={float(row['f1']):.4f}"
            )
    if not probe_resid.empty:
        best_resid = probe_resid.sort_values("residual_mae").head(5)
        lines.append("")
        lines.append("Top residual probes by residual MAE:")
        for row in best_resid.to_dict("records"):
            lines.append(
                f"- {row['feature_set']} / {row['regressor']}: "
                f"mae={float(row['residual_mae']):.4f}, r2={float(row['r2']):.4f}"
            )
    if not route_df.empty:
        overall = route_df[route_df["group"] == "overall"].sort_values("mae")
        lines.append("")
        lines.append("Route intervention overall MAE:")
        for row in overall.to_dict("records"):
            lines.append(f"- {row['route_mode']}: {float(row['mae']):.6f}")
    if not swap_df.empty:
        lines.append("")
        lines.append("Environment swap summary:")
        for row in swap_df.to_dict("records"):
            lines.append(
                f"- {row['swap_type']}: mean_abs_change={float(row['mean_abs_pred_swap_minus_original']):.6f}, "
                f"delta_mae={float(row['delta_mae']):.6f}"
            )
    lines.extend(["", "## Files", ""])
    for key, value in files.items():
        if key in {"split", "frozen_params"}:
            continue
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    random.seed(cli.random_state)
    np.random.seed(cli.random_state)
    torch.manual_seed(cli.random_state)

    ckpt_path = Path(resolve_project_path(cli.ckpt_path) or cli.ckpt_path)
    if cli.exp_dir:
        exp_dir = Path(resolve_project_path(cli.exp_dir) or cli.exp_dir)
    else:
        exp_dir = ckpt_path.parent
    if cli.output_dir:
        output_dir = Path(resolve_project_path(cli.output_dir) or cli.output_dir)
    else:
        output_dir = exp_dir / "case_study"
    case_dir = output_dir / "case_outputs"
    case_dir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] project_root={PROJECT_ROOT}")
    log(f"[INFO] checkpoint path={ckpt_path}")
    log(f"[INFO] exp_dir={exp_dir}")
    log(f"[INFO] output_dir={case_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(str(ckpt_path), cli.device)
    args = make_runtime_args(ckpt, cli, exp_dir)
    init_seed(int(getattr(args, "seed", cli.random_state)))
    log(f"[INFO] args fpem_use_pretrained_inv_agcrn={getattr(args, 'fpem_use_pretrained_inv_agcrn', None)}")
    log(f"[INFO] args pretrained path={getattr(args, 'fpem_pretrained_inv_agcrn_path', None)}")
    log(f"[INFO] route K={getattr(args, 'fpem_env_route_k', None)}")

    loaders, scaler, counts = build_tds_data(args)
    del loaders
    log(f"[INFO] data counts={json.dumps(counts, ensure_ascii=False)}")

    graph = load_graph(args.graph_file, device=args.device)
    model, _lr = build_model(args, graph)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    frozen_params = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    log(f"[INFO] load_state missing={list(missing)} unexpected={list(unexpected)}")
    log(f"[INFO] frozen_params={frozen_params} trainable_params={trainable_params}")

    analysis_loader = make_analysis_loader(args, cli.split, scaler, num_workers=cli.num_workers)
    max_batches = int(cli.max_batches)
    extracted = extract_features(model, analysis_loader, scaler, args, max_batches, cli.random_state)
    log(
        "[INFO] extracted "
        f"z_inv={extracted.arrays['z_inv'].shape} "
        f"e_env={extracted.arrays['e_env'].shape} "
        f"route_weight={extracted.arrays['route_weight'].shape}"
    )

    features_path = case_dir / "features.npz"
    np.savez_compressed(features_path, **extracted.arrays)
    per_sample_path = case_dir / "per_sample.csv"
    extracted.per_sample.to_csv(per_sample_path, index=False)

    probe_env_path = case_dir / "probe_env_results.csv"
    probe_env_df = probe_classification(extracted.arrays, probe_env_path)
    inv_raw_vs_projected_probe_path = case_dir / "inv_raw_vs_projected_probe.csv"
    if probe_env_df.empty:
        probe_env_df.to_csv(inv_raw_vs_projected_probe_path, index=False)
    else:
        probe_env_df[
            probe_env_df["feature_set"].isin(["z_inv_raw", "h_inv", "e_env"])
        ].to_csv(inv_raw_vs_projected_probe_path, index=False)
    probe_resid_path = case_dir / "probe_residual_results.csv"
    probe_resid_df = probe_residual(extracted.arrays, probe_resid_path)

    crosstab_path = case_dir / "expert_env_crosstab.csv"
    tables = expert_crosstabs(extracted.arrays, crosstab_path)
    route_usage_path = case_dir / "route_usage_summary.csv"
    route_usage_df = route_usage_summary(extracted.arrays, route_usage_path)
    hyper_head_statistics_path = case_dir / "hyper_head_statistics.csv"
    hyper_head_statistics_df = hyper_head_statistics(extracted.arrays, hyper_head_statistics_path)
    hyper_head_per_environment_mae_path = case_dir / "hyper_head_per_environment_mae.csv"
    hyper_head_per_environment_mae_df = hyper_head_per_environment_mae(
        extracted.arrays,
        hyper_head_per_environment_mae_path,
    )
    hyper_route_usage_summary_path = case_dir / "hyper_route_usage_summary.csv"
    hyper_route_usage_summary_df = hyper_route_usage_summary(
        extracted.arrays,
        hyper_route_usage_summary_path,
    )
    prototype_similarity_path = case_dir / "prototype_similarity.csv"
    prototype_similarity_df = prototype_similarity(extracted.arrays, prototype_similarity_path)
    prototype_env_distribution_path = case_dir / "prototype_env_distribution.csv"
    prototype_env_distribution_df = prototype_env_distribution(extracted.arrays, prototype_env_distribution_path)
    heatmap_workday = case_dir / "expert_by_workday_holiday.png"
    heatmap_hour = case_dir / "expert_by_hour.png"
    heatmap_rush = case_dir / "expert_by_rush_hour.png"
    draw_heatmap(tables["workday_holiday"], heatmap_workday, "P(expert | workday/holiday)")
    draw_heatmap(tables["hour"], heatmap_hour, "P(expert | hour)")
    draw_heatmap(tables["rush_hour"], heatmap_rush, "P(expert | rush_hour)")

    expert_mae_path = case_dir / "expert_per_env_mae.csv"
    expert_mae_df = expert_per_env_mae(extracted.arrays, expert_mae_path)
    if "y_expert" in extracted.arrays:
        log(f"[INFO] extracted y_expert={extracted.arrays['y_expert'].shape}")
    else:
        log("[WARN] y_route_heads unavailable; expert_per_env_mae contains assignment-only final MAE")

    route_intervention_path = case_dir / "route_intervention_mae.csv"
    route_df = route_intervention_mae(extracted.arrays, extracted.route_preds, route_intervention_path)

    swap_path = case_dir / "env_swap_results.csv"
    swap_df, swap_values = env_swap_analysis(
        model,
        extracted.arrays,
        scaler,
        args,
        swap_path,
        seed=cli.random_state,
        batch_size=int(args.test_batch_size),
    )
    swap_plot = case_dir / "env_swap_boxplot.png"
    draw_boxplot(swap_values, swap_plot, "|pred_swap - pred_original|")
    hyper_swap_analysis_path = case_dir / "hyper_swap_analysis.csv"
    hyper_swap_analysis_df = hyper_swap_analysis(swap_df, extracted.arrays, hyper_swap_analysis_path)

    viz_files = make_visualizations(extracted.arrays, case_dir, cli.viz_method, cli.random_state)

    files: dict[str, Path | str] = {
        "split": cli.split,
        "frozen_params": str(frozen_params),
        "features": features_path,
        "per_sample": per_sample_path,
        "probe_env_results": probe_env_path,
        "inv_raw_vs_projected_probe": inv_raw_vs_projected_probe_path,
        "probe_residual_results": probe_resid_path,
        "expert_env_crosstab": crosstab_path,
        "route_usage_summary": route_usage_path,
        "hyper_head_statistics": hyper_head_statistics_path,
        "hyper_head_per_environment_mae": hyper_head_per_environment_mae_path,
        "hyper_route_usage_summary": hyper_route_usage_summary_path,
        "prototype_similarity": prototype_similarity_path,
        "prototype_env_distribution": prototype_env_distribution_path,
        "expert_by_workday_holiday": heatmap_workday,
        "expert_by_hour": heatmap_hour,
        "expert_by_rush_hour": heatmap_rush,
        "expert_per_env_mae": expert_mae_path,
        "route_intervention_mae": route_intervention_path,
        "env_swap_results": swap_path,
        "env_swap_boxplot": swap_plot,
        "hyper_swap_analysis": hyper_swap_analysis_path,
    }
    files.update(viz_files)
    readme_path = case_dir / "README.md"
    write_readme(readme_path, ckpt_path, args, extracted, probe_env_df, probe_resid_df, route_df, swap_df, files)

    manifest = {
        "checkpoint": str(ckpt_path),
        "exp_dir": str(exp_dir),
        "output_dir": str(case_dir),
        "split": cli.split,
        "max_batches": max_batches,
        "num_samples": int(extracted.arrays["y_true"].shape[0]),
        "route_k": int(extracted.arrays["route_weight"].shape[1]),
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "eval_metrics": extracted.eval_metrics,
        "route_usage_summary": route_usage_df.to_dict("records"),
        "hyper_head_statistics": hyper_head_statistics_df.to_dict("records"),
        "hyper_head_per_environment_mae": hyper_head_per_environment_mae_df.to_dict("records"),
        "hyper_route_usage_summary": hyper_route_usage_summary_df.to_dict("records"),
        "hyper_swap_analysis": hyper_swap_analysis_df.to_dict("records"),
        "prototype_similarity_rows": int(prototype_similarity_df.shape[0]),
        "prototype_env_distribution_rows": int(prototype_env_distribution_df.shape[0]),
        "files": {k: str(v) for k, v in files.items() if k not in {"split", "frozen_params"}},
    }
    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[INFO] wrote manifest={manifest_path}")
    log(f"[INFO] wrote README={readme_path}")
    log("[DONE] case study analysis complete")


if __name__ == "__main__":
    main()
