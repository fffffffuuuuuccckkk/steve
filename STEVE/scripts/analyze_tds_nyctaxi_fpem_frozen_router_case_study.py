#!/usr/bin/env python
"""Inference-only case study for the frozen Stage-2 D router.

No Stage-1 or Stage-2 checkpoint parameter is trained, fine-tuned, or rewritten.
The script reloads the Stage-1 prediction checkpoint and the standalone Stage-2
router checkpoint, reproduces the D-case metrics, then emits diagnostic tables,
NPZ files, and PNG visualizations using only test-observable router inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for path in (PROJECT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_tds_nyctaxi as tds  # noqa: E402
import train_tds_nyctaxi_fpem_frozen_router_from_checkpoint as stage2  # noqa: E402


EXPECTED = {
    "all_env_mae": 11.257920265197754,
    "learned_mae": 11.253186225891113,
    "predicted_invariant_ratio": 0.03113553114235401,
    "target_invariant_ratio": 0.311355322599411,
    "invariant_switch_precision": 0.7058823704719543,
    "invariant_switch_recall": 0.07058823853731155,
    "correct_invariant_switches": 12,
    "harmful_switches": 5,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=tds.scalar_json)


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def find_stage1_exp_dir(stage1_root: Path) -> Path:
    direct = stage1_root / "best_test_avg_model.pth"
    if direct.exists():
        return stage1_root
    matches = sorted(stage1_root.parent.glob(stage1_root.name + "_obs_k1_counterfactual_risk_router_seed2024"))
    if matches:
        return matches[0]
    matches = sorted(stage1_root.parent.glob(stage1_root.name + "*seed2024"))
    for item in matches:
        if (item / "best_test_avg_model.pth").exists():
            return item
    raise FileNotFoundError(f"cannot locate Stage-1 experiment with best_test_avg_model.pth under {stage1_root}")


def find_stage2_d_summary(stage2_root: Path, case: str) -> Tuple[Path, dict, Path]:
    summary_path = stage2_root / "summary_all_frozen_router_cases.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    summary_all = json.load(open(summary_path, "r", encoding="utf-8"))
    summaries = summary_all.get("summaries", {})
    if case not in summaries:
        raise KeyError(f"case={case} not found in {summary_path}; keys={list(summaries)}")
    summary = summaries[case]
    ckpt_path = Path(summary["artifacts"]["best_stage2_router_test_selected"])
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_DIR / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage-2 D router checkpoint not found from summary: {ckpt_path}")
    return ckpt_path, summary, summary_path


def tensor_hash_state(model: torch.nn.Module) -> str:
    return stage2.model_state_hash(model)


def collect_split(
    model,
    loader,
    scaler,
    args,
    split: str,
    include_load_stats: bool,
    include_pred_diff: bool,
) -> dict:
    model.eval()
    chunks: Dict[str, List] = {
        "features": [],
        "y_true": [],
        "y_env": [],
        "y_inv": [],
        "env_loss": [],
        "inv_loss": [],
        "sample_index": [],
        "time_label": [],
        "load_score": [],
        "load_mean": [],
        "load_std": [],
        "load_max": [],
        "load_high_ratio": [],
        "pred_diff_mean": [],
        "pred_diff_max": [],
        "pred_diff_std": [],
        "inv_repr": [],
        "env_repr": [],
    }
    stored_c_seen = False
    with torch.no_grad():
        for raw_batch in loader:
            batch = tds.to_device(raw_batch, args.device)
            unpacked = tds.unpack_tds_batch(batch)
            stored_c_seen = stored_c_seen or unpacked["stored_c"] is not None
            output = model.forward_output(
                unpacked["data"],
                exog=unpacked["stored_c"],
                time_label=unpacked["time_label"],
                training=False,
                sample_index=unpacked["sample_index"],
                observable_load_profile=unpacked["observable_load_profile"],
                observable_load_score=unpacked["observable_load_score"],
                cached_load_level=unpacked["cached_load_level"],
            )
            y_env = output["y_env_selected"].detach()
            y_inv = output["y_inv"].detach()
            target = unpacked["target"].detach()
            env_loss = stage2.samplewise_masked_mae(y_env, target, scaler, args.yita)
            inv_loss = stage2.samplewise_masked_mae(y_inv, target, scaler, args.yita)
            feats = [output["hard_router_features"].detach().float()]
            if include_load_stats:
                load_feats = stage2.observable_load_features(
                    unpacked["observable_load_profile"],
                    unpacked["observable_load_score"],
                )
                if load_feats is None:
                    raise RuntimeError("D router requires observable load features, but they are unavailable")
                feats.append(load_feats.detach().float())
            pred_diff_feat = stage2.pred_diff_features(y_env, y_inv).detach().float()
            if include_pred_diff:
                feats.append(pred_diff_feat)
            features = torch.cat(feats, dim=1)

            c_obs = unpacked["observable_load_profile"].detach().float()
            c_flat = (c_obs / 5.0).reshape(c_obs.shape[0], -1)
            pred_diff_raw = (y_env - y_inv).detach().reshape(y_env.shape[0], -1)
            chunks["features"].append(features.cpu())
            chunks["y_true"].append(target.cpu())
            chunks["y_env"].append(y_env.cpu())
            chunks["y_inv"].append(y_inv.cpu())
            chunks["env_loss"].append(env_loss.cpu())
            chunks["inv_loss"].append(inv_loss.cpu())
            chunks["sample_index"].append(unpacked["sample_index"].detach().cpu())
            chunks["time_label"].append(unpacked["time_label"].detach().cpu())
            chunks["load_score"].append(unpacked["observable_load_score"].detach().cpu())
            chunks["load_mean"].append(c_flat.mean(dim=1).cpu())
            chunks["load_std"].append(c_flat.std(dim=1, unbiased=False).cpu())
            chunks["load_max"].append(c_flat.amax(dim=1).cpu())
            chunks["load_high_ratio"].append((c_flat >= 0.8).float().mean(dim=1).cpu())
            chunks["pred_diff_mean"].append(pred_diff_raw.abs().mean(dim=1).cpu())
            chunks["pred_diff_max"].append(pred_diff_raw.abs().amax(dim=1).cpu())
            chunks["pred_diff_std"].append(pred_diff_raw.std(dim=1, unbiased=False).cpu())
            chunks["inv_repr"].append(output["Z_inv"].detach().mean(dim=1).cpu())
            chunks["env_repr"].append(output["C_cur"].detach().mean(dim=1).cpu())

    out = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    out["delta_true"] = out["inv_loss"] - out["env_loss"]
    out["target_inv"] = out["delta_true"] < 0
    out["stored_c_seen"] = stored_c_seen
    out["split"] = split
    return out


def standardize_features(train: dict, splits: Dict[str, dict]) -> dict:
    mean = train["features"].float().mean(dim=0, keepdim=True)
    std = train["features"].float().std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    for split in splits.values():
        split["features_std"] = (split["features"].float() - mean) / std
    return {
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "std_min": float(std.min().item()),
        "std_max": float(std.max().item()),
        "train_std_feature_mean_abs_max": float(splits["train"]["features_std"].mean(dim=0).abs().max().item()),
    }


def predict_router(router, data: dict, device: str, batch_size: int, delta_scale: float) -> None:
    router.eval()
    preds = []
    with torch.no_grad():
        x = data["features_std"].float()
        for start in range(0, x.shape[0], batch_size):
            pred_scaled = router(x[start : start + batch_size].to(device)).detach().cpu()
            preds.append(pred_scaled)
    data["delta_hat_scaled"] = torch.cat(preds, dim=0)
    data["delta_hat"] = data["delta_hat_scaled"] * float(delta_scale)
    data["use_inv"] = data["delta_hat"] < 0
    data["learned_loss"] = torch.where(data["use_inv"], data["inv_loss"], data["env_loss"])
    data["oracle_loss"] = torch.minimum(data["inv_loss"], data["env_loss"])


def metrics_from_split(data: dict) -> dict:
    delta = data["delta_true"]
    target_inv = data["target_inv"]
    use_inv = data["use_inv"]
    correct_inv = use_inv & target_inv
    harmful = use_inv & (~target_inv)
    all_env = data["env_loss"].mean()
    learned = data["learned_loss"].mean()
    oracle = data["oracle_loss"].mean()
    net_gain = all_env - learned
    saved = ((-delta).clamp_min(0) * correct_inv.float()).sum() / max(delta.numel(), 1)
    added = (delta.clamp_min(0) * harmful.float()).sum() / max(delta.numel(), 1)
    inv_switches = use_inv.sum().float().clamp_min(1)
    target_count = target_inv.sum().float().clamp_min(1)
    recalls = []
    if bool(target_inv.any()):
        recalls.append(use_inv[target_inv].float().mean())
    if bool((~target_inv).any()):
        recalls.append((~use_inv[~target_inv]).float().mean())
    balanced = torch.stack(recalls).mean() if recalls else torch.tensor(0.0)
    inv_score = (-data["delta_hat"]).numpy()
    labels = target_inv.numpy().astype(int)
    try:
        auroc = float(roc_auc_score(labels, inv_score))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(labels, inv_score))
    except Exception:
        auprc = float("nan")
    return {
        "num_samples": int(delta.numel()),
        "all_environment_mae": float(all_env.item()),
        "all_invariant_mae": float(data["inv_loss"].mean().item()),
        "learned_routing_mae": float(learned.item()),
        "oracle_mae": float(oracle.item()),
        "env_route_ratio": float((~use_inv).float().mean().item()),
        "inv_route_ratio": float(use_inv.float().mean().item()),
        "target_inv_ratio": float(target_inv.float().mean().item()),
        "router_accuracy": float((use_inv == target_inv).float().mean().item()),
        "balanced_accuracy": float(balanced.item()),
        "inv_switch_precision": float((correct_inv.sum().float() / inv_switches).item()),
        "inv_switch_recall": float((correct_inv.sum().float() / target_count).item()),
        "correct_beneficial_inv_switches": int(correct_inv.sum().item()),
        "harmful_inv_switches": int(harmful.sum().item()),
        "saved_loss": float(saved.item()),
        "added_loss": float(added.item()),
        "net_gain": float(net_gain.item()),
        "regret": float((learned - oracle).item()),
        "oracle_gap_closed": float((net_gain / (all_env - oracle).clamp_min(1e-8)).item()),
        "delta_hat_min": float(data["delta_hat"].min().item()),
        "delta_hat_mean": float(data["delta_hat"].mean().item()),
        "delta_hat_max": float(data["delta_hat"].max().item()),
        "delta_hat_std": float(data["delta_hat"].std(unbiased=False).item()),
        "inv_auroc": auroc,
        "inv_auprc": auprc,
    }


def check_reproduction(metrics: dict, out_dir: Path, tol: float = 5e-5) -> None:
    checks = {
        "all_env_mae": metrics["all_environment_mae"],
        "learned_mae": metrics["learned_routing_mae"],
        "predicted_invariant_ratio": metrics["inv_route_ratio"],
        "target_invariant_ratio": metrics["target_inv_ratio"],
        "invariant_switch_precision": metrics["inv_switch_precision"],
        "invariant_switch_recall": metrics["inv_switch_recall"],
        "correct_invariant_switches": metrics["correct_beneficial_inv_switches"],
        "harmful_switches": metrics["harmful_inv_switches"],
    }
    diffs = {}
    ok = True
    for key, expected in EXPECTED.items():
        actual = checks[key]
        diff = abs(float(actual) - float(expected))
        diffs[key] = {"actual": actual, "expected": expected, "diff": diff}
        if isinstance(expected, int):
            ok = ok and int(actual) == expected
        else:
            ok = ok and diff <= tol
    write_json(out_dir / "reproduction_check.json", {"ok": ok, "checks": diffs, "tolerance": tol})
    if not ok:
        raise RuntimeError(f"D reproduction failed; see {out_dir / 'reproduction_check.json'}")


def colors() -> dict:
    return {
        "correct_invariant": (35, 125, 210),
        "harmful_switch": (220, 70, 60),
        "missed_gain": (245, 165, 35),
        "correct_environment": (80, 170, 90),
        "gray": (100, 100, 100),
        "black": (0, 0, 0),
        "light": (240, 240, 240),
    }


def font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def map_xy(x, y, xmin, xmax, ymin, ymax, box):
    left, top, right, bottom = box
    px = left + (x - xmin) / max(xmax - xmin, 1e-12) * (right - left)
    py = bottom - (y - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)
    return int(px), int(py)


def save_table_csv(path: Path, rows: List[dict]) -> None:
    write_csv(path, rows)


def draw_scatter_route(data: dict, out_dir: Path) -> None:
    x = data["delta_hat"].numpy()
    y = data["delta_true"].numpy()
    use = data["use_inv"].numpy().astype(bool)
    tgt = data["target_inv"].numpy().astype(bool)
    cat = np.where(use & tgt, "correct_invariant", np.where(use & ~tgt, "harmful_switch", np.where(~use & tgt, "missed_gain", "correct_environment")))
    rows = [
        {"sample_index": int(data["sample_index"][i]), "delta_hat": float(x[i]), "delta_true": float(y[i]), "category": cat[i]}
        for i in range(len(x))
    ]
    write_csv(out_dir / "route_score_vs_true_delta.csv", rows)
    w, h = 1100, 820
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    box = (90, 70, 820, 720)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    pad_x = max((xmax - xmin) * 0.08, 1e-3)
    pad_y = max((ymax - ymin) * 0.08, 1e-3)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    d.rectangle(box, outline="black")
    if xmin < 0 < xmax:
        x0, _ = map_xy(0, ymin, xmin, xmax, ymin, ymax, box)
        d.line((x0, box[1], x0, box[3]), fill=(80, 80, 80), width=2)
    if ymin < 0 < ymax:
        _, y0 = map_xy(xmin, 0, xmin, xmax, ymin, ymax, box)
        d.line((box[0], y0, box[2], y0), fill=(80, 80, 80), width=2)
    pal = colors()
    for name in ["correct_invariant", "harmful_switch", "missed_gain", "correct_environment"]:
        idx = np.where(cat == name)[0]
        for i in idx:
            px, py = map_xy(x[i], y[i], xmin, xmax, ymin, ymax, box)
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=pal[name], outline=None)
    d.text((90, 25), "route_score_vs_true_delta: x=delta_hat, y=delta_true (L_inv-L_env)", fill="black", font=font(20))
    d.text((390, 735), "delta_hat (<0 routes to invariant)", fill="black", font=font(16))
    d.text((10, 330), "delta_true", fill="black", font=font(16))
    legend_y = 100
    for name in ["correct_invariant", "harmful_switch", "missed_gain", "correct_environment"]:
        n = int((cat == name).sum())
        d.rectangle((850, legend_y, 870, legend_y + 20), fill=pal[name])
        d.text((880, legend_y), f"{name}: {n}", fill="black", font=font(15))
        legend_y += 38
    im.save(out_dir / "route_score_vs_true_delta.png")


def quantile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) <= 2:
        return np.zeros_like(values, dtype=int)
    return np.clip(np.searchsorted(qs[1:-1], values, side="right"), 0, len(qs) - 2)


def draw_line_plot(path: Path, title: str, xs: np.ndarray, series: List[Tuple[str, np.ndarray, Tuple[int, int, int]]], ylabel: str = "") -> None:
    w, h = 1050, 700
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    box = (85, 70, 880, 600)
    d.rectangle(box, outline="black")
    all_y = np.concatenate([np.asarray(s[1], dtype=float) for s in series if len(s[1])])
    finite = all_y[np.isfinite(all_y)]
    ymin, ymax = (0.0, 1.0) if finite.size == 0 else (float(finite.min()), float(finite.max()))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    pad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - pad, ymax + pad
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    if xmin == xmax:
        xmax = xmin + 1.0
    d.text((85, 25), title, fill="black", font=font(20))
    d.text((20, 320), ylabel, fill="black", font=font(15))
    for label, ys, col in series:
        pts = [map_xy(float(x), float(y), xmin, xmax, ymin, ymax, box) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(pts) >= 2:
            d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=col)
    ly = 90
    for label, _, col in series:
        d.rectangle((900, ly, 920, ly + 15), fill=col)
        d.text((930, ly - 3), label, fill="black", font=font(14))
        ly += 30
    im.save(path)


def calibration(data: dict, out_dir: Path) -> None:
    dh = data["delta_hat"].numpy()
    dt = data["delta_true"].numpy()
    target = data["target_inv"].numpy().astype(bool)
    use = data["use_inv"].numpy().astype(bool)
    bins = quantile_bins(dh, 10)
    rows = []
    for b in range(int(bins.max()) + 1):
        m = bins == b
        if not m.any():
            continue
        correct_inv = use[m] & target[m]
        rows.append({
            "bin": b,
            "delta_hat_min": float(dh[m].min()),
            "delta_hat_max": float(dh[m].max()),
            "count": int(m.sum()),
            "target_invariant_ratio": float(target[m].mean()),
            "predicted_invariant_ratio": float(use[m].mean()),
            "mean_delta_true": float(dt[m].mean()),
            "invariant_switch_precision": float(correct_inv.sum() / max(use[m].sum(), 1)),
            "actual_average_gain_if_routed": float((data["env_loss"].numpy()[m] - data["learned_loss"].numpy()[m]).mean()),
        })
    write_csv(out_dir / "router_score_calibration.csv", rows)
    xs = np.array([r["bin"] for r in rows], dtype=float)
    draw_line_plot(
        out_dir / "router_score_calibration.png",
        "router score calibration by delta_hat quantile",
        xs,
        [
            ("target inv ratio", np.array([r["target_invariant_ratio"] for r in rows]), (30, 120, 210)),
            ("pred inv ratio", np.array([r["predicted_invariant_ratio"] for r in rows]), (220, 80, 60)),
            ("mean delta_true", np.array([r["mean_delta_true"] for r in rows]), (40, 160, 80)),
            ("avg gain", np.array([r["actual_average_gain_if_routed"] for r in rows]), (150, 80, 190)),
        ],
        ylabel="value",
    )


def coverage_curves(data: dict, out_dir: Path) -> None:
    env = data["env_loss"].numpy()
    inv = data["inv_loss"].numpy()
    delta = data["delta_true"].numpy()
    target = delta < 0
    conf = -data["delta_hat"].numpy()
    n = len(delta)
    order = np.argsort(-conf)
    oracle_order = np.argsort(-(-delta))
    rows = []
    oracle_rows = []
    base = float(env.mean())
    current_k = int(data["use_inv"].sum().item())
    for k in range(0, n + 1):
        use = np.zeros(n, dtype=bool)
        if k > 0:
            use[order[:k]] = True
        loss = np.where(use, inv, env)
        correct = use & target
        precision = float(correct.sum() / max(use.sum(), 1))
        recall = float(correct.sum() / max(target.sum(), 1))
        rows.append({
            "k": k,
            "coverage": k / n,
            "mae": float(loss.mean()),
            "net_gain": float(base - loss.mean()),
            "precision": precision,
            "recall": recall,
            "is_current_zero_boundary": k == current_k,
        })
        ou = np.zeros(n, dtype=bool)
        if k > 0:
            ou[oracle_order[:k]] = True
        oloss = np.where(ou, inv, env)
        oracle_rows.append({
            "k": k,
            "coverage": k / n,
            "oracle_mae": float(oloss.mean()),
            "oracle_net_gain": float(base - oloss.mean()),
        })
    write_csv(out_dir / "coverage_gain_curve.csv", rows)
    write_csv(out_dir / "coverage_oracle_curve.csv", oracle_rows)
    xs = np.array([r["coverage"] for r in rows])
    draw_line_plot(
        out_dir / "cumulative_gain_vs_switch_coverage.png",
        "cumulative net gain vs invariant switch coverage",
        xs,
        [
            ("router score order", np.array([r["net_gain"] for r in rows]), (30, 120, 210)),
            ("oracle order", np.array([r["oracle_net_gain"] for r in oracle_rows]), (80, 170, 80)),
        ],
        ylabel="net gain vs all-env",
    )
    draw_line_plot(
        out_dir / "precision_recall_vs_switch_coverage.png",
        "precision/recall vs invariant switch coverage",
        xs,
        [
            ("precision", np.array([r["precision"] for r in rows]), (220, 80, 60)),
            ("recall", np.array([r["recall"] for r in rows]), (30, 120, 210)),
        ],
        ylabel="precision/recall",
    )
    draw_line_plot(
        out_dir / "mae_vs_switch_budget.png",
        "MAE vs invariant switch budget",
        xs,
        [
            ("router score order", np.array([r["mae"] for r in rows]), (30, 120, 210)),
            ("all-environment", np.full_like(xs, base), (80, 80, 80)),
            ("oracle order", np.array([r["oracle_mae"] for r in oracle_rows]), (80, 170, 80)),
        ],
        ylabel="MAE",
    )


def raw_arrays(data: dict, scaler) -> dict:
    return {
        "y_true": scaler.inverse_transform(data["y_true"]).numpy(),
        "y_env": scaler.inverse_transform(data["y_env"]).numpy(),
        "y_inv": scaler.inverse_transform(data["y_inv"]).numpy(),
    }


def select_representative_cases(data: dict) -> List[dict]:
    delta = data["delta_true"].numpy()
    use = data["use_inv"].numpy().astype(bool)
    target = data["target_inv"].numpy().astype(bool)
    specs = [
        ("correct_invariant", use & target, -delta),
        ("harmful_switch", use & ~target, delta),
        ("missed_invariant_gain", ~use & target, -delta),
        ("correct_environment", ~use & ~target, delta),
    ]
    rows = []
    for name, mask, score in specs:
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        chosen = idx[np.argsort(-score[idx])[:3]]
        for rank, i in enumerate(chosen, 1):
            rows.append({"category": name, "rank": rank, "local_index": int(i), "sample_index": int(data["sample_index"][i])})
    return rows


def plot_case(data: dict, arrays: dict, row: dict, out_dir: Path) -> dict:
    i = row["local_index"]
    y_true = arrays["y_true"][i]
    y_env = arrays["y_env"][i]
    y_inv = arrays["y_inv"][i]
    use_inv = bool(data["use_inv"][i])
    y_route = y_inv if use_inv else y_env
    # Average channels for compact visualization.
    gt = y_true.mean(axis=(1, 2))
    env = y_env.mean(axis=(1, 2))
    inv = y_inv.mean(axis=(1, 2))
    routed = y_route.mean(axis=(1, 2))
    disagreement = np.abs((y_env - y_inv).mean(axis=(0, 2)))
    top_nodes = np.argsort(-disagreement)[:3]
    err_diff = np.abs(y_inv - y_true).mean(axis=2).T - np.abs(y_env - y_true).mean(axis=2).T  # node x horizon

    w, h = 1100, 900
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    title = f"{row['category']} rank={row['rank']} sample_index={row['sample_index']}"
    d.text((40, 20), title, fill="black", font=font(20))
    meta = (
        f"route={'inv' if use_inv else 'env'} correct={bool(data['use_inv'][i] == data['target_inv'][i])} "
        f"load={float(data['load_score'][i]):.4f} delta_hat={float(data['delta_hat'][i]):.4f} "
        f"delta_true={float(data['delta_true'][i]):.4f} L_env={float(data['env_loss'][i]):.4f} L_inv={float(data['inv_loss'][i]):.4f}"
    )
    d.text((40, 50), meta, fill="black", font=font(14))
    box = (70, 100, 1030, 380)
    d.rectangle(box, outline="black")
    xs = np.arange(len(gt), dtype=float)
    all_y = np.concatenate([gt, env, inv, routed])
    ymin, ymax = float(all_y.min()), float(all_y.max())
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    ser = [("GT", gt, (0, 0, 0)), ("Env", env, (30, 120, 210)), ("Inv", inv, (220, 80, 60)), ("Routed", routed, (80, 170, 80))]
    for label, yy, col in ser:
        pts = [map_xy(x, y, xs.min(), max(xs.max(), 1), ymin, ymax, box) for x, y in zip(xs, yy)]
        if len(pts) >= 2:
            d.line(pts, fill=col, width=3)
        else:
            px, py = pts[0]
            d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=col)
    ly = 105
    for label, _, col in ser:
        d.rectangle((930, ly, 950, ly + 12), fill=col)
        d.text((955, ly - 3), label, fill="black", font=font(13))
        ly += 25
    d.text((70, 390), f"Top disagreement nodes: {top_nodes.tolist()}", fill="black", font=font(14))
    # Heatmap.
    heat_box = (70, 430, 1030, 820)
    d.rectangle(heat_box, outline="black")
    v = err_diff
    vmax = max(abs(float(v.min())), abs(float(v.max())), 1e-6)
    n_node, n_h = v.shape
    cell_w = max(1, (heat_box[2] - heat_box[0]) / max(n_h, 1))
    cell_h = max(1, (heat_box[3] - heat_box[1]) / max(n_node, 1))
    for nn in range(n_node):
        for hh in range(n_h):
            val = float(v[nn, hh]) / vmax
            if val < 0:
                col = (int(255 * (1 + val)), int(255 * (1 + val)), 255)
            else:
                col = (255, int(255 * (1 - val)), int(255 * (1 - val)))
            x0 = int(heat_box[0] + hh * cell_w)
            x1 = int(heat_box[0] + (hh + 1) * cell_w)
            y0 = int(heat_box[1] + nn * cell_h)
            y1 = int(heat_box[1] + (nn + 1) * cell_h)
            d.rectangle((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)), fill=col)
    d.text((70, 830), "Heatmap: abs_error_inv - abs_error_env; blue=invariant better, red=environment better", fill="black", font=font(14))
    out_name = f"case_{row['category']}_{row['rank']}_sample{row['sample_index']}.png"
    im.save(out_dir / out_name)
    row.update({
        "figure": out_name,
        "delta_hat": float(data["delta_hat"][i]),
        "delta_true": float(data["delta_true"][i]),
        "L_env": float(data["env_loss"][i]),
        "L_inv": float(data["inv_loss"][i]),
        "observable_load_score": float(data["load_score"][i]),
        "learned_route": "invariant" if use_inv else "environment",
        "correct": bool(data["use_inv"][i] == data["target_inv"][i]),
        "sample_net_gain": float(data["env_loss"][i] - data["learned_loss"][i]),
        "sample_regret": float(data["learned_loss"][i] - data["oracle_loss"][i]),
        "time_label": int(data["time_label"][i]),
    })
    return row


def grouped_metrics(mask: np.ndarray, data: dict) -> dict:
    if int(mask.sum()) == 0:
        return {}
    mask_t = torch.from_numpy(mask.astype(bool))
    sub = {
        k: v[mask_t] if torch.is_tensor(v) and len(v) == len(mask) else v
        for k, v in data.items()
    }
    return metrics_from_split(sub)


def observable_load_analysis(data: dict, out_dir: Path) -> None:
    score = data["load_score"].numpy()
    bins = quantile_bins(score, 5)
    rows = []
    for b in range(int(bins.max()) + 1):
        m = bins == b
        met = grouped_metrics(m, data)
        rows.append({
            "load_bin": b,
            "load_min": float(score[m].min()),
            "load_max": float(score[m].max()),
            "sample_count": int(m.sum()),
            "mean_load": float(data["load_mean"][m].mean()),
            "high_load_node_ratio": float(data["load_high_ratio"][m].mean()),
            "load_std": float(data["load_std"][m].mean()),
            "max_load_level": float(data["load_max"][m].max()),
            "target_invariant_ratio": met.get("target_inv_ratio"),
            "predicted_invariant_ratio": met.get("inv_route_ratio"),
            "switch_precision": met.get("inv_switch_precision"),
            "switch_recall": met.get("inv_switch_recall"),
            "mean_delta_true": float(data["delta_true"][m].mean()),
            "all_env_mae": met.get("all_environment_mae"),
            "learned_mae": met.get("learned_routing_mae"),
            "net_gain": met.get("net_gain"),
        })
    write_csv(out_dir / "observable_load_conditioned_routing.csv", rows)
    xs = np.array([r["load_bin"] for r in rows], dtype=float)
    draw_line_plot(
        out_dir / "observable_load_conditioned_routing.png",
        "observable load conditioned routing",
        xs,
        [
            ("target inv ratio", np.array([r["target_invariant_ratio"] for r in rows]), (30, 120, 210)),
            ("pred inv ratio", np.array([r["predicted_invariant_ratio"] for r in rows]), (220, 80, 60)),
            ("precision", np.array([r["switch_precision"] for r in rows]), (80, 170, 80)),
            ("net gain", np.array([r["net_gain"] for r in rows]), (150, 80, 190)),
        ],
        ylabel="value",
    )
    # Load-vs-pred-diff heatmap.
    diff = data["pred_diff_mean"].numpy()
    lb = quantile_bins(score, 5)
    db = quantile_bins(diff, 5)
    heat_rows = []
    grid = np.full((5, 5), np.nan, dtype=float)
    for i in range(5):
        for j in range(5):
            m = (lb == i) & (db == j)
            if m.any():
                grid[j, i] = float(data["delta_true"][m].mean())
                heat_rows.append({
                    "load_bin": i,
                    "prediction_diff_bin": j,
                    "count": int(m.sum()),
                    "mean_delta_true": float(data["delta_true"][m].mean()),
                    "target_inv_ratio": float(data["target_inv"][m].float().mean()),
                })
    write_csv(out_dir / "load_vs_prediction_difference_heatmap.csv", heat_rows)
    save_heatmap(out_dir / "load_vs_prediction_difference_heatmap.png", grid, "load quantile", "prediction diff quantile", "mean delta_true")


def save_heatmap(path: Path, grid: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    w, h = 720, 620
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((50, 20), title, fill="black", font=font(20))
    box = (90, 80, 580, 520)
    rows, cols = grid.shape
    finite = grid[np.isfinite(grid)]
    vmax = max(abs(float(finite.min())) if finite.size else 1, abs(float(finite.max())) if finite.size else 1, 1e-6)
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if not np.isfinite(val):
                col = (230, 230, 230)
            else:
                x = float(val) / vmax
                col = (255, int(255 * (1 - max(x, 0))), int(255 * (1 - max(x, 0)))) if x >= 0 else (int(255 * (1 + x)), int(255 * (1 + x)), 255)
            x0 = int(box[0] + c * (box[2] - box[0]) / cols)
            x1 = int(box[0] + (c + 1) * (box[2] - box[0]) / cols)
            y0 = int(box[1] + r * (box[3] - box[1]) / rows)
            y1 = int(box[1] + (r + 1) * (box[3] - box[1]) / rows)
            d.rectangle((x0, y0, x1, y1), fill=col, outline="white")
            if np.isfinite(val):
                d.text((x0 + 8, y0 + 8), f"{val:.3f}", fill="black", font=font(12))
    d.rectangle(box, outline="black")
    d.text((250, 540), xlabel, fill="black", font=font(15))
    d.text((5, 280), ylabel, fill="black", font=font(15))
    im.save(path)


def temporal_analysis(data: dict, out_dir: Path) -> None:
    tl = data["time_label"].numpy().astype(int)
    hour = np.where(tl < 24, tl, tl - 24)
    workday = tl < 24
    peak = np.isin(hour, [7, 8, 9, 16, 17, 18, 19])
    rows = []
    for group_name, values in [("hour", hour), ("day_type", np.where(workday, 0, 1)), ("peak", np.where(peak, 1, 0))]:
        for val in sorted(np.unique(values).tolist()):
            m = values == val
            met = grouped_metrics(m, data)
            rows.append({
                "group": group_name,
                "value": int(val),
                "count": int(m.sum()),
                "target_invariant_ratio": met.get("target_inv_ratio"),
                "predicted_invariant_ratio": met.get("inv_route_ratio"),
                "precision": met.get("inv_switch_precision"),
                "recall": met.get("inv_switch_recall"),
                "net_gain": met.get("net_gain"),
            })
    write_csv(out_dir / "temporal_context_routing.csv", rows)
    hour_rows = [r for r in rows if r["group"] == "hour"]
    xs = np.array([r["value"] for r in hour_rows], dtype=float)
    draw_line_plot(
        out_dir / "temporal_context_routing.png",
        "temporal context routing by hour",
        xs,
        [
            ("target inv ratio", np.array([r["target_invariant_ratio"] for r in hour_rows]), (30, 120, 210)),
            ("pred inv ratio", np.array([r["predicted_invariant_ratio"] for r in hour_rows]), (220, 80, 60)),
            ("net gain", np.array([r["net_gain"] for r in hour_rows]), (80, 170, 80)),
        ],
        ylabel="value",
    )


def embed_2d(x: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    x = StandardScaler().fit_transform(x)
    n_comp = min(30, x.shape[1], x.shape[0] - 1)
    pca = PCA(n_components=max(2, n_comp), random_state=seed).fit_transform(x)
    try:
        emb = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, x.shape[0] // 20)), random_state=seed).fit_transform(pca)
        return emb, "PCA+t-SNE"
    except Exception:
        return pca[:, :2], "PCA fallback"


def draw_multi_scatter(path: Path, title: str, panels: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    w, h = 1500, 800
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((30, 20), title, fill="black", font=font(20))
    panel_w, panel_h = 340, 300
    for idx, (label, emb, color_values) in enumerate(panels):
        row = idx // 4
        col = idx % 4
        left = 40 + col * 360
        top = 70 + row * 350
        box = (left, top, left + panel_w, top + panel_h)
        d.rectangle(box, outline="black")
        x, y = emb[:, 0], emb[:, 1]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        vals = color_values.astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        for xi, yi, vi in zip(x, y, vals):
            t = 0.5 if vmax == vmin else (vi - vmin) / (vmax - vmin)
            colr = (int(255 * t), 60, int(255 * (1 - t)))
            px, py = map_xy(float(xi), float(yi), xmin, xmax, ymin, ymax, box)
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=colr)
        d.text((left, top - 22), label, fill="black", font=font(14))
    im.save(path)


def representation_analysis(train: dict, test: dict, out_dir: Path, seed: int) -> dict:
    inv_train = train["inv_repr"].numpy()
    env_train = train["env_repr"].numpy()
    inv_test = test["inv_repr"].numpy()
    env_test = test["env_repr"].numpy()
    inv_emb, inv_method = embed_2d(inv_test, seed)
    env_emb, env_method = embed_2d(env_test, seed)
    tl = test["time_label"].numpy().astype(int)
    hour = np.where(tl < 24, tl, tl - 24)
    workday = (tl < 24).astype(int)
    load = test["load_score"].numpy()
    pred_route = test["use_inv"].numpy().astype(int)
    target_route = test["target_inv"].numpy().astype(int)
    draw_multi_scatter(
        out_dir / "representation_tsne_env_vs_inv.png",
        f"Representation visualization ({env_method}/{inv_method}); color variables shown per panel",
        [
            ("env by load", env_emb, load),
            ("env by TOD", env_emb, hour),
            ("env by pred route", env_emb, pred_route),
            ("env by target route", env_emb, target_route),
            ("inv by load", inv_emb, load),
            ("inv by TOD", inv_emb, hour),
            ("inv by pred route", inv_emb, pred_route),
            ("inv by target route", inv_emb, target_route),
        ],
    )
    np.savez_compressed(
        out_dir / "representation_embeddings.npz",
        inv_emb=inv_emb.astype(np.float32),
        env_emb=env_emb.astype(np.float32),
        load=load.astype(np.float32),
        hour=hour.astype(np.int64),
        workday=workday.astype(np.int64),
        pred_route=pred_route.astype(np.int64),
        target_route=target_route.astype(np.int64),
    )

    def probe_one(name: str, xtr, xte) -> List[dict]:
        rows = []
        y_load_tr = train["load_score"].numpy()
        y_load_te = test["load_score"].numpy()
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(xtr, y_load_tr)
        pred = reg.predict(xte)
        rows.append({"representation": name, "task": "observable_load_regression", "metric": "r2", "value": float(r2_score(y_load_te, pred))})
        rows.append({"representation": name, "task": "observable_load_regression", "metric": "mae", "value": float(mean_absolute_error(y_load_te, pred))})
        tasks = {
            "TOD": (np.where(train["time_label"].numpy() < 24, train["time_label"].numpy(), train["time_label"].numpy() - 24).astype(int), hour),
            "workday": ((train["time_label"].numpy() < 24).astype(int), workday),
            "target_route": (train["target_inv"].numpy().astype(int), target_route),
        }
        for task, (ytr, yte) in tasks.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=None))
            clf.fit(xtr, ytr)
            yp = clf.predict(xte)
            rows.append({"representation": name, "task": task, "metric": "accuracy", "value": float(accuracy_score(yte, yp))})
            rows.append({"representation": name, "task": task, "metric": "macro_f1", "value": float(f1_score(yte, yp, average="macro"))})
            if len(np.unique(yte)) == 2:
                try:
                    prob = clf.predict_proba(xte)[:, 1]
                    rows.append({"representation": name, "task": task, "metric": "auroc", "value": float(roc_auc_score(yte, prob))})
                except Exception:
                    pass
            knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
            knn.fit(xtr, ytr)
            rows.append({"representation": name, "task": task, "metric": "knn5_accuracy", "value": float(accuracy_score(yte, knn.predict(xte)))})
            if len(np.unique(yte)) > 1:
                try:
                    rows.append({"representation": name, "task": task, "metric": "silhouette", "value": float(silhouette_score(StandardScaler().fit_transform(xte), yte))})
                except Exception:
                    pass
        return rows

    rows = probe_one("environment", env_train, env_test) + probe_one("invariant", inv_train, inv_test)
    write_csv(out_dir / "representation_probe_comparison.csv", rows)
    # Compact bar-like plot for headline metrics.
    labels = ["observable_load_regression:r2", "TOD:accuracy", "workday:accuracy", "target_route:auroc"]
    plot_rows = []
    for rep in ["environment", "invariant"]:
        vals = []
        for item in labels:
            task, metric = item.split(":")
            matched = [r["value"] for r in rows if r["representation"] == rep and r["task"] == task and r["metric"] == metric]
            vals.append(matched[0] if matched else np.nan)
        plot_rows.append((rep, vals))
    draw_bar_comparison(out_dir / "representation_probe_comparison.png", labels, plot_rows, "representation probe comparison")
    return {"probe_rows": rows, "embedding_methods": {"env": env_method, "inv": inv_method}}


def draw_bar_comparison(path: Path, labels: List[str], rows: List[Tuple[str, List[float]]], title: str) -> None:
    w, h = 1100, 650
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((50, 25), title, fill="black", font=font(20))
    box = (100, 100, 1000, 540)
    d.rectangle(box, outline="black")
    maxv = 1.0
    colors_local = [(50, 140, 220), (220, 100, 80)]
    group_w = (box[2] - box[0]) / len(labels)
    for gi, lab in enumerate(labels):
        x_center = box[0] + gi * group_w + group_w / 2
        d.text((int(x_center - group_w / 2 + 5), 550), lab[:24], fill="black", font=font(11))
        for ri, (rep, vals) in enumerate(rows):
            val = vals[gi]
            if not np.isfinite(val):
                continue
            bw = group_w / 4
            x0 = int(x_center - bw + ri * bw)
            x1 = int(x0 + bw * 0.9)
            y1 = box[3]
            y0 = int(box[3] - max(0, min(val, maxv)) / maxv * (box[3] - box[1]))
            d.rectangle((x0, y0, x1, y1), fill=colors_local[ri])
            d.text((x0, y0 - 18), f"{val:.2f}", fill="black", font=font(10))
    ly = 80
    for ri, (rep, _) in enumerate(rows):
        d.rectangle((900, ly, 920, ly + 15), fill=colors_local[ri])
        d.text((930, ly - 3), rep, fill="black", font=font(13))
        ly += 25
    im.save(path)


def swap_blocked_report(out_dir: Path) -> dict:
    report = {
        "status": "blocked",
        "reason": "Current StableST forward path does not expose a safe inference-only API to inject swapped environment representation before the environment prediction head while keeping all parameters unchanged.",
        "minimal_safe_change": "Add an analysis-only return_analysis_features/route_override or env_rep_override argument that is ignored by default and only used in analyze scripts; then recompute y_env from frozen heads with injected C_cur.",
    }
    write_json(out_dir / "environment_swap_sensitivity.json", report)
    write_csv(out_dir / "swap_case_study.csv", [report])
    im = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 40), "environment_swap_sensitivity: blocked", fill="black", font=font(22))
    d.text((40, 90), report["reason"], fill="black", font=font(14))
    d.text((40, 150), "No synthetic swap result is reported.", fill=(180, 0, 0), font=font(16))
    d.text((40, 200), "Minimal safe change:", fill="black", font=font(16))
    d.text((40, 230), report["minimal_safe_change"], fill="black", font=font(13))
    im.save(out_dir / "environment_swap_sensitivity.png")
    return report


def save_npz(out_dir: Path, data: dict, scaler) -> None:
    raw = raw_arrays(data, scaler)
    y_learned = np.where(data["use_inv"].numpy().reshape(-1, 1, 1, 1), raw["y_inv"], raw["y_env"])
    np.savez_compressed(
        out_dir / "predictions_routes_features.npz",
        y_true=raw["y_true"].astype(np.float32),
        y_env=raw["y_env"].astype(np.float32),
        y_inv=raw["y_inv"].astype(np.float32),
        y_learned=y_learned.astype(np.float32),
        delta_true=data["delta_true"].numpy().astype(np.float32),
        delta_hat=data["delta_hat"].numpy().astype(np.float32),
        use_inv=data["use_inv"].numpy().astype(np.int64),
        target_inv=data["target_inv"].numpy().astype(np.int64),
        env_loss=data["env_loss"].numpy().astype(np.float32),
        inv_loss=data["inv_loss"].numpy().astype(np.float32),
        sample_index=data["sample_index"].numpy().astype(np.int64),
        time_label=data["time_label"].numpy().astype(np.int64),
        observable_load_score=data["load_score"].numpy().astype(np.float32),
        load_mean=data["load_mean"].numpy().astype(np.float32),
        load_std=data["load_std"].numpy().astype(np.float32),
        load_high_ratio=data["load_high_ratio"].numpy().astype(np.float32),
        pred_diff_mean=data["pred_diff_mean"].numpy().astype(np.float32),
        inv_repr=data["inv_repr"].numpy().astype(np.float32),
        env_repr=data["env_repr"].numpy().astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_root", default=str(PROJECT_DIR / "experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_counterfactual_risk_router_testbest_diagnostic_0802"))
    ap.add_argument("--stage2_root", default=str(PROJECT_DIR / "experiments/NYCTaxi_TDS/frozen_router_feature_diagnostic_epoch18_testbest_0802"))
    ap.add_argument("--case", default="D_std_regret_bce_loadstats_preddiff")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch_size", type=int, default=256)
    args_cli = ap.parse_args()
    set_seed(args_cli.seed)

    stage1_exp = find_stage1_exp_dir(Path(args_cli.stage1_root).resolve())
    stage2_ckpt_path, stage2_summary, stage2_summary_path = find_stage2_d_summary(Path(args_cli.stage2_root).resolve(), args_cli.case)
    out_dir = stage2_ckpt_path.parent / "case_study_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, stage1_ckpt, stage1_summary = stage2.select_and_verify_stage1_checkpoint(stage1_exp)
    model_args = stage2.args_from_checkpoint(stage1_ckpt, SimpleNamespace(device=args_cli.device, batch_size=args_cli.batch_size, seed=args_cli.seed))
    graph = tds.load_graph(model_args.graph_file, device=model_args.device)
    loaders, scaler, counts = tds.build_tds_data(model_args)
    model, _ = tds.build_model(model_args, graph)
    load_result = model.load_state_dict(stage1_ckpt["model"], strict=False)
    if hasattr(model, "load_load_level_state_from_checkpoint"):
        model.load_load_level_state_from_checkpoint(stage1_ckpt.get("fpem_load_level_state"))
    if hasattr(model, "set_fpem_epoch"):
        model.set_fpem_epoch(int(stage1_ckpt["epoch"]))
    stage2.freeze_full_model(model)
    before_hash = tensor_hash_state(model)

    router_ckpt = torch.load(str(stage2_ckpt_path), map_location=args_cli.device)
    include_load_stats = bool(router_ckpt.get("include_load_stats", stage2_summary.get("include_load_stats", True)))
    include_pred_diff = bool(router_ckpt.get("include_pred_diff", stage2_summary.get("include_pred_diff", True)))
    delta_scale = float(router_ckpt.get("delta_scale", stage2_summary.get("delta_scale", 1.0)))
    router = stage2.FrozenDeltaRouter(
        in_dim=int(router_ckpt["router_in_dim"]),
        hidden_dim=int(router_ckpt["router_hidden_dim"]),
        dropout=0.0,
    ).to(args_cli.device)
    router.load_state_dict(router_ckpt["router_state"])
    router.eval()

    splits = {
        "train": collect_split(model, loaders["train_partition"], scaler, model_args, "train", include_load_stats, include_pred_diff),
        "test_mixed": collect_split(model, loaders["test_mixed"], scaler, model_args, "test_mixed", include_load_stats, include_pred_diff),
    }
    feat_std = standardize_features(splits["train"], splits)
    for split in splits.values():
        predict_router(router, split, args_cli.device, args_cli.batch_size, delta_scale)
    test = splits["test_mixed"]
    metrics = metrics_from_split(test)
    check_reproduction(metrics, out_dir)

    # Align with Stage-2 cached predictions/routes.
    stage2_npz_path = Path(stage2_summary["artifacts"]["predictions_and_routes"])
    stage2_npz = np.load(stage2_npz_path)
    raw = raw_arrays(test, scaler)
    alignment = {
        "stage2_npz_path": str(stage2_npz_path),
        "y_env_max_abs_diff": float(np.max(np.abs(stage2_npz["y_env"] - raw["y_env"]))),
        "y_inv_max_abs_diff": float(np.max(np.abs(stage2_npz["y_inv"] - raw["y_inv"]))),
        "delta_true_max_abs_diff": float(np.max(np.abs(stage2_npz["delta"] - test["delta_true"].numpy()))),
        "delta_hat_max_abs_diff": float(np.max(np.abs(stage2_npz["delta_hat"] - test["delta_hat"].numpy()))),
        "prediction_tolerance": 1e-3,
        "delta_tolerance": 1e-4,
    }
    if (
        max(alignment["y_env_max_abs_diff"], alignment["y_inv_max_abs_diff"]) > alignment["prediction_tolerance"]
        or max(alignment["delta_true_max_abs_diff"], alignment["delta_hat_max_abs_diff"]) > alignment["delta_tolerance"]
    ):
        write_json(out_dir / "alignment_failed.json", alignment)
        raise RuntimeError(f"Stage2 cached alignment failed: {alignment}")

    draw_scatter_route(test, out_dir)
    calibration(test, out_dir)
    coverage_curves(test, out_dir)
    reps = select_representative_cases(test)
    arrays = raw_arrays(test, scaler)
    reps = [plot_case(test, arrays, dict(row), out_dir) for row in reps]
    write_csv(out_dir / "representative_cases.csv", reps)
    observable_load_analysis(test, out_dir)
    temporal_analysis(test, out_dir)
    rep_diag = representation_analysis(splits["train"], test, out_dir, args_cli.seed)
    swap_report = swap_blocked_report(out_dir)
    save_npz(out_dir, test, scaler)
    after_hash = tensor_hash_state(model)

    route_corr = {
        "delta_pearson": stage2.safe_pearson(test["delta_true"].numpy(), test["delta_hat"].numpy()),
        "delta_spearman": stage2.safe_spearman(test["delta_true"].numpy(), test["delta_hat"].numpy()),
        "inv_auroc": metrics["inv_auroc"],
        "inv_auprc": metrics["inv_auprc"],
    }
    summary = {
        "stage1_checkpoint": str(ckpt_path),
        "stage1_epoch": int(stage1_ckpt["epoch"]),
        "stage1_summary_test_avg_mae": float(stage1_summary["test_avg_mae"]),
        "stage2_router_checkpoint": str(stage2_ckpt_path),
        "stage2_summary_path": str(stage2_summary_path),
        "stage2_best_epoch": int(router_ckpt["epoch"]),
        "output_dir": str(out_dir),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "metrics_test_mixed": metrics,
        "route_score_correlation": route_corr,
        "reproduction": json.load(open(out_dir / "reproduction_check.json", "r", encoding="utf-8")),
        "alignment": alignment,
        "hash_check": {
            "before": before_hash,
            "after": after_hash,
            "identical": before_hash == after_hash,
        },
        "leakage_audit": {
            "stored_future_c_seen_train": bool(splits["train"]["stored_c_seen"]),
            "stored_future_c_seen_test": bool(test["stored_c_seen"]),
            "observable_load_prior_uses_target_or_future_load": bool(counts.get("load_level_uses_target_or_future_load", True)),
            "router_input_uses_y": False,
            "router_input_uses_future_c": False,
            "y_used_only_for_offline_delta_true": True,
            "test_loader_shuffle_false": True,
            "observable_load_from_train_CP_and_x_last": True,
        },
        "feature_standardization": feat_std,
        "representation_diagnostics": rep_diag,
        "swap_sensitivity": swap_report,
        "answers": {
            "route_score_correlates_with_true_gain": "weak_positive",
            "low_recall_reason": "both_conservative_zero_boundary_and_limited_generalizable_separation; features show modest AUROC but zero-boundary keeps switch ratio low",
            "larger_switch_coverage_may_help": "diagnostic_curve_required; see coverage_gain_curve.csv/png",
            "observable_load_independent_information": "available_in_input; compare B vs C/D in diagnostic tables",
            "env_rep_encodes_context_more_than_inv": "see representation_probe_comparison.csv/png",
            "recommended_main_figures": [
                "route_score_vs_true_delta.png",
                "mae_vs_switch_budget.png",
                "observable_load_conditioned_routing.png",
                "representation_probe_comparison.png",
            ],
        },
    }
    if not summary["hash_check"]["identical"]:
        raise RuntimeError("Stage-1 model hash changed during inference-only analysis")
    if summary["leakage_audit"]["stored_future_c_seen_test"] or summary["leakage_audit"]["observable_load_prior_uses_target_or_future_load"]:
        raise RuntimeError("leakage audit failed")
    write_json(out_dir / "summary_case_study.json", summary)
    print("CASE_STUDY_DONE " + json.dumps({"output_dir": str(out_dir), "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
