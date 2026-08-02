#!/usr/bin/env python
"""Independent Stage-2 frozen prediction-branch router training.

This script intentionally does not continue the STEVE optimizer.  It loads a
completed Stage-1 checkpoint, freezes the full STEVE model, precomputes
test-observable router features and counterfactual masked-MAE targets, then
trains a standalone scalar Delta router:

    delta = L_inv - L_env
    delta_hat < 0 -> invariant
    delta_hat >= 0 -> environment

The Stage-1 prediction modules are never updated and the Stage-2 best checkpoint
contains only the new router plus audit metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_tds_nyctaxi as tds  # noqa: E402
from models.fpem.losses import head_prediction_losses  # noqa: E402


EXPECTED_STAGE1_EPOCH = 18
EXPECTED_STAGE1_TEST_AVG_MAE = 10.602606773376465


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode("utf-8"))
    h.update(str(t.dtype).encode("utf-8"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_full_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def enforce_stage2_modes(model: nn.Module, router: nn.Module) -> None:
    # Requirement: every Stage-2 epoch calls model.train(), then immediately
    # restores frozen modules to eval(); only the standalone router is train().
    model.train()
    model.eval()
    router.train()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_and_verify_stage1_checkpoint(base_exp_dir: Path) -> Tuple[Path, dict, dict]:
    summary_path = base_exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing Stage-1 summary.json: {summary_path}")
    summary = load_json(summary_path)

    candidates = [base_exp_dir / "best_test_avg_model.pth"]
    if not candidates[0].exists():
        candidates.append(base_exp_dir / "best_val_model.pth")

    last_model = base_exp_dir / "last_model.pth"
    for ckpt_path in candidates:
        if not ckpt_path.exists():
            continue
        if ckpt_path.name == last_model.name:
            raise RuntimeError("BUG: last_model.pth must never be selected")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        monitor = ckpt.get("monitor", {}) if isinstance(ckpt, dict) else {}
        ckpt_epoch = int(ckpt.get("epoch", -1))
        monitor_epoch = int(monitor.get("epoch", ckpt_epoch))
        monitor_mae = float(monitor.get("test_avg_mae", float("nan")))
        summary_epoch = int(summary.get("best_epoch", -1))
        summary_mae = float(summary.get("test_avg_mae", float("nan")))
        epoch_ok = (
            ckpt_epoch == EXPECTED_STAGE1_EPOCH
            and monitor_epoch == EXPECTED_STAGE1_EPOCH
            and summary_epoch == EXPECTED_STAGE1_EPOCH
        )
        metric_ok = (
            math.isfinite(monitor_mae)
            and abs(monitor_mae - EXPECTED_STAGE1_TEST_AVG_MAE) <= 1e-9
            and math.isfinite(summary_mae)
            and abs(summary_mae - EXPECTED_STAGE1_TEST_AVG_MAE) <= 1e-9
        )
        if not epoch_ok or not metric_ok:
            raise RuntimeError(
                "Stage-1 checkpoint metadata mismatch: "
                f"path={ckpt_path} ckpt_epoch={ckpt_epoch} "
                f"monitor_epoch={monitor_epoch} monitor_test_avg_mae={monitor_mae} "
                f"summary_epoch={summary_epoch} summary_test_avg_mae={summary_mae}; "
                f"expected epoch={EXPECTED_STAGE1_EPOCH}, "
                f"test_avg_mae={EXPECTED_STAGE1_TEST_AVG_MAE}"
            )
        return ckpt_path, ckpt, summary
    raise FileNotFoundError(
        f"neither best_test_avg_model.pth nor best_val_model.pth exists in {base_exp_dir}"
    )


def args_from_checkpoint(ckpt: dict, cli) -> SimpleNamespace:
    ckpt_args = dict(ckpt.get("args", {}))
    ckpt_args["device"] = cli.device
    ckpt_args["batch_size"] = int(cli.batch_size)
    ckpt_args["test_batch_size"] = int(cli.batch_size)
    ckpt_args["seed"] = int(cli.seed)
    ckpt_args["data_dir"] = str(Path(ckpt_args.get("data_dir", PROJECT_DIR / "data")).resolve())
    graph_file = ckpt_args.get("graph_file", PROJECT_DIR / "data/NYCTaxi_TDS/adj_mx.npz")
    ckpt_args["graph_file"] = str(Path(graph_file).resolve())
    ckpt_args["mode"] = "eval"
    ckpt_args["resume"] = False
    ckpt_args["debug"] = False
    # Keep legal observable c_obs behavior from Stage-1.
    ckpt_args["fpem_use_observable_load_prior"] = True
    ckpt_args["fpem_ignore_future_c"] = True
    return SimpleNamespace(**ckpt_args)


def samplewise_masked_mae(pred: torch.Tensor, target: torch.Tensor, scaler, yita: float) -> torch.Tensor:
    losses = head_prediction_losses(pred.unsqueeze(1), target, scaler, yita)
    return losses[:, 0].detach()


def pred_diff_features(y_env: torch.Tensor, y_inv: torch.Tensor) -> torch.Tensor:
    diff = y_env - y_inv
    flat_diff = diff.reshape(diff.shape[0], -1)
    flat_env = y_env.reshape(y_env.shape[0], -1)
    flat_inv = y_inv.reshape(y_inv.shape[0], -1)
    return torch.stack(
        [
            flat_diff.abs().mean(dim=1),
            flat_diff.abs().amax(dim=1),
            flat_diff.std(dim=1, unbiased=False),
            flat_env.mean(dim=1),
            flat_inv.mean(dim=1),
        ],
        dim=1,
    )


def observable_load_features(
    observable_load_profile: Optional[torch.Tensor],
    observable_load_score: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if observable_load_profile is None:
        return None
    profile = observable_load_profile.detach().float()
    # c_obs is cached in [0, 5]; divide by 5 for stable numeric scale before
    # train-set standardization.  Shape is [B, 1, N, C].
    profile = profile / 5.0
    bsz = profile.shape[0]
    flat = profile.reshape(bsz, -1)
    ch = profile.reshape(bsz, -1, profile.shape[-1])
    feats = [
        flat.mean(dim=1, keepdim=True),
        flat.std(dim=1, unbiased=False, keepdim=True),
        flat.amin(dim=1, keepdim=True),
        flat.amax(dim=1, keepdim=True),
        ch.mean(dim=1),
        ch.std(dim=1, unbiased=False),
        ch.amin(dim=1),
        ch.amax(dim=1),
    ]
    if observable_load_score is not None:
        feats.insert(0, observable_load_score.detach().float().view(bsz, 1))
    return torch.cat(feats, dim=1)


@dataclass
class SplitCache:
    split: str
    features: torch.Tensor
    delta: torch.Tensor
    env_loss: torch.Tensor
    inv_loss: torch.Tensor
    target_inv: torch.Tensor
    y_true: torch.Tensor
    y_env: torch.Tensor
    y_inv: torch.Tensor
    sample_index: torch.Tensor
    observable_load_score: torch.Tensor
    stored_c_seen: bool
    train_forward_default_env_exact: bool


def extract_split_cache(
    model,
    loader,
    scaler,
    args,
    split: str,
    include_pred_diff: bool,
    include_load_stats: bool,
) -> SplitCache:
    model.eval()
    features: List[torch.Tensor] = []
    deltas: List[torch.Tensor] = []
    env_losses: List[torch.Tensor] = []
    inv_losses: List[torch.Tensor] = []
    y_true: List[torch.Tensor] = []
    y_envs: List[torch.Tensor] = []
    y_invs: List[torch.Tensor] = []
    sample_indices: List[torch.Tensor] = []
    load_scores: List[torch.Tensor] = []
    stored_c_seen = False
    default_env_flags: List[bool] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = tds.to_device(raw_batch, args.device)
            unpacked = tds.unpack_tds_batch(batch)
            stored_c_seen = stored_c_seen or unpacked["stored_c"] is not None
            output = model.forward_output(
                unpacked["data"],
                exog=unpacked["stored_c"],
                time_label=unpacked["time_label"],
                # All splits use inference-mode features; the Stage-2 router
                # never receives train-only routing behavior as an input.
                training=False,
                sample_index=unpacked["sample_index"],
                observable_load_profile=unpacked["observable_load_profile"],
                observable_load_score=unpacked["observable_load_score"],
                cached_load_level=unpacked["cached_load_level"],
            )
            y_env = output["y_env_selected"].detach()
            y_inv = output["y_inv"].detach()
            target = unpacked["target"].detach()
            env_loss = samplewise_masked_mae(y_env, target, scaler, args.yita)
            inv_loss = samplewise_masked_mae(y_inv, target, scaler, args.yita)
            delta = (inv_loss - env_loss).detach()
            base_features = output["hard_router_features"].detach().float()
            if include_load_stats:
                load_feats = observable_load_features(
                    unpacked["observable_load_profile"],
                    unpacked["observable_load_score"],
                )
                if load_feats is None:
                    raise RuntimeError(
                        "include_load_stats=true but observable c_obs features are unavailable"
                    )
                base_features = torch.cat(
                    [base_features, load_feats.detach().float()],
                    dim=1,
                )
            if include_pred_diff:
                base_features = torch.cat(
                    [base_features, pred_diff_features(y_env, y_inv).detach().float()],
                    dim=1,
                )
            features.append(base_features.cpu())
            deltas.append(delta.cpu())
            env_losses.append(env_loss.cpu())
            inv_losses.append(inv_loss.cpu())
            y_true.append(target.cpu())
            y_envs.append(y_env.cpu())
            y_invs.append(y_inv.cpu())
            sample_indices.append(unpacked["sample_index"].detach().cpu())
            if unpacked["observable_load_score"] is not None:
                load_scores.append(unpacked["observable_load_score"].detach().cpu())
            if split == "train":
                default_env_flags.append(
                    bool(torch.allclose(output["prediction"], y_env, atol=0.0, rtol=0.0))
                )
    delta_all = torch.cat(deltas, dim=0)
    return SplitCache(
        split=split,
        features=torch.cat(features, dim=0),
        delta=delta_all,
        env_loss=torch.cat(env_losses, dim=0),
        inv_loss=torch.cat(inv_losses, dim=0),
        target_inv=(delta_all < 0.0).float(),
        y_true=torch.cat(y_true, dim=0),
        y_env=torch.cat(y_envs, dim=0),
        y_inv=torch.cat(y_invs, dim=0),
        sample_index=torch.cat(sample_indices, dim=0),
        observable_load_score=torch.cat(load_scores, dim=0) if load_scores else torch.empty(0),
        stored_c_seen=stored_c_seen,
        train_forward_default_env_exact=all(default_env_flags) if default_env_flags else True,
    )


class FrozenDeltaRouter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize_feature_caches(caches: Dict[str, SplitCache]) -> dict:
    train_features = caches["train"].features.float()
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    for cache in caches.values():
        cache.features = (cache.features.float() - mean) / std
    return {
        "feature_mean_shape": list(mean.shape),
        "feature_std_shape": list(std.shape),
        "feature_std_min": float(std.min().item()),
        "feature_std_max": float(std.max().item()),
        "feature_train_standardized_mean_abs_max": float(caches["train"].features.mean(dim=0).abs().max().item()),
        "feature_train_standardized_std_min": float(caches["train"].features.std(dim=0, unbiased=False).min().item()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _rankdata(scores.astype(np.float64))
    pos_rank_sum = float(ranks[labels].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return float(auc)


def binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores.astype(np.float64), kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Average precision / step-wise PR area; avoids depending on sklearn or
    # deprecated/removed numpy.trapz variants.
    return float((precision * sorted_labels).sum() / max(float(pos), 1.0))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return safe_pearson(_rankdata(x), _rankdata(y))


def route_metrics(delta_hat_scaled: torch.Tensor, cache: SplitCache, delta_scale: float = 1.0) -> dict:
    delta_hat = delta_hat_scaled.float() * float(delta_scale)
    delta = cache.delta.float()
    env_loss = cache.env_loss.float()
    inv_loss = cache.inv_loss.float()
    target_inv = delta < 0.0
    use_inv = delta_hat.float() < 0.0
    learned_loss = torch.where(use_inv, inv_loss, env_loss)
    oracle_loss = torch.minimum(inv_loss, env_loss)
    all_env = env_loss.mean()
    all_inv = inv_loss.mean()
    learned = learned_loss.mean()
    oracle = oracle_loss.mean()
    env_target = ~target_inv
    correct = use_inv == target_inv
    recalls = []
    if bool(target_inv.any()):
        recalls.append(use_inv[target_inv].float().mean())
    if bool(env_target.any()):
        recalls.append((~use_inv[env_target]).float().mean())
    balanced = torch.stack(recalls).mean() if recalls else torch.tensor(0.0)
    correct_inv = use_inv & target_inv
    harmful_inv = use_inv & env_target
    inv_switches = use_inv.sum().float()
    target_inv_count = target_inv.sum().float()
    saved_loss = ((-delta).clamp_min(0.0) * correct_inv.float()).sum() / max(delta.numel(), 1)
    added_loss = (delta.clamp_min(0.0) * harmful_inv.float()).sum() / max(delta.numel(), 1)
    net_gain = all_env - learned
    regret = learned - oracle
    oracle_gap = (all_env - oracle).clamp_min(1e-8)
    target_np = target_inv.detach().cpu().numpy().astype(np.int64)
    inv_score_np = (-delta_hat).detach().cpu().numpy()
    delta_np = delta.detach().cpu().numpy()
    delta_hat_np = delta_hat.detach().cpu().numpy()
    out = {
        "all_environment_mae": float(all_env.item()),
        "all_invariant_mae": float(all_inv.item()),
        "learned_routing_mae": float(learned.item()),
        "oracle_mae": float(oracle.item()),
        "env_route_ratio": float((~use_inv).float().mean().item()),
        "inv_route_ratio": float(use_inv.float().mean().item()),
        "target_inv_ratio": float(target_inv.float().mean().item()),
        "router_accuracy": float(correct.float().mean().item()),
        "balanced_accuracy": float(balanced.item()),
        "inv_switch_precision": float((correct_inv.sum().float() / inv_switches.clamp_min(1.0)).item()),
        "inv_switch_recall": float((correct_inv.sum().float() / target_inv_count.clamp_min(1.0)).item()),
        "correct_beneficial_inv_switches": int(correct_inv.sum().item()),
        "harmful_inv_switches": int(harmful_inv.sum().item()),
        "saved_loss": float(saved_loss.item()),
        "added_loss": float(added_loss.item()),
        "net_gain": float(net_gain.item()),
        "regret": float(regret.item()),
        "oracle_gap_closed": float((net_gain / oracle_gap).item()),
        "num_samples": int(delta.numel()),
        "over_selects_invariant": bool(use_inv.float().mean().item() > target_inv.float().mean().item() + 1e-9),
        "delta_hat_min": float(delta_hat.min().item()),
        "delta_hat_mean": float(delta_hat.mean().item()),
        "delta_hat_max": float(delta_hat.max().item()),
        "delta_hat_std": float(delta_hat.std(unbiased=False).item()),
        "inv_auroc": binary_auroc(target_np, inv_score_np),
        "inv_auprc": binary_auprc(target_np, inv_score_np),
        "delta_pearson": safe_pearson(delta_np, delta_hat_np),
        "delta_spearman": safe_spearman(delta_np, delta_hat_np),
    }
    return out


def predict_delta(router: nn.Module, cache: SplitCache, device: str, batch_size: int) -> torch.Tensor:
    router.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, cache.features.shape[0], batch_size):
            x = cache.features[start : start + batch_size].to(device)
            preds.append(router(x).detach().cpu())
    return torch.cat(preds, dim=0)


def evaluate_router(
    router: nn.Module,
    caches: Dict[str, SplitCache],
    device: str,
    batch_size: int,
    delta_scale: float = 1.0,
) -> dict:
    metrics = {}
    for split, cache in caches.items():
        delta_hat = predict_delta(router, cache, device, batch_size)
        metrics[split] = route_metrics(delta_hat, cache, delta_scale=delta_scale)
    if "test_workday" in metrics and "test_holiday" in metrics:
        metrics["test_avg_mae"] = float(
            0.5
            * (
                metrics["test_workday"]["learned_routing_mae"]
                + metrics["test_holiday"]["learned_routing_mae"]
            )
        )
    elif "test_mixed" in metrics:
        metrics["test_avg_mae"] = metrics["test_mixed"]["learned_routing_mae"]
    return metrics


def train_router(cli, router: nn.Module, caches: Dict[str, SplitCache], out_dir: Path) -> Tuple[dict, List[dict]]:
    train_cache = caches["train"]
    train_mean_abs_delta = float(train_cache.delta.abs().mean().clamp_min(1e-8).item())
    train_delta_std = float(train_cache.delta.std(unbiased=False).clamp_min(1e-8).item())
    delta_scale = train_delta_std if bool(cli.scale_delta_by_train_std) else 1.0
    opt = torch.optim.Adam(router.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    optimizer_param_count = sum(p.numel() for group in opt.param_groups for p in group["params"])
    router_param_count = sum(p.numel() for p in router.parameters())
    if optimizer_param_count != router_param_count:
        raise RuntimeError("optimizer contains parameters outside the standalone router")

    best_metric = float("inf")
    best_epoch = 0
    best_summary = {}
    not_improved = 0
    curve: List[dict] = []
    x_all = train_cache.features.float()
    y_all = train_cache.delta.float()
    y_scaled_all = y_all / float(delta_scale)
    target_inv_all = (y_all < 0.0).float()
    n = x_all.shape[0]
    rng = torch.Generator()
    rng.manual_seed(int(cli.seed))
    for epoch in range(1, cli.epochs + 1):
        enforce_stage2_modes(cli.frozen_model, router)
        order = torch.randperm(n, generator=rng)
        total_loss = 0.0
        total_mse = 0.0
        total_bce = 0.0
        total_seen = 0
        for start in range(0, n, cli.batch_size):
            idx = order[start : start + cli.batch_size]
            x = x_all[idx].to(cli.device)
            delta = y_all[idx].to(cli.device)
            delta_scaled = y_scaled_all[idx].to(cli.device)
            target_inv = target_inv_all[idx].to(cli.device)
            delta_hat_scaled = router(x)
            mse = F.mse_loss(delta_hat_scaled, delta_scaled)
            if cli.bce_weight > 0.0:
                inv_logit = -delta_hat_scaled / max(float(cli.temperature), 1e-8)
                regret_weight = (delta.abs() / train_mean_abs_delta).clamp(
                    min=0.0, max=float(cli.weight_max)
                )
                bce_each = F.binary_cross_entropy_with_logits(
                    inv_logit, target_inv, reduction="none"
                )
                bce = (bce_each * regret_weight).sum() / regret_weight.sum().clamp_min(1e-8)
            else:
                bce = delta_hat_scaled.new_zeros(())
            loss = float(cli.mse_weight) * mse + float(cli.bce_weight) * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cli.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), cli.grad_clip)
            opt.step()
            bs = int(idx.numel())
            total_seen += bs
            total_loss += float(loss.item()) * bs
            total_mse += float(mse.item()) * bs
            total_bce += float(bce.item()) * bs

        metrics = evaluate_router(
            router,
            caches,
            cli.device,
            cli.eval_batch_size,
            delta_scale=delta_scale,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "train_mse": total_mse / max(total_seen, 1),
            "train_regret_bce": total_bce / max(total_seen, 1),
            "test_avg_mae": metrics["test_avg_mae"],
        }
        for split_name in ["train", "val", "test_mixed"]:
            split_metrics = metrics[split_name]
            prefix = "test" if split_name == "test_mixed" else split_name
            for key in [
                "learned_routing_mae",
                "all_environment_mae",
                "net_gain",
                "inv_route_ratio",
                "inv_switch_precision",
                "delta_hat_min",
                "delta_hat_mean",
                "delta_hat_max",
                "delta_hat_std",
                "inv_auroc",
                "inv_auprc",
                "delta_pearson",
                "delta_spearman",
            ]:
                row[f"{prefix}_{key}"] = split_metrics.get(key)
        curve.append(row)
        improved = metrics["test_avg_mae"] < best_metric - 1e-12
        if improved:
            best_metric = metrics["test_avg_mae"]
            best_epoch = epoch
            best_summary = metrics
            not_improved = 0
            torch.save(
                {
                    "stage": "frozen_delta_router_stage2",
                    "case": cli.case,
                    "epoch": epoch,
                    "selection_metric": "test_avg_mae_DIAGNOSTIC",
                    "test_avg_mae": best_metric,
                    "router_state": router.state_dict(),
                    "router_in_dim": int(x_all.shape[1]),
                    "router_hidden_dim": int(cli.hidden_dim),
                    "include_pred_diff": bool(cli.include_pred_diff),
                    "include_load_stats": bool(cli.include_load_stats),
                    "standardize_features": bool(cli.standardize_features),
                    "scale_delta_by_train_std": bool(cli.scale_delta_by_train_std),
                    "mse_weight": float(cli.mse_weight),
                    "bce_weight": float(cli.bce_weight),
                    "train_mean_abs_delta": train_mean_abs_delta,
                    "train_delta_std": train_delta_std,
                    "delta_scale": delta_scale,
                    "optimizer": opt.state_dict(),
                },
                str(out_dir / "best_stage2_router_test_selected.pth"),
            )
        else:
            not_improved += 1
        print(
            "STAGE2_EPOCH "
            + json.dumps(
                {
                    "case": cli.case,
                    **row,
                    "best_epoch": best_epoch,
                    "best_test_avg_mae": best_metric,
                    "not_improved": not_improved,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not_improved >= int(cli.patience):
            break
    return best_summary, curve


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions_npz(
    router: nn.Module,
    cache: SplitCache,
    scaler,
    path: Path,
    device: str,
    batch_size: int,
    delta_scale: float = 1.0,
) -> None:
    delta_hat_scaled = predict_delta(router, cache, device, batch_size)
    delta_hat = delta_hat_scaled * float(delta_scale)
    use_inv = delta_hat < 0.0
    y_env_raw = scaler.inverse_transform(cache.y_env).numpy()
    y_inv_raw = scaler.inverse_transform(cache.y_inv).numpy()
    y_true_raw = scaler.inverse_transform(cache.y_true).numpy()
    y_learned = np.where(
        use_inv.numpy().reshape(-1, 1, 1, 1),
        y_inv_raw,
        y_env_raw,
    )
    np.savez_compressed(
        path,
        sample_index=cache.sample_index.numpy(),
        y_true=y_true_raw.astype(np.float32),
        y_env=y_env_raw.astype(np.float32),
        y_inv=y_inv_raw.astype(np.float32),
        y_learned=y_learned.astype(np.float32),
        delta=cache.delta.numpy().astype(np.float32),
        delta_hat=delta_hat.numpy().astype(np.float32),
        delta_hat_scaled=delta_hat_scaled.numpy().astype(np.float32),
        target_inv=cache.target_inv.numpy().astype(np.float32),
        use_inv=use_inv.numpy().astype(np.int64),
        env_loss=cache.env_loss.numpy().astype(np.float32),
        inv_loss=cache.inv_loss.numpy().astype(np.float32),
        observable_load_score=cache.observable_load_score.numpy().astype(np.float32),
    )


def verify_prediction_consistency(
    model,
    loader,
    scaler,
    args,
    reference: SplitCache,
    include_pred_diff: bool,
    include_load_stats: bool,
) -> dict:
    after = extract_split_cache(
        model,
        loader,
        scaler,
        args,
        reference.split,
        include_pred_diff,
        include_load_stats,
    )
    return {
        f"{reference.split}_y_env_max_abs_diff": float((after.y_env - reference.y_env).abs().max().item()),
        f"{reference.split}_y_inv_max_abs_diff": float((after.y_inv - reference.y_inv).abs().max().item()),
        f"{reference.split}_delta_max_abs_diff": float((after.delta - reference.delta).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_exp_dir", default=str(PROJECT_DIR / "experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_counterfactual_risk_router_testbest_diagnostic_0802_obs_k1_counterfactual_risk_router_seed2024"))
    parser.add_argument("--output_root", default=str(PROJECT_DIR / "experiments/NYCTaxi_TDS/frozen_router_from_counterfactual_risk_stage1_epoch18_testbest_0802"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--weight_max", type=float, default=20.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=0.0)
    parser.add_argument("--include_pred_diff", type=str2bool, default=False)
    parser.add_argument("--include_load_stats", type=str2bool, default=False)
    parser.add_argument("--standardize_features", type=str2bool, default=True)
    parser.add_argument("--scale_delta_by_train_std", type=str2bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    cli = parser.parse_args()

    set_seed(cli.seed)
    base_exp_dir = Path(cli.base_exp_dir).resolve()
    output_root = Path(cli.output_root).resolve()
    out_dir = output_root / f"{cli.case}_seed{cli.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, ckpt, stage1_summary = select_and_verify_stage1_checkpoint(base_exp_dir)
    model_args = args_from_checkpoint(ckpt, cli)
    graph = tds.load_graph(model_args.graph_file, device=model_args.device)
    loaders, scaler, counts = tds.build_tds_data(model_args)
    model, _lr = tds.build_model(model_args, graph)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    if hasattr(model, "load_load_level_state_from_checkpoint"):
        model.load_load_level_state_from_checkpoint(ckpt.get("fpem_load_level_state"))
    if hasattr(model, "set_fpem_epoch"):
        model.set_fpem_epoch(int(ckpt.get("epoch", EXPECTED_STAGE1_EPOCH)))
    freeze_full_model(model)
    before_hash = model_state_hash(model)

    print(
        "CHECKPOINT_VERIFIED "
        + json.dumps(
            {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_epoch": int(ckpt["epoch"]),
                "monitor_test_avg_mae": float(ckpt["monitor"]["test_avg_mae"]),
                "summary_best_epoch": int(stage1_summary["best_epoch"]),
                "summary_test_avg_mae": float(stage1_summary["test_avg_mae"]),
                "load_missing_keys": list(load_result.missing_keys),
                "load_unexpected_keys": list(load_result.unexpected_keys),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    caches = {
        "train": extract_split_cache(model, loaders["train_partition"], scaler, model_args, "train", cli.include_pred_diff, cli.include_load_stats),
        "val": extract_split_cache(model, loaders["val"], scaler, model_args, "val", cli.include_pred_diff, cli.include_load_stats),
        "test_mixed": extract_split_cache(model, loaders["test_mixed"], scaler, model_args, "test_mixed", cli.include_pred_diff, cli.include_load_stats),
        "test_workday": extract_split_cache(model, loaders["test_workday"], scaler, model_args, "test_workday", cli.include_pred_diff, cli.include_load_stats),
        "test_holiday": extract_split_cache(model, loaders["test_holiday"], scaler, model_args, "test_holiday", cli.include_pred_diff, cli.include_load_stats),
    }
    feature_standardization = (
        standardize_feature_caches(caches)
        if bool(cli.standardize_features)
        else {"standardize_features": False}
    )
    leakage_audit = {
        "stored_future_c_seen_in_any_split": bool(any(c.stored_c_seen for c in caches.values())),
        "observable_load_prior_uses_target_or_future_load": bool(counts.get("load_level_uses_target_or_future_load", True)),
        "router_features_use_target": False,
        "pred_diff_features_use_target": False,
        "observable_load_score_in_router_input": bool(cli.include_load_stats),
        "c_obs_channel_stats_in_router_input": bool(cli.include_load_stats),
        "inference_route_uses_target": False,
        "fixed_zero_boundary_no_threshold_search": True,
        "train_forward_prediction_default_environment": bool(caches["train"].train_forward_default_env_exact),
    }
    if leakage_audit["stored_future_c_seen_in_any_split"] or leakage_audit["observable_load_prior_uses_target_or_future_load"]:
        raise RuntimeError(f"future-c/target leakage audit failed: {leakage_audit}")

    router = FrozenDeltaRouter(
        in_dim=int(caches["train"].features.shape[1]),
        hidden_dim=int(cli.hidden_dim),
        dropout=float(cli.dropout),
    ).to(cli.device)
    cli.frozen_model = model
    best_metrics, curve = train_router(cli, router, caches, out_dir)

    best_ckpt = torch.load(str(out_dir / "best_stage2_router_test_selected.pth"), map_location=cli.device)
    router.load_state_dict(best_ckpt["router_state"])
    delta_scale = float(best_ckpt.get("delta_scale", 1.0))
    final_metrics = evaluate_router(
        router,
        caches,
        cli.device,
        cli.eval_batch_size,
        delta_scale=delta_scale,
    )
    after_hash = model_state_hash(model)
    consistency = verify_prediction_consistency(
        model,
        loaders["test_mixed"],
        scaler,
        model_args,
        caches["test_mixed"],
        cli.include_pred_diff,
        cli.include_load_stats,
    )
    hash_check = {
        "model_state_hash_before": before_hash,
        "model_state_hash_after": after_hash,
        "non_router_params_and_buffers_hash_identical": before_hash == after_hash,
    }
    if not hash_check["non_router_params_and_buffers_hash_identical"]:
        raise RuntimeError("frozen STEVE model parameter/buffer hash changed during Stage2")
    if max(consistency.values()) != 0.0:
        raise RuntimeError(f"y_env/y_inv consistency check failed: {consistency}")

    save_predictions_npz(
        router,
        caches["test_mixed"],
        scaler,
        out_dir / "predictions_and_routes.npz",
        cli.device,
        cli.eval_batch_size,
        delta_scale=delta_scale,
    )
    write_csv(out_dir / "router_training_curve.csv", curve)

    summary = {
        "case": cli.case,
        "seed": cli.seed,
        "base_exp_dir": str(base_exp_dir),
        "stage1_checkpoint": str(ckpt_path),
        "stage1_checkpoint_epoch": int(ckpt["epoch"]),
        "stage1_monitor_test_avg_mae": float(ckpt["monitor"]["test_avg_mae"]),
        "stage1_summary_test_avg_mae": float(stage1_summary["test_avg_mae"]),
        "best_stage2_epoch": int(best_ckpt["epoch"]),
        "best_stage2_selection_metric": "test_avg_mae_DIAGNOSTIC",
        "best_stage2_test_avg_mae": float(best_ckpt["test_avg_mae"]),
        "router_input_dim": int(caches["train"].features.shape[1]),
        "router_param_count": int(sum(p.numel() for p in router.parameters())),
        "optimizer_param_count": int(sum(p.numel() for p in router.parameters())),
        "include_pred_diff": bool(cli.include_pred_diff),
        "include_load_stats": bool(cli.include_load_stats),
        "standardize_features": bool(cli.standardize_features),
        "scale_delta_by_train_std": bool(cli.scale_delta_by_train_std),
        "mse_weight": float(cli.mse_weight),
        "bce_weight": float(cli.bce_weight),
        "temperature": float(cli.temperature),
        "weight_max": float(cli.weight_max),
        "train_mean_abs_delta": float(caches["train"].delta.abs().mean().item()),
        "train_delta_std": float(caches["train"].delta.std(unbiased=False).item()),
        "delta_scale": float(delta_scale),
        "train_num_samples": int(caches["train"].delta.numel()),
        "train": final_metrics["train"],
        "val": final_metrics["val"],
        "test_mixed": final_metrics["test_mixed"],
        "test_workday": final_metrics["test_workday"],
        "test_holiday": final_metrics["test_holiday"],
        "test_avg_mae": final_metrics["test_avg_mae"],
        "hash_check": hash_check,
        "prediction_consistency": consistency,
        "leakage_audit": leakage_audit,
        "feature_standardization": feature_standardization,
        "data_counts": counts,
        "artifacts": {
            "best_stage2_router_test_selected": str(out_dir / "best_stage2_router_test_selected.pth"),
            "predictions_and_routes": str(out_dir / "predictions_and_routes.npz"),
            "router_training_curve": str(out_dir / "router_training_curve.csv"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=tds.scalar_json)
    print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False, default=tds.scalar_json), flush=True)


if __name__ == "__main__":
    main()
