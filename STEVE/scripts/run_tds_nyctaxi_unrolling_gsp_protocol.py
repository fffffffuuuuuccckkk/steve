#!/usr/bin/env python
"""Run Unrolling-GSP-STForecast on the STEVE NYCTaxi_TDS protocol.

This is intentionally decoupled from FPEM/STEVE model files.  It imports the
paper implementation from ``baselines/unrolling_gsp_stforecast`` and only
shares the dataset, metric convention, and experiment/checkpoint layout used by
the surrounding STEVE repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_DIR / "baselines" / "unrolling_gsp_stforecast"
sys.path.insert(0, str(PAPER_DIR))

try:
    import yaml
except Exception:  # pragma: no cover - fallback for minimal envs
    yaml = None

from lib.unrolling_model import UnrollingModel  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def load_yaml_config(path: Path) -> Dict:
    if yaml is None or not path.exists():
        return {
            "model": {
                "kNN": 6,
                "interval": 6,
                "num_blocks": 5,
                "num_layers": 25,
                "CG_iters": 3,
                "PGD_iters": 3,
                "num_heads": 4,
                "feature_channels": 6,
                "use_extrapolation": True,
                "use_one_channel": True,
                "sharedM": False,
                "sharedQ": True,
                "diff_interval": True,
            },
            "ADMM_params": {"mu_u": 3, "mu_d1": 3, "mu_d2": 3},
            "st_emb_info": {"spatial_dim": 5, "t_dim": 10, "tid_dim": 6, "diw_dim": 4},
        }
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TDSWindowDataset(Dataset):
    """Dataset adapter for STEVE's pre-windowed train/val/test.npz files."""

    def __init__(
        self,
        npz_path: Path,
        use_one_channel: bool = False,
        time_step_per_hour: int = 12,
    ) -> None:
        z = np.load(npz_path)
        self.x = z["x"].astype(np.float32)
        self.y = z["y"].astype(np.float32)
        self.time_label = z["time_label"].astype(np.int64) if "time_label" in z.files else None
        self.c = z["c"].astype(np.float32) if "c" in z.files else None
        self.use_one_channel = bool(use_one_channel)
        if self.use_one_channel:
            self.x = self.x[..., :1]
            self.y = self.y[..., :1]
        self.t_in = int(self.x.shape[1])
        self.t_out = int(self.y.shape[1])
        self.T = self.t_in + self.t_out
        self.n_nodes = int(self.x.shape[2])
        self.signal_channel = int(self.x.shape[-1])
        self.time_step_per_hour = int(time_step_per_hour)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def _time_index(self, idx: int) -> np.ndarray:
        if self.time_label is None:
            base = idx * self.time_step_per_hour
        else:
            label = int(self.time_label[idx])
            # Same semantics as STEVE/FPEM: label<24 is one day type, label>=24
            # is the other, and label % 24 is hour-of-day.
            day_type = 1 if label < 24 else 6
            hour = label % 24
            base = (14 + day_type) * 288 + hour * self.time_step_per_hour
        offsets = np.arange(-(self.t_in - 1), self.t_out + 1, dtype=np.int64)
        if offsets.shape[0] != self.T:
            offsets = np.arange(self.T, dtype=np.int64) - (self.t_in - 1)
        return base + offsets * self.time_step_per_hour

    def __getitem__(self, idx: int):
        hist = torch.from_numpy(self.x[idx])
        future = torch.from_numpy(self.y[idx])
        full = torch.cat([hist, future], dim=0)
        time_idx = torch.from_numpy(self._time_index(idx)).long()
        return hist, full, future, time_idx, torch.tensor(idx, dtype=torch.long)


class WindowStandardScaler:
    def __init__(self, train_set: TDSWindowDataset, device: torch.device) -> None:
        full = np.concatenate([train_set.x, train_set.y], axis=1)
        mean = full.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = full.std(axis=(0, 1), keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def load_graph_info(graph_file: Path, n_nodes: int, device: torch.device) -> Dict[str, torch.Tensor]:
    z = np.load(graph_file, allow_pickle=True)
    if "adj_mx" in z.files:
        adj = np.asarray(z["adj_mx"], dtype=np.float32)
    else:
        adj = np.asarray(z[z.files[0]], dtype=np.float32)
    if adj.shape[0] != n_nodes:
        raise ValueError(f"adj nodes {adj.shape[0]} != data nodes {n_nodes}")
    adj = np.maximum(adj, adj.T)
    edges = set()
    dists = {}
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        w = float(adj[i, j])
        dist = 1.0 if w >= 0.999 else float(np.sqrt(max(-np.log(max(w, 1e-8)), 1e-6)))
        edges.add((i, j))
        dists[(i, j)] = dist
    # Add a light bidirectional ring so every node is reachable even if the
    # provided graph has disconnected components.
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        for a, b in ((i, j), (j, i)):
            if (a, b) not in edges:
                edges.add((a, b))
                dists[(a, b)] = 1.0
    edge_arr = np.asarray(sorted(edges), dtype=np.int64)
    dist_arr = np.asarray([dists[tuple(e)] for e in edge_arr], dtype=np.float32)
    return {
        "n_nodes": int(n_nodes),
        "u_edges": torch.from_numpy(edge_arr).long().to(device),
        "u_dist": torch.from_numpy(dist_arr).float().to(device),
    }


def masked_channel_mae(pred: torch.Tensor, target: torch.Tensor, channel: int, threshold: float) -> torch.Tensor:
    pred_c = pred[..., channel]
    target_c = target[..., channel]
    mask = target_c > float(threshold)
    if bool(mask.any()):
        return (pred_c - target_c).abs().masked_select(mask).mean()
    return (pred_c - target_c).abs().mean()


def weighted_masked_mae(pred: torch.Tensor, target: torch.Tensor, yita: float, threshold: float) -> torch.Tensor:
    if pred.shape[-1] == 1:
        return masked_channel_mae(pred, target, 0, threshold)
    return (
        float(yita) * masked_channel_mae(pred, target, 0, threshold)
        + (1.0 - float(yita)) * masked_channel_mae(pred, target, 1, threshold)
    )


def weighted_rmse(pred: torch.Tensor, target: torch.Tensor, yita: float, threshold: float) -> torch.Tensor:
    if pred.shape[-1] == 1:
        mask = target[..., 0] > threshold
        err = (pred[..., 0] - target[..., 0]).pow(2)
        return torch.sqrt(err.masked_select(mask).mean() if bool(mask.any()) else err.mean())
    values = []
    weights = [float(yita), 1.0 - float(yita)]
    for c, w in enumerate(weights):
        mask = target[..., c] > threshold
        err = (pred[..., c] - target[..., c]).pow(2)
        val = torch.sqrt(err.masked_select(mask).mean() if bool(mask.any()) else err.mean())
        values.append(w * val)
    return values[0] + values[1]


def weighted_mape(pred: torch.Tensor, target: torch.Tensor, yita: float, threshold: float) -> torch.Tensor:
    if pred.shape[-1] == 1:
        mask = target[..., 0] > threshold
        denom = target[..., 0].clamp_min(1e-6)
        err = ((pred[..., 0] - target[..., 0]).abs() / denom)
        return (err.masked_select(mask).mean() if bool(mask.any()) else err.mean()) * 100.0
    values = []
    weights = [float(yita), 1.0 - float(yita)]
    for c, w in enumerate(weights):
        mask = target[..., c] > threshold
        denom = target[..., c].clamp_min(1e-6)
        err = ((pred[..., c] - target[..., c]).abs() / denom)
        values.append(w * (err.masked_select(mask).mean() if bool(mask.any()) else err.mean()) * 100.0)
    return values[0] + values[1]


@dataclass
class EpochMetrics:
    mae: float
    rmse: float
    mape: float
    loss: float


def evaluate(
    model: UnrollingModel,
    loader: DataLoader,
    scaler: WindowStandardScaler,
    device: torch.device,
    args,
    max_batches: int = -1,
) -> EpochMetrics:
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.HuberLoss(delta=float(args.huber_delta))
    with torch.no_grad():
        for batch_idx, (hist, full, future, time_idx, _sample_idx) in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
                break
            hist = hist.to(device)
            full = full.to(device)
            future = future.to(device)
            time_idx = time_idx.to(device)
            norm_hist = scaler.normalize(hist)
            pred_full = scaler.recover(model(norm_hist, time_idx))
            pred_future = pred_full[:, hist.shape[1]: hist.shape[1] + future.shape[1]]
            if args.loss_scope == "full":
                loss = loss_fn(pred_full, full)
            else:
                loss = loss_fn(pred_future, future)
            bsz = int(hist.shape[0])
            total_loss += float(loss.detach().cpu()) * bsz
            total_mae += float(weighted_masked_mae(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
            total_rmse += float(weighted_rmse(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
            total_mape += float(weighted_mape(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
            total_n += bsz
    total_n = max(total_n, 1)
    return EpochMetrics(
        mae=total_mae / total_n,
        rmse=total_rmse / total_n,
        mape=total_mape / total_n,
        loss=total_loss / total_n,
    )


def train_one_epoch(
    model: UnrollingModel,
    loader: DataLoader,
    scaler: WindowStandardScaler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    max_batches: int = -1,
) -> EpochMetrics:
    model.train()
    loss_fn = nn.HuberLoss(delta=float(args.huber_delta))
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    total_n = 0
    for batch_idx, (hist, full, future, time_idx, _sample_idx) in enumerate(loader):
        if max_batches >= 0 and batch_idx >= max_batches:
            break
        hist = hist.to(device)
        full = full.to(device)
        future = future.to(device)
        time_idx = time_idx.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_full = scaler.recover(model(scaler.normalize(hist), time_idx))
        pred_future = pred_full[:, hist.shape[1]: hist.shape[1] + future.shape[1]]
        if args.loss_scope == "full":
            loss = loss_fn(pred_full, full)
        else:
            loss = loss_fn(pred_future, future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        if float(args.clamp) > 0 and hasattr(model, "clamp_param"):
            model.clamp_param(float(args.clamp))
        bsz = int(hist.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        with torch.no_grad():
            total_mae += float(weighted_masked_mae(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
            total_rmse += float(weighted_rmse(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
            total_mape += float(weighted_mape(pred_future, future, args.yita, args.mask_threshold).detach().cpu()) * bsz
        total_n += bsz
    total_n = max(total_n, 1)
    return EpochMetrics(
        mae=total_mae / total_n,
        rmse=total_rmse / total_n,
        mape=total_mape / total_n,
        loss=total_loss / total_n,
    )


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, args, extra: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val),
            "args": vars(args),
            "extra": extra,
        },
        str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NYCTaxi_TDS")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--graph_file", default="data/NYCTaxi_TDS/adj_mx.npz")
    parser.add_argument("--result_root", default="experiments/NYCTaxi_TDS")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optim", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--loss_scope", choices=["full", "pred"], default="full")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=5.0)
    parser.add_argument("--yita", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--clamp", type=float, default=1.2)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--time_step_per_hour", type=int, default=12)

    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--cg_iters", type=int, default=None)
    parser.add_argument("--pgd_iters", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--feature_channels", type=int, default=None)
    parser.add_argument("--extrapolation_layers", type=int, default=1)
    parser.add_argument("--use_one_channel", type=str2bool, default=False)
    parser.add_argument("--le_emb", type=str2bool, default=False)
    parser.add_argument("--sharedM", type=str2bool, default=None)
    parser.add_argument("--sharedQ", type=str2bool, default=None)
    parser.add_argument("--diff_interval", type=str2bool, default=None)
    parser.add_argument("--ablation", default="None")
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    data_root = PROJECT_DIR / args.data_dir / args.dataset
    train_set = TDSWindowDataset(data_root / "train.npz", args.use_one_channel, args.time_step_per_hour)
    val_set = TDSWindowDataset(data_root / "val.npz", args.use_one_channel, args.time_step_per_hour)
    test_set = TDSWindowDataset(data_root / "test.npz", args.use_one_channel, args.time_step_per_hour)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    scaler = WindowStandardScaler(train_set, device)

    cfg = load_yaml_config(PAPER_DIR / "config.yaml")
    model_cfg = cfg["model"]
    t_in, t_out, T = train_set.t_in, train_set.t_out, train_set.T
    graph_info = load_graph_info(PROJECT_DIR / args.graph_file, train_set.n_nodes, device)
    blocks = int(args.blocks if args.blocks is not None else model_cfg.get("num_blocks", 5))
    layers = int(args.layers if args.layers is not None else model_cfg.get("num_layers", 25))
    cg_iters = int(args.cg_iters if args.cg_iters is not None else model_cfg.get("CG_iters", 3))
    pgd_iters = int(args.pgd_iters if args.pgd_iters is not None else model_cfg.get("PGD_iters", 3))
    heads = int(args.heads if args.heads is not None else model_cfg.get("num_heads", 4))
    feature_channels = int(args.feature_channels if args.feature_channels is not None else model_cfg.get("feature_channels", 6))
    neighbors = int(args.neighbors if args.neighbors is not None else model_cfg.get("kNN", 6))
    interval = int(args.interval if args.interval is not None else model_cfg.get("interval", 6))
    sharedM = bool(model_cfg.get("sharedM", False) if args.sharedM is None else args.sharedM)
    sharedQ = bool(model_cfg.get("sharedQ", True) if args.sharedQ is None else args.sharedQ)
    diff_interval = bool(model_cfg.get("diff_interval", True) if args.diff_interval is None else args.diff_interval)
    admm_info = {
        "ADMM_iters": layers,
        "CG_iters": cg_iters,
        "PGD_iters": pgd_iters,
        "mu_u_init": cfg.get("ADMM_params", {}).get("mu_u", 3),
        "mu_d1_init": cfg.get("ADMM_params", {}).get("mu_d1", 3),
        "mu_d2_init": cfg.get("ADMM_params", {}).get("mu_d2", 3),
    }

    model = UnrollingModel(
        blocks,
        device,
        T,
        t_in,
        heads,
        interval,
        train_set.signal_channel,
        feature_channels,
        GNN_layers=2,
        graph_info=graph_info,
        ADMM_info=admm_info,
        k_hop=neighbors,
        ablation=args.ablation,
        st_emb_info=cfg.get("st_emb_info", {"spatial_dim": 5, "t_dim": 10, "tid_dim": 6, "diw_dim": 4}),
        use_extrapolation=bool(model_cfg.get("use_extrapolation", True)),
        extrapolation_agg_layers=int(args.extrapolation_layers),
        use_one_channel=bool(args.use_one_channel),
        sharedM=sharedM,
        sharedQ=sharedQ,
        diff_interval=diff_interval,
        predict_only=False,
        le_emb=bool(args.le_emb),
    ).to(device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.exp_name or f"unrolling_gsp_{args.dataset.lower()}_seed{args.seed}"
    exp_dir = PROJECT_DIR / args.result_root / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "tds_unrolling_gsp_run.log"
    summary_path = exp_dir / "summary.json"
    last_ckpt = exp_dir / "last_model.pth"
    best_ckpt = exp_dir / "best_val_model.pth"

    start_epoch = 1
    best_val = float("inf")
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(str(last_ckpt), map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))

    command = " ".join(sys.argv)
    with log_path.open("a", encoding="utf-8") as log:
        print(json.dumps({
            "command": command,
            "args": vars(args),
            "paper_dir": str(PAPER_DIR),
            "counts": {
                "train_total": len(train_set),
                "val_total": len(val_set),
                "test_total": len(test_set),
                "train_shuffle": True,
                "test_shuffle": False,
            },
            "shape": {
                "t_in": t_in,
                "t_out": t_out,
                "n_nodes": train_set.n_nodes,
                "channels": train_set.signal_channel,
            },
            "model": {
                "blocks": blocks,
                "layers": layers,
                "cg_iters": cg_iters,
                "pgd_iters": pgd_iters,
                "heads": heads,
                "feature_channels": feature_channels,
                "neighbors": neighbors,
                "interval": interval,
                "use_one_channel": bool(args.use_one_channel),
                "sharedM": sharedM,
                "sharedQ": sharedQ,
                "diff_interval": diff_interval,
                "ablation": args.ablation,
            },
            "trainable_params": trainable_params,
            "total_params": total_params,
            "resume_start_epoch": start_epoch,
        }, indent=2), file=log, flush=True)

    best_epoch = start_epoch - 1
    bad_epochs = 0
    history = []
    test_metrics = None
    tic = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, scaler, optimizer, device, args, args.max_train_batches)
        val_metrics = evaluate(model, val_loader, scaler, device, args, args.max_eval_batches)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as log:
            print(
                "epoch={:03d} train_mae={:.6f} val_mae={:.6f} val_rmse={:.6f} val_mape={:.6f}".format(
                    epoch, train_metrics.mae, val_metrics.mae, val_metrics.rmse, val_metrics.mape
                ),
                file=log,
                flush=True,
            )
        improved = val_metrics.mae < best_val - 1e-8
        if improved:
            best_val = val_metrics.mae
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_val, args, {"val": asdict(val_metrics)})
            test_metrics = evaluate(model, test_loader, scaler, device, args, args.max_eval_batches)
            torch.save(model.state_dict(), str(exp_dir / "best_test_avg_model.pth"))
        else:
            bad_epochs += 1
        save_checkpoint(last_ckpt, model, optimizer, epoch, best_val, args, {"last_val": asdict(val_metrics)})
        if bad_epochs >= args.patience:
            break

    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
    val_metrics = evaluate(model, val_loader, scaler, device, args, args.max_eval_batches)
    test_metrics = evaluate(model, test_loader, scaler, device, args, args.max_eval_batches)

    summary = {
        "finished": True,
        "run_name": run_name,
        "model": "unrolling_gsp_stforecast",
        "dataset": args.dataset,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "val_avg_mae": float(val_metrics.mae),
        "val_avg_rmse": float(val_metrics.rmse),
        "val_avg_mape": float(val_metrics.mape),
        "test_avg_mae": float(test_metrics.mae),
        "test_mixed_mae": float(test_metrics.mae),
        "test_avg_rmse": float(test_metrics.rmse),
        "test_avg_mape": float(test_metrics.mape),
        "mask_threshold": float(args.mask_threshold),
        "yita": float(args.yita),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "paper_source": "/data/OuXiaoyu/Unrolling-GSP-STForecast",
        "paper_copy": str(PAPER_DIR),
        "elapsed_seconds": float(time.time() - tic),
        "args": vars(args),
        "model_config": {
            "t_in": t_in,
            "t_out": t_out,
            "blocks": blocks,
            "layers": layers,
            "cg_iters": cg_iters,
            "pgd_iters": pgd_iters,
            "heads": heads,
            "feature_channels": feature_channels,
            "neighbors": neighbors,
            "interval": interval,
            "use_one_channel": bool(args.use_one_channel),
            "sharedM": sharedM,
            "sharedQ": sharedQ,
            "diff_interval": diff_interval,
            "ablation": args.ablation,
        },
        "history_tail": history[-10:],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        print("[DONE] " + json.dumps(summary, sort_keys=True), file=log, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
