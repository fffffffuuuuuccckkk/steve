from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline_adapters.common_eval import Timer, ensure_dir, parameter_counts, peak_gpu_memory_mb, save_json
from baseline_adapters.graph_adapter import load_graph_npz_or_npy
from baseline_adapters.largest_dataset import make_largest_npz_loaders
from baseline_adapters.metric_adapter import MetricAccumulator, masked_mae
from baselines.epod.model import EpoDAGCRNForecast, build_support_mask


PAPER = "Improving Generalization of Dynamic Graph Learning via Environment Prompt"
OFFICIAL_REPO = "No official implementation found during the initial audit pass"
SOURCE_COMMIT = "non-official-paper-adapter"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def prompt_target_from_history(x_norm: torch.Tensor, horizon: int) -> torch.Tensor:
    if x_norm.shape[1] >= horizon:
        return x_norm[:, -horizon:, :, :]
    return x_norm[:, -1:, :, :].expand(-1, horizon, -1, -1)


def compute_losses(model, batch, support_mask, prompt_weight: float, beta: float):
    pred_norm, debug = model(batch["x"], support_mask)
    forecast_loss = F.l1_loss(pred_norm, batch["y"])
    prompt_target = prompt_target_from_history(batch["x"], pred_norm.shape[1])
    prompt_loss = F.l1_loss(debug["prompt_recon"], prompt_target)
    kl_loss = debug["prompt_kl"].to(dtype=forecast_loss.dtype)
    total = forecast_loss + float(prompt_weight) * prompt_loss + float(beta) * kl_loss
    logs = {
        "loss": total.detach(),
        "forecast_loss": forecast_loss.detach(),
        "prompt_loss": prompt_loss.detach(),
        "prompt_kl": kl_loss.detach(),
        "dynamic_adj_mean": debug["dynamic_adj_mean"],
        "dynamic_adj_std": debug["dynamic_adj_std"],
        "prompt_attention_mean": debug["prompt_attention_mean"],
    }
    return total, pred_norm, logs


def evaluate(model, loader, scaler, support_mask, device, mask_value: float = 5.0, max_batches: int = -1) -> Dict[str, float]:
    model.eval()
    acc = MetricAccumulator(mask_value=mask_value)
    total_loss = 0.0
    seen = 0
    dyn_mean = []
    dyn_std = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches >= 0 and i >= max_batches:
                break
            batch = to_device(batch, device)
            pred_norm, debug = model(batch["x"], support_mask)
            y_norm = batch["y"]
            loss = F.l1_loss(pred_norm, y_norm)
            pred = scaler.inverse_transform(pred_norm)
            target = scaler.inverse_transform(y_norm)
            acc.update(pred, target)
            bsz = int(pred.shape[0])
            total_loss += float(loss.detach().cpu()) * bsz
            seen += bsz
            dyn_mean.append(float(debug["dynamic_adj_mean"].detach().cpu()))
            dyn_std.append(float(debug["dynamic_adj_std"].detach().cpu()))
    metrics = acc.compute()
    metrics["loss"] = total_loss / max(seen, 1)
    metrics["dynamic_adj_mean"] = float(np.mean(dyn_mean)) if dyn_mean else 0.0
    metrics["dynamic_adj_std"] = float(np.mean(dyn_std)) if dyn_std else 0.0
    return metrics


def train_one_epoch(model, loader, scaler, optimizer, support_mask, device, prompt_weight: float, beta: float, max_batches: int = -1) -> Dict[str, float]:
    del scaler
    model.train()
    totals = {"loss": 0.0, "forecast_loss": 0.0, "prompt_loss": 0.0, "prompt_kl": 0.0}
    dyn_std = []
    seen = 0
    for i, batch in enumerate(loader):
        if max_batches >= 0 and i >= max_batches:
            break
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _pred, logs = compute_losses(model, batch, support_mask, prompt_weight, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bsz = int(batch["x"].shape[0])
        seen += bsz
        for key in totals:
            totals[key] += float(logs[key].detach().cpu()) * bsz
        dyn_std.append(float(logs["dynamic_adj_std"].detach().cpu()))
    denom = max(seen, 1)
    out = {key: value / denom for key, value in totals.items()}
    out["dynamic_adj_std"] = float(np.mean(dyn_std)) if dyn_std else 0.0
    return out


def write_curve(path: str, rows) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/LargeST-SD_TDS")
    parser.add_argument("--graph_file", default="data/LargeST-SD_TDS/adj_mx.npz")
    parser.add_argument("--output_dir", default="experiments/LargeST_SD_OOD/EpoD/seed2024")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--cheb_k", type=int, default=2)
    parser.add_argument("--graph_hops", type=int, default=5)
    parser.add_argument("--prompt_noise_std", type=float, default=0.01)
    parser.add_argument("--prompt_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--mask_value", type=float, default=5.0)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    ensure_dir(args.output_dir)

    bundle = make_largest_npz_loaders(
        args.dataset_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )
    train_shape = bundle.shapes["train"]["x"]
    y_shape = bundle.shapes["train"]["y"]
    input_length = int(train_shape[1])
    horizon = int(y_shape[1])
    num_nodes = int(train_shape[2])
    input_dim = int(train_shape[3])
    output_dim = int(y_shape[3])
    graph = load_graph_npz_or_npy(args.graph_file, device=device)
    support_mask = build_support_mask(graph, num_nodes=num_nodes, hops=args.graph_hops).to(device=device)

    model = EpoDAGCRNForecast(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        cheb_k=args.cheb_k,
        num_layers=args.num_layers,
        graph_hops=args.graph_hops,
        prompt_noise_std=args.prompt_noise_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = parameter_counts(model)

    protocol = {
        "method": "EpoD",
        "implementation_status": "non-official-paper-adapter",
        "paper": PAPER,
        "official_repo": OFFICIAL_REPO,
        "source_commit": SOURCE_COMMIT,
        "dataset": bundle.meta.get("dataset", os.path.basename(args.dataset_dir)),
        "num_nodes": num_nodes,
        "train_period": bundle.meta.get("requested_split", {}).get("train", "unknown"),
        "val_period": bundle.meta.get("requested_split", {}).get("val", "unknown"),
        "test_period": bundle.meta.get("requested_split", {}).get("test_ood", "unknown"),
        "input_length": input_length,
        "output_length": horizon,
        "target_channels": output_dim,
        "seed": args.seed,
        "scaler": bundle.scaler.state_dict(),
        "graph_file": args.graph_file,
        "graph_hops": int(args.graph_hops),
        "support_edge_ratio": float(support_mask.float().mean().detach().cpu()),
        "checkpoint_selection": "lowest validation MAE",
        "uses_environment_labels": False,
        "uses_external_features": False,
        "uses_test_time_adaptation": False,
        "fixed_graph": True,
        **counts,
    }
    save_json(os.path.join(args.output_dir, "protocol.json"), protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)

    rows = []
    best_val = float("inf")
    best_epoch = -1
    best_path = os.path.join(args.output_dir, "best_val_model.pth")
    bad_epochs = 0
    max_epochs = 1 if args.smoke else int(args.epochs)
    with Timer() as timer:
        for epoch in range(1, max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                bundle.loaders["train"],
                bundle.scaler,
                optimizer,
                support_mask,
                device,
                prompt_weight=args.prompt_loss_weight,
                beta=args.beta,
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate(
                model,
                bundle.loaders["val"],
                bundle.scaler,
                support_mask,
                device,
                mask_value=args.mask_value,
                max_batches=args.max_eval_batches,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_forecast_loss": train_metrics["forecast_loss"],
                "train_prompt_loss": train_metrics["prompt_loss"],
                "train_prompt_kl": train_metrics["prompt_kl"],
                "train_dynamic_adj_std": train_metrics["dynamic_adj_std"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mape": val_metrics["mape"],
                "val_dynamic_adj_std": val_metrics["dynamic_adj_std"],
            }
            rows.append(row)
            print("EPOCH " + json.dumps(row, sort_keys=True), flush=True)
            if val_metrics["mae"] < best_val:
                best_val = val_metrics["mae"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "protocol": protocol,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.patience):
                    break

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        val_metrics = evaluate(model, bundle.loaders["val"], bundle.scaler, support_mask, device, args.mask_value, args.max_eval_batches)
        test_metrics = evaluate(model, bundle.loaders["test"], bundle.scaler, support_mask, device, args.mask_value, args.max_eval_batches)

    summary = {
        **protocol,
        "best_epoch": int(best_epoch),
        "val_mae": float(val_metrics["mae"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_mape": float(val_metrics["mape"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mape": float(test_metrics["mape"]),
        "test_dynamic_adj_mean": float(test_metrics["dynamic_adj_mean"]),
        "test_dynamic_adj_std": float(test_metrics["dynamic_adj_std"]),
        "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
        "training_time": float(timer.elapsed),
        "smoke": bool(args.smoke),
        "train_shapes": {k: {kk: list(vv) for kk, vv in shape.items()} for k, shape in bundle.shapes.items()},
    }
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_curve(os.path.join(args.output_dir, "training_curve.csv"), rows)
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

