import argparse
import gc
import json
import os
import subprocess
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

from lib.dataloader import StandardScaler
from lib.metrics import mae_torch, test_metrics
from lib.utils import dwa, get_project_path, init_seed, load_graph
from models.fpem.agcrn_adapter import AGCRNEncoder
from models.our_model import StableST


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_cli_value(value):
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "yes", "y", "on"}:
        return True
    if lower in {"false", "no", "n", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_unknown_overrides(items):
    overrides = {}
    i = 0
    while i < len(items):
        item = items[i]
        if not item.startswith("--"):
            i += 1
            continue
        key = item[2:].replace("-", "_")
        if "=" in key:
            key, value = key.split("=", 1)
        elif i + 1 < len(items) and not items[i + 1].startswith("--"):
            value = items[i + 1]
            i += 1
        else:
            value = "true"
        overrides[key] = parse_cli_value(value)
        i += 1
    return overrides


def as_float32_array(data):
    return np.ascontiguousarray(data, dtype=np.float32)


def as_int64_array(data):
    return np.ascontiguousarray(data, dtype=np.int64)


class TDSDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, time_label, c, include_time, include_index=True):
        self.x = torch.from_numpy(as_float32_array(x))
        self.y = torch.from_numpy(as_float32_array(y))
        self.c = torch.from_numpy(as_float32_array(c))
        self.include_time = include_time
        self.time_label = None if time_label is None else torch.from_numpy(as_int64_array(time_label))
        self.include_index = include_index
        self.sample_index = torch.arange(self.x.shape[0], dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.include_time:
            item = (self.x[idx], self.y[idx], self.time_label[idx], self.c[idx])
        else:
            item = (self.x[idx], self.y[idx], self.c[idx])
        if self.include_index:
            item = item + (self.sample_index[idx],)
        return item


def to_device(batch, device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def make_loader(x, y, time_label, c, batch_size, shuffle, include_time, drop_last=False, pin_memory=False, include_index=True):
    dataset = TDSDataset(x, y, time_label, c, include_time=include_time, include_index=include_index)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def standard_scaler_from_arrays(arrays):
    total = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        total += flat.size
        total_sum += float(flat.sum(dtype=np.float64))
        total_sq_sum += float(np.dot(flat, flat))
    mean = total_sum / max(total, 1)
    var = max(total_sq_sum / max(total, 1) - mean * mean, 0.0)
    return StandardScaler(mean=np.float64(mean), std=np.float64(np.sqrt(var)))


def standard_transform_float32(data, scaler):
    out = np.empty(data.shape, dtype=np.float32)
    np.subtract(data, scaler.mean, out=out, casting="unsafe")
    np.divide(out, scaler.std, out=out, casting="unsafe")
    return np.ascontiguousarray(out)


def load_npz(path):
    with np.load(path) as z:
        return {"x": z["x"], "y": z["y"], "time_label": z["time_label"], "c": z["c"]}


def subset_pack(pack, idx):
    n = len(pack["x"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == n else v) for k, v in pack.items()}


def build_tds_data(args):
    root = os.path.join(args.data_dir, args.dataset)
    if not os.path.isdir(root) and args.dataset == "NYCTaxi_TDS":
        fallback = os.path.join(args.data_dir, "NYCTaxi")
        if os.path.isdir(fallback):
            root = fallback
    raw = {cat: load_npz(os.path.join(root, f"{cat}.npz")) for cat in ["train", "val", "test"]}

    work = raw["train"]["time_label"] < 24
    holiday = ~work
    work_idx_all = np.where(work)[0]
    holiday_idx = np.where(holiday)[0]
    work_count = min(len(work_idx_all), int(round(len(holiday_idx) * args.train_work_per_holiday)))
    rng = np.random.default_rng(getattr(args, "data_seed", args.seed))
    work_idx = np.sort(rng.choice(work_idx_all, size=work_count, replace=False))
    train_idx = np.sort(np.concatenate([work_idx, holiday_idx]))

    packs = {
        "train": subset_pack(raw["train"], train_idx),
        "val": raw["val"],
        "test_mixed": raw["test"],
        "test_workday": subset_pack(raw["test"], np.where(raw["test"]["time_label"] < 24)[0]),
        "test_holiday": subset_pack(raw["test"], np.where(raw["test"]["time_label"] >= 24)[0]),
    }
    del raw

    scaler = standard_scaler_from_arrays([packs["train"]["x"], packs["val"]["x"]])
    for pack in packs.values():
        pack["x"] = standard_transform_float32(pack["x"], scaler)
        pack["y"] = standard_transform_float32(pack["y"], scaler)
        pack["c"] = as_float32_array(pack["c"])

    pin_memory = str(args.device).startswith("cuda")
    train_route_mode = str(getattr(args, "fpem_env_route_train_mode", "")).lower()
    use_chronological_train = (
        train_route_mode == "hard_prediction_environment_sinkhorn"
        and float(getattr(args, "fpem_env_sinkhorn_temporal_lambda", 0.0)) > 0.0
    )
    train_shuffle = not use_chronological_train
    loaders = {
        "train": make_loader(
            packs["train"]["x"], packs["train"]["y"], packs["train"]["time_label"], packs["train"]["c"],
            args.batch_size, shuffle=train_shuffle, include_time=True, drop_last=True, pin_memory=pin_memory,
        ),
        "train_partition": make_loader(
            packs["train"]["x"], packs["train"]["y"], packs["train"]["time_label"], packs["train"]["c"],
            args.test_batch_size, shuffle=False, include_time=True, drop_last=False, pin_memory=pin_memory,
        ),
        "val": make_loader(
            packs["val"]["x"], packs["val"]["y"], packs["val"]["time_label"], packs["val"]["c"],
            args.test_batch_size, shuffle=False, include_time=True, drop_last=False, pin_memory=pin_memory,
        ),
        "test_mixed": make_loader(
            packs["test_mixed"]["x"], packs["test_mixed"]["y"], packs["test_mixed"]["time_label"], packs["test_mixed"]["c"],
            args.test_batch_size, shuffle=False, include_time=True, drop_last=False, pin_memory=pin_memory,
        ),
        "test_workday": make_loader(
            packs["test_workday"]["x"], packs["test_workday"]["y"], packs["test_workday"]["time_label"], packs["test_workday"]["c"],
            args.test_batch_size, shuffle=False, include_time=True, drop_last=False, pin_memory=pin_memory,
        ),
        "test_holiday": make_loader(
            packs["test_holiday"]["x"], packs["test_holiday"]["y"], packs["test_holiday"]["time_label"], packs["test_holiday"]["c"],
            args.test_batch_size, shuffle=False, include_time=True, drop_last=False, pin_memory=pin_memory,
        ),
        "scaler": scaler,
    }
    counts = {
        "train_total": int(len(train_idx)),
        "train_workday": int(len(work_idx)),
        "train_holiday": int(len(holiday_idx)),
        "train_workday_holiday_ratio": float(len(work_idx) / max(len(holiday_idx), 1)),
        "val_total": int(len(packs["val"]["x"])),
        "val_workday": int((packs["val"]["time_label"] < 24).sum()),
        "val_holiday": int((packs["val"]["time_label"] >= 24).sum()),
        "test_total": int(len(packs["test_mixed"]["x"])),
        "test_workday": int(len(packs["test_workday"]["x"])),
        "test_holiday": int(len(packs["test_holiday"]["x"])),
        "train_shuffle": bool(train_shuffle),
        "train_chronological_for_env_sinkhorn_temporal": bool(use_chronological_train),
    }
    del packs
    gc.collect()
    return loaders, scaler, counts


class AGCRNForecast(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_node = args.num_nodes
        self.output_dim = args.d_output
        self.horizon = 1
        self.encoder = AGCRNEncoder(
            args.num_nodes,
            args.d_input,
            args.agcrn_rnn_units,
            args.agcrn_cheb_k,
            args.agcrn_embed_dim,
            args.agcrn_num_layers,
        )
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, args.agcrn_rnn_units), bias=True)

    def forward(self, source, targets=None):
        output = self.encoder(source)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        return output.permute(0, 1, 3, 2)


def init_agcrn_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)


def build_model(args, graph):
    if args.model == "agcrn":
        model = AGCRNForecast(args)
        init_agcrn_params(model)
        lr = args.agcrn_lr
    elif args.model == "steve":
        model = StableST(
            args=args,
            adj=graph,
            in_channels=args.d_input,
            embed_size=args.d_model,
            T_dim=args.input_length,
            output_T_dim=1,
            output_dim=args.d_output,
            device=args.device,
        )
        lr = args.lr_init
    else:
        raise ValueError(f"unsupported model: {args.model}")
    return model.to(args.device), lr


def weighted_flow_mae(pred, target, scaler, yita=0.5):
    pred = scaler.inverse_transform(pred)
    target = scaler.inverse_transform(target)
    loss = yita * mae_torch(pred[..., 0], target[..., 0], mask_value=5.0)
    loss = loss + (1.0 - yita) * mae_torch(pred[..., 1], target[..., 1], mask_value=5.0)
    return loss


def forecast_loss(model, batch, scaler, args):
    if len(batch) == 5:
        data, target, _time_label, _c, _sample_index = batch
    elif len(batch) == 4:
        data, target, _time_label, _c = batch
    else:
        data, target, _c = batch
    pred = model(data, target)
    return weighted_flow_mae(pred, target, scaler, args.yita), pred


def steve_loss(model, batch, scaler, args, loss_weights, epoch):
    sample_index = None
    if len(batch) == 5:
        data, target, time_label, c, sample_index = batch
    elif len(batch) == 4:
        data, target, time_label, c = batch
    else:
        data, target, c = batch
        time_label = None
    p = epoch / max(float(args.epochs), 1.0)
    z, h = model(data, c=c, time_label=time_label)
    loss, sep_loss, _lm = model.calculate_loss(
        z, h, target, c, time_label, scaler, loss_weights, p, True, sample_index=sample_index
    )
    return loss, sep_loss


def predict_batch(model, batch, args):
    sample_index = None
    if len(batch) == 5:
        data, target, time_label, c, sample_index = batch
    elif len(batch) == 4:
        data, target, time_label, c = batch
    else:
        data, target, c = batch
        time_label = None
    if args.model == "steve":
        output = model.forward_output(data, exog=c, time_label=time_label, training=False, sample_index=sample_index)
        model._last_eval_output = output
        pred = output["prediction"]
    else:
        pred = model(data, target)
    return pred, target


def evaluate(model, loader, scaler, args, max_batches=None):
    model.eval()
    y_pred = []
    y_true = []
    progressive_clusters = []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = to_device(raw_batch, args.device)
            pred, target = predict_batch(model, batch, args)
            latest_eval_output = getattr(model, "_last_eval_output", None)
            if isinstance(latest_eval_output, dict) and torch.is_tensor(latest_eval_output.get("progressive_cluster_id")):
                progressive_clusters.append(latest_eval_output["progressive_cluster_id"].detach().cpu())
            total_loss += weighted_flow_mae(pred, target, scaler, args.yita).item()
            count += 1
            y_pred.append(pred.squeeze(1))
            y_true.append(target.squeeze(1))
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
    mae, mape = test_metrics(y_pred, y_true)
    result = {"loss": total_loss / max(count, 1), "mae": float(mae), "mape": float(mape)}
    if progressive_clusters:
        clusters = torch.cat(progressive_clusters, dim=0).long()
        if clusters.numel() > 1:
            result["progressive_route_switch_frequency"] = float((clusters[1:] != clusters[:-1]).float().mean().item())
        else:
            result["progressive_route_switch_frequency"] = 0.0
        result["progressive_selected_k_observed"] = int(clusters.max().item() + 1) if clusters.numel() else 0
    return result


def save_checkpoint(path, model, optimizer, epoch, monitor, args):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "monitor": monitor,
        "args": vars(args),
    }
    if hasattr(model, "get_progressive_gmm_state_for_checkpoint"):
        payload["fpem_progressive_gmm_state"] = model.get_progressive_gmm_state_for_checkpoint()
    torch.save(payload, path)


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def report_load_state_warnings(context, result, log_path=None):
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if not missing and not unexpected:
        return
    msg = (
        f"CHECKPOINT_LOAD_WARNING context={context} "
        f"missing_keys={missing} unexpected_keys={unexpected}"
    )
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def parse_epoch_rows(log_path):
    rows = []
    if not os.path.exists(log_path):
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("EPOCH "):
                continue
            try:
                rows.append(json.loads(line[len("EPOCH "):]))
            except json.JSONDecodeError:
                continue
    return rows


def rebuild_best_from_rows(rows):
    best = {
        "val": {"mae": float("inf"), "epoch": 0},
        "test_avg": {"mae": float("inf"), "epoch": 0},
        "workday": {"mae": float("inf"), "epoch": 0},
        "holiday": {"mae": float("inf"), "epoch": 0},
    }
    not_improved = 0
    for row in rows:
        epoch = int(row.get("epoch", 0))
        val_loss = float(row["val"]["loss"])
        if val_loss < best["val"]["mae"]:
            best["val"] = {"mae": val_loss, "epoch": epoch}
            not_improved = 0
        else:
            not_improved += 1
        test_avg = float(row["test_avg_mae"])
        if test_avg < best["test_avg"]["mae"]:
            best["test_avg"] = {"mae": test_avg, "epoch": epoch}
        workday = float(row["test_workday"]["mae"])
        if workday < best["workday"]["mae"]:
            best["workday"] = {"mae": workday, "epoch": epoch}
        holiday = float(row["test_holiday"]["mae"])
        if holiday < best["holiday"]["mae"]:
            best["holiday"] = {"mae": holiday, "epoch": epoch}
    return best, not_improved


def load_resume_state(args, model, optimizer, log_path, last_path, best_val_path):
    fresh_best, fresh_not_improved = rebuild_best_from_rows([])
    if not args.resume:
        return 1, fresh_best, fresh_not_improved
    ckpt_path = last_path if os.path.exists(last_path) else best_val_path if os.path.exists(best_val_path) else None
    if ckpt_path is None:
        return 1, fresh_best, fresh_not_improved
    rows = parse_epoch_rows(log_path)
    best, not_improved = rebuild_best_from_rows(rows)
    ckpt = load_checkpoint(ckpt_path, args.device)
    result = model.load_state_dict(ckpt["model"], strict=False)
    report_load_state_warnings(f"resume:{ckpt_path}", result, log_path)
    if "fpem_progressive_gmm_state" in ckpt and hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
        model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
    if getattr(args, "fpem_use_pretrained_inv_agcrn", False):
        if hasattr(model, "_load_pretrained_inv_agcrn"):
            model._load_pretrained_inv_agcrn()
        if hasattr(model, "freeze_invariant_encoder"):
            model.freeze_invariant_encoder()
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            pass
    logged_epoch = max([int(row.get("epoch", 0)) for row in rows], default=0)
    ckpt_epoch = int(ckpt.get("epoch", 0))
    return max(logged_epoch, ckpt_epoch) + 1, best, not_improved


def git_commit_hash(project):
    try:
        result = subprocess.run(
            ["git", "-C", project, "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def scalar_json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


MECHANISM_SUMMARY_KEYS = {
    "env_day_acc": "fpem/env_day_acc",
    "env_hour_acc": "fpem/env_hour_acc",
    "env_rush_acc": "fpem/env_rush_acc",
    "inv_day_acc": "fpem/inv_day_acc",
    "inv_hour_acc": "fpem/inv_hour_acc",
    "inv_rush_acc": "fpem/inv_rush_acc",
    "effective_expert_number": "fpem/effective_expert_number",
    "max_expert_usage_ratio": "fpem/max_expert_usage_ratio",
    "min_expert_usage_ratio": "fpem/min_expert_usage_ratio",
    "hyper_alpha_mean": "fpem/hyper_alpha_mean",
    "hyper_delta_norm": "fpem/hyper_delta_norm",
    "loss_future_mi": "fpem/future_mi_loss",
    "loss_swap": "fpem/swap_loss",
    "loss_club_upper": "fpem/club_upper",
    "swap_prediction_delta": "fpem/swap_prediction_delta",
    "swap_route_delta": "fpem/swap_route_delta",
    "swap_hyper_alpha_delta": "fpem/swap_hyper_alpha_delta",
    "swap_hyper_delta_norm": "fpem/swap_hyper_delta_norm",
    "expert0_soft_usage": "fpem/route_soft_mean_expert_0",
    "expert1_soft_usage": "fpem/route_soft_mean_expert_1",
    "expert2_soft_usage": "fpem/route_soft_mean_expert_2",
    "expert3_soft_usage": "fpem/route_soft_mean_expert_3",
    "expert_soft_usage_0": "fpem/route_soft_mean_expert_0",
    "expert_soft_usage_1": "fpem/route_soft_mean_expert_1",
    "expert_soft_usage_2": "fpem/route_soft_mean_expert_2",
    "expert_soft_usage_3": "fpem/route_soft_mean_expert_3",
    "expert0_hard_count": "fpem/route_hard_count_expert_0",
    "expert1_hard_count": "fpem/route_hard_count_expert_1",
    "expert2_hard_count": "fpem/route_hard_count_expert_2",
    "expert3_hard_count": "fpem/route_hard_count_expert_3",
    "expert_hard_count_0": "fpem/route_hard_count_expert_0",
    "expert_hard_count_1": "fpem/route_hard_count_expert_1",
    "expert_hard_count_2": "fpem/route_hard_count_expert_2",
    "expert_hard_count_3": "fpem/route_hard_count_expert_3",
    "hyper_alpha_head_0": "fpem/hyper_alpha_head_0",
    "hyper_alpha_head_1": "fpem/hyper_alpha_head_1",
    "hyper_alpha_head_2": "fpem/hyper_alpha_head_2",
    "hyper_alpha_head_3": "fpem/hyper_alpha_head_3",
    "hyper_gamma_norm_head_0": "fpem/hyper_gamma_norm_head_0",
    "hyper_gamma_norm_head_1": "fpem/hyper_gamma_norm_head_1",
    "hyper_gamma_norm_head_2": "fpem/hyper_gamma_norm_head_2",
    "hyper_gamma_norm_head_3": "fpem/hyper_gamma_norm_head_3",
    "hyper_beta_norm_head_0": "fpem/hyper_beta_norm_head_0",
    "hyper_beta_norm_head_1": "fpem/hyper_beta_norm_head_1",
    "hyper_beta_norm_head_2": "fpem/hyper_beta_norm_head_2",
    "hyper_beta_norm_head_3": "fpem/hyper_beta_norm_head_3",
    "expert_collapse_warning": "fpem/expert_collapse_warning",
    "prototype_pairwise_cosine": "fpem/prototype_pairwise_cosine",
    "expert_prediction_pairwise_cosine": "fpem/expert_prediction_pairwise_cosine",
    "loss_env_day": "fpem/loss_env_day",
    "loss_env_hour": "fpem/loss_env_hour",
    "loss_env_rush": "fpem/loss_env_rush",
    "loss_env_supcon": "fpem/loss_env_supcon",
    "loss_inv_env_adv": "fpem/loss_inv_env_adv",
    "loss_cross_cov_sep": "fpem/loss_cross_cov_sep",
    "fpem_env_use_exogenous": "fpem/fpem_env_use_exogenous",
    "env_exogenous_available": "fpem/env_exogenous_available",
    "env_exogenous_time_available": "fpem/env_exogenous_time_available",
    "env_exogenous_load_available": "fpem/env_exogenous_load_available",
    "env_exogenous_feature_dim": "fpem/env_exogenous_feature_dim",
    "env_exogenous_embedding_norm": "fpem/env_exogenous_embedding_norm",
    "env_exogenous_load_embedding_norm": "fpem/env_exogenous_load_embedding_norm",
    "backbone_agcrn": "fpem/backbone_agcrn",
    "backbone_graphwavenet": "fpem/backbone_graphwavenet",
    "backbone_staeformer": "fpem/backbone_staeformer",
    "fpem_force_uniform_route": "fpem/fpem_force_uniform_route",
    "env_rep_ablation_mode_id": "fpem/env_rep_ablation_mode_id",
    "env_rep_ablation_zero": "fpem/env_rep_ablation_zero",
    "env_rep_ablation_shuffle_batch": "fpem/env_rep_ablation_shuffle_batch",
    "hard_sinkhorn_enabled": "fpem/hard_sinkhorn_enabled",
    "sinkhorn_soft_entropy": "fpem/sinkhorn_soft_entropy",
    "sinkhorn_selected_loss": "fpem/sinkhorn_selected_loss",
    "env_sinkhorn_prediction_alpha": "fpem/env_sinkhorn_prediction_alpha",
    "env_sinkhorn_environment_beta": "fpem/env_sinkhorn_environment_beta",
    "env_sinkhorn_schedule_progress": "fpem/env_sinkhorn_schedule_progress",
    "env_sinkhorn_temporal_lambda": "fpem/env_sinkhorn_temporal_lambda",
    "env_sinkhorn_temporal_valid_fraction": "fpem/env_sinkhorn_temporal_valid_fraction",
    "env_sinkhorn_prediction_cost_mean": "fpem/env_sinkhorn_prediction_cost_mean",
    "env_sinkhorn_environment_cost_mean": "fpem/env_sinkhorn_environment_cost_mean",
    "env_sinkhorn_temporal_cost_mean": "fpem/env_sinkhorn_temporal_cost_mean",
    "env_sinkhorn_gaussian_initialized": "fpem/env_sinkhorn_gaussian_initialized",
    "env_sinkhorn_gaussian_feature_dim": "fpem/env_sinkhorn_gaussian_feature_dim",
    "env_route_inference_mode_gaussian": "fpem/env_route_inference_mode_gaussian",
    "warmup_risk_stage_warmup": "fpem/warmup_risk_stage_warmup",
    "warmup_risk_stage_soft": "fpem/warmup_risk_stage_soft",
    "warmup_risk_stage_hard": "fpem/warmup_risk_stage_hard",
    "warmup_risk_temperature": "fpem/warmup_risk_temperature",
    "warmup_risk_lambda_common": "fpem/warmup_risk_lambda_common",
    "warmup_risk_selected_loss": "fpem/warmup_risk_selected_loss",
    "warmup_risk_uniform_common_loss": "fpem/warmup_risk_uniform_common_loss",
    "uniform_expert_mae": "fpem/uniform_expert_mae",
    "risk_routed_mae": "fpem/risk_routed_mae",
    "risk_prediction_mae": "fpem/risk_prediction_mae",
    "risk_loss": "fpem/risk_loss",
    "risk_ranking_accuracy": "fpem/risk_ranking_accuracy",
    "risk_oracle_argmin_accuracy": "fpem/risk_oracle_argmin_accuracy",
    "progressive_selected_k": "fpem/progressive_selected_k",
    "progressive_partition_ready": "fpem/progressive_partition_ready",
    "progressive_active_expert_count": "fpem/progressive_active_expert_count",
    "progressive_last_partition_epoch": "fpem/progressive_last_partition_epoch",
    "progressive_assignment_uses_target_or_prediction_error": "fpem/progressive_assignment_uses_target_or_prediction_error",
    "progressive_routed_mae": "fpem/progressive_routed_mae",
    "progressive_compactness_loss": "fpem/progressive_compactness_loss",
    "progressive_consistency_loss": "fpem/progressive_consistency_loss",
    "progressive_cluster_entropy": "fpem/progressive_cluster_entropy",
    "progressive_assignment_change_ratio": "fpem/progressive_assignment_change_ratio",
    "progressive_hungarian_overlap_score": "fpem/progressive_hungarian_overlap_score",
    "progressive_bic_k1": "fpem/progressive_bic_k1",
    "progressive_bic_k2": "fpem/progressive_bic_k2",
    "progressive_bic_k3": "fpem/progressive_bic_k3",
    "progressive_bic_k4": "fpem/progressive_bic_k4",
    "progressive_bic_k5": "fpem/progressive_bic_k5",
    "progressive_bic_k6": "fpem/progressive_bic_k6",
    "router_assignment_accuracy": "fpem/router_assignment_accuracy",
    "router_assignment_agreement": "fpem/router_assignment_agreement",
    "oracle_hard_mae": "fpem/oracle_hard_mae",
    "router_hard_mae": "fpem/router_hard_mae",
    "router_regret": "fpem/router_regret",
    "hard_count_expert_0": "fpem/hard_count_expert_0",
    "hard_count_expert_1": "fpem/hard_count_expert_1",
    "hard_count_expert_2": "fpem/hard_count_expert_2",
    "sinkhorn_soft_col_mass_expert_0": "fpem/sinkhorn_soft_col_mass_expert_0",
    "sinkhorn_soft_col_mass_expert_1": "fpem/sinkhorn_soft_col_mass_expert_1",
    "sinkhorn_soft_col_mass_expert_2": "fpem/sinkhorn_soft_col_mass_expert_2",
    "expert_cross_mae_group_0_expert_0": "fpem/expert_cross_mae_group_0_expert_0",
    "expert_cross_mae_group_0_expert_1": "fpem/expert_cross_mae_group_0_expert_1",
    "expert_cross_mae_group_0_expert_2": "fpem/expert_cross_mae_group_0_expert_2",
    "expert_cross_mae_group_1_expert_0": "fpem/expert_cross_mae_group_1_expert_0",
    "expert_cross_mae_group_1_expert_1": "fpem/expert_cross_mae_group_1_expert_1",
    "expert_cross_mae_group_1_expert_2": "fpem/expert_cross_mae_group_1_expert_2",
    "expert_cross_mae_group_2_expert_0": "fpem/expert_cross_mae_group_2_expert_0",
    "expert_cross_mae_group_2_expert_1": "fpem/expert_cross_mae_group_2_expert_1",
    "expert_cross_mae_group_2_expert_2": "fpem/expert_cross_mae_group_2_expert_2",
}


def mechanism_summary_from_final(final):
    monitor = (final.get("val") or {}).get("monitor", {})
    fpem_logs = monitor.get("fpem_logs", {}) if isinstance(monitor, dict) else {}
    result = {"best_val_fpem_logs": fpem_logs}
    if isinstance(fpem_logs, dict):
        for out_key, log_key in MECHANISM_SUMMARY_KEYS.items():
            result[out_key] = fpem_logs.get(log_key)
    return result


def train_one(args):
    init_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "tds_run.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    loaders, scaler, counts = build_tds_data(args)
    graph = load_graph(args.graph_file, device=args.device)
    model, lr = build_model(args, graph)
    main_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("mi_net.")
    ]
    optimizer = torch.optim.Adam(main_params, lr=lr, eps=1.0e-8, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        min_lr=0.000005,
        eps=1e-08,
    )

    best_paths = {
        "val": os.path.join(args.log_dir, "best_val_model.pth"),
        "test_avg": os.path.join(args.log_dir, "best_test_avg_model.pth"),
        "workday": os.path.join(args.log_dir, "best_test_workday_model.pth"),
        "holiday": os.path.join(args.log_dir, "best_test_holiday_model.pth"),
    }
    last_path = os.path.join(args.log_dir, "last_model.pth")
    start_epoch, best, not_improved = load_resume_state(args, model, optimizer, log_path, last_path, best_paths["val"])
    if args.resume and start_epoch > 1 and args.resume_reset_patience:
        not_improved = 0
    route_train_mode = str(getattr(args, "fpem_env_route_train_mode", "")).lower()
    if route_train_mode == "warmup_risk_sinkhorn":
        top_level_prediction_source = "risk_router_soft"
    elif route_train_mode == "progressive_gmm_environment":
        top_level_prediction_source = "progressive_gmm_teacher_hard"
    else:
        top_level_prediction_source = "model_forward_prediction"

    log(json.dumps({
        "args": vars(args),
        "counts": counts,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_params": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "optimizer_params": int(sum(p.numel() for group in optimizer.param_groups for p in group["params"])),
        "resume_start_epoch": start_epoch,
        "top_level_prediction_source": top_level_prediction_source,
    }, ensure_ascii=False, indent=2))
    start_time = time.time()
    loss_tm1 = loss_t = np.ones(3)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.model == "steve" and args.use_dwa:
            loss_tm2 = loss_tm1
            loss_tm1 = loss_t
            loss_weights = dwa(loss_tm1, loss_tm1 if epoch <= 2 else loss_tm2, args.temp)
        else:
            loss_weights = np.ones(3)

        train_losses = []
        sep_losses = []
        fpem_log_sums = {}
        fpem_log_count = 0
        for batch_idx, raw_batch in enumerate(loaders["train"]):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            batch = to_device(raw_batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "steve":
                loss, sep_loss = steve_loss(model, batch, scaler, args, loss_weights, epoch)
                sep_losses.append(sep_loss)
                primary_loss = None
                latest_outputs = getattr(model, "latest_fpem_outputs", {})
                if isinstance(latest_outputs, dict):
                    primary_loss = latest_outputs.get("primary_loss")
                if hasattr(model, "prepare_fpem_gc_pred_loss_only"):
                    model.prepare_fpem_gc_pred_loss_only(primary_loss)
                gc_handles = model.register_fpem_grad_consensus_hooks(epoch) if hasattr(model, "register_fpem_grad_consensus_hooks") else []
                try:
                    loss.backward()
                finally:
                    for handle in gc_handles:
                        handle.remove()
                    if hasattr(model, "clear_fpem_gc_pred_loss_only"):
                        model.clear_fpem_gc_pred_loss_only()
                latest_logs = getattr(model, "latest_fpem_logs", None)
                if latest_logs:
                    fpem_log_count += 1
                    for key, value in latest_logs.items():
                        fpem_log_sums[key] = fpem_log_sums.get(key, 0.0) + float(value)
            else:
                loss, _pred = forecast_loss(model, batch, scaler, args)
                loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch={epoch} batch={batch_idx}: {loss.item()}")
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if hasattr(model, "update_progressive_teacher_ema"):
                model.update_progressive_teacher_ema()
            if hasattr(model, "clear_fpem_runtime_cache"):
                model.clear_fpem_runtime_cache()
            else:
                if hasattr(model, "latest_fpem_outputs"):
                    model.latest_fpem_outputs = {}
                if hasattr(model, "_fpem_gc_primary_grads"):
                    model._fpem_gc_primary_grads = {}
            train_losses.append(float(loss.item()))

        if sep_losses:
            loss_t = np.nan_to_num(np.asarray(sep_losses, dtype=np.float64).mean(axis=0), nan=1.0, posinf=1.0, neginf=1.0)

        progressive_partition_info = None
        if args.model == "steve" and hasattr(model, "maybe_update_progressive_gmm_partition"):
            progressive_partition_info = model.maybe_update_progressive_gmm_partition(
                loaders.get("train_partition", loaders["train"]),
                epoch=epoch,
                total_epochs=args.epochs,
                device=args.device,
                max_batches=getattr(args, "fpem_env_partition_max_batches", -1),
            )
            if progressive_partition_info:
                log("PROGRESSIVE_GMM_PARTITION " + json.dumps(progressive_partition_info, ensure_ascii=False, default=scalar_json))

        val = evaluate(model, loaders["val"], scaler, args, max_batches=args.max_eval_batches)
        mixed = evaluate(model, loaders["test_mixed"], scaler, args, max_batches=args.max_eval_batches)
        workday = evaluate(model, loaders["test_workday"], scaler, args, max_batches=args.max_eval_batches)
        holiday = evaluate(model, loaders["test_holiday"], scaler, args, max_batches=args.max_eval_batches)
        test_avg_mae = (workday["mae"] + holiday["mae"]) / 2.0
        scheduler.step(float(val["loss"]))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val": val,
            "test_mixed": mixed,
            "test_workday": workday,
            "test_holiday": holiday,
            "test_avg_mae": test_avg_mae,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if fpem_log_count > 0:
            row["fpem_logs"] = {key: value / fpem_log_count for key, value in fpem_log_sums.items()}
        if progressive_partition_info:
            row["progressive_partition"] = progressive_partition_info
        log("EPOCH " + json.dumps(row, ensure_ascii=False))

        improved_val = val["loss"] < best["val"]["mae"]
        if improved_val:
            best["val"] = {"mae": val["loss"], "epoch": epoch}
            save_checkpoint(best_paths["val"], model, optimizer, epoch, row, args)
            not_improved = 0
        else:
            not_improved += 1
        if test_avg_mae < best["test_avg"]["mae"]:
            best["test_avg"] = {"mae": test_avg_mae, "epoch": epoch}
            save_checkpoint(best_paths["test_avg"], model, optimizer, epoch, row, args)
        if workday["mae"] < best["workday"]["mae"]:
            best["workday"] = {"mae": workday["mae"], "epoch": epoch}
            save_checkpoint(best_paths["workday"], model, optimizer, epoch, row, args)
        if holiday["mae"] < best["holiday"]["mae"]:
            best["holiday"] = {"mae": holiday["mae"], "epoch": epoch}
            save_checkpoint(best_paths["holiday"], model, optimizer, epoch, row, args)
        save_checkpoint(last_path, model, optimizer, epoch, row, args)

        test_mae_stop_epoch = int(getattr(args, "early_stop_test_avg_mae_epoch", 0))
        test_mae_threshold = float(getattr(args, "early_stop_test_avg_mae_threshold", 0.0))
        if (
            test_mae_stop_epoch > 0
            and epoch >= test_mae_stop_epoch
            and test_avg_mae >= test_mae_threshold
        ):
            log(
                f"EARLY_STOP_TEST_MAE epoch={epoch} "
                f"test_avg_mae={test_avg_mae:.6f} threshold={test_mae_threshold:.6f}"
            )
            break

        if args.early_stop and not_improved >= args.early_stop_patience:
            log(f"EARLY_STOP epoch={epoch} patience={args.early_stop_patience}")
            break

    final = {}
    for name, path in best_paths.items():
        if not os.path.exists(path):
            continue
        ckpt = load_checkpoint(path, args.device)
        result = model.load_state_dict(ckpt["model"], strict=False)
        report_load_state_warnings(f"final:{path}", result, log_path)
        if "fpem_progressive_gmm_state" in ckpt and hasattr(model, "load_progressive_gmm_state_from_checkpoint"):
            model.load_progressive_gmm_state_from_checkpoint(ckpt.get("fpem_progressive_gmm_state"))
        final[name] = {
            "selected_epoch": int(ckpt["epoch"]),
            "monitor": ckpt.get("monitor", {}),
            "val": evaluate(model, loaders["val"], scaler, args, max_batches=args.max_eval_batches),
            "test_mixed": evaluate(model, loaders["test_mixed"], scaler, args, max_batches=args.max_eval_batches),
            "test_workday": evaluate(model, loaders["test_workday"], scaler, args, max_batches=args.max_eval_batches),
            "test_holiday": evaluate(model, loaders["test_holiday"], scaler, args, max_batches=args.max_eval_batches),
        }
        final[name]["test_avg_mae"] = (final[name]["test_workday"]["mae"] + final[name]["test_holiday"]["mae"]) / 2.0

    selected = final.get("val") or next(iter(final.values()))
    summary = {
        "exp_name": args.run_name,
        "seed": int(args.seed),
        "dataset": args.dataset,
        "model": args.model,
        "ablation": args.ablation,
        "best_epoch": int(best["val"]["epoch"]),
        "best_val_loss": float(best["val"]["mae"]),
        "test_mixed_mae": float(selected["test_mixed"]["mae"]),
        "test_workday_mae": float(selected["test_workday"]["mae"]),
        "test_holiday_mae": float(selected["test_holiday"]["mae"]),
        "test_avg_mae": float(selected["test_avg_mae"]),
        "fpem_backbone": str(getattr(args, "fpem_backbone", "agcrn")),
        "route_head_mode": str(getattr(args, "fpem_env_route_head_mode", "concat_input")),
        "fpem_use_env_fusion": bool(getattr(args, "fpem_use_env_fusion", True)),
        "fpem_env_route_use_inv_fallback_expert": bool(getattr(args, "fpem_env_route_use_inv_fallback_expert", True)),
        "fpem_env_route_target_mode": str(getattr(args, "fpem_env_route_target_mode", "prediction_oracle")),
        "fpem_env_route_train_mode": str(getattr(args, "fpem_env_route_train_mode", "soft_oracle")),
        "fpem_env_route_inference_mode": str(getattr(args, "fpem_env_route_inference_mode", "mlp")),
        "top_level_prediction_source": top_level_prediction_source,
        "fpem_force_uniform_route": bool(getattr(args, "fpem_force_uniform_route", False)),
        "fpem_env_rep_ablation": str(getattr(args, "fpem_env_rep_ablation", "normal")),
        "finished": True,
        "counts": counts,
        "best": best,
        "final": final,
        "elapsed_min": (time.time() - start_time) / 60.0,
        "args": vars(args),
        "git_commit": args.git_commit,
    }
    if hasattr(model, "get_progressive_gmm_state_for_checkpoint"):
        progressive_state = model.get_progressive_gmm_state_for_checkpoint()
        summary["progressive_partition_history"] = progressive_state.get("partition_history", [])
        summary["progressive_latest_partition"] = progressive_state.get("latest_partition_logs", {})
    summary.update(mechanism_summary_from_final(final))
    with open(os.path.join(args.log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=scalar_json)
    log("SUMMARY " + json.dumps({k: summary[k] for k in [
        "exp_name", "model", "ablation", "best_epoch", "best_val_loss",
        "test_mixed_mae", "test_workday_mae", "test_holiday_mae", "test_avg_mae",
        "top_level_prediction_source", "finished",
    ]}, ensure_ascii=False))


def resolve_path(project, path):
    return path if os.path.isabs(path) else os.path.join(project, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="configs/NYCTaxi.yaml")
    parser.add_argument("--model", choices=["steve", "agcrn"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--dataset", default="NYCTaxi_TDS")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--graph_file", default="data/NYCTaxi_TDS/adj_mx.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--early_stop_test_avg_mae_epoch", type=int, default=0)
    parser.add_argument("--early_stop_test_avg_mae_threshold", type=float, default=0.0)
    parser.add_argument("--train_work_per_holiday", type=float, default=2.5)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--result_root", default=os.path.join("experiments", "NYCTaxi_TDS"))
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--resume_reset_patience", type=str2bool, default=True)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--fpem_backbone", choices=["agcrn", "graphwavenet", "staeformer"], default="agcrn")
    parser.add_argument("--agcrn_lr", type=float, default=0.003)
    parser.add_argument("--agcrn_embed_dim", type=int, default=10)
    parser.add_argument("--agcrn_rnn_units", type=int, default=64)
    parser.add_argument("--agcrn_num_layers", type=int, default=2)
    parser.add_argument("--agcrn_cheb_k", type=int, default=2)
    parser.add_argument("--graphwavenet_layers", type=int, default=4)
    parser.add_argument("--graphwavenet_kernel_size", type=int, default=2)
    parser.add_argument("--graphwavenet_dropout", type=float, default=0.1)
    parser.add_argument("--staeformer_layers", type=int, default=2)
    parser.add_argument("--staeformer_heads", type=int, default=4)
    parser.add_argument("--staeformer_dropout", type=float, default=0.1)
    parser.add_argument("--staeformer_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--staeformer_input_length_max", type=int, default=512)
    parser.add_argument("--fpem_use_pretrained_inv_agcrn", type=str2bool, default=False)
    parser.add_argument(
        "--fpem_pretrained_inv_agcrn_path",
        default=os.path.join("experiments", "NYCTaxi_TDS", "pure_agcrn_seed2024", "best_val_model.pth"),
    )
    parser.add_argument("--fpem_use_env_supervision", type=str2bool, default=False)
    parser.add_argument("--fpem_lambda_env_day_cls", type=float, default=0.0)
    parser.add_argument("--fpem_lambda_env_hour_cls", type=float, default=0.0)
    parser.add_argument("--fpem_lambda_env_rush_cls", type=float, default=0.0)
    parser.add_argument("--fpem_env_use_exogenous", type=str2bool, default=True)
    parser.add_argument("--fpem_env_exogenous_hour_dim", type=int, default=8)
    parser.add_argument("--fpem_env_exogenous_day_dim", type=int, default=4)
    parser.add_argument("--fpem_env_exogenous_rush_dim", type=int, default=4)
    parser.add_argument("--fpem_env_exogenous_load_dim", type=int, default=8)
    parser.add_argument("--fpem_env_exogenous_load_num_embeddings", type=int, default=6)
    parser.add_argument("--fpem_env_exogenous_scale", type=float, default=0.1)
    parser.add_argument("--fpem_use_env_supcon", type=str2bool, default=False)
    parser.add_argument("--fpem_lambda_env_supcon", type=float, default=0.0)
    parser.add_argument("--fpem_env_supcon_temperature", type=float, default=0.1)
    parser.add_argument("--fpem_use_inv_projector", type=str2bool, default=False)
    parser.add_argument("--fpem_use_inv_env_adversarial", type=str2bool, default=False)
    parser.add_argument("--fpem_lambda_inv_env_adv", type=float, default=0.0)
    parser.add_argument("--fpem_grl_alpha", type=float, default=1.0)
    parser.add_argument(
        "--fpem_env_route_head_mode",
        choices=[
            "concat_input",
            "hyper_inv_film",
            "hyper_inv_film_proto",
            "hyper_inv_film_proto_concat",
            "hyper_inv_film_proto_input_concat",
            "hyper_inv_film_proto_input_add",
        ],
        default="concat_input",
    )
    parser.add_argument("--fpem_use_env_prototype_router", type=str2bool, default=False)
    parser.add_argument(
        "--fpem_env_route_target_mode",
        choices=["prediction_oracle", "env_prototype", "hybrid"],
        default="prediction_oracle",
    )
    parser.add_argument(
        "--fpem_env_route_train_mode",
        choices=[
            "soft_oracle",
            "gradient_compat_route",
            "hard_prediction_sinkhorn",
            "hard_prediction_environment_sinkhorn",
            "warmup_risk_sinkhorn",
            "progressive_gmm_environment",
        ],
        default="soft_oracle",
    )
    parser.add_argument(
        "--fpem_env_route_inference_mode",
        choices=["mlp", "nearest_prototype", "gaussian", "gaussian_viterbi"],
        default="mlp",
    )
    parser.add_argument("--fpem_env_prototype_temperature", type=float, default=1.0)
    parser.add_argument("--fpem_env_route_hybrid_alpha", type=float, default=1.0)
    parser.add_argument("--fpem_env_route_hybrid_alpha_start", type=float, default=1.0)
    parser.add_argument("--fpem_env_route_hybrid_alpha_end", type=float, default=0.5)
    parser.add_argument("--fpem_env_route_hybrid_alpha_decay_epochs", type=int, default=30)
    parser.add_argument("--fpem_use_sinkhorn_route", type=str2bool, default=False)
    parser.add_argument("--fpem_force_uniform_route", type=str2bool, default=False)
    parser.add_argument("--fpem_env_rep_ablation", choices=["normal", "zero", "shuffle_batch"], default="normal")
    parser.add_argument("--fpem_env_route_sinkhorn_tau", type=float, default=1.0)
    parser.add_argument("--fpem_env_route_sinkhorn_iters", type=int, default=5)
    parser.add_argument("--fpem_env_sinkhorn_prediction_alpha_start", type=float, default=0.2)
    parser.add_argument("--fpem_env_sinkhorn_prediction_alpha_final", type=float, default=1.0)
    parser.add_argument("--fpem_env_sinkhorn_environment_beta_start", type=float, default=1.0)
    parser.add_argument("--fpem_env_sinkhorn_environment_beta_final", type=float, default=0.2)
    parser.add_argument("--fpem_env_sinkhorn_schedule_start_epoch", type=int, default=5)
    parser.add_argument("--fpem_env_sinkhorn_schedule_end_epoch", type=int, default=15)
    parser.add_argument("--fpem_env_sinkhorn_temporal_lambda", type=float, default=0.05)
    parser.add_argument("--fpem_env_sinkhorn_gaussian_ema", type=float, default=0.05)
    parser.add_argument("--fpem_env_sinkhorn_gaussian_min_var", type=float, default=1e-4)
    parser.add_argument("--fpem_sinkhorn_warmup_epochs", type=int, default=10)
    parser.add_argument("--fpem_sinkhorn_soft_end_epoch", type=int, default=20)
    parser.add_argument("--fpem_sinkhorn_temperature_start", type=float, default=1.0)
    parser.add_argument("--fpem_sinkhorn_temperature_final", type=float, default=0.3)
    parser.add_argument("--fpem_sinkhorn_lambda_common", type=float, default=0.1)
    parser.add_argument("--fpem_risk_router_temperature", type=float, default=1.0)
    parser.add_argument("--fpem_risk_router_lambda", type=float, default=0.5)
    parser.add_argument("--fpem_risk_router_pairwise_lambda", type=float, default=0.0)
    parser.add_argument("--fpem_env_teacher_ema_momentum", type=float, default=0.995)
    parser.add_argument("--fpem_env_partition_start_epoch", type=int, default=5)
    parser.add_argument("--fpem_env_partition_update_interval", type=int, default=5)
    parser.add_argument("--fpem_env_partition_freeze_last_epochs", type=int, default=15)
    parser.add_argument("--fpem_env_partition_max_batches", type=int, default=-1)
    parser.add_argument("--fpem_env_max_clusters", type=int, default=6)
    parser.add_argument("--fpem_env_min_cluster_ratio", type=float, default=0.08)
    parser.add_argument("--fpem_env_gmm_n_init", type=int, default=10)
    parser.add_argument("--fpem_env_gmm_variance_floor", type=float, default=1e-4)
    parser.add_argument("--fpem_env_progressive_lambda_common", type=float, default=0.2)
    parser.add_argument("--fpem_env_cluster_compactness_lambda", type=float, default=0.01)
    parser.add_argument("--fpem_env_cluster_consistency_lambda", type=float, default=0.05)
    parser.add_argument("--fpem_env_split_perturb_std", type=float, default=1e-5)
    parser.add_argument("--fpem_sinkhorn_iters", type=int, default=3)
    parser.add_argument("--fpem_sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--fpem_expert_uniform_warmup_epochs", type=int, default=0)
    parser.add_argument("--fpem_env_route_balance_warmup_epochs", type=int, default=0)
    parser.add_argument("--fpem_env_route_initial_temperature", type=float, default=1.0)
    parser.add_argument("--fpem_env_route_final_temperature", type=float, default=0.3)
    parser.add_argument("--fpem_use_cross_cov_sep", type=str2bool, default=False)
    parser.add_argument("--fpem_lambda_cross_cov_sep", type=float, default=0.0)
    args_cli, unknown = parser.parse_known_args()

    project = get_project_path()
    config_path = resolve_path(project, args_cli.config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg.setdefault("ablation", "full")
    cfg.update(vars(args_cli))
    cfg.update(parse_unknown_overrides(unknown))
    if cfg.get("data_seed") is None:
        cfg["data_seed"] = cfg.get("seed", 2024)
    cfg["config_filename"] = config_path
    cfg["data_dir"] = resolve_path(project, cfg["data_dir"])
    cfg["graph_file"] = resolve_path(project, cfg["graph_file"])
    if cfg.get("fpem_pretrained_inv_agcrn_path"):
        cfg["fpem_pretrained_inv_agcrn_path"] = resolve_path(project, cfg["fpem_pretrained_inv_agcrn_path"])
    if cfg["dataset"] == "NYCTaxi_TDS" and not os.path.exists(cfg["graph_file"]):
        cfg["graph_file"] = resolve_path(project, os.path.join("data", "NYCTaxi", "adj_mx.npz"))
    cfg["git_commit"] = git_commit_hash(project)
    if not torch.cuda.is_available():
        cfg["device"] = "cpu"
    run_name = args_cli.exp_name or args_cli.run_name or f"tds_nyctaxi_{cfg['model']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cfg["run_name"] = run_name
    result_root = cfg["result_root"]
    if not os.path.isabs(result_root):
        result_root = os.path.join(project, result_root)
    cfg["result_root"] = result_root
    cfg["log_dir"] = os.path.join(result_root, run_name)
    return SimpleNamespace(**cfg)


if __name__ == "__main__":
    train_one(parse_args())
