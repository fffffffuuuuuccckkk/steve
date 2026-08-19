#!/usr/bin/env python
"""Prepare external datasets into the STEVE TDS npz format.

Output format per dataset directory:

    train.npz / val.npz / test.npz with keys:
        x          [num_samples, input_len, num_nodes, d_input]
        y          [num_samples, horizon,   num_nodes, d_output]
        time_label [num_samples]  # weekday: hour, weekend: 24 + hour
        c          [num_samples, horizon,   num_nodes, d_output]
    adj_mx.npz with key:
        adj_mx     [num_nodes, num_nodes]
    meta.json

The default split policy follows the requested OOD protocol:

* LargeST-SD/GBA:
    2019 -> Train
    2020-01-01 .. 2020-06-30 23:xx -> Val
    2020-07-01 .. 2020-12-31 23:xx -> OOD Test
* KnowAir:
    2015-2016 -> Train (the provided files start at 2016-01-01)
    2017 -> Val
    2018 -> OOD Test

This script is intentionally conservative: if LargeST 2019/2020 raw files are
not present, it reports the missing files instead of silently using 2017 data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover - message quality matters here
        raise RuntimeError(
            f"Missing dependency '{module_name}'. {install_hint}"
        ) from exc


def time_label_from_timestamps(times: Sequence[pd.Timestamp]) -> np.ndarray:
    labels = []
    for t in pd.DatetimeIndex(times):
        hour = int(t.hour)
        labels.append(hour if int(t.dayofweek) < 5 else 24 + hour)
    return np.asarray(labels, dtype=np.int64)


def split_indices_by_latest_and_target_end(
    times: pd.DatetimeIndex,
    input_len: int,
    horizon: int,
    start: str,
    end: str,
) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    latest_start = input_len - 1
    latest_end_exclusive = len(times) - horizon
    if latest_end_exclusive <= latest_start:
        return np.zeros((0,), dtype=np.int64)
    latest_idx = np.arange(latest_start, latest_end_exclusive, dtype=np.int64)
    latest_time = times[latest_idx]
    target_end_time = times[latest_idx + horizon]
    keep = (latest_time >= start_ts) & (target_end_time <= end_ts)
    return latest_idx[np.asarray(keep)]


def make_windows(
    data: np.ndarray,
    times: pd.DatetimeIndex,
    latest_indices: np.ndarray,
    input_len: int,
    horizon: int,
    train_load_thresholds: Optional[np.ndarray] = None,
    c_levels: int = 6,
) -> Dict[str, np.ndarray]:
    """Create TDS windows.

    ``data`` is [T,N,C]. ``latest_indices`` points to the latest input time.
    """

    if len(latest_indices) == 0:
        raise ValueError("split produced zero samples")
    x = np.empty((len(latest_indices), input_len, data.shape[1], data.shape[2]), dtype=np.float32)
    y = np.empty((len(latest_indices), horizon, data.shape[1], data.shape[2]), dtype=np.float32)
    for i, t in enumerate(latest_indices):
        x[i] = data[t - input_len + 1 : t + 1]
        y[i] = data[t + 1 : t + 1 + horizon]
    labels = time_label_from_timestamps(times[latest_indices])
    # c is deliberately based on latest historical x only, not future y.
    latest_score = x[:, -1].mean(axis=(1, 2), dtype=np.float64)
    if train_load_thresholds is None:
        quantiles = np.arange(1, c_levels, dtype=np.float64) / float(c_levels)
        train_load_thresholds = np.quantile(latest_score, quantiles)
    levels = np.digitize(latest_score, train_load_thresholds, right=True).astype(np.float32)
    c = np.repeat(levels[:, None, None, None], horizon, axis=1)
    c = np.repeat(c, data.shape[1], axis=2)
    c = np.repeat(c, data.shape[2], axis=3)
    return {
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "time_label": np.ascontiguousarray(labels),
        "c": np.ascontiguousarray(c.astype(np.float32)),
        "_latest_load_score": latest_score,
        "_thresholds": np.asarray(train_load_thresholds, dtype=np.float64),
    }


def save_split(path: Path, pack: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        x=pack["x"],
        y=pack["y"],
        time_label=pack["time_label"],
        c=pack["c"],
    )


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371.0088
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def distance_adjacency(lon: np.ndarray, lat: np.ndarray, topk: int = 10) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    n = len(lon)
    dist = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    nonzero = dist[dist > 0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    adj = np.exp(-np.square(dist / max(sigma, 1.0e-6))).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    if topk > 0 and topk < n:
        keep = np.zeros_like(adj, dtype=bool)
        for i in range(n):
            idx = np.argsort(dist[i])[: topk + 1]
            keep[i, idx] = True
        keep = keep | keep.T
        adj = np.where(keep, adj, 0.0).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
    return adj


def write_meta(out_dir: Path, payload: Dict) -> None:
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_dataset_config(
    dataset_name: str,
    num_nodes: int,
    d_input: int,
    d_output: int,
    input_len: int,
    horizon: int,
) -> Path:
    """Write a lightweight config so the dataset can be selected by scripts.

    ``run_tds_nyctaxi.py`` historically has argparse defaults for dataset and
    graph_file, so launch scripts should still pass ``--dataset`` and
    ``--graph_file`` explicitly. This config records the dataset-specific
    dimensions and gives a canonical path for future launchers.
    """

    cfg_dir = PROJECT_ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace("-", "_")
    cfg_path = cfg_dir / f"{safe_name}.yaml"
    graph_file = f"data/{dataset_name}/adj_mx.npz"
    text = f"""# Auto-generated by scripts/prepare_tds_external_datasets.py
## global
seed: 31
device: cuda:0
mode: train
best_path: ""
debug: False

## data
dataset: {dataset_name}
data_dir: data
graph_file: {graph_file}
num_nodes: {num_nodes}
input_length: {input_len}
output_T_dim: {horizon}
batch_size: 64
test_batch_size: 64
len_closeness: 8
len_period: 27
len_trend: 0
result_root: experiments/{dataset_name}

## model
d_input: {d_input}
d_output: {d_output}
d_model: 64
dropout: 0.1
percent: 0.1
shm_temp: 0.5
nmb_prototype: 4
yita: 0.5
layers: 3
K: 64
bank_gamma: 0.2
kw: 1
mi_w: 2

## train
epochs: 100
lr_init: 0.001
early_stop: True
early_stop_patience: 50
grad_norm: True
max_grad_norm: 5
use_dwa: True
temp: 2
MMI: False
lr_patience: 20
lr_mode: only
use_RevIN: True

## latent-confounder structural regularization
confounder_dep_mode: none
confounder_extractor: token_attention
confounder_dim: 8
confounder_num_heads: 4
confounder_attention_dropout: 0.0
confounder_graph_embed_dim: 16
confounder_variational: True
confounder_kl_weight: 0.0001
confounder_dep_warmup_epochs: 5
confounder_dep_ramp_epochs: 10
confounder_projection_ridge: 0.001
confounder_dep_detach_target: True
gci_weight: 0.001
gci_graph_align_weight: 0.1
gci_edge_preserve_beta: 0.1
gci_graph_hops: 1
gci_symmetrize_adj: True
gci_add_self_loops: True
scd_weight: 0.001
dep_eps: 1.0e-6
confounder_injection: none
"""
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def prepare_knowair_region(
    region: str,
    input_len: int,
    horizon: int,
    out_name: Optional[str],
    topk: int,
) -> Path:
    netCDF4 = _import_or_raise(
        "netCDF4",
        "Install with: python -m pip install --only-binary=:all: netCDF4",
    )
    raw_root = DATA_ROOT / "kown_air"
    nc_path = raw_root / f"dataset_{region}.nc"
    station_path = raw_root / f"stations_{region}.csv"
    if not nc_path.exists() or not station_path.exists():
        raise FileNotFoundError(f"KnowAir files not found: {nc_path}, {station_path}")
    out_dir = DATA_ROOT / (out_name or f"KnowAir-{region.upper()}_TDS")
    with netCDF4.Dataset(str(nc_path), "r") as ds:
        time_var = ds.variables["time"]
        units = getattr(time_var, "units", "hours since 2016-01-01")
        times = pd.DatetimeIndex(netCDF4.num2date(time_var[:], units=units, only_use_cftime_datetimes=False))
        channels = []
        channel_names = ["PM2.5", "O3"]
        for var in channel_names:
            arr = np.asarray(ds.variables[var][:], dtype=np.float32)
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            channels.append(arr)
        data = np.stack(channels, axis=-1).astype(np.float32)  # [T,N,2]
    if np.isnan(data).any():
        # Simple deterministic imputation: station/channel median, then global zero fallback.
        med = np.nanmedian(data, axis=0)
        global_med = float(np.nanmedian(data)) if np.isfinite(np.nanmedian(data)) else 0.0
        med = np.nan_to_num(med, nan=global_med).astype(np.float32)
        for node_idx in range(data.shape[1]):
            for channel_idx in range(data.shape[2]):
                mask = np.isnan(data[:, node_idx, channel_idx])
                if mask.any():
                    data[mask, node_idx, channel_idx] = med[node_idx, channel_idx]
        data = np.nan_to_num(data, nan=global_med).astype(np.float32)
    train_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2015-01-01 00:00:00", "2016-12-31 23:59:59"
    )
    val_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2017-01-01 00:00:00", "2017-12-31 23:59:59"
    )
    test_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2018-01-01 00:00:00", "2018-12-31 23:59:59"
    )
    train = make_windows(data, times, train_idx, input_len, horizon)
    val = make_windows(data, times, val_idx, input_len, horizon, train["_thresholds"])
    test = make_windows(data, times, test_idx, input_len, horizon, train["_thresholds"])
    for split, pack in [("train", train), ("val", val), ("test", test)]:
        save_split(out_dir / f"{split}.npz", pack)
    stations = pd.read_csv(station_path)
    adj = distance_adjacency(stations["lon"].values, stations["lat"].values, topk=topk)
    np.savez_compressed(out_dir / "adj_mx.npz", adj_mx=adj)
    meta = {
        "dataset": out_dir.name,
        "source": str(nc_path),
        "station_file": str(station_path),
        "region": region,
        "channels": channel_names,
        "input_len": input_len,
        "horizon": horizon,
        "num_nodes": int(data.shape[1]),
        "d_input": int(data.shape[2]),
        "d_output": int(data.shape[2]),
        "time_range": [str(times[0]), str(times[-1])],
        "requested_split": {
            "train": "2015-2016; available data starts at 2016-01-01",
            "val": "2017",
            "test_ood": "2018",
        },
        "samples": {
            "train": int(len(train["x"])),
            "val": int(len(val["x"])),
            "test": int(len(test["x"])),
        },
        "c_source": "latest historical x load quantile; no future y/c leakage",
        "load_thresholds": train["_thresholds"].astype(float).tolist(),
        "adjacency": f"haversine Gaussian topk={topk}, symmetrized",
    }
    write_meta(out_dir, meta)
    cfg_path = write_dataset_config(
        out_dir.name,
        num_nodes=int(data.shape[1]),
        d_input=int(data.shape[2]),
        d_output=int(data.shape[2]),
        input_len=input_len,
        horizon=horizon,
    )
    with open(out_dir / "config_path.txt", "w", encoding="utf-8") as f:
        f.write(str(cfg_path) + "\n")
    return out_dir


def _read_largest_hdf(path: Path) -> pd.DataFrame:
    # LargeST official generator uses pandas.read_hdf. Keep this separate so
    # missing PyTables gives a clear error message.
    try:
        return pd.read_hdf(str(path))
    except ImportError as exc:
        raise RuntimeError(
            "Reading LargeST .h5 requires PyTables. Install with: python -m pip install tables"
        ) from exc


def _extract_zip_if_needed(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    target = Path(str(path)[: -len(".zip")])
    if target.exists() and target.stat().st_size > 0:
        return target
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"LargeST archive is not a valid zip: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        h5_members = [m for m in zf.namelist() if m.lower().endswith((".h5", ".npy"))]
        if len(h5_members) != 1:
            raise RuntimeError(f"Expected exactly one h5/npy member in {path}, got {h5_members}")
        member = h5_members[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        with zf.open(member, "r") as src, open(tmp_target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024 * 64)
                if not chunk:
                    break
                dst.write(chunk)
        os.replace(tmp_target, target)
    return target


def _candidate_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return _extract_zip_if_needed(p)
    return None


def _largest_subset_meta(dataset: str) -> pd.DataFrame:
    meta_path = DATA_ROOT / "LargeST" / "ca_meta.csv"
    if not meta_path.exists():
        # Official repository layout fallback.
        meta_path = DATA_ROOT / "LargeST" / "data" / "ca" / "ca_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError("LargeST ca_meta.csv not found")
    meta = pd.read_csv(meta_path)
    district = {"sd": 11, "gba": 4}[dataset]
    sub = meta[meta["District"] == district].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"LargeST subset {dataset} produced zero sensors from ca_meta.csv")
    return sub


def _read_largest_subset_year(dataset: str, year: int) -> Tuple[pd.DataFrame, str]:
    dataset = dataset.lower()
    subset_candidates = [
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset}_his_{year}.h5",
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset.upper()}_his_{year}.h5",
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset}_his_{year}.h5.zip",
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset.upper()}_his_{year}.h5.zip",
        DATA_ROOT / "LargeST" / f"{dataset}_his_{year}.h5",
        DATA_ROOT / "LargeST" / f"{dataset.upper()}_his_{year}.h5",
        DATA_ROOT / "LargeST" / f"{dataset}_his_{year}.h5.zip",
        DATA_ROOT / "LargeST" / f"{dataset.upper()}_his_{year}.h5.zip",
    ]
    subset_path = _candidate_existing(subset_candidates)
    if subset_path is not None:
        df = _read_largest_hdf(subset_path)
        return df, str(subset_path)

    ca_candidates = [
        DATA_ROOT / "LargeST" / f"ca_his_{year}.h5",
        DATA_ROOT / "LargeST" / f"ca_his_raw_{year}.h5",
        DATA_ROOT / "LargeST" / f"ca_his_{year}.h5.zip",
        DATA_ROOT / "LargeST" / f"ca_his_raw_{year}.h5.zip",
        DATA_ROOT / "LargeST" / "data" / "ca" / f"ca_his_{year}.h5",
        DATA_ROOT / "LargeST" / "data" / "ca" / f"ca_his_raw_{year}.h5",
        DATA_ROOT / "LargeST" / "data" / "ca" / f"ca_his_{year}.h5.zip",
        DATA_ROOT / "LargeST" / "data" / "ca" / f"ca_his_raw_{year}.h5.zip",
    ]
    ca_path = _candidate_existing(ca_candidates)
    if ca_path is not None:
        df = _read_largest_hdf(ca_path)
        if "raw" in ca_path.name:
            # Match the official process_ca_his.ipynb: 15-minute resampling,
            # rounded flow counts, deterministic missing-value handling.
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.resample("15min").mean().round(0).fillna(0)
        meta = _largest_subset_meta(dataset)
        cols = [str(x) for x in meta["ID"].values.tolist()]
        df.columns = [str(c) for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"LargeST CA history {ca_path} is missing {len(missing)} {dataset.upper()} sensors; "
                f"first missing IDs: {missing[:10]}"
            )
        return df.loc[:, cols], f"{ca_path} filtered by District={int(meta['District'].iloc[0])}"
    available = _describe_largest_available()
    raise FileNotFoundError(
        "Missing LargeST raw/processed h5 for "
        f"dataset={dataset}, year={year}. Expected one of:\n"
        + "\n".join(str(p) for p in subset_candidates + ca_candidates)
        + "\n\nCurrently visible LargeST files:\n"
        + available
    )


def _describe_largest_available() -> str:
    root = DATA_ROOT / "LargeST"
    if not root.exists():
        return f"{root} does not exist"
    lines: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if not any(token in p.name.lower() for token in ["2017", "2018", "2019", "2020", ".h5", ".zip", "adj"]):
            continue
        rel = p.relative_to(root)
        desc = f"{rel} ({p.stat().st_size} bytes)"
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    members = ", ".join(info.filename for info in zf.infolist()[:8])
                    desc += f" contains: {members}"
            except Exception as exc:
                desc += f" zip-list-error={exc}"
        lines.append(desc)
    return "\n".join(lines[:80]) if lines else "no matching LargeST h5/zip/adj files found"


def prepare_largest_subset(dataset: str, input_len: int, horizon: int, out_name: Optional[str]) -> Path:
    dataset = dataset.lower()
    if dataset not in {"sd", "gba"}:
        raise ValueError("LargeST subset must be 'sd' or 'gba'")
    out_dir = DATA_ROOT / (out_name or f"LargeST-{dataset.upper()}_TDS")
    df_2019, source_2019 = _read_largest_subset_year(dataset, 2019)
    df_2020, source_2020 = _read_largest_subset_year(dataset, 2020)
    df = pd.concat([df_2019, df_2020], axis=0)
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    data = df.values.astype(np.float32)[:, :, None]
    times = pd.DatetimeIndex(df.index)
    train_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2019-01-01 00:00:00", "2019-12-31 23:59:59"
    )
    val_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2020-01-01 00:00:00", "2020-06-30 23:59:59"
    )
    test_idx = split_indices_by_latest_and_target_end(
        times, input_len, horizon, "2020-07-01 00:00:00", "2020-12-31 23:59:59"
    )
    train = make_windows(data, times, train_idx, input_len, horizon)
    val = make_windows(data, times, val_idx, input_len, horizon, train["_thresholds"])
    test = make_windows(data, times, test_idx, input_len, horizon, train["_thresholds"])
    for split, pack in [("train", train), ("val", val), ("test", test)]:
        save_split(out_dir / f"{split}.npz", pack)
    # Prefer official subset adjacency; otherwise cut the CA road-network
    # adjacency by the same ID2 indices as the official notebooks.
    n = data.shape[1]
    adj = None
    subset_adj_candidates = [
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset}_rn_adj.npy",
        DATA_ROOT / "LargeST" / "data" / dataset / f"{dataset}_rn_adj.npy.zip",
        DATA_ROOT / "LargeST" / f"{dataset}_rn_adj.npy",
        DATA_ROOT / "LargeST" / f"{dataset}_rn_adj.npy.zip",
    ]
    ca_adj_candidates = [
        DATA_ROOT / "LargeST" / "ca_rn_adj.npy",
        DATA_ROOT / "LargeST" / "ca_rn_adj.npy.zip",
        DATA_ROOT / "LargeST" / "data" / "ca" / "ca_rn_adj.npy",
        DATA_ROOT / "LargeST" / "data" / "ca" / "ca_rn_adj.npy.zip",
    ]
    adj_source = None
    for p0 in subset_adj_candidates:
        if not p0.exists():
            continue
        p = _extract_zip_if_needed(p0)
        try:
            cand = np.load(str(p), allow_pickle=True)
            cand = np.asarray(cand, dtype=np.float32)
            if cand.shape == (n, n):
                adj = cand
                adj_source = str(p)
                break
        except Exception:
            continue
    if adj is None:
        for p0 in ca_adj_candidates:
            if not p0.exists():
                continue
            p = _extract_zip_if_needed(p0)
            cand = np.asarray(np.load(str(p), allow_pickle=True), dtype=np.float32)
            meta = _largest_subset_meta(dataset)
            id2 = meta["ID2"].astype(int).values
            adj = cand[id2][:, id2].astype(np.float32)
            adj_source = f"{p} filtered by {dataset.upper()} ID2"
            break
    if adj is None:
        adj = np.eye(n, dtype=np.float32)
        adj_source = "identity fallback; official LargeST adjacency not found"
    np.savez_compressed(out_dir / "adj_mx.npz", adj_mx=adj)
    write_meta(
        out_dir,
        {
            "dataset": out_dir.name,
            "source": [source_2019, source_2020],
            "subset": dataset,
            "channels": ["flow"],
            "input_len": input_len,
            "horizon": horizon,
            "num_nodes": int(n),
            "d_input": 1,
            "d_output": 1,
            "requested_split": {
                "train": "2019",
                "val": "2020 first half",
                "test_ood": "2020 second half",
            },
            "samples": {
                "train": int(len(train["x"])),
                "val": int(len(val["x"])),
                "test": int(len(test["x"])),
            },
            "c_source": "latest historical x load quantile; no future y/c leakage",
            "load_thresholds": train["_thresholds"].astype(float).tolist(),
            "adjacency": adj_source,
        },
    )
    cfg_path = write_dataset_config(
        out_dir.name,
        num_nodes=int(n),
        d_input=1,
        d_output=1,
        input_len=input_len,
        horizon=horizon,
    )
    with open(out_dir / "config_path.txt", "w", encoding="utf-8") as f:
        f.write(str(cfg_path) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        default="knowair_yrd,knowair_bthsa,largest_sd,largest_gba",
        help="Comma-separated: knowair_yrd, knowair_bthsa, largest_sd, largest_gba",
    )
    ap.add_argument("--input_len", type=int, default=35)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--knowair_topk", type=int, default=10)
    ap.add_argument("--skip_missing", action="store_true")
    args = ap.parse_args()
    done = []
    failed = {}
    for item in [x.strip().lower() for x in args.datasets.split(",") if x.strip()]:
        try:
            if item == "knowair_yrd":
                done.append(str(prepare_knowair_region("yrd", args.input_len, args.horizon, None, args.knowair_topk)))
            elif item == "knowair_bthsa":
                done.append(str(prepare_knowair_region("bthsa", args.input_len, args.horizon, None, args.knowair_topk)))
            elif item == "largest_sd":
                done.append(str(prepare_largest_subset("sd", args.input_len, args.horizon, None)))
            elif item == "largest_gba":
                done.append(str(prepare_largest_subset("gba", args.input_len, args.horizon, None)))
            else:
                raise ValueError(f"unknown dataset key: {item}")
        except Exception as exc:
            if not args.skip_missing:
                raise
            failed[item] = str(exc)
    print(json.dumps({"generated": done, "failed": failed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
