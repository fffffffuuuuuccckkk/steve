from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metric_adapter import StandardScaler


@dataclass
class DatasetBundle:
    loaders: Dict[str, DataLoader]
    scaler: StandardScaler
    meta: Dict
    shapes: Dict[str, Dict[str, tuple]]


class NpzForecastDataset(Dataset):
    """Forecasting dataset backed by pre-built split npz files.

    The dataset exposes normalized ``x`` and ``y`` tensors shaped
    ``[T,N,C]`` and ``[H,N,C]``.  Optional fields such as ``time_label`` and
    ``c`` are returned for baselines that explicitly need them, but baseline
    models must opt into using them in their own adapter/wrapper.
    """

    def __init__(self, npz_path: str, scaler: StandardScaler, transform_y: bool = True):
        self.npz_path = npz_path
        pack = np.load(npz_path, allow_pickle=True)
        self.x = scaler.transform_np(pack["x"].astype(np.float32, copy=False))
        self.y = (
            scaler.transform_np(pack["y"].astype(np.float32, copy=False))
            if transform_y
            else pack["y"].astype(np.float32, copy=False)
        )
        self.time_label = pack["time_label"].astype(np.int64, copy=False) if "time_label" in pack.files else None
        self.c = pack["c"].astype(np.float32, copy=False) if "c" in pack.files else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.from_numpy(self.y[idx]),
        }
        if self.time_label is not None:
            item["time_label"] = torch.tensor(int(self.time_label[idx]), dtype=torch.long)
        if self.c is not None:
            item["c"] = torch.from_numpy(self.c[idx])
        return item


def load_meta(dataset_dir: str) -> Dict:
    meta_path = os.path.join(dataset_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_train_scaler(dataset_dir: str) -> StandardScaler:
    train = np.load(os.path.join(dataset_dir, "train.npz"), allow_pickle=True)
    return StandardScaler.fit(train["x"].astype(np.float32, copy=False))


def make_largest_npz_loaders(
    dataset_dir: str,
    batch_size: int,
    test_batch_size: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 2024,
    pin_memory: bool = False,
) -> DatasetBundle:
    scaler = fit_train_scaler(dataset_dir)
    test_batch_size = int(test_batch_size or batch_size)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    datasets = {
        split: NpzForecastDataset(os.path.join(dataset_dir, f"{split}.npz"), scaler)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            drop_last=False,
        ),
    }
    shapes = {
        split: {
            "x": tuple(datasets[split].x.shape),
            "y": tuple(datasets[split].y.shape),
        }
        for split in ("train", "val", "test")
    }
    return DatasetBundle(loaders=loaders, scaler=scaler, meta=load_meta(dataset_dir), shapes=shapes)

