from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


class StandardScaler:
    """Dataset scaler fitted on training data only."""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 0 else 1.0

    @classmethod
    def fit(cls, data: np.ndarray) -> "StandardScaler":
        return cls(float(np.nanmean(data)), float(np.nanstd(data)))

    def transform_np(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32, copy=False)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - data.new_tensor(self.mean)) / data.new_tensor(self.std)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * data.new_tensor(self.std) + data.new_tensor(self.mean)

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "fit_split": "train"}


def _valid_mask(target: torch.Tensor, mask_value: Optional[float]) -> torch.Tensor:
    if mask_value is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        mask = target > target.new_tensor(float(mask_value))
    return mask & torch.isfinite(target)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask_value: Optional[float] = 5.0) -> torch.Tensor:
    mask = _valid_mask(target, mask_value)
    if not bool(mask.any()):
        return torch.mean(torch.abs(pred - target))
    return torch.mean(torch.abs(pred[mask] - target[mask]))


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask_value: Optional[float] = 5.0) -> torch.Tensor:
    mask = _valid_mask(target, mask_value)
    if not bool(mask.any()):
        return torch.sqrt(torch.mean((pred - target).pow(2)))
    return torch.sqrt(torch.mean((pred[mask] - target[mask]).pow(2)))


def masked_mape(pred: torch.Tensor, target: torch.Tensor, mask_value: Optional[float] = 5.0, eps: float = 1.0e-5) -> torch.Tensor:
    mask = _valid_mask(target, mask_value)
    denom = target.abs().clamp_min(float(eps))
    value = torch.abs((pred - target) / denom)
    if not bool(mask.any()):
        return torch.mean(value)
    return torch.mean(value[mask])


@dataclass
class MetricAccumulator:
    mask_value: Optional[float] = 5.0

    def __post_init__(self) -> None:
        self.count = 0
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        bsz = int(pred.shape[0])
        self.count += bsz
        self.mae_sum += float(masked_mae(pred, target, self.mask_value).detach().cpu()) * bsz
        self.rmse_sum += float(masked_rmse(pred, target, self.mask_value).detach().cpu()) * bsz
        self.mape_sum += float(masked_mape(pred, target, self.mask_value).detach().cpu()) * bsz

    def compute(self) -> Dict[str, float]:
        denom = max(self.count, 1)
        return {
            "mae": self.mae_sum / denom,
            "rmse": self.rmse_sum / denom,
            "mape": self.mape_sum / denom,
        }

