"""Atomic, reusable observable-load prior for NYCTaxi.

The prior is derived exclusively from raw historical inputs.  Stored dataset
``c`` and future targets are deliberately absent from this module's API.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


CACHE_VERSION = 1
SPLITS = ("train", "val", "test")
DEFAULT_RANDOM_SEED = 314159


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _raw_x(raw_splits, split):
    value = raw_splits[split]
    x = value["x"] if isinstance(value, dict) else value
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"{split}_x must have shape [S,T,N,F], got {x.shape}")
    return x


def _compute_c_obs(x, capacity):
    latest = np.asarray(x[:, -1], dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    ratio = np.zeros_like(latest, dtype=np.float64)
    np.divide(5.0 * latest, capacity, out=ratio, where=capacity != 0)
    c_obs = np.clip(np.ceil(ratio), 0, 5)
    c_obs[:, capacity == 0] = 0
    return np.ascontiguousarray(c_obs, dtype=np.uint8)


def _metadata(raw_splits, random_seed):
    return {
        "cache_version": CACHE_VERSION,
        "formula": "c_obs=clip(ceil(5*x[:,-1]/CP),0,5); CP=max(train_x,axis=(0,1)); CP==0 -> c_obs=0",
        "load_score": "mean(c_obs/5 over nodes and channels)",
        "threshold_fit_split": "train",
        "quantiles": [1.0 / 3.0, 2.0 / 3.0],
        "random_assignment_seed": int(random_seed),
        "source_x_shapes": {
            split: list(_raw_x(raw_splits, split).shape) for split in SPLITS
        },
        "uses_stored_c": False,
        "uses_target_y": False,
    }


def _compute_payload(raw_splits, random_seed):
    train_x = _raw_x(raw_splits, "train")
    capacity = np.max(np.asarray(train_x, dtype=np.float64), axis=(0, 1))
    payload = {
        "cache_version": np.asarray(CACHE_VERSION, dtype=np.int64),
        "random_assignment_seed": np.asarray(random_seed, dtype=np.int64),
        "CP": np.ascontiguousarray(capacity, dtype=np.float64),
    }
    for split in SPLITS:
        c_obs = _compute_c_obs(_raw_x(raw_splits, split), capacity)
        score = np.mean(c_obs.astype(np.float64) / 5.0, axis=(1, 2))
        payload[f"{split}_c_obs"] = c_obs
        payload[f"{split}_load_score"] = np.ascontiguousarray(score, dtype=np.float64)

    thresholds = np.quantile(
        payload["train_load_score"], [1.0 / 3.0, 2.0 / 3.0]
    ).astype(np.float64)
    payload["thresholds"] = np.ascontiguousarray(thresholds)
    rng = np.random.default_rng(int(random_seed))
    for split in SPLITS:
        expert_id = np.digitize(
            payload[f"{split}_load_score"], thresholds, right=True
        ).astype(np.int64)
        payload[f"{split}_expert_id"] = np.ascontiguousarray(expert_id)
        payload[f"{split}_random_expert_id"] = np.ascontiguousarray(
            rng.permutation(expert_id), dtype=np.int64
        )
    payload["metadata_json"] = np.asarray(
        json.dumps(_metadata(raw_splits, random_seed), sort_keys=True)
    )
    return payload


def _load_and_validate(path, raw_splits):
    required = {
        "cache_version",
        "random_assignment_seed",
        "metadata_json",
        "CP",
        "thresholds",
    }
    for split in SPLITS:
        required.update(
            {
                f"{split}_c_obs",
                f"{split}_load_score",
                f"{split}_expert_id",
                f"{split}_random_expert_id",
            }
        )

    with np.load(path, allow_pickle=False) as cache:
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"observable-load cache is missing keys: {missing}")
        payload = {key: cache[key].copy() for key in required}

    if int(payload["cache_version"]) != CACHE_VERSION:
        raise ValueError(
            f"cache version {int(payload['cache_version'])} != {CACHE_VERSION}"
        )
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("uses_stored_c") is not False or metadata.get("uses_target_y") is not False:
        raise ValueError("observable-load cache metadata permits future leakage")
    expected_shapes = {
        split: tuple(_raw_x(raw_splits, split).shape) for split in SPLITS
    }
    if metadata.get("source_x_shapes") != {
        split: list(shape) for split, shape in expected_shapes.items()
    }:
        raise ValueError("observable-load cache source shapes do not match raw data")

    node_channel_shape = expected_shapes["train"][2:]
    if tuple(payload["CP"].shape) != node_channel_shape:
        raise ValueError(
            f"CP shape {payload['CP'].shape} != {node_channel_shape}"
        )
    thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
    if thresholds.shape != (2,) or bool(np.diff(thresholds).min() < 0):
        raise ValueError(f"invalid K=3 thresholds: {thresholds.tolist()}")
    if not bool(np.isfinite(thresholds).all()):
        raise ValueError("observable-load thresholds are not finite")

    for split, x_shape in expected_shapes.items():
        sample_count, _, node_count, channel_count = x_shape
        c_obs = payload[f"{split}_c_obs"]
        score = np.asarray(payload[f"{split}_load_score"], dtype=np.float64)
        expert_id = np.asarray(payload[f"{split}_expert_id"], dtype=np.int64)
        random_id = np.asarray(
            payload[f"{split}_random_expert_id"], dtype=np.int64
        )
        if tuple(c_obs.shape) != (sample_count, node_count, channel_count):
            raise ValueError(f"{split}_c_obs has invalid shape {c_obs.shape}")
        if score.shape != (sample_count,):
            raise ValueError(f"{split}_load_score has invalid shape {score.shape}")
        if expert_id.shape != (sample_count,) or random_id.shape != (sample_count,):
            raise ValueError(f"{split} expert IDs have invalid shapes")
        if c_obs.dtype != np.uint8 or bool((c_obs > 5).any()):
            raise ValueError(f"{split}_c_obs must be uint8 in [0,5]")
        expected_score = np.mean(c_obs.astype(np.float64) / 5.0, axis=(1, 2))
        if not np.array_equal(score, expected_score):
            raise ValueError(f"{split}_load_score is inconsistent with cached c_obs")
        expected_id = np.digitize(score, thresholds, right=True).astype(np.int64)
        if not np.array_equal(expert_id, expected_id):
            raise ValueError(f"{split}_expert_id is inconsistent with thresholds")
        if bool((expert_id < 0).any()) or bool((expert_id > 2).any()):
            raise ValueError(f"{split}_expert_id is outside [0,2]")
        if not np.array_equal(
            np.bincount(random_id, minlength=3),
            np.bincount(expert_id, minlength=3),
        ):
            raise ValueError(
                f"{split} random assignment does not preserve regime counts"
            )

    payload["metadata"] = metadata
    payload["path"] = str(Path(path).resolve())
    return payload


def ensure_observable_load_prior_cache(
    path, raw_splits, random_seed=DEFAULT_RANDOM_SEED
):
    """Load a valid cache or atomically create it under an exclusive lock."""
    path = Path(path)
    lock_path = Path(str(path) + ".lock")
    with _exclusive_file_lock(lock_path):
        if path.exists():
            payload = _load_and_validate(path, raw_splits)
            payload["created"] = False
            return payload

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _compute_payload(raw_splits, random_seed)
        temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with temp_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        loaded = _load_and_validate(path, raw_splits)
        loaded["created"] = True
        return loaded


def validate_observable_load_prior_cache(path, raw_splits):
    return _load_and_validate(Path(path), raw_splits)


def cache_summary(payload):
    result = {
        "path": payload["path"],
        "cache_version": int(payload["cache_version"]),
        "created": bool(payload.get("created", False)),
        "CP_shape": list(payload["CP"].shape),
        "CP_zero_count": int(np.sum(payload["CP"] == 0)),
        "thresholds": np.asarray(payload["thresholds"]).astype(float).tolist(),
        "random_assignment_seed": int(payload["random_assignment_seed"]),
        "metadata": payload["metadata"],
        "splits": {},
    }
    for split in SPLITS:
        result["splits"][split] = {
            "samples": int(payload[f"{split}_load_score"].shape[0]),
            "c_obs_shape": list(payload[f"{split}_c_obs"].shape),
            "score_min": float(np.min(payload[f"{split}_load_score"])),
            "score_max": float(np.max(payload[f"{split}_load_score"])),
            "expert_counts": np.bincount(
                payload[f"{split}_expert_id"], minlength=3
            ).astype(int).tolist(),
            "random_expert_counts": np.bincount(
                payload[f"{split}_random_expert_id"], minlength=3
            ).astype(int).tolist(),
        }
    return result
