#!/usr/bin/env python3
"""Build or validate the shared NYCTaxi observable-load prior cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.fpem.observable_load_prior import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    cache_summary,
    ensure_observable_load_prior_cache,
    validate_observable_load_prior_cache,
)


def load_raw(data_dir):
    raw = {}
    for split in ("train", "val", "test"):
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as data:
            raw[split] = {"x": data["x"].copy()}
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "data" / "NYCTaxi"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT / "data" / "NYCTaxi" / "observable_load_prior_k3_v1.npz",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    raw = load_raw(args.data_dir)
    if args.validate_only:
        payload = validate_observable_load_prior_cache(args.cache, raw)
        payload["created"] = False
    else:
        payload = ensure_observable_load_prior_cache(
            args.cache, raw, random_seed=args.random_seed
        )
    print(json.dumps(cache_summary(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
