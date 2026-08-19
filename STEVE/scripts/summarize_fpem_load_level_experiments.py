#!/usr/bin/env python3
"""Aggregate the validation-selected FPEM load-level experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DEFAULT_CASES = (
    "k1_environment_expert",
    "k3_unsupervised_experts",
    "k3_load_level_always_env",
    "k3_load_level_gate",
    "k3_random_load_level_gate",
)
METRICS = (
    "best_val_loss",
    "test_mixed_mae",
    "test_workday_mae",
    "test_holiday_mae",
    "test_avg_mae",
)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite metric, got {value!r}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/NYCTaxi_TDS"),
    )
    parser.add_argument(
        "--prefix",
        default="fpem_agcrn_aligned_pretrained_inv_load_level_expert_gate_0727",
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <result-root>/<prefix>_aggregate.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial report instead of failing when a run is missing.",
    )
    args = parser.parse_args()

    cases = comma_list(args.cases)
    seeds = [int(seed) for seed in comma_list(args.seeds)]
    output_dir = args.output_dir or args.result_root / f"{args.prefix}_aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for case in cases:
        for seed in seeds:
            exp_name = f"{args.prefix}_{case}_seed{seed}"
            summary_path = args.result_root / exp_name / "summary.json"
            if not summary_path.is_file():
                missing.append(str(summary_path))
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if not summary.get("finished", False):
                missing.append(f"{summary_path} (finished=false)")
                continue
            row: dict[str, object] = {
                "case": case,
                "seed": seed,
                "exp_name": exp_name,
                "best_epoch": int(summary["best_epoch"]),
                "elapsed_min": finite_float(summary["elapsed_min"]),
                "top_level_prediction_source": summary.get(
                    "top_level_prediction_source", ""
                ),
                "load_level_thresholds": json.dumps(
                    summary.get("load_level_state", {}).get("thresholds", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for metric in METRICS:
                row[metric] = finite_float(summary[metric])
            rows.append(row)

    if missing and not args.allow_incomplete:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"incomplete experiment matrix:\n{joined}")

    run_fields = [
        "case",
        "seed",
        "best_epoch",
        *METRICS,
        "elapsed_min",
        "top_level_prediction_source",
        "load_level_thresholds",
        "exp_name",
    ]
    with (output_dir / "per_seed_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, object]] = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        if not case_rows:
            continue
        aggregate: dict[str, object] = {
            "case": case,
            "completed_seeds": len(case_rows),
            "seeds": ",".join(str(row["seed"]) for row in case_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in case_rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)

    aggregate_fields = [
        "case",
        "completed_seeds",
        "seeds",
        *[
            field
            for metric in METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ],
    ]
    with (output_dir / "aggregate_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    payload = {
        "selection_protocol": "best validation MAE checkpoint; test evaluated once",
        "std_definition": "population standard deviation (ddof=0)",
        "expected_cases": cases,
        "expected_seeds": seeds,
        "complete": not missing,
        "missing": missing,
        "per_seed": rows,
        "aggregate": aggregate_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runs": len(rows),
                "expected_runs": len(cases) * len(seeds),
                "complete": not missing,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
