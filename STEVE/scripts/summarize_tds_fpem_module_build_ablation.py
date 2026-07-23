#!/usr/bin/env python3
"""Summarize FPEM module build-up, leave-one-out, cumulative, and hyper runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any


GROUP_ORDER = {"loo": 0, "bu": 1, "cum": 2, "hyper": 3}

CONFIG_FIELDS = [
    "fpem_env_route_k",
    "fpem_lambda_inv_pred",
    "fpem_use_env_route",
    "fpem_use_env_fusion",
    "fpem_use_confounder_extractor",
    "fpem_use_club_mi",
    "fpem_lambda_club_mi",
    "fpem_use_future_mi",
    "fpem_lambda_future_mi",
    "fpem_use_swap",
    "fpem_lambda_swap",
    "fpem_env_route_head_mode",
    "fpem_use_grad_consensus",
    "fpem_use_pretrained_inv_agcrn",
    "fpem_pretrained_inv_agcrn_path",
    "batch_size",
    "test_batch_size",
]

COLUMNS = [
    "group",
    "name",
    "seed",
    "finished",
    "best_epoch",
    "best_val_loss",
    "test_mixed_mae",
    "test_workday_mae",
    "test_holiday_mae",
    "test_avg_mae",
] + CONFIG_FIELDS

AGGREGATE_COLUMNS = [
    "group",
    "name",
    "num_finished",
    "mean_test_avg_mae",
    "std_test_avg_mae",
    "mean_test_workday_mae",
    "mean_test_holiday_mae",
]

NA = "NA"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return NA


def nested(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def group_for_name(name: str) -> str:
    group = name.split("_", 1)[0]
    return group if group in GROUP_ORDER else "other"


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_runs(result_root: Path, run_prefix: str) -> list[tuple[str, str]]:
    pattern = re.compile(rf"^{re.escape(run_prefix)}_(.+)_seed([0-9]+)$")
    discovered: list[tuple[str, str]] = []
    if not result_root.is_dir():
        return discovered
    for path in result_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            discovered.append((match.group(1), match.group(2)))
    return discovered


def build_targets(
    result_root: Path,
    run_prefix: str,
    names: list[str],
    seeds: list[str],
) -> list[tuple[str, str]]:
    discovered = discover_runs(result_root, run_prefix)
    if names and seeds:
        targets = [(name, seed) for name in names for seed in seeds]
        known = set(targets)
        targets.extend(item for item in discovered if item not in known)
        return targets
    if discovered:
        return discovered
    return [(name, seed) for name in names for seed in seeds]


def flatten_run(result_root: Path, run_prefix: str, name: str, seed: str) -> dict[str, Any]:
    run_name = f"{run_prefix}_{name}_seed{seed}"
    run_dir = result_root / run_name
    summary = read_json(run_dir / "summary.json")
    launch = read_json(run_dir / "launch_config.json") or {}
    args = summary.get("args", {}) if summary and isinstance(summary.get("args"), dict) else {}

    row: dict[str, Any] = {
        "group": group_for_name(name),
        "name": name,
        "seed": first_present(summary.get("seed") if summary else None, launch.get("seed"), seed),
        "finished": first_present(summary.get("finished") if summary else None, False),
        "best_epoch": first_present(summary.get("best_epoch") if summary else None),
        "best_val_loss": first_present(
            summary.get("best_val_loss") if summary else None,
            nested(summary, "best", "val", "mae"),
        ),
        "test_mixed_mae": first_present(
            summary.get("test_mixed_mae") if summary else None,
            nested(summary, "final", "val", "test_mixed", "mae"),
        ),
        "test_workday_mae": first_present(
            summary.get("test_workday_mae") if summary else None,
            nested(summary, "final", "val", "test_workday", "mae"),
        ),
        "test_holiday_mae": first_present(
            summary.get("test_holiday_mae") if summary else None,
            nested(summary, "final", "val", "test_holiday", "mae"),
        ),
        "test_avg_mae": first_present(
            summary.get("test_avg_mae") if summary else None,
            nested(summary, "final", "val", "test_avg_mae"),
        ),
    }
    for field in CONFIG_FIELDS:
        row[field] = first_present(
            summary.get(field) if summary else None,
            args.get(field),
            launch.get(field),
        )
    return row


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else NA
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, NA)) for column in columns) + " |")
    return "\n".join(lines)


def write_outputs(
    rows: list[dict[str, Any]],
    columns: list[str],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column, NA)) for column in columns})
    with markdown_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(markdown_table(rows, columns) + "\n")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["group"]), str(row["name"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (group, name), group_rows in grouped.items():
        finished = [row for row in group_rows if row.get("finished") is True]

        def values(field: str) -> list[float]:
            result = [as_float(row.get(field)) for row in finished]
            return [value for value in result if value is not None]

        avg_values = values("test_avg_mae")
        workday_values = values("test_workday_mae")
        holiday_values = values("test_holiday_mae")
        aggregates.append({
            "group": group,
            "name": name,
            "num_finished": len(finished),
            "mean_test_avg_mae": statistics.fmean(avg_values) if avg_values else NA,
            "std_test_avg_mae": statistics.pstdev(avg_values) if avg_values else NA,
            "mean_test_workday_mae": statistics.fmean(workday_values) if workday_values else NA,
            "mean_test_holiday_mae": statistics.fmean(holiday_values) if holiday_values else NA,
        })
    return sorted(
        aggregates,
        key=lambda row: (GROUP_ORDER.get(str(row["group"]), 99), str(row["name"])),
    )


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        default=str(root / "experiments" / "NYCTaxi_TDS"),
    )
    parser.add_argument(
        "--run_prefix",
        default=os.environ.get("RUN_PREFIX", "fpem_agcrn_aligned_module_build"),
    )
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", "2024,2025,2026"))
    parser.add_argument("--names", default=None)
    parser.add_argument("--csv_out", default=None)
    parser.add_argument("--md_out", default=None)
    parser.add_argument("--aggregate_csv_out", default=None)
    parser.add_argument("--aggregate_md_out", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    names = parse_csv_list(args.names)
    seeds = parse_csv_list(args.seeds)
    targets = build_targets(result_root, args.run_prefix, names, seeds)
    name_order = {name: index for index, name in enumerate(names)}
    rows = [flatten_run(result_root, args.run_prefix, name, seed) for name, seed in targets]
    rows.sort(key=lambda row: (
        GROUP_ORDER.get(str(row["group"]), 99),
        name_order.get(str(row["name"]), 9999),
        str(row["name"]),
        int(row["seed"]) if str(row["seed"]).isdigit() else str(row["seed"]),
    ))
    aggregates = aggregate_rows(rows)

    csv_path = Path(args.csv_out) if args.csv_out else result_root / "fpem_agcrn_aligned_module_build_summary.csv"
    md_path = Path(args.md_out) if args.md_out else result_root / "fpem_agcrn_aligned_module_build_summary.md"
    aggregate_csv = (
        Path(args.aggregate_csv_out)
        if args.aggregate_csv_out
        else result_root / "fpem_agcrn_aligned_module_build_aggregate.csv"
    )
    aggregate_md = (
        Path(args.aggregate_md_out)
        if args.aggregate_md_out
        else result_root / "fpem_agcrn_aligned_module_build_aggregate.md"
    )

    write_outputs(rows, COLUMNS, csv_path, md_path)
    write_outputs(aggregates, AGGREGATE_COLUMNS, aggregate_csv, aggregate_md)
    print(markdown_table(rows, COLUMNS))
    print("\nAGGREGATE ACROSS SEEDS")
    print(markdown_table(aggregates, AGGREGATE_COLUMNS))
    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Aggregate Markdown: {aggregate_md}")


if __name__ == "__main__":
    main()
