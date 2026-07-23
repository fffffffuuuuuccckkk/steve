#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
from pathlib import Path


MAIN_ABLATIONS = [
    "full",
    "inv_only",
    "k3",
    "with_inv_loss",
    "no_future_mi",
    "no_swap",
    "no_club",
    "no_confounder_extractor",
]

HYPER_ABLATIONS = [
    "hyper_full",
    "hyper_no_hyper_reg",
    "hyper_no_alpha_gate",
    "hyper_no_swap_fallback",
]

CONFIG_FIELDS = [
    "fpem_env_route_k",
    "fpem_lambda_inv_pred",
    "fpem_use_future_mi",
    "fpem_use_swap",
    "fpem_use_club_mi",
    "fpem_use_confounder_extractor",
    "fpem_env_route_head_mode",
]

COLUMNS = [
    "name",
    "model",
    "seed",
    "finished",
    "best_epoch",
    "best_val_loss",
    "test_mixed_mae",
    "test_workday_mae",
    "test_holiday_mae",
    "test_avg_mae",
] + CONFIG_FIELDS


def project_root():
    return Path(__file__).resolve().parents[1]


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None


def nested(summary, *keys):
    current = summary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def flatten_summary(summary, name, fallback_model="steve", fallback_seed=""):
    row = {
        "name": name,
        "model": fallback_model,
        "seed": fallback_seed,
        "finished": False,
        "best_epoch": "",
        "best_val_loss": "",
        "test_mixed_mae": "",
        "test_workday_mae": "",
        "test_holiday_mae": "",
        "test_avg_mae": "",
        "summary_path": "",
    }
    row.update({field: "" for field in CONFIG_FIELDS})
    if summary is None:
        return row

    args = summary.get("args", {}) if isinstance(summary.get("args"), dict) else {}
    final_val = nested(summary, "final", "val")
    row.update({
        "name": summary.get("ablation", args.get("ablation", name)),
        "model": summary.get("model", args.get("model", fallback_model)),
        "seed": summary.get("seed", args.get("seed", fallback_seed)),
        "finished": summary.get("finished", False),
        "best_epoch": summary.get("best_epoch", nested(summary, "best", "val", "epoch")),
        "best_val_loss": summary.get("best_val_loss", nested(summary, "best", "val", "mae")),
        "test_mixed_mae": summary.get("test_mixed_mae"),
        "test_workday_mae": summary.get("test_workday_mae"),
        "test_holiday_mae": summary.get("test_holiday_mae"),
        "test_avg_mae": summary.get("test_avg_mae"),
    })
    for field in CONFIG_FIELDS:
        row[field] = summary.get(field, args.get(field, ""))

    if isinstance(final_val, dict):
        fallbacks = {
            "test_mixed_mae": nested(final_val, "test_mixed", "mae"),
            "test_workday_mae": nested(final_val, "test_workday", "mae"),
            "test_holiday_mae": nested(final_val, "test_holiday", "mae"),
            "test_avg_mae": final_val.get("test_avg_mae"),
        }
        for field, value in fallbacks.items():
            if row[field] in (None, ""):
                row[field] = value

    if row["test_avg_mae"] in (None, ""):
        workday = row["test_workday_mae"]
        holiday = row["test_holiday_mae"]
        if workday not in (None, "") and holiday not in (None, ""):
            try:
                row["test_avg_mae"] = (float(workday) + float(holiday)) / 2.0
            except (TypeError, ValueError):
                pass
    return row


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else ""
    return str(value)


def markdown_table(rows):
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in COLUMNS) + " |")
    return "\n".join(lines)


def collect_rows(result_root, run_prefix, ablations, seed):
    rows = []
    for name in ablations:
        run_name = f"{run_prefix}_{name}_seed{seed}"
        path = result_root / run_name / "summary.json"
        row = flatten_summary(read_json(path), name, fallback_seed=seed)
        row["summary_path"] = str(path) if path.exists() else ""
        rows.append(row)
    return rows


def write_outputs(rows, csv_path, markdown_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=COLUMNS + ["summary_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS + ["summary_path"]})
    table = markdown_table(rows)
    with markdown_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser()
    root = project_root()
    parser.add_argument("--result_root", default=str(root / "experiments" / "NYCTaxi_TDS"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--run_prefix",
        default=os.environ.get("RUN_PREFIX", "fpem_agcrn_aligned_k1_noinv"),
    )
    parser.add_argument("--baseline_summary", default=None)
    parser.add_argument("--csv_out", default=None)
    parser.add_argument("--md_out", default=None)
    parser.add_argument("--include_hyper", action="store_true")
    parser.add_argument("--hyper_csv_out", default=None)
    parser.add_argument("--hyper_md_out", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    prefix_tag = args.run_prefix.replace("/", "_")
    baseline_path = (
        Path(args.baseline_summary)
        if args.baseline_summary
        else result_root / "agcrn_tds_nyctaxi_20260617" / "summary.json"
    )
    baseline = flatten_summary(
        read_json(baseline_path), "baseline", fallback_model="agcrn", fallback_seed=args.seed
    )
    baseline["summary_path"] = str(baseline_path) if baseline_path.exists() else ""

    main_rows = [baseline] + collect_rows(
        result_root, args.run_prefix, MAIN_ABLATIONS, args.seed
    )
    csv_path = Path(args.csv_out) if args.csv_out else result_root / f"{prefix_tag}_summary.csv"
    md_path = Path(args.md_out) if args.md_out else result_root / f"{prefix_tag}_summary.md"
    main_table = write_outputs(main_rows, csv_path, md_path)
    print("MAIN ALIGNED ABLATIONS")
    print(main_table)
    print(f"\nCSV: {csv_path}\nMarkdown: {md_path}")

    if args.include_hyper:
        hyper_rows = collect_rows(
            result_root, args.run_prefix, HYPER_ABLATIONS, args.seed
        )
        hyper_csv = (
            Path(args.hyper_csv_out)
            if args.hyper_csv_out
            else result_root / f"{prefix_tag}_hyper_summary.csv"
        )
        hyper_md = (
            Path(args.hyper_md_out)
            if args.hyper_md_out
            else result_root / f"{prefix_tag}_hyper_summary.md"
        )
        hyper_table = write_outputs(hyper_rows, hyper_csv, hyper_md)
        print("\nHYPER ABLATIONS (compare only against hyper_full)")
        print(hyper_table)
        print(f"\nCSV: {hyper_csv}\nMarkdown: {hyper_md}")


if __name__ == "__main__":
    main()
