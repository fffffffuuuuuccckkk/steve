#!/usr/bin/env python3
"""Summarize NYCTaxi-TDS FPEM input-add module-validity experiments.

This script only reads experiment summaries. It never selects a recipe by test
MAE; aggregate values are descriptive mean/std over finished seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RUN_PREFIX = "fpem_agcrn_aligned_pretrained_inv_input_add_module_validity"
DEFAULT_CASES = [
    "add_single_no_hyper",
    "inv_single_hyper",
    "add_single_hyper",
    "concat_single_hyper",
    "add_k1_plain",
    "add_k3_uniform_plain",
    "add_k3_prediction_router_plain",
    "add_k3_proto_softmax_plain",
    "add_k3_proto_sinkhorn_plain",
    "add_obj_none",
    "add_obj_future",
    "add_obj_swap",
    "add_obj_full",
    "add_full_exogenous_on",
    "add_full_exogenous_off",
    "add_full_env_zero",
    "add_full_env_shuffle",
    "add_full_club_001",
    "add_full_club_01",
    "add_full_k2",
    "add_full_k4",
    "add_full_no_balance",
    "add_full_no_diverse",
    "add_full_no_proto_align",
    "add_full_no_hyper_reg",
    "add_full_no_route_regs",
]

METRIC_FIELDS = [
    "best_epoch",
    "best_val_loss",
    "test_avg_mae",
    "test_mixed_mae",
    "test_mixed_rmse",
    "test_mixed_mape",
    "route_entropy_mean",
    "effective_expert_number",
    "max_expert_usage_ratio",
    "min_expert_usage_ratio",
    "prototype_pairwise_cosine",
    "expert_prediction_pairwise_cosine",
    "hyper_alpha_mean",
    "hyper_delta_norm",
    "loss_future_mi",
    "loss_swap",
    "loss_club_upper",
]

DETAIL_FIELDS = [
    "case",
    "seed",
    "status",
    "run_name",
    "summary_path",
    "route_head_mode",
    "fpem_env_route_target_mode",
    "fpem_force_uniform_route",
    "fpem_env_rep_ablation",
    "fpem_env_use_exogenous",
    "args_fpem_env_route_k",
    "args_fpem_use_env_prototype_router",
    "args_fpem_use_sinkhorn_route",
    "args_fpem_use_future_mi",
    "args_fpem_use_swap",
    "args_fpem_use_club_mi",
    *METRIC_FIELDS,
]


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _finite_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get_nested(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_case_seed(run_name: str, run_prefix: str) -> Optional[tuple[str, int]]:
    pattern = re.escape(run_prefix) + r"_(?P<case>.+)_seed(?P<seed>\d+)$"
    match = re.match(pattern, run_name)
    if match:
        return match.group("case"), int(match.group("seed"))
    generic = re.match(r".*input_add_module_validity_(?P<case>.+)_seed(?P<seed>\d+)$", run_name)
    if generic:
        return generic.group("case"), int(generic.group("seed"))
    return None


def _find_run_dir(result_root: Path, run_prefix: str, case: str, seed: int) -> Optional[Path]:
    exact = result_root / f"{run_prefix}_{case}_seed{seed}"
    if exact.exists():
        return exact
    matches = sorted(result_root.glob(f"*input_add_module_validity*_{case}_seed{seed}"))
    return matches[0] if matches else None


def _extract_detail(
    result_root: Path,
    run_prefix: str,
    case: str,
    seed: int,
    run_dir: Optional[Path],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {field: "" for field in DETAIL_FIELDS}
    row.update({"case": case, "seed": seed, "status": "missing"})
    if run_dir is None:
        return row

    summary_path = run_dir / "summary.json"
    row["run_name"] = run_dir.name
    row["summary_path"] = str(summary_path)
    data = _read_json(summary_path)
    if data is None:
        row["status"] = "incomplete"
        return row

    test_avg = _finite_float(data.get("test_avg_mae"))
    row["status"] = "ok" if bool(data.get("finished")) and test_avg is not None else "incomplete"
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    fpem_logs = data.get("best_val_fpem_logs") if isinstance(data.get("best_val_fpem_logs"), dict) else {}

    simple_keys = [
        "route_head_mode",
        "fpem_env_route_target_mode",
        "fpem_force_uniform_route",
        "fpem_env_rep_ablation",
        "fpem_env_use_exogenous",
        "best_epoch",
        "best_val_loss",
        "test_avg_mae",
        "test_mixed_mae",
        "effective_expert_number",
        "max_expert_usage_ratio",
        "min_expert_usage_ratio",
        "prototype_pairwise_cosine",
        "expert_prediction_pairwise_cosine",
        "hyper_alpha_mean",
        "hyper_delta_norm",
        "loss_future_mi",
        "loss_swap",
    ]
    for key in simple_keys:
        row[key] = data.get(key, "")

    row["test_mixed_rmse"] = (
        _get_nested(data, "final.val.test_mixed.rmse")
        or _get_nested(data, "final.val.test_mixed.RMSE")
        or data.get("test_mixed_rmse", "")
    )
    row["test_mixed_mape"] = (
        _get_nested(data, "final.val.test_mixed.mape")
        or _get_nested(data, "final.val.test_mixed.MAPE")
        or data.get("test_mixed_mape", "")
    )
    row["route_entropy_mean"] = data.get("route_entropy_mean", fpem_logs.get("fpem/route_entropy_mean", ""))
    row["loss_club_upper"] = data.get("loss_club_upper", fpem_logs.get("fpem/club_upper_bound", ""))

    for key in [
        "fpem_env_route_k",
        "fpem_use_env_prototype_router",
        "fpem_use_sinkhorn_route",
        "fpem_use_future_mi",
        "fpem_use_swap",
        "fpem_use_club_mi",
    ]:
        row[f"args_{key}"] = args.get(key, "")
    return row


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _aggregate(rows: List[Dict[str, Any]], cases: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for case in cases:
        case_rows = [r for r in rows if r["case"] == case]
        ok_rows = [r for r in case_rows if r["status"] == "ok"]
        agg: Dict[str, Any] = {
            "case": case,
            "n_ok": len(ok_rows),
            "n_total": len(case_rows),
            "status": "ok" if len(ok_rows) == len(case_rows) and case_rows else "incomplete",
        }
        for metric in METRIC_FIELDS:
            vals = [_finite_float(r.get(metric)) for r in ok_rows]
            vals = [v for v in vals if v is not None]
            agg[f"{metric}_mean"] = statistics.mean(vals) if vals else ""
            agg[f"{metric}_std"] = statistics.stdev(vals) if len(vals) >= 2 else ""
        out.append(agg)
    return out


def _write_markdown(path: Path, detailed_rows: List[Dict[str, Any]], aggregate_rows: List[Dict[str, Any]]) -> None:
    ok_count = sum(1 for r in detailed_rows if r["status"] == "ok")
    total_count = len(detailed_rows)
    lines = [
        "# FPEM input-add module-validity summary",
        "",
        f"- Finished seeds: {ok_count}/{total_count}",
        "- Aggregates are mean/sample-std over finished seeds only.",
        "- This file is descriptive; do not select recipes by test MAE alone.",
        "",
        "## Aggregate by case",
        "",
        "| case | n_ok/n_total | test_avg_mae mean | test_avg_mae std | best_val_loss mean | route entropy mean | effective experts mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        def fmt(key: str) -> str:
            value = row.get(key, "")
            if value == "":
                return ""
            try:
                return f"{float(value):.6g}"
            except Exception:
                return str(value)

        lines.append(
            f"| {row['case']} | {row['n_ok']}/{row['n_total']} | "
            f"{fmt('test_avg_mae_mean')} | {fmt('test_avg_mae_std')} | "
            f"{fmt('best_val_loss_mean')} | {fmt('route_entropy_mean_mean')} | "
            f"{fmt('effective_expert_number_mean')} |"
        )

    missing = [r for r in detailed_rows if r["status"] != "ok"]
    if missing:
        lines.extend(["", "## Missing / incomplete", ""])
        for row in missing:
            lines.append(f"- {row['case']} seed{row['seed']}: {row['status']} ({row.get('run_name') or 'no dir'})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default="/data/OuXiaoyu/STEVE_CODE/STEVE")
    parser.add_argument("--result_root", default="experiments/NYCTaxi_TDS")
    parser.add_argument("--run_prefix", default=os.environ.get("RUN_PREFIX", DEFAULT_RUN_PREFIX))
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", "2024,2025,2026"))
    parser.add_argument("--cases", default=os.environ.get("CASES", "all"))
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = project_dir / result_root

    seeds = [int(x) for x in _parse_csv_list(args.seeds)]
    cases = DEFAULT_CASES if args.cases == "all" else _parse_csv_list(args.cases)

    detailed_rows: List[Dict[str, Any]] = []
    for case in cases:
        for seed in seeds:
            run_dir = _find_run_dir(result_root, args.run_prefix, case, seed)
            detailed_rows.append(_extract_detail(result_root, args.run_prefix, case, seed, run_dir))

    aggregate_rows = _aggregate(detailed_rows, cases)

    detail_path = result_root / "fpem_input_add_module_validity_detailed.csv"
    aggregate_path = result_root / "fpem_input_add_module_validity_aggregate.csv"
    md_path = result_root / "fpem_input_add_module_validity_summary.md"
    agg_fields = ["case", "n_ok", "n_total", "status"]
    for metric in METRIC_FIELDS:
        agg_fields.extend([f"{metric}_mean", f"{metric}_std"])

    _write_csv(detail_path, detailed_rows, DETAIL_FIELDS)
    _write_csv(aggregate_path, aggregate_rows, agg_fields)
    _write_markdown(md_path, detailed_rows, aggregate_rows)

    print(f"[summary] detailed={detail_path}")
    print(f"[summary] aggregate={aggregate_path}")
    print(f"[summary] markdown={md_path}")


if __name__ == "__main__":
    main()
