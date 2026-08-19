#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-basicts}"
fi

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
BASE_EXP_DIR="${BASE_EXP_DIR:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_counterfactual_risk_router_testbest_diagnostic_0802_obs_k1_counterfactual_risk_router_seed2024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/frozen_router_feature_diagnostic_epoch18_testbest_0802}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ROUTER_DROPOUT="${ROUTER_DROPOUT:-0.0}"
WEIGHT_MAX="${WEIGHT_MAX:-20.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
if [ "${#GPU_POOL[@]}" -lt 4 ]; then
  echo "[ERROR] need 4 GPU ids for the 4 frozen-router cases; got GPU_IDS=${GPU_IDS}" >&2
  exit 2
fi

CASES=(
  A_std_delta_mse
  B_std_regret_bce
  C_std_regret_bce_loadstats
  D_std_regret_bce_loadstats_preddiff
)
MSE_WEIGHTS=(1.0 0.0 0.0 0.0)
BCE_WEIGHTS=(0.0 1.0 1.0 1.0)
LOAD_STATS=(false false true true)
PRED_DIFF=(false false false true)

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] BASE_EXP_DIR=${BASE_EXP_DIR}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] cases=${CASES[*]} gpu_ids=${GPU_IDS} seed=${SEED}"

pids=()
for idx in "${!CASES[@]}"; do
  case_name="${CASES[$idx]}"
  gpu="${GPU_POOL[$idx]}"
  log_file="${OUTPUT_ROOT}/${case_name}_seed${SEED}.log"
  echo "[LAUNCH] case=${case_name} gpu=${gpu} log=${log_file}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" scripts/train_tds_nyctaxi_fpem_frozen_router_from_checkpoint.py \
      --base_exp_dir "$BASE_EXP_DIR" \
      --output_root "$OUTPUT_ROOT" \
      --case "$case_name" \
      --device cuda:0 \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --batch_size "$BATCH_SIZE" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$ROUTER_DROPOUT" \
      --lr "$LR" \
      --temperature "$TEMPERATURE" \
      --weight_max "$WEIGHT_MAX" \
      --mse_weight "${MSE_WEIGHTS[$idx]}" \
      --bce_weight "${BCE_WEIGHTS[$idx]}" \
      --include_load_stats "${LOAD_STATS[$idx]}" \
      --standardize_features true \
      --scale_delta_by_train_std true \
      --include_pred_diff "${PRED_DIFF[$idx]}"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = [
    "A_std_delta_mse",
    "B_std_regret_bce",
    "C_std_regret_bce_loadstats",
    "D_std_regret_bce_loadstats_preddiff",
]
rows = []
summaries = {}
for case in cases:
    path = root / f"{case}_seed2024" / "summary.json"
    if not path.exists():
        rows.append({"case": case, "status": "MISSING"})
        continue
    data = json.load(open(path, "r", encoding="utf-8"))
    test = data["test_mixed"]
    train = data["train"]
    row = {
        "case": case,
        "status": "OK",
        "best_stage2_epoch": data["best_stage2_epoch"],
        "router_input_dim": data["router_input_dim"],
        "mse_weight": data["mse_weight"],
        "bce_weight": data["bce_weight"],
        "include_load_stats": data["include_load_stats"],
        "include_pred_diff": data["include_pred_diff"],
        "test_avg_mae": data["test_avg_mae"],
        "all_environment_mae": test["all_environment_mae"],
        "all_invariant_mae": test["all_invariant_mae"],
        "learned_routing_mae": test["learned_routing_mae"],
        "oracle_mae": test["oracle_mae"],
        "env_route_ratio": test["env_route_ratio"],
        "inv_route_ratio": test["inv_route_ratio"],
        "target_inv_ratio": test["target_inv_ratio"],
        "router_accuracy": test["router_accuracy"],
        "balanced_accuracy": test["balanced_accuracy"],
        "inv_switch_precision": test["inv_switch_precision"],
        "inv_switch_recall": test["inv_switch_recall"],
        "correct_beneficial_inv_switches": test["correct_beneficial_inv_switches"],
        "harmful_inv_switches": test["harmful_inv_switches"],
        "saved_loss": test["saved_loss"],
        "added_loss": test["added_loss"],
        "net_gain": test["net_gain"],
        "regret": test["regret"],
        "oracle_gap_closed": test["oracle_gap_closed"],
        "test_delta_hat_min": test["delta_hat_min"],
        "test_delta_hat_mean": test["delta_hat_mean"],
        "test_delta_hat_max": test["delta_hat_max"],
        "test_delta_hat_std": test["delta_hat_std"],
        "test_inv_auroc": test["inv_auroc"],
        "test_inv_auprc": test["inv_auprc"],
        "test_delta_pearson": test["delta_pearson"],
        "test_delta_spearman": test["delta_spearman"],
        "train_inv_route_ratio": train["inv_route_ratio"],
        "train_router_accuracy": train["router_accuracy"],
        "train_balanced_accuracy": train["balanced_accuracy"],
        "train_inv_switch_precision": train["inv_switch_precision"],
        "train_inv_switch_recall": train["inv_switch_recall"],
        "train_net_gain": train["net_gain"],
        "train_inv_auroc": train["inv_auroc"],
        "train_inv_auprc": train["inv_auprc"],
        "train_delta_pearson": train["delta_pearson"],
        "train_delta_spearman": train["delta_spearman"],
        "learned_better_than_all_env": test["learned_routing_mae"] < test["all_environment_mae"],
        "net_gain_positive": test["net_gain"] > 0,
        "over_selects_invariant": test["over_selects_invariant"],
        "hash_identical": data["hash_check"]["non_router_params_and_buffers_hash_identical"],
        "stored_future_c_seen": data["leakage_audit"]["stored_future_c_seen_in_any_split"],
        "future_load_leakage": data["leakage_audit"]["observable_load_prior_uses_target_or_future_load"],
    }
    rows.append(row)
    summaries[case] = data

csv_path = root / "summary_all_frozen_router_cases.csv"
json_path = root / "summary_all_frozen_router_cases.json"
if rows:
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "summaries": summaries}, f, ensure_ascii=False, indent=2)
print(f"[SUMMARY] wrote {csv_path}")
print(f"[SUMMARY] wrote {json_path}")
PY

if [ "$fail" -ne 0 ]; then
  echo "[DONE] FAIL=${fail}" >&2
  exit 1
fi
echo "[DONE] OK"
