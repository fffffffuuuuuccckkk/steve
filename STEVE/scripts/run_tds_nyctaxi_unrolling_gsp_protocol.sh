#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
fi
cd "$PROJECT_DIR"

export PYTHON="${PYTHON:-python}"
export RUN_PREFIX="${RUN_PREFIX:-unrolling_gsp_nyctaxi_tds_protocol}"
export RESULT_ROOT="${RESULT_ROOT:-experiments/NYCTaxi_TDS}"
export LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_logs}"
export SEEDS="${SEEDS:-2024,2025,2026}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"
export MAX_EPOCH="${MAX_EPOCH:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export RESUME="${RESUME:-true}"
export MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:--1}"
export MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:--1}"
export UNROLLING_LR="${UNROLLING_LR:-0.0005}"
export UNROLLING_PATIENCE="${UNROLLING_PATIENCE:-20}"
export UNROLLING_BLOCKS="${UNROLLING_BLOCKS:-5}"
export UNROLLING_LAYERS="${UNROLLING_LAYERS:-25}"
export UNROLLING_CG_ITERS="${UNROLLING_CG_ITERS:-3}"
export UNROLLING_PGD_ITERS="${UNROLLING_PGD_ITERS:-3}"
export UNROLLING_HEADS="${UNROLLING_HEADS:-4}"
export UNROLLING_FEATURE_CHANNELS="${UNROLLING_FEATURE_CHANNELS:-6}"
export UNROLLING_NEIGHBORS="${UNROLLING_NEIGHBORS:-6}"
export UNROLLING_INTERVAL="${UNROLLING_INTERVAL:-6}"
export UNROLLING_USE_ONE_CHANNEL="${UNROLLING_USE_ONE_CHANNEL:-false}"
export UNROLLING_LOSS_SCOPE="${UNROLLING_LOSS_SCOPE:-full}"

mkdir -p "$LOG_ROOT"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

completed_summary_valid() {
  local summary_json="$1"
  [ -f "$summary_json" ] || return 1
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
try:
    data=json.load(open(sys.argv[1]))
    ok=bool(data.get("finished")) and math.isfinite(float(data.get("test_avg_mae")))
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

summary_metric_values() {
  local summary_json="${1:-}"
  if [ -z "$summary_json" ] || [ ! -f "$summary_json" ]; then
    printf 'NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA'
    return
  fi
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
data=json.load(open(sys.argv[1]))
keys=["best_epoch","best_val_loss","test_avg_mae","test_mixed_mae","test_avg_rmse","test_avg_mape","trainable_params","elapsed_seconds"]
def fmt(v):
    if v is None: return "NA"
    if isinstance(v,float) and (math.isnan(v) or math.isinf(v)): return "NA"
    return str(v)
print("\t".join(fmt(data.get(k)) for k in keys), end="")
PY
}

record_summary() {
  local seed="$1"; shift
  local status="$1"; shift
  local run_name="$1"; shift
  local detail="${1:-}"; shift || true
  local summary_json="${1:-}"
  (
    flock 8
    if [ ! -s "$SUMMARY_FILE" ]; then
      printf 'seed\tstatus\trun_name\tbest_epoch\tbest_val_loss\ttest_avg_mae\ttest_mixed_mae\ttest_avg_rmse\ttest_avg_mape\ttrainable_params\telapsed_seconds\tdetail\n' > "$SUMMARY_FILE"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$status" "$run_name" "$(summary_metric_values "$summary_json")" "$detail" >> "$SUMMARY_FILE"
  ) 8>"${LOG_ROOT}/summary.lock"
}

run_one() {
  local gpu_id="$1"; shift
  local seed="$1"; shift
  local run_name="${RUN_PREFIX}_seed${seed}"
  local exp_dir="${PROJECT_DIR}/${RESULT_ROOT}/${run_name}"
  local summary_json="${exp_dir}/summary.json"
  local log_file="${LOG_ROOT}/${run_name}.log"

  if completed_summary_valid "$summary_json"; then
    echo "[SKIP] seed=$seed run=$run_name"
    record_summary "$seed" "SKIP_DONE" "$run_name" "completed summary exists" "$summary_json"
    return 0
  fi

  local -a cmd=(
    "$PYTHON" scripts/run_tds_nyctaxi_unrolling_gsp_protocol.py
    --dataset NYCTaxi_TDS
    --data_dir data
    --graph_file data/NYCTaxi_TDS/adj_mx.npz
    --result_root "$RESULT_ROOT"
    --exp_name "$run_name"
    --seed "$seed"
    --device cuda:0
    --epochs "$MAX_EPOCH"
    --batch_size "$BATCH_SIZE"
    --test_batch_size "$TEST_BATCH_SIZE"
    --lr "$UNROLLING_LR"
    --patience "$UNROLLING_PATIENCE"
    --resume "$RESUME"
    --max_train_batches "$MAX_TRAIN_BATCHES"
    --max_eval_batches "$MAX_EVAL_BATCHES"
    --blocks "$UNROLLING_BLOCKS"
    --layers "$UNROLLING_LAYERS"
    --cg_iters "$UNROLLING_CG_ITERS"
    --pgd_iters "$UNROLLING_PGD_ITERS"
    --heads "$UNROLLING_HEADS"
    --feature_channels "$UNROLLING_FEATURE_CHANNELS"
    --neighbors "$UNROLLING_NEIGHBORS"
    --interval "$UNROLLING_INTERVAL"
    --use_one_channel "$UNROLLING_USE_ONE_CHANNEL"
    --loss_scope "$UNROLLING_LOSS_SCOPE"
  )

  echo "[LAUNCH] seed=$seed gpu=$gpu_id run=$run_name"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id" > "${LOG_ROOT}/${run_name}.cmd"
  printf '%q ' "${cmd[@]}" >> "${LOG_ROOT}/${run_name}.cmd"
  printf '\n' >> "${LOG_ROOT}/${run_name}.cmd"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] && completed_summary_valid "$summary_json"; then
    record_summary "$seed" "OK" "$run_name" "rc=0" "$summary_json"
    return 0
  fi
  record_summary "$seed" "FAIL" "$run_name" "rc=$rc" "$summary_json"
  return "$rc"
}

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"

echo "[INFO] RUN_PREFIX=$RUN_PREFIX seeds=$SEEDS gpu_ids=$GPU_IDS max_parallel=$MAX_PARALLEL"

job_index=0
running=0
for seed in "${SEED_LIST[@]}"; do
  gpu="${GPU_POOL[$((job_index % ${#GPU_POOL[@]}))]}"
  run_one "$gpu" "$seed" &
  running=$((running + 1))
  job_index=$((job_index + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n || true
    running=$((running - 1))
  fi
done
while [ "$running" -gt 0 ]; do
  wait -n || true
  running=$((running - 1))
done

ok=$(awk -F'\t' 'NR>1 && $2=="OK"{c++} END{print c+0}' "$SUMMARY_FILE" 2>/dev/null || echo 0)
skip=$(awk -F'\t' 'NR>1 && $2=="SKIP_DONE"{c++} END{print c+0}' "$SUMMARY_FILE" 2>/dev/null || echo 0)
fail=$(awk -F'\t' 'NR>1 && $2=="FAIL"{c++} END{print c+0}' "$SUMMARY_FILE" 2>/dev/null || echo 0)
echo "[DONE] OK=$ok SKIP_DONE=$skip FAIL=$fail summary=$SUMMARY_FILE"
