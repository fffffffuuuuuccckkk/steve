#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="/data/OuXiaoyu/STEVE_CODE/STEVE"
source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
if ! conda activate "$CONDA_ENV"; then
  if [ "$CONDA_ENV" = "basicts" ]; then
    echo "[WARN] conda env 'basicts' is unavailable; falling back to 'tslib'." >&2
    conda activate tslib
  else
    exit 1
  fi
fi
cd "$PROJECT_DIR" || exit 1

export PYTHON=${PYTHON:-python}
export RUN_PREFIX=${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_best_recipe}
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export MAX_PARALLEL=${MAX_PARALLEL:-4}
export SEEDS=${SEEDS:-2024,2025,2026}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-16}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-16}
export FPEM_USE_GRAD_CONSENSUS=${FPEM_USE_GRAD_CONSENSUS:-false}
export FPEM_USE_PRETRAINED_INV_AGCRN=${FPEM_USE_PRETRAINED_INV_AGCRN:-true}
export FPEM_PRETRAINED_INV_AGCRN_PATH=${FPEM_PRETRAINED_INV_AGCRN_PATH:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}
export PLAN_ONLY=${PLAN_ONLY:-false}
export MAX_RETRY=${MAX_RETRY:-2}
export RETRY_SLEEP=${RETRY_SLEEP:-180}
export MEMORY_RETRY_FOREVER=${MEMORY_RETRY_FOREVER:-false}
export GPU_MAX_USED_MB=${GPU_MAX_USED_MB:-1024}
export GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
export FORCE=${FORCE:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}

RESULT_ROOT=${RESULT_ROOT:-experiments/NYCTaxi_TDS}
LOG_ROOT=${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_logs}
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
mkdir -p "$LOG_ROOT"

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if truthy "$FORCE"; then
  echo "[ERROR] FORCE=true is intentionally unsupported: this script never deletes experiment directories." >&2
  echo "[ERROR] Use a new RUN_PREFIX when a clean rerun is required." >&2
  exit 2
fi

if [ ! -e data/NYCTaxi_TDS ]; then
  ln -s NYCTaxi data/NYCTaxi_TDS
fi

if truthy "$FPEM_USE_PRETRAINED_INV_AGCRN" && [ ! -f "$FPEM_PRETRAINED_INV_AGCRN_PATH" ]; then
  echo "[ERROR] pretrained AGCRN checkpoint not found: $FPEM_PRETRAINED_INV_AGCRN_PATH" >&2
  echo "[ERROR] Run pure_agcrn_seed2024 first, or set FPEM_PRETRAINED_INV_AGCRN_PATH=/path/to/best_val_model.pth" >&2
  exit 2
fi

# Prevent two launchers with the same prefix from scheduling duplicate jobs.
exec 9>"${LOG_ROOT}/scheduler.lock"
if ! flock -n 9; then
  echo "[ERROR] another launcher is already using RUN_PREFIX=${RUN_PREFIX}" >&2
  exit 2
fi

cleanup_memory() {
  local gpu_id="$1"
  echo "[CLEANUP] gpu=$gpu_id date=$(date)"
  nvidia-smi -i "$gpu_id" || true
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" - <<'PY' || true
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception as exc:
    print("cuda cleanup skipped:", exc)
PY
  nvidia-smi -i "$gpu_id" || true
}

is_memory_error() {
  local attempt_log="$1"
  local exit_code="${2:-0}"
  if [ "$exit_code" -eq 132 ] || [ "$exit_code" -eq 134 ] || \
     [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 139 ]; then
    return 0
  fi
  grep -Eqi "CUDA out of memory|out of memory|CUDNN_STATUS_ALLOC_FAILED|cublas.*alloc|NCCL|Killed|bus error|illegal instruction|SIGILL|aborted|SIGABRT|illegal memory access|cuda runtime error|segmentation fault|segfault|SIGSEGV" "$attempt_log"
}

group_for_name() {
  printf '%s' "${1%%_*}"
}

record_summary() {
  local name="$1"
  local seed="$2"
  local status="$3"
  local run_name="$4"
  local detail="${5:-}"
  local group tmp_file
  group="$(group_for_name "$name")"
  tmp_file="${SUMMARY_FILE}.tmp.$$.${RANDOM}"
  (
    flock -x 8
    if [ -s "$SUMMARY_FILE" ]; then
      awk -F '\t' -v group="$group" -v name="$name" -v seed="$seed" \
        'NR == 1 || !(($1 == group) && ($2 == name) && ($3 == seed))' \
        "$SUMMARY_FILE" > "$tmp_file"
    else
      printf 'group\tname\tseed\tstatus\trun_name\tdetail\n' > "$tmp_file"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$group" "$name" "$seed" "$status" "$run_name" "$detail" >> "$tmp_file"
    mv "$tmp_file" "$SUMMARY_FILE"
  ) 8>"${SUMMARY_FILE}.lock"
}

# Stable aligned full configuration. concat_input does not use the hyper
# fallback expert; fallback-router supervision is enabled only in hyper_* jobs.
BASE_ARGS=(
  --config_filename configs/NYCTaxi.yaml
  --dataset NYCTaxi_TDS
  --data_dir data
  --graph_file data/NYCTaxi_TDS/adj_mx.npz
  --model steve
  --epochs "$MAX_EPOCH"
  --batch_size "$BATCH_SIZE"
  --test_batch_size "$TEST_BATCH_SIZE"
  --device cuda:0
  --train_work_per_holiday 2.5
  --result_root "$RESULT_ROOT"
  --resume true
  --resume_reset_patience true
  --early_stop_test_avg_mae_epoch 40
  --early_stop_test_avg_mae_threshold 12

  --fpem_backbone agcrn
  --fpem_use_pretrained_inv_agcrn "$FPEM_USE_PRETRAINED_INV_AGCRN"
  --fpem_pretrained_inv_agcrn_path "$FPEM_PRETRAINED_INV_AGCRN_PATH"
  --agcrn_embed_dim 10
  --agcrn_num_layers 2
  --agcrn_cheb_k 2

  --fpem_use_env_mask true
  --fpem_confounder_use_mask false
  --fpem_env_mask_hidden_dim 64
  --fpem_env_mask_temperature 1.0
  --fpem_env_mask_warmup_epochs 5
  --fpem_lambda_mask_sparse 0.0005
  --fpem_lambda_mask_entropy 0.0005

  --fpem_use_env_route true
  --fpem_use_env_fusion true
  --fpem_env_route_k 1
  --fpem_env_route_head_mode concat_input
  --fpem_env_route_mode confidence_mix
  --fpem_env_route_use_inv_fallback_expert false
  --fpem_env_route_tau 1.0
  --fpem_env_route_oracle_tau 0.3
  --fpem_env_route_train_mode soft_oracle
  --fpem_env_route_hidden_dim 64
  --fpem_env_route_warmup_epochs 5
  --fpem_env_route_lambda_final 1.0
  --fpem_env_route_lambda_global 0.0
  --fpem_env_route_lambda_route_soft 0.5
  --fpem_env_route_lambda_expert 0.2
  --fpem_env_route_lambda_router_oracle 1.0
  --fpem_env_route_lambda_balance 0.0
  --fpem_env_route_lambda_diverse 0.0
  --fpem_env_route_lambda_entropy 0.0

  --fpem_lambda_inv_pred 0.0
  --fpem_hyper_alpha_mode sample_gate
  --fpem_lambda_hyper_delta_norm 0.0001

  --fpem_use_confounder_extractor true
  --fpem_use_club_mi true
  --fpem_lambda_club_mi 0.01

  --fpem_use_future_mi true
  --fpem_lambda_future_mi 0.02
  --fpem_future_mi_target_mode env_encoder
  --fpem_future_mi_warmup_epochs 5
  --fpem_future_mi_hidden_dim 64
  --fpem_future_mi_detach_target true

  --fpem_use_swap true
  --fpem_lambda_swap 0.01
  --fpem_swap_warmup_epochs 30
  --fpem_swap_margin 0.01
  --fpem_swap_gain_eta 0.0
  --fpem_swap_gain_tau 0.05
  --fpem_lambda_swap_diff 1.0
  --fpem_lambda_swap_same 0.05
  --fpem_swap_only_diff_route true
  --fpem_swap_detach_inv true
  --fpem_swap_detach_env false
  --fpem_use_swap_fallback_router_loss false
  --fpem_lambda_swap_fallback_router 0.0
  --fpem_swap_fallback_warmup_epochs 30

  --fpem_use_grad_consensus "$FPEM_USE_GRAD_CONSENSUS"
  --fpem_gc_pred_loss_only true
  --fpem_gc_inv_rho 0.3
  --fpem_gc_env_rho 0.3
  --fpem_gc_tau 0.5
  --fpem_gc_temp 0.1
  --fpem_gc_min_keep 0.2
  --fpem_gc_warmup_epochs 10
  --fpem_gc_route_min_samples 2
  --fpem_use_gradcompat_aux false
  --fpem_lambda_gradcompat_aux 0.0
)

BEST_RECIPE_NAMES=(
  no_conf_k3_mask_on
  no_conf_k3_mask_off
  no_conf_k3_no_mask_no_swap_no_club
  no_conf_k3_no_mask_no_swap_no_club_no_future
  no_conf_k2_no_mask_no_swap_no_club
  no_conf_k4_no_mask_no_swap_no_club
)

# Return the last override for --key, matching parse_unknown_overrides behavior.
effective_value() {
  local key="$1"
  local value="$2"
  shift 2
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--${key}" ] && [ "$#" -ge 2 ]; then
      value="$2"
      shift 2
    else
      shift
    fi
  done
  printf '%s' "$value"
}

write_launch_config() {
  local path="$1"
  local name="$2"
  local seed="$3"
  local gpu="$4"
  local run_name="$5"
  local env_route="$6"
  local env_fusion="$7"
  local route_k="$8"
  local inv_loss="$9"
  shift 9
  local confounder="$1" club="$2" club_lambda="$3" future_mi="$4" future_lambda="$5"
  local swap="$6" swap_lambda="$7" head_mode="$8" grad_consensus="$9"
  local pretrained_inv_agcrn="${10}" pretrained_inv_agcrn_path="${11}"
  "$PYTHON" - "$path" "$name" "$seed" "$gpu" "$run_name" \
    "$env_route" "$env_fusion" "$route_k" "$inv_loss" "$confounder" \
    "$club" "$club_lambda" "$future_mi" "$future_lambda" "$swap" \
    "$swap_lambda" "$head_mode" "$grad_consensus" "$pretrained_inv_agcrn" \
    "$pretrained_inv_agcrn_path" "$BATCH_SIZE" "$TEST_BATCH_SIZE" <<'PY'
import json
import sys
from datetime import datetime, timezone

def parse(value):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

keys = [
    "name", "seed", "gpu", "run_name", "fpem_use_env_route",
    "fpem_use_env_fusion", "fpem_env_route_k", "fpem_lambda_inv_pred",
    "fpem_use_confounder_extractor", "fpem_use_club_mi", "fpem_lambda_club_mi",
    "fpem_use_future_mi", "fpem_lambda_future_mi", "fpem_use_swap",
    "fpem_lambda_swap", "fpem_env_route_head_mode", "fpem_use_grad_consensus",
    "fpem_use_pretrained_inv_agcrn", "fpem_pretrained_inv_agcrn_path",
    "batch_size", "test_batch_size",
]
data = {key: parse(value) for key, value in zip(keys, sys.argv[2:])}
data["updated_at"] = datetime.now(timezone.utc).isoformat()
with open(sys.argv[1], "w", encoding="utf-8") as file_obj:
    json.dump(data, file_obj, ensure_ascii=False, indent=2)
PY
}

run_one() {
  local gpu_id="$1"
  local name="$2"
  local seed="$3"
  shift 3
  local extra_args=("$@")
  local run_name="${RUN_PREFIX}_${name}_seed${seed}"
  local exp_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${exp_dir}/launcher.log"
  local config_file="${exp_dir}/launch_config.json"
  local attempt=0 exit_code attempt_log start_time end_time
  local env_route env_fusion route_k inv_loss confounder club club_lambda
  local future_mi future_lambda swap swap_lambda head_mode grad_consensus
  local pretrained_inv_agcrn pretrained_inv_agcrn_path

  mkdir -p "$exp_dir"
  env_route="$(effective_value fpem_use_env_route true "${extra_args[@]}")"
  env_fusion="$(effective_value fpem_use_env_fusion true "${extra_args[@]}")"
  route_k="$(effective_value fpem_env_route_k 1 "${extra_args[@]}")"
  inv_loss="$(effective_value fpem_lambda_inv_pred 0.0 "${extra_args[@]}")"
  confounder="$(effective_value fpem_use_confounder_extractor true "${extra_args[@]}")"
  club="$(effective_value fpem_use_club_mi true "${extra_args[@]}")"
  club_lambda="$(effective_value fpem_lambda_club_mi 0.01 "${extra_args[@]}")"
  future_mi="$(effective_value fpem_use_future_mi true "${extra_args[@]}")"
  future_lambda="$(effective_value fpem_lambda_future_mi 0.02 "${extra_args[@]}")"
  swap="$(effective_value fpem_use_swap true "${extra_args[@]}")"
  swap_lambda="$(effective_value fpem_lambda_swap 0.01 "${extra_args[@]}")"
  head_mode="$(effective_value fpem_env_route_head_mode concat_input "${extra_args[@]}")"
  grad_consensus="$(effective_value fpem_use_grad_consensus "$FPEM_USE_GRAD_CONSENSUS" "${extra_args[@]}")"
  pretrained_inv_agcrn="$(effective_value fpem_use_pretrained_inv_agcrn "$FPEM_USE_PRETRAINED_INV_AGCRN" "${extra_args[@]}")"
  pretrained_inv_agcrn_path="$(effective_value fpem_pretrained_inv_agcrn_path "$FPEM_PRETRAINED_INV_AGCRN_PATH" "${extra_args[@]}")"

  write_launch_config "$config_file" "$name" "$seed" "$gpu_id" "$run_name" \
    "$env_route" "$env_fusion" "$route_k" "$inv_loss" "$confounder" \
    "$club" "$club_lambda" "$future_mi" "$future_lambda" "$swap" \
    "$swap_lambda" "$head_mode" "$grad_consensus" "$pretrained_inv_agcrn" \
    "$pretrained_inv_agcrn_path"

  {
    echo "============================================================"
    echo "[EXPERIMENT] name=$name seed=$seed gpu=$gpu_id run_name=$run_name"
    echo "[MODULES] fpem_use_env_route=$env_route fpem_use_env_fusion=$env_fusion fpem_env_route_k=$route_k"
    echo "[MODULES] fpem_lambda_inv_pred=$inv_loss fpem_use_confounder_extractor=$confounder"
    echo "[MODULES] fpem_use_club_mi=$club fpem_use_future_mi=$future_mi fpem_use_swap=$swap"
    echo "[MODULES] fpem_env_route_head_mode=$head_mode fpem_use_grad_consensus=$grad_consensus"
    echo "[PRETRAIN] fpem_use_pretrained_inv_agcrn=$pretrained_inv_agcrn path=$pretrained_inv_agcrn_path"
    echo "[LOG] $log_file"
    echo "============================================================"
  } | tee -a "$log_file"

  if [ -f "${exp_dir}/summary.json" ]; then
    if "$PYTHON" - "${exp_dir}/summary.json" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if summary.get("finished") is True else 1)
PY
    then
      echo "[SKIP_DONE] ${run_name}" | tee -a "$log_file"
      record_summary "$name" "$seed" "OK" "$run_name" "existing_summary=${exp_dir}/summary.json"
      return 0
    fi
  fi

  while true; do
    attempt_log="${exp_dir}/attempt_${attempt}.log"
    : > "$attempt_log"
    start_time="$(date --iso-8601=seconds)"
    local cmd=(
      "$PYTHON" run_tds_nyctaxi.py
      "${BASE_ARGS[@]}"
      --seed "$seed"
      --exp_name "$run_name"
      --ablation "$name"
      "${extra_args[@]}"
    )
    {
      echo "============================================================"
      echo "[START] $start_time"
      echo "[ATTEMPT] $attempt"
      printf '[COMMAND] CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      echo "============================================================"
    } | tee -a "$log_file" "$attempt_log"

    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee -a "$log_file" "$attempt_log"
    exit_code=${PIPESTATUS[0]}
    end_time="$(date --iso-8601=seconds)"
    echo "[END] $end_time exit_code=$exit_code" | tee -a "$log_file" "$attempt_log"

    if [ "$exit_code" -eq 0 ]; then
      echo "[OK] $name seed=$seed" | tee -a "$log_file"
      record_summary "$name" "$seed" "OK" "$run_name"
      return 0
    fi

    echo "[FAIL] $name seed=$seed exit_code=$exit_code" | tee -a "$log_file"
    if is_memory_error "$attempt_log" "$exit_code"; then
      if truthy "$MEMORY_RETRY_FOREVER" || [ "$attempt" -lt "$MAX_RETRY" ]; then
        echo "[MEMORY_ERROR] wait ${RETRY_SLEEP}s, cleanup, then resume the same experiment" | tee -a "$log_file"
        sleep "$RETRY_SLEEP"
        cleanup_memory "$gpu_id" 2>&1 | tee -a "$log_file"
        attempt=$((attempt + 1))
        continue
      fi
      record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code};memory_error=max_retry"
      return "$exit_code"
    fi

    record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code}"
    return "$exit_code"
  done
}

run_named_ablation() {
  local gpu_id="$1" name="$2" seed="$3"
  case "$name" in
    no_conf_k3_mask_on) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask true \
      --fpem_env_route_k 3 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    no_conf_k3_mask_off) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask false \
      --fpem_lambda_mask_sparse 0.0 \
      --fpem_lambda_mask_entropy 0.0 \
      --fpem_env_route_k 3 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    no_conf_k3_no_mask_no_swap_no_club) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask false \
      --fpem_lambda_mask_sparse 0.0 \
      --fpem_lambda_mask_entropy 0.0 \
      --fpem_use_swap false \
      --fpem_lambda_swap 0.0 \
      --fpem_use_club_mi false \
      --fpem_lambda_club_mi 0.0 \
      --fpem_env_route_k 3 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    no_conf_k3_no_mask_no_swap_no_club_no_future) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask false \
      --fpem_lambda_mask_sparse 0.0 \
      --fpem_lambda_mask_entropy 0.0 \
      --fpem_use_swap false \
      --fpem_lambda_swap 0.0 \
      --fpem_use_club_mi false \
      --fpem_lambda_club_mi 0.0 \
      --fpem_use_future_mi false \
      --fpem_lambda_future_mi 0.0 \
      --fpem_env_route_k 3 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    no_conf_k2_no_mask_no_swap_no_club) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask false \
      --fpem_lambda_mask_sparse 0.0 \
      --fpem_lambda_mask_entropy 0.0 \
      --fpem_use_swap false \
      --fpem_lambda_swap 0.0 \
      --fpem_use_club_mi false \
      --fpem_lambda_club_mi 0.0 \
      --fpem_env_route_k 2 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    no_conf_k4_no_mask_no_swap_no_club) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_confounder_extractor false \
      --fpem_use_env_mask false \
      --fpem_lambda_mask_sparse 0.0 \
      --fpem_lambda_mask_entropy 0.0 \
      --fpem_use_swap false \
      --fpem_lambda_swap 0.0 \
      --fpem_use_club_mi false \
      --fpem_lambda_club_mi 0.0 \
      --fpem_env_route_k 4 \
      --fpem_env_route_lambda_balance 0.01 \
      --fpem_env_route_lambda_diverse 0.01 ;;
    *) echo "[ERROR] unknown ablation: $name" >&2; return 2 ;;
  esac
}

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
if [ "${#SEED_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] SEEDS is empty" >&2
  exit 2
fi
for seed in "${SEED_ARRAY[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid seed in SEEDS: $seed" >&2
    exit 2
  fi
done

JOB_NAMES=()
JOB_SEEDS=()
ENABLED_NAMES=()
add_group_jobs() {
  local name seed
  for name in "$@"; do
    ENABLED_NAMES+=("$name")
    for seed in "${SEED_ARRAY[@]}"; do
      JOB_NAMES+=("$name")
      JOB_SEEDS+=("$seed")
    done
  done
}

add_group_jobs "${BEST_RECIPE_NAMES[@]}"

TOTAL_JOBS=${#JOB_NAMES[@]}
if [ "$TOTAL_JOBS" -eq 0 ]; then
  echo "[ERROR] no experiment group is enabled" >&2
  exit 2
fi

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi

detect_gpu_pool() {
  local requested gpu
  if [ "$GPU_IDS" = "auto" ]; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
    return
  fi
  requested="$(printf '%s' "$GPU_IDS" | tr ',' ' ')"
  for gpu in $requested; do
    if nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
      printf '%s\n' "$gpu"
    else
      echo "[WARN] ignoring unavailable GPU id: $gpu" >&2
    fi
  done
}

mapfile -t GPU_POOL < <(detect_gpu_pool)
if [ "${#GPU_POOL[@]}" -eq 0 ]; then
  echo "[ERROR] no NVIDIA GPU is available" >&2
  exit 2
fi

gpu_memory_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d '[:space:]'
}

gpu_is_available() {
  local used
  used="$(gpu_memory_used_mb "$1")"
  [ -n "$used" ] && [ "$used" -le "$GPU_MAX_USED_MB" ]
}

echo "============================================================"
echo "[EFFECTIVE_CONFIG]"
echo "RUN_PREFIX=$RUN_PREFIX"
echo "GPU_IDS=$GPU_IDS"
echo "DETECTED_GPUS=${GPU_POOL[*]}"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "SEEDS=$SEEDS"
echo "MAX_EPOCH=$MAX_EPOCH"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
echo "FPEM_USE_GRAD_CONSENSUS=$FPEM_USE_GRAD_CONSENSUS"
echo "FPEM_USE_PRETRAINED_INV_AGCRN=$FPEM_USE_PRETRAINED_INV_AGCRN"
echo "FPEM_PRETRAINED_INV_AGCRN_PATH=$FPEM_PRETRAINED_INV_AGCRN_PATH"
echo "EXPERIMENT_NAMES=$(IFS=,; printf '%s' "${BEST_RECIPE_NAMES[*]}")"
echo "TOTAL_JOBS=$TOTAL_JOBS"
echo "============================================================"

if truthy "$PLAN_ONLY"; then
  echo "[PLAN_ONLY] no training will be started"
  for ((job_index = 0; job_index < TOTAL_JOBS; job_index++)); do
    echo "[PLAN] job=$job_index name=${JOB_NAMES[$job_index]} seed=${JOB_SEEDS[$job_index]}"
  done
  exit 0
fi

if [ ! -s "$SUMMARY_FILE" ]; then
  printf 'group\tname\tseed\tstatus\trun_name\tdetail\n' > "$SUMMARY_FILE"
fi

FAILED=0
NEXT_JOB=0
declare -A WORKER_PID=()
declare -A WORKER_JOB=()

stop_workers() {
  local gpu_id pid
  for gpu_id in "${GPU_POOL[@]}"; do
    pid="${WORKER_PID[$gpu_id]:-}"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap 'stop_workers; exit 130' INT TERM

while true; do
  made_progress=false
  running=0

  for gpu_id in "${GPU_POOL[@]}"; do
    pid="${WORKER_PID[$gpu_id]:-}"
    if [ -n "$pid" ]; then
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      else
        job_index="${WORKER_JOB[$gpu_id]}"
        if wait "$pid"; then
          echo "[SCHEDULER] finished job=$job_index name=${JOB_NAMES[$job_index]} seed=${JOB_SEEDS[$job_index]} gpu=$gpu_id status=0"
        else
          status=$?
          echo "[SCHEDULER] finished job=$job_index name=${JOB_NAMES[$job_index]} seed=${JOB_SEEDS[$job_index]} gpu=$gpu_id status=$status"
          FAILED=1
        fi
        unset 'WORKER_PID[$gpu_id]'
        unset 'WORKER_JOB[$gpu_id]'
        made_progress=true
      fi
    fi
  done

  for gpu_id in "${GPU_POOL[@]}"; do
    [ "$NEXT_JOB" -ge "$TOTAL_JOBS" ] && break
    [ "$running" -ge "$MAX_PARALLEL" ] && break
    [ -n "${WORKER_PID[$gpu_id]:-}" ] && continue
    if gpu_is_available "$gpu_id"; then
      job_index="$NEXT_JOB"
      name="${JOB_NAMES[$job_index]}"
      seed="${JOB_SEEDS[$job_index]}"
      echo "[SCHEDULER] launch job=$job_index name=$name seed=$seed gpu=$gpu_id used_mb=$(gpu_memory_used_mb "$gpu_id")"
      run_named_ablation "$gpu_id" "$name" "$seed" &
      WORKER_PID[$gpu_id]=$!
      WORKER_JOB[$gpu_id]="$job_index"
      NEXT_JOB=$((NEXT_JOB + 1))
      running=$((running + 1))
      made_progress=true
    fi
  done

  if [ "$NEXT_JOB" -ge "$TOTAL_JOBS" ] && [ "$running" -eq 0 ]; then
    break
  fi
  if [ "$made_progress" = false ]; then
    sleep "$GPU_POLL_SECONDS"
  fi
done

ENABLED_NAMES_CSV=$(IFS=,; printf '%s' "${ENABLED_NAMES[*]}")
NORMALIZED_SEEDS=$(IFS=,; printf '%s' "${SEED_ARRAY[*]}")
if ! "$PYTHON" scripts/summarize_tds_fpem_module_build_ablation.py \
  --result_root "$RESULT_ROOT" \
  --run_prefix "$RUN_PREFIX" \
  --seeds "$NORMALIZED_SEEDS" \
  --names "$ENABLED_NAMES_CSV" \
  --csv_out "${RESULT_ROOT}/${RUN_PREFIX}_summary.csv" \
  --md_out "${RESULT_ROOT}/${RUN_PREFIX}_summary.md" \
  --aggregate_csv_out "${RESULT_ROOT}/${RUN_PREFIX}_aggregate.csv" \
  --aggregate_md_out "${RESULT_ROOT}/${RUN_PREFIX}_aggregate.md"; then
  echo "[ERROR] summarizer failed" >&2
  FAILED=1
fi

echo "============================================================"
echo "[PARTIAL_STATUS]"
cat "$SUMMARY_FILE"
echo "============================================================"
exit "$FAILED"
