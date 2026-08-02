#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
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
fi
cd "$PROJECT_DIR" || exit 1

export PYTHON=${PYTHON:-python}
export RUN_PREFIX=${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_hyper_proto}
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export MAX_PARALLEL=${MAX_PARALLEL:-4}
export SEEDS=${SEEDS:-2024,2025,2026}
export CASES=${CASES:-all}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-16}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-16}
export RESUME=${RESUME:-true}
export PLAN_ONLY=${PLAN_ONLY:-false}
export MAX_RETRY=${MAX_RETRY:-2}
export RETRY_SLEEP=${RETRY_SLEEP:-180}
export MEMORY_RETRY_FOREVER=${MEMORY_RETRY_FOREVER:-false}
export GPU_MAX_USED_MB=${GPU_MAX_USED_MB:-1024}
export GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}
export FPEM_USE_GRAD_CONSENSUS=${FPEM_USE_GRAD_CONSENSUS:-false}
export FPEM_USE_PRETRAINED_INV_AGCRN=${FPEM_USE_PRETRAINED_INV_AGCRN:-true}
export FPEM_PRETRAINED_INV_AGCRN_PATH=${FPEM_PRETRAINED_INV_AGCRN_PATH:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}
export SWAP_LAMBDA=${SWAP_LAMBDA:-0.01}
export MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-}
export MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-}
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

if [ ! -e data/NYCTaxi_TDS ] && [ -e data/NYCTaxi ]; then
  ln -s NYCTaxi data/NYCTaxi_TDS
fi

if truthy "$FPEM_USE_PRETRAINED_INV_AGCRN" && [ ! -f "$FPEM_PRETRAINED_INV_AGCRN_PATH" ]; then
  echo "[ERROR] pretrained AGCRN checkpoint not found: $FPEM_PRETRAINED_INV_AGCRN_PATH" >&2
  exit 2
fi

exec 9>"${LOG_ROOT}/scheduler.lock"
if ! flock -n 9; then
  echo "[ERROR] another launcher is already using RUN_PREFIX=${RUN_PREFIX}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

ALL_CASES=(
  hyper_proto_input_concat
  hyper_proto_input_add
  concat_proto_reference
  hyper_prediction_router_reference
  hyper_proto
  hyper_proto_sinkhorn
  hyper_proto_sinkhorn_future
  hyper_proto_sinkhorn_swap
  hyper_proto_sinkhorn_future_swap
  hyper_full_no_exogenous
  hyper_full_k2
  hyper_full_k4
)

case_enabled() {
  local name="$1"
  if [ "$CASES" = "all" ]; then
    return 0
  fi
  local item
  IFS=',' read -r -a requested <<< "$CASES"
  for item in "${requested[@]}"; do
    [ "$item" = "$name" ] && return 0
  done
  return 1
}

gpu_memory_used_mb() {
  local gpu_id="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -n 1 | tr -d ' '
}

gpu_is_available() {
  local gpu_id="$1"
  local used
  used="$(gpu_memory_used_mb "$gpu_id")"
  [ -z "$used" ] && return 0
  [ "$used" -le "$GPU_MAX_USED_MB" ]
}

cleanup_memory() {
  local gpu_id="$1"
  echo "[CLEANUP] gpu=$gpu_id date=$(date)"
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
}

is_memory_error() {
  local attempt_log="$1"
  local exit_code="${2:-0}"
  if [ "$exit_code" -eq 132 ] || [ "$exit_code" -eq 134 ] || \
     [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 139 ]; then
    return 0
  fi
  grep -Eqi "CUDA out of memory|out of memory|CUDNN_STATUS_ALLOC_FAILED|cublas.*alloc|Killed|bus error|illegal instruction|SIGILL|aborted|SIGABRT|illegal memory access|cuda runtime error|segmentation fault|segfault|SIGSEGV" "$attempt_log"
}

summary_metric_values() {
  local summary_json="${1:-}"
  if [ -z "$summary_json" ] || [ ! -f "$summary_json" ]; then
    printf 'NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA'
    return
  fi
  "$PYTHON" - "$summary_json" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

keys = [
    "route_head_mode",
    "fpem_use_env_fusion",
    "fpem_env_route_use_inv_fallback_expert",
    "fpem_env_route_target_mode",
    "test_avg_mae",
    "test_workday_mae",
    "test_holiday_mae",
    "hyper_alpha_mean",
    "hyper_alpha_head_0",
    "hyper_alpha_head_1",
    "hyper_alpha_head_2",
    "hyper_delta_norm",
    "hyper_gamma_norm_head_0",
    "hyper_gamma_norm_head_1",
    "hyper_gamma_norm_head_2",
    "hyper_beta_norm_head_0",
    "hyper_beta_norm_head_1",
    "hyper_beta_norm_head_2",
    "effective_expert_number",
    "expert_soft_usage_0",
    "expert_soft_usage_1",
    "expert_soft_usage_2",
    "expert_hard_count_0",
    "expert_hard_count_1",
    "expert_hard_count_2",
    "max_expert_usage_ratio",
    "min_expert_usage_ratio",
    "prototype_pairwise_cosine",
    "loss_future_mi",
    "loss_swap",
    "swap_prediction_delta",
    "swap_route_delta",
]

def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NA"
    return str(v)

print("\t".join(fmt(data.get(k)) for k in keys), end="")
PY
}

record_summary() {
  local name="$1"
  local seed="$2"
  local status="$3"
  local run_name="$4"
  local detail="${5:-}"
  local summary_json="${6:-}"
  (
    flock 8
    local tmp_file="${SUMMARY_FILE}.tmp"
    if [ ! -s "$SUMMARY_FILE" ]; then
      printf 'name\tseed\tstatus\trun_name\troute_head_mode\tfpem_use_env_fusion\tfpem_env_route_use_inv_fallback_expert\tfpem_env_route_target_mode\ttest_avg_mae\ttest_workday_mae\ttest_holiday_mae\thyper_alpha_mean\thyper_alpha_head_0\thyper_alpha_head_1\thyper_alpha_head_2\thyper_delta_norm\thyper_gamma_norm_head_0\thyper_gamma_norm_head_1\thyper_gamma_norm_head_2\thyper_beta_norm_head_0\thyper_beta_norm_head_1\thyper_beta_norm_head_2\teffective_expert_number\texpert_soft_usage_0\texpert_soft_usage_1\texpert_soft_usage_2\texpert_hard_count_0\texpert_hard_count_1\texpert_hard_count_2\tmax_expert_usage_ratio\tmin_expert_usage_ratio\tprototype_pairwise_cosine\tloss_future_mi\tloss_swap\tswap_prediction_delta\tswap_route_delta\tdetail\n' > "$tmp_file"
    else
      cp "$SUMMARY_FILE" "$tmp_file"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$seed" "$status" "$run_name" "$(summary_metric_values "$summary_json")" "$detail" >> "$tmp_file"
    mv "$tmp_file" "$SUMMARY_FILE"
  ) 8>"${SUMMARY_FILE}.lock"
}

OPTIONAL_ARGS=()
[ -n "$MAX_TRAIN_BATCHES" ] && OPTIONAL_ARGS+=(--max_train_batches "$MAX_TRAIN_BATCHES")
[ -n "$MAX_EVAL_BATCHES" ] && OPTIONAL_ARGS+=(--max_eval_batches "$MAX_EVAL_BATCHES")

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
  --resume "$RESUME"
  --resume_reset_patience true
  --early_stop_test_avg_mae_epoch 40
  --early_stop_test_avg_mae_threshold 12

  --fpem_backbone agcrn
  --fpem_use_pretrained_inv_agcrn "$FPEM_USE_PRETRAINED_INV_AGCRN"
  --fpem_pretrained_inv_agcrn_path "$FPEM_PRETRAINED_INV_AGCRN_PATH"
  --agcrn_embed_dim 10
  --agcrn_num_layers 2
  --agcrn_cheb_k 2

  --fpem_use_confounder_extractor false
  --fpem_use_env_mask false
  --fpem_confounder_use_mask false
  --fpem_lambda_mask_sparse 0.0
  --fpem_lambda_mask_entropy 0.0
  --fpem_lambda_inv_pred 0.0

  --fpem_use_env_route true
  --fpem_env_route_k 3
  --fpem_env_route_warmup_epochs 0
  --fpem_env_route_tau 1.0
  --fpem_env_route_oracle_tau 0.3
  --fpem_env_route_train_mode soft_oracle
  --fpem_env_route_hidden_dim 64
  --fpem_env_route_lambda_final 1.0
  --fpem_env_route_lambda_global 0.0
  --fpem_env_route_lambda_route_soft 0.5
  --fpem_env_route_lambda_expert 0.2
  --fpem_env_route_lambda_router_oracle 1.0
  --fpem_env_route_lambda_balance 0.1
  --fpem_env_route_lambda_diverse 0.02
  --fpem_env_route_lambda_proto_align 0.01
  --fpem_env_route_lambda_entropy 0.0
  --fpem_hyper_alpha_mode sample_gate
  --fpem_lambda_hyper_delta_norm 0.0001

  --fpem_use_env_prototype_router true
  --fpem_env_route_target_mode env_prototype
  --fpem_env_prototype_temperature 1.0
  --fpem_use_sinkhorn_route true
  --fpem_sinkhorn_iters 3
  --fpem_sinkhorn_epsilon 0.05
  --fpem_expert_uniform_warmup_epochs 5
  --fpem_env_route_balance_warmup_epochs 10
  --fpem_env_route_initial_temperature 1.0
  --fpem_env_route_final_temperature 0.3

  --fpem_env_use_exogenous true
  --fpem_use_env_supervision false
  --fpem_lambda_env_day_cls 0.0
  --fpem_lambda_env_hour_cls 0.0
  --fpem_lambda_env_rush_cls 0.0
  --fpem_use_env_supcon false
  --fpem_lambda_env_supcon 0.0
  --fpem_use_inv_projector false
  --fpem_use_inv_env_adversarial false
  --fpem_use_cross_cov_sep false
  --fpem_use_club_mi false
  --fpem_lambda_club_mi 0.0

  --fpem_use_env_fusion false
  --fpem_env_route_use_inv_fallback_expert false

  --fpem_use_future_mi false
  --fpem_lambda_future_mi 0.0
  --fpem_future_mi_target_mode env_encoder
  --fpem_future_mi_warmup_epochs 5
  --fpem_future_mi_hidden_dim 64
  --fpem_future_mi_detach_target true

  --fpem_use_swap false
  --fpem_lambda_swap 0.0
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

  --fpem_use_grad_consensus "$FPEM_USE_GRAD_CONSENSUS"
  --fpem_gc_pred_loss_only true
)

write_launch_config() {
  local path="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local gpu_id="$1"; shift
  "$PYTHON" - "$path" "$name" "$seed" "$gpu_id" "$RUN_PREFIX" "$@" <<'PY'
import json
import sys
path, name, seed, gpu_id, run_prefix, *args = sys.argv[1:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({
        "name": name,
        "seed": int(seed),
        "gpu_id": gpu_id,
        "run_prefix": run_prefix,
        "extra_args": args,
    }, f, ensure_ascii=False, indent=2)
PY
}

run_one() {
  local gpu_id="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local run_name="${RUN_PREFIX}_${name}_seed${seed}"
  local exp_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.launcher.log"
  local config_file="${LOG_ROOT}/${run_name}.launch_config.json"
  local attempt=1
  local extra_args=("$@")
  mkdir -p "$exp_dir"
  write_launch_config "$config_file" "$name" "$seed" "$gpu_id" "${extra_args[@]}"

  {
    echo "============================================================"
    echo "[EXPERIMENT] name=$name seed=$seed gpu=$gpu_id run_name=$run_name"
    echo "[EXTRA_ARGS] ${extra_args[*]}"
    echo "[LOG] $log_file"
    echo "============================================================"
  } | tee -a "$log_file"

  if [ -f "${exp_dir}/summary.json" ]; then
    if "$PYTHON" - "${exp_dir}/summary.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        summary = json.load(f)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if summary.get("finished") is True else 1)
PY
    then
      echo "[SKIP_DONE] ${run_name}" | tee -a "$log_file"
      record_summary "$name" "$seed" "OK" "$run_name" "existing_summary=${exp_dir}/summary.json" "${exp_dir}/summary.json"
      return 0
    fi
  fi

  while true; do
    local attempt_log="${exp_dir}/attempt_${attempt}_$(date +%Y%m%d-%H%M%S).log"
    local cmd=(
      "$PYTHON" run_tds_nyctaxi.py
      "${BASE_ARGS[@]}"
      "${OPTIONAL_ARGS[@]}"
      --seed "$seed"
      --exp_name "$run_name"
      --ablation "$name"
      "${extra_args[@]}"
    )
    {
      echo "============================================================"
      echo "[START] $(date --iso-8601=seconds)"
      echo "[ATTEMPT] $attempt"
      printf '[COMMAND] CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      echo "============================================================"
    } | tee -a "$log_file" "$attempt_log"

    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee -a "$log_file" "$attempt_log"
    local exit_code=${PIPESTATUS[0]}
    echo "[END] $(date --iso-8601=seconds) exit_code=$exit_code" | tee -a "$log_file" "$attempt_log"

    if [ "$exit_code" -eq 0 ]; then
      record_summary "$name" "$seed" "OK" "$run_name" "" "${exp_dir}/summary.json"
      return 0
    fi

    if is_memory_error "$attempt_log" "$exit_code"; then
      if truthy "$MEMORY_RETRY_FOREVER" || [ "$attempt" -lt "$MAX_RETRY" ]; then
        echo "[MEMORY_ERROR] cleanup and retry with resume=true" | tee -a "$log_file"
        sleep "$RETRY_SLEEP"
        cleanup_memory "$gpu_id" 2>&1 | tee -a "$log_file"
        attempt=$((attempt + 1))
        continue
      fi
      record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code};memory_error=max_retry" "${exp_dir}/summary.json"
      return "$exit_code"
    fi
    record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code}" "${exp_dir}/summary.json"
    return "$exit_code"
  done
}

case_args() {
  local name="$1"
  case "$name" in
    hyper_proto_input_concat)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto_input_concat \
        --fpem_env_use_exogenous true \
        --fpem_use_env_fusion false \
        --fpem_use_sinkhorn_route true \
        --fpem_expert_uniform_warmup_epochs 5 \
        --fpem_env_route_balance_warmup_epochs 10 \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_proto_input_add)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto_input_add \
        --fpem_env_use_exogenous true \
        --fpem_use_env_fusion false \
        --fpem_use_sinkhorn_route true \
        --fpem_expert_uniform_warmup_epochs 5 \
        --fpem_env_route_balance_warmup_epochs 10 \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_proto_concat_fusion_first)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto_concat \
        --fpem_use_env_fusion true \
        --fpem_use_sinkhorn_route true \
        --fpem_expert_uniform_warmup_epochs 5 \
        --fpem_env_route_balance_warmup_epochs 10 \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    concat_proto_reference)
      printf '%s\n' \
        --fpem_env_route_head_mode concat_input \
        --fpem_use_env_fusion true \
        --fpem_env_route_use_inv_fallback_expert false \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_prediction_router_reference)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film \
        --fpem_use_env_prototype_router false \
        --fpem_use_sinkhorn_route false \
        --fpem_use_env_fusion false \
        --fpem_env_route_use_inv_fallback_expert false \
        --fpem_env_route_target_mode prediction_oracle \
        --fpem_use_future_mi false \
        --fpem_lambda_future_mi 0.0 \
        --fpem_use_swap false \
        --fpem_lambda_swap 0.0
      ;;
    hyper_proto)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route false \
        --fpem_expert_uniform_warmup_epochs 0 \
        --fpem_env_route_balance_warmup_epochs 0 \
        --fpem_use_future_mi false \
        --fpem_lambda_future_mi 0.0 \
        --fpem_use_swap false \
        --fpem_lambda_swap 0.0
      ;;
    hyper_proto_sinkhorn)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route true \
        --fpem_expert_uniform_warmup_epochs 5 \
        --fpem_env_route_balance_warmup_epochs 10 \
        --fpem_use_future_mi false \
        --fpem_lambda_future_mi 0.0 \
        --fpem_use_swap false \
        --fpem_lambda_swap 0.0
      ;;
    hyper_proto_sinkhorn_future)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap false \
        --fpem_lambda_swap 0.0
      ;;
    hyper_proto_sinkhorn_swap)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi false \
        --fpem_lambda_future_mi 0.0 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_proto_sinkhorn_future_swap)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_full_no_exogenous)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA" \
        --fpem_env_use_exogenous false
      ;;
    hyper_full_k2)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_env_route_k 2 \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    hyper_full_k4)
      printf '%s\n' \
        --fpem_env_route_head_mode hyper_inv_film_proto \
        --fpem_env_route_k 4 \
        --fpem_use_sinkhorn_route true \
        --fpem_use_future_mi true \
        --fpem_lambda_future_mi 0.02 \
        --fpem_use_swap true \
        --fpem_lambda_swap "$SWAP_LAMBDA"
      ;;
    *)
      echo "[ERROR] unknown case: $name" >&2
      return 2
      ;;
  esac
}

JOB_NAMES=()
JOB_SEEDS=()
for name in "${ALL_CASES[@]}"; do
  if case_enabled "$name"; then
    for seed in "${SEED_LIST[@]}"; do
      JOB_NAMES+=("$name")
      JOB_SEEDS+=("$seed")
    done
  fi
done
TOTAL_JOBS=${#JOB_NAMES[@]}

echo "============================================================"
echo "[CONFIG] PROJECT_DIR=$PROJECT_DIR"
echo "[CONFIG] RUN_PREFIX=$RUN_PREFIX"
echo "[CONFIG] GPU_IDS=$GPU_IDS MAX_PARALLEL=$MAX_PARALLEL"
echo "[CONFIG] SEEDS=$SEEDS CASES=$CASES TOTAL_JOBS=$TOTAL_JOBS"
echo "[CONFIG] SUMMARY_FILE=$SUMMARY_FILE"
echo "============================================================"

if truthy "$PLAN_ONLY"; then
  for ((i = 0; i < TOTAL_JOBS; i++)); do
    echo "[PLAN] job=$i name=${JOB_NAMES[$i]} seed=${JOB_SEEDS[$i]}"
  done
  exit 0
fi

if [ ! -s "$SUMMARY_FILE" ]; then
  printf 'name\tseed\tstatus\trun_name\troute_head_mode\tfpem_use_env_fusion\tfpem_env_route_use_inv_fallback_expert\tfpem_env_route_target_mode\ttest_avg_mae\ttest_workday_mae\ttest_holiday_mae\thyper_alpha_mean\thyper_alpha_head_0\thyper_alpha_head_1\thyper_alpha_head_2\thyper_delta_norm\thyper_gamma_norm_head_0\thyper_gamma_norm_head_1\thyper_gamma_norm_head_2\thyper_beta_norm_head_0\thyper_beta_norm_head_1\thyper_beta_norm_head_2\teffective_expert_number\texpert_soft_usage_0\texpert_soft_usage_1\texpert_soft_usage_2\texpert_hard_count_0\texpert_hard_count_1\texpert_hard_count_2\tmax_expert_usage_ratio\tmin_expert_usage_ratio\tprototype_pairwise_cosine\tloss_future_mi\tloss_swap\tswap_prediction_delta\tswap_route_delta\tdetail\n' > "$SUMMARY_FILE"
fi

FAILED=0
NEXT_JOB=0
declare -A WORKER_PID=()
declare -A WORKER_JOB=()

stop_workers() {
  local gpu_id pid
  for gpu_id in "${GPU_POOL[@]}"; do
    pid="${WORKER_PID[$gpu_id]:-}"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
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
      mapfile -t extra_args < <(case_args "$name")
      echo "[SCHEDULER] launch job=$job_index name=$name seed=$seed gpu=$gpu_id used_mb=$(gpu_memory_used_mb "$gpu_id")"
      run_one "$gpu_id" "$name" "$seed" "${extra_args[@]}" &
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

echo "============================================================"
echo "[PARTIAL_STATUS]"
cat "$SUMMARY_FILE"
echo "============================================================"
ok_count="$(awk -F '\t' 'NR > 1 && $3 == "OK" {c++} END {print c+0}' "$SUMMARY_FILE")"
fail_count="$(awk -F '\t' 'NR > 1 && $3 == "FAIL" {c++} END {print c+0}' "$SUMMARY_FILE")"
echo "[DONE] OK=$ok_count FAIL=$fail_count summary=$SUMMARY_FILE"
exit "$FAILED"
