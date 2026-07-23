#!/usr/bin/env bash
set -euo pipefail

# Main Progressive-GMM K=3/common=0.20 run:
#   SEEDS=2024,2025,2026 CASES=add_progressive_gmm_kmax3_common020 GPU_IDS=0,1,2 MAX_PARALLEL=3 RUN_ROUTE_EVAL=1 \
#     bash scripts/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh
#
# Evaluation-only for existing best_val_model.pth checkpoints:
#   SEEDS=2024,2025,2026 CASES=add_progressive_gmm_kmax3_common020 ROUTE_EVAL_ONLY=1 \
#     bash scripts/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh

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
cd "$PROJECT_DIR"

export PYTHON=${PYTHON:-python}
export CASES=${CASES:-all}
if [ -z "${RUN_PREFIX+x}" ] && [ "$CASES" = "add_progressive_gmm_kmax3_common020" ]; then
  export RUN_PREFIX=fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_progressive_gmm_0720
else
  export RUN_PREFIX=${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_input_add_module_validity}
fi
export GPU_IDS=${GPU_IDS:-0,1,2,3}
export MAX_PARALLEL=${MAX_PARALLEL:-4}
export SEEDS=${SEEDS:-2024,2025,2026}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-16}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-16}
export RESUME=${RESUME:-true}
export DRY_RUN=${DRY_RUN:-false}
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
export RUN_ROUTE_EVAL=${RUN_ROUTE_EVAL:-false}
export ROUTE_EVAL_ONLY=${ROUTE_EVAL_ONLY:-false}
export ROUTE_EVAL_CHECKPOINT_PREFIX=${ROUTE_EVAL_CHECKPOINT_PREFIX:-$RUN_PREFIX}
export ROUTE_EVAL_CASE=${ROUTE_EVAL_CASE:-}
export ROUTE_EVAL_CASES=${ROUTE_EVAL_CASES:-$CASES}
export ROUTE_EVAL_OUTPUT_ROOT=${ROUTE_EVAL_OUTPUT_ROOT:-${RESULT_ROOT:-experiments/NYCTaxi_TDS}/${ROUTE_EVAL_CHECKPOINT_PREFIX}_online_route_eval}
export ROUTE_EVAL_MAX_BATCHES=${ROUTE_EVAL_MAX_BATCHES:--1}
export ROUTE_EVAL_RANDOM_TRIALS=${ROUTE_EVAL_RANDOM_TRIALS:-20}
export ROUTE_EVAL_RANDOM_SEED_BASE=${ROUTE_EVAL_RANDOM_SEED_BASE:-20260721}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}

RESULT_ROOT=${RESULT_ROOT:-experiments/NYCTaxi_TDS}
LOG_ROOT=${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_logs}
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
STATUS_DIR="${LOG_ROOT}/status"
ROUTE_EVAL_STATUS_DIR="${LOG_ROOT}/route_eval_status"
mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$ROUTE_EVAL_STATUS_DIR"

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

# NOTE: the legacy concat_input path below is the old z_inv + e_useful path,
# not the requested real concat(z_inv, e_env) hyper-head input.
ALL_CASES=(
  concat_single_hyper
  add_single_hyper
  add_single_no_hyper
  inv_single_hyper
  add_k1_plain
  add_k3_uniform_plain
  add_k3_prediction_router_plain
  add_k3_inv_fallback_prediction_router_plain
  add_k3_hard_prediction_sinkhorn
  add_k3_warmup_risk_sinkhorn_common005
  add_k3_warmup_risk_sinkhorn_common010
  add_k3_warmup_risk_sinkhorn_common020
  add_progressive_gmm_kmax3_common020
  add_k3_prediction_environment_sinkhorn_gaussian
  add_k3_environment_only_sinkhorn_gaussian
  add_k3_prediction_environment_sinkhorn_no_temporal
  add_k3_proto_softmax_plain
  add_k3_proto_sinkhorn_plain
  add_obj_none
  add_obj_future
  add_obj_swap
  add_obj_full
  add_full_exogenous_on
  add_full_exogenous_off
  add_full_env_zero
  add_full_env_shuffle
  add_full_club_001
  add_full_club_01
  add_full_k2
  add_full_k4
  add_full_no_balance
  add_full_no_diverse
  add_full_no_proto_align
  add_full_no_hyper_reg
  add_full_no_route_regs
  add_full_graphwavenet_backbone
  add_full_staeformer_backbone
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

completed_summary_valid() {
  local summary_json="$1"
  [ -f "$summary_json" ] || return 1
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mae = float(data.get("test_avg_mae"))
    ok = bool(data.get("finished")) and math.isfinite(mae)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

summary_metric_values() {
  local summary_json="${1:-}"
  if [ -z "$summary_json" ] || [ ! -f "$summary_json" ]; then
    printf 'NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA'
    return
  fi
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
keys = [
    "best_epoch", "best_val_loss", "test_avg_mae", "test_mixed_mae",
    "route_head_mode", "fpem_env_route_target_mode", "fpem_force_uniform_route",
    "fpem_env_rep_ablation", "effective_expert_number",
    "prototype_pairwise_cosine", "loss_future_mi", "loss_swap",
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
    if [ ! -s "$SUMMARY_FILE" ]; then
      printf 'case\tseed\tstatus\trun_name\tbest_epoch\tbest_val_loss\ttest_avg_mae\ttest_mixed_mae\troute_head_mode\tfpem_env_route_target_mode\tfpem_force_uniform_route\tfpem_env_rep_ablation\teffective_expert_number\tprototype_pairwise_cosine\tloss_future_mi\tloss_swap\tdetail\n' > "$SUMMARY_FILE"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$seed" "$status" "$run_name" "$(summary_metric_values "$summary_json")" "$detail" >> "$SUMMARY_FILE"
  ) 8>"${LOG_ROOT}/summary.lock"
}

arg_value() {
  local key="$1"
  local value="$2"
  shift 2
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "$key" ]; then
      shift
      value="${1:-$value}"
    fi
    shift || true
  done
  printf '%s' "$value"
}

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
  --fpem_env_route_head_mode hyper_inv_film_proto_input_add
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
  --fpem_force_uniform_route false
  --fpem_env_rep_ablation normal
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

  --fpem_use_future_mi true
  --fpem_lambda_future_mi 0.02
  --fpem_future_mi_target_mode env_encoder
  --fpem_future_mi_warmup_epochs 5
  --fpem_future_mi_hidden_dim 64
  --fpem_future_mi_detach_target true

  --fpem_use_swap true
  --fpem_lambda_swap "$SWAP_LAMBDA"
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

OPTIONAL_ARGS=()
if [ -n "$MAX_TRAIN_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi
if [ -n "$MAX_EVAL_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_eval_batches "$MAX_EVAL_BATCHES")
fi

case_args() {
  local name="$1"
  case "$name" in
    add_single_no_hyper)
      printf '%s\n' --fpem_env_route_head_mode concat_input --fpem_env_route_k 1 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0
      ;;
    inv_single_hyper)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto --fpem_env_route_k 1 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0
      ;;
    add_single_hyper)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 1 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0
      ;;
    concat_single_hyper)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_concat --fpem_env_route_k 1 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0
      ;;
    add_k1_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 1 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_k3_uniform_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_force_uniform_route true --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_k3_prediction_router_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_k3_inv_fallback_prediction_router_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_use_inv_fallback_expert true --fpem_lambda_inv_pred 0.2 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_train_mode soft_oracle --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_k3_hard_prediction_sinkhorn)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode hard_prediction_sinkhorn --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.5 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_warmup_risk_sinkhorn_common005)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode warmup_risk_sinkhorn --fpem_sinkhorn_warmup_epochs 10 --fpem_sinkhorn_soft_end_epoch 20 --fpem_sinkhorn_temperature_start 1.0 --fpem_sinkhorn_temperature_final 0.3 --fpem_sinkhorn_lambda_common 0.05 --fpem_risk_router_temperature 1.0 --fpem_risk_router_lambda 0.5 --fpem_risk_router_pairwise_lambda 0.0 --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_warmup_risk_sinkhorn_common010)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode warmup_risk_sinkhorn --fpem_sinkhorn_warmup_epochs 10 --fpem_sinkhorn_soft_end_epoch 20 --fpem_sinkhorn_temperature_start 1.0 --fpem_sinkhorn_temperature_final 0.3 --fpem_sinkhorn_lambda_common 0.10 --fpem_risk_router_temperature 1.0 --fpem_risk_router_lambda 0.5 --fpem_risk_router_pairwise_lambda 0.0 --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_warmup_risk_sinkhorn_common020)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode warmup_risk_sinkhorn --fpem_sinkhorn_warmup_epochs 10 --fpem_sinkhorn_soft_end_epoch 20 --fpem_sinkhorn_temperature_start 1.0 --fpem_sinkhorn_temperature_final 0.3 --fpem_sinkhorn_lambda_common 0.20 --fpem_risk_router_temperature 1.0 --fpem_risk_router_lambda 0.5 --fpem_risk_router_pairwise_lambda 0.0 --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_progressive_gmm_kmax3_common020)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode progressive_gmm_environment --fpem_env_max_clusters 3 --fpem_env_teacher_ema_momentum 0.995 --fpem_env_partition_start_epoch 5 --fpem_env_partition_update_interval 5 --fpem_env_partition_freeze_last_epochs 15 --fpem_env_min_cluster_ratio 0.08 --fpem_env_gmm_n_init 10 --fpem_env_gmm_variance_floor 0.0001 --fpem_env_progressive_lambda_common 0.20 --fpem_env_cluster_compactness_lambda 0.01 --fpem_env_cluster_consistency_lambda 0.05 --fpem_force_uniform_route false --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_progressive_gmm_kmax6_common020)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 6 --fpem_env_route_train_mode progressive_gmm_environment --fpem_env_max_clusters 6 --fpem_env_teacher_ema_momentum 0.995 --fpem_env_partition_start_epoch 5 --fpem_env_partition_update_interval 5 --fpem_env_partition_freeze_last_epochs 15 --fpem_env_min_cluster_ratio 0.08 --fpem_env_gmm_n_init 10 --fpem_env_gmm_variance_floor 0.0001 --fpem_env_progressive_lambda_common 0.20 --fpem_env_cluster_compactness_lambda 0.01 --fpem_env_cluster_consistency_lambda 0.05 --fpem_force_uniform_route false --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_progressive_gmm_kmax6_common010)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 6 --fpem_env_route_train_mode progressive_gmm_environment --fpem_env_max_clusters 6 --fpem_env_teacher_ema_momentum 0.995 --fpem_env_partition_start_epoch 5 --fpem_env_partition_update_interval 5 --fpem_env_partition_freeze_last_epochs 15 --fpem_env_min_cluster_ratio 0.08 --fpem_env_gmm_n_init 10 --fpem_env_gmm_variance_floor 0.0001 --fpem_env_progressive_lambda_common 0.10 --fpem_env_cluster_compactness_lambda 0.01 --fpem_env_cluster_consistency_lambda 0.05 --fpem_force_uniform_route false --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_prediction_environment_sinkhorn_gaussian)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode hard_prediction_environment_sinkhorn --fpem_env_route_inference_mode gaussian --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_env_sinkhorn_prediction_alpha_start 0.2 --fpem_env_sinkhorn_prediction_alpha_final 1.0 --fpem_env_sinkhorn_environment_beta_start 1.0 --fpem_env_sinkhorn_environment_beta_final 0.2 --fpem_env_sinkhorn_schedule_start_epoch 5 --fpem_env_sinkhorn_schedule_end_epoch 15 --fpem_env_sinkhorn_temporal_lambda 0.05 --fpem_env_sinkhorn_gaussian_ema 0.05 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_environment_only_sinkhorn_gaussian)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode hard_prediction_environment_sinkhorn --fpem_env_route_inference_mode gaussian --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_env_sinkhorn_prediction_alpha_start 0.0 --fpem_env_sinkhorn_prediction_alpha_final 0.0 --fpem_env_sinkhorn_environment_beta_start 1.0 --fpem_env_sinkhorn_environment_beta_final 1.0 --fpem_env_sinkhorn_schedule_start_epoch 5 --fpem_env_sinkhorn_schedule_end_epoch 15 --fpem_env_sinkhorn_temporal_lambda 0.05 --fpem_env_sinkhorn_gaussian_ema 0.05 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_prediction_environment_sinkhorn_no_temporal)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_env_route_train_mode hard_prediction_environment_sinkhorn --fpem_env_route_inference_mode gaussian --fpem_force_uniform_route false --fpem_env_route_sinkhorn_tau 1.0 --fpem_env_route_sinkhorn_iters 5 --fpem_env_sinkhorn_prediction_alpha_start 0.2 --fpem_env_sinkhorn_prediction_alpha_final 1.0 --fpem_env_sinkhorn_environment_beta_start 1.0 --fpem_env_sinkhorn_environment_beta_final 0.2 --fpem_env_sinkhorn_schedule_start_epoch 5 --fpem_env_sinkhorn_schedule_end_epoch 15 --fpem_env_sinkhorn_temporal_lambda 0.0 --fpem_env_sinkhorn_gaussian_ema 0.05 --fpem_use_env_prototype_router false --fpem_use_sinkhorn_route false --fpem_env_route_target_mode prediction_oracle --fpem_env_route_lambda_router_oracle 0.0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0 --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0 --fpem_env_route_lambda_route_soft 0.0 --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_k3_proto_softmax_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_use_env_prototype_router true --fpem_env_route_target_mode env_prototype --fpem_use_sinkhorn_route false --fpem_expert_uniform_warmup_epochs 0 --fpem_env_route_balance_warmup_epochs 0 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_k3_proto_sinkhorn_plain)
      printf '%s\n' --fpem_env_route_head_mode hyper_inv_film_proto_input_add --fpem_env_route_k 3 --fpem_use_env_prototype_router true --fpem_env_route_target_mode env_prototype --fpem_use_sinkhorn_route true --fpem_expert_uniform_warmup_epochs 5 --fpem_env_route_balance_warmup_epochs 10 --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_obj_none)
      printf '%s\n' --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_obj_future)
      printf '%s\n' --fpem_use_future_mi true --fpem_lambda_future_mi 0.02 --fpem_use_swap false --fpem_lambda_swap 0.0 --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_obj_swap)
      printf '%s\n' --fpem_use_future_mi false --fpem_lambda_future_mi 0.0 --fpem_use_swap true --fpem_lambda_swap "$SWAP_LAMBDA" --fpem_use_club_mi false --fpem_lambda_club_mi 0.0
      ;;
    add_obj_full|add_full_exogenous_on)
      :
      ;;
    add_full_exogenous_off)
      printf '%s\n' --fpem_env_use_exogenous false
      ;;
    add_full_env_zero)
      printf '%s\n' --fpem_env_rep_ablation zero
      ;;
    add_full_env_shuffle)
      printf '%s\n' --fpem_env_rep_ablation shuffle_batch
      ;;
    add_full_club_001)
      printf '%s\n' --fpem_use_club_mi true --fpem_lambda_club_mi 0.001
      ;;
    add_full_club_01)
      printf '%s\n' --fpem_use_club_mi true --fpem_lambda_club_mi 0.01
      ;;
    add_full_k2)
      printf '%s\n' --fpem_env_route_k 2
      ;;
    add_full_k4)
      printf '%s\n' --fpem_env_route_k 4
      ;;
    add_full_no_balance)
      printf '%s\n' --fpem_env_route_lambda_balance 0.0
      ;;
    add_full_no_diverse)
      printf '%s\n' --fpem_env_route_lambda_diverse 0.0
      ;;
    add_full_no_proto_align)
      printf '%s\n' --fpem_env_route_lambda_proto_align 0.0
      ;;
    add_full_no_hyper_reg)
      printf '%s\n' --fpem_lambda_hyper_delta_norm 0.0
      ;;
    add_full_no_route_regs)
      printf '%s\n' --fpem_env_route_lambda_balance 0.0 --fpem_env_route_lambda_diverse 0.0 --fpem_env_route_lambda_proto_align 0.0 --fpem_env_route_lambda_entropy 0.0
      ;;
    add_full_graphwavenet_backbone)
      printf '%s\n' --fpem_backbone graphwavenet --fpem_use_pretrained_inv_agcrn false --graphwavenet_layers 4 --graphwavenet_kernel_size 2 --graphwavenet_dropout 0.1
      ;;
    add_full_staeformer_backbone)
      printf '%s\n' --fpem_backbone staeformer --fpem_use_pretrained_inv_agcrn false --staeformer_layers 2 --staeformer_heads 4 --staeformer_dropout 0.1 --staeformer_mlp_ratio 2.0
      ;;
    *)
      echo "[ERROR] unknown case: $name" >&2
      return 1
      ;;
  esac
}

write_launch_config() {
  local path="$1"; shift
  {
    printf 'project_dir=%s\n' "$PROJECT_DIR"
    printf 'run_prefix=%s\n' "$RUN_PREFIX"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } > "$path"
}

run_one() {
  local gpu_id="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local run_name="${RUN_PREFIX}_${name}_seed${seed}"
  local exp_dir="${PROJECT_DIR}/${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local summary_json="${exp_dir}/summary.json"
  local status_file="${STATUS_DIR}/${run_name}.status"
  local attempt=1
  local rc=0
  local should_retry=0
  local detail=""
  local -a extra_args=()
  mapfile -t extra_args < <(case_args "$name")

  if completed_summary_valid "$summary_json"; then
    echo "[SKIP] completed case=$name seed=$seed run=$run_name"
    record_summary "$name" "$seed" "SKIP_DONE" "$run_name" "summary.json already has finite test_avg_mae" "$summary_json"
    printf 'SKIP_DONE\n' > "$status_file"
    return 0
  fi

  local -a cmd=(
    "$PYTHON" run_tds_nyctaxi.py
    "${BASE_ARGS[@]}"
    "${OPTIONAL_ARGS[@]}"
    --seed "$seed"
    --exp_name "$run_name"
    --ablation "$name"
    "${extra_args[@]}"
  )

  local backbone head_mode route_k proto sinkhorn future_mi swap club env_ablation force_uniform
  backbone="$(arg_value --fpem_backbone agcrn "${BASE_ARGS[@]}" "${extra_args[@]}")"
  head_mode="$(arg_value --fpem_env_route_head_mode hyper_inv_film_proto_input_add "${BASE_ARGS[@]}" "${extra_args[@]}")"
  route_k="$(arg_value --fpem_env_route_k 3 "${BASE_ARGS[@]}" "${extra_args[@]}")"
  proto="$(arg_value --fpem_use_env_prototype_router true "${BASE_ARGS[@]}" "${extra_args[@]}")"
  sinkhorn="$(arg_value --fpem_use_sinkhorn_route true "${BASE_ARGS[@]}" "${extra_args[@]}")"
  future_mi="$(arg_value --fpem_use_future_mi true "${BASE_ARGS[@]}" "${extra_args[@]}")"
  swap="$(arg_value --fpem_use_swap true "${BASE_ARGS[@]}" "${extra_args[@]}")"
  club="$(arg_value --fpem_use_club_mi false "${BASE_ARGS[@]}" "${extra_args[@]}")"
  env_ablation="$(arg_value --fpem_env_rep_ablation normal "${BASE_ARGS[@]}" "${extra_args[@]}")"
  force_uniform="$(arg_value --fpem_force_uniform_route false "${BASE_ARGS[@]}" "${extra_args[@]}")"

  echo "[LAUNCH] case=$name seed=$seed gpu=$gpu_id exp_dir=$exp_dir backbone=$backbone head=$head_mode K=$route_k proto=$proto sinkhorn=$sinkhorn future_mi=$future_mi swap=$swap club=$club env_ablation=$env_ablation force_uniform=$force_uniform"
  write_launch_config "${LOG_ROOT}/${run_name}.cmd" "${cmd[@]}"

  if truthy "$DRY_RUN" || truthy "$PLAN_ONLY"; then
    printf '[DRY_RUN] CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    record_summary "$name" "$seed" "DRY_RUN" "$run_name" "not launched" ""
    printf 'DRY_RUN\n' > "$status_file"
    return 0
  fi

  while :; do
    local attempt_log="${LOG_ROOT}/${run_name}.attempt${attempt}.log"
    echo "[RUN] case=$name seed=$seed attempt=$attempt gpu=$gpu_id date=$(date)"
    cleanup_memory "$gpu_id"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" > "$attempt_log" 2>&1
    rc=$?
    set -e
    cat "$attempt_log" >> "$log_file"
    if [ "$rc" -eq 0 ] && completed_summary_valid "$summary_json"; then
      record_summary "$name" "$seed" "OK" "$run_name" "attempt=$attempt" "$summary_json"
      printf 'OK\n' > "$status_file"
      return 0
    fi
    should_retry=0
    if is_memory_error "$attempt_log" "$rc"; then
      should_retry=1
      detail="memory_or_segfault rc=$rc attempt=$attempt"
    else
      detail="non_memory_failure rc=$rc attempt=$attempt"
    fi
    if [ "$should_retry" -eq 1 ] && { [ "$attempt" -lt "$MAX_RETRY" ] || truthy "$MEMORY_RETRY_FOREVER"; }; then
      echo "[RETRY] $run_name $detail sleep=${RETRY_SLEEP}s"
      cleanup_memory "$gpu_id"
      sleep "$RETRY_SLEEP"
      attempt=$((attempt + 1))
      continue
    fi
    record_summary "$name" "$seed" "FAIL" "$run_name" "$detail" "$summary_json"
    printf 'FAIL\n' > "$status_file"
    return "$rc"
  done
}

run_route_eval_one() {
  local gpu_id="$1"; shift
  local case_name="$1"; shift
  local seed="$1"; shift
  local ckpt_run="${ROUTE_EVAL_CHECKPOINT_PREFIX}_${case_name}_seed${seed}"
  local ckpt_path="${PROJECT_DIR}/${RESULT_ROOT}/${ckpt_run}/best_val_model.pth"
  local output_dir="${PROJECT_DIR}/${ROUTE_EVAL_OUTPUT_ROOT}/${ckpt_run}"
  local status_file="${ROUTE_EVAL_STATUS_DIR}/${ckpt_run}.status"
  local log_file="${LOG_ROOT}/${ckpt_run}.route_eval.log"

  if [ -f "${output_dir}/online_route_results.json" ]; then
    echo "[ROUTE_EVAL_SKIP] seed=$seed output=${output_dir}/online_route_results.json"
    printf 'ROUTE_EVAL_OK\n' > "$status_file"
    return 0
  fi
  if [ ! -f "$ckpt_path" ]; then
    echo "[ROUTE_EVAL_FAIL] missing best_val_model.pth case=$case_name seed=$seed ckpt=$ckpt_path" | tee "$log_file"
    printf 'ROUTE_EVAL_FAIL\n' > "$status_file"
    return 2
  fi

  echo "[ROUTE_EVAL_RUNNING] case=$case_name seed=$seed gpu=$gpu_id ckpt=$ckpt_path output=$output_dir"
  printf 'ROUTE_EVAL_RUNNING\n' > "$status_file"
  if truthy "$DRY_RUN" || truthy "$PLAN_ONLY"; then
    printf '[DRY_RUN_ROUTE_EVAL] CUDA_VISIBLE_DEVICES=%q CHECKPOINT_PATH=%q OUTPUT_DIR=%q DEVICE=cuda:0 BATCH_SIZE=%q MAX_BATCHES=%q NUM_RANDOM_TRIALS=%q RANDOM_SEED_BASE=%q bash scripts/evaluate_tds_nyctaxi_online_expert_routing.sh\n' \
      "$gpu_id" "$ckpt_path" "$output_dir" "$TEST_BATCH_SIZE" "$ROUTE_EVAL_MAX_BATCHES" "$ROUTE_EVAL_RANDOM_TRIALS" "$((ROUTE_EVAL_RANDOM_SEED_BASE + seed * 1000))"
    return 0
  fi

  cleanup_memory "$gpu_id"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    CHECKPOINT_PATH="$ckpt_path" \
    OUTPUT_DIR="$output_dir" \
    DEVICE=cuda:0 \
    BATCH_SIZE="$TEST_BATCH_SIZE" \
    MAX_BATCHES="$ROUTE_EVAL_MAX_BATCHES" \
    NUM_RANDOM_TRIALS="$ROUTE_EVAL_RANDOM_TRIALS" \
    RANDOM_SEED_BASE="$((ROUTE_EVAL_RANDOM_SEED_BASE + seed * 1000))" \
    bash scripts/evaluate_tds_nyctaxi_online_expert_routing.sh > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ -f "${output_dir}/online_route_results.json" ]; then
    echo "[ROUTE_EVAL_OK] case=$case_name seed=$seed output=$output_dir"
    printf 'ROUTE_EVAL_OK\n' > "$status_file"
    return 0
  fi
  echo "[ROUTE_EVAL_FAIL] case=$case_name seed=$seed rc=$rc log=$log_file"
  printf 'ROUTE_EVAL_FAIL\n' > "$status_file"
  return "$rc"
}

run_route_eval_all() {
  local eval_job_index=0
  local eval_running=0
  local seed gpu case_name
  local -a eval_cases=()
  if [ -n "$ROUTE_EVAL_CASE" ]; then
    eval_cases=("$ROUTE_EVAL_CASE")
  elif [ "$ROUTE_EVAL_CASES" = "all" ]; then
    eval_cases=("${ALL_CASES[@]}")
  else
    IFS=',' read -r -a eval_cases <<< "$ROUTE_EVAL_CASES"
  fi
  for case_name in "${eval_cases[@]}"; do
    if [ -z "$ROUTE_EVAL_CASE" ]; then
      case_enabled "$case_name" || continue
    fi
    for seed in "${SEED_LIST[@]}"; do
      gpu="${GPU_POOL[$((eval_job_index % ${#GPU_POOL[@]}))]}"
      if ! truthy "$DRY_RUN" && ! truthy "$PLAN_ONLY"; then
        while ! gpu_is_available "$gpu"; do
          echo "[WAIT_GPU_ROUTE_EVAL] gpu=$gpu memory used above ${GPU_MAX_USED_MB}MB"
          sleep "$GPU_POLL_SECONDS"
        done
      fi
      if truthy "$DRY_RUN" || truthy "$PLAN_ONLY"; then
        run_route_eval_one "$gpu" "$case_name" "$seed"
      else
        while [ "$eval_running" -ge "$MAX_PARALLEL" ]; do
          if ! wait -n; then
            true
          fi
          eval_running=$((eval_running - 1))
        done
        run_route_eval_one "$gpu" "$case_name" "$seed" &
        eval_running=$((eval_running + 1))
      fi
      eval_job_index=$((eval_job_index + 1))
    done
  done
  while [ "$eval_running" -gt 0 ]; do
    if ! wait -n; then
      true
    fi
    eval_running=$((eval_running - 1))
  done
}

write_route_eval_summary() {
  local output_root="${PROJECT_DIR}/${ROUTE_EVAL_OUTPUT_ROOT}"
  local summary_tsv="${PROJECT_DIR}/${ROUTE_EVAL_OUTPUT_ROOT}/route_eval_seed_summary.tsv"
  "$PYTHON" - "$output_root" "$summary_tsv" <<'PY' || true
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

root, out_path = sys.argv[1], sys.argv[2]
groups = defaultdict(list)
for path in sorted(glob.glob(os.path.join(root, "*", "online_route_results.json"))):
    run_name = os.path.basename(os.path.dirname(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    case_name = run_name.rsplit("_seed", 1)[0]
    for row in data.get("test_results", []):
        method = row.get("routing_method")
        mae = row.get("test_avg_mae")
        seed = row.get("seed", data.get("seed"))
        if method is None or mae is None:
            continue
        groups[(case_name, method)].append((int(seed), float(mae)))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("case\trouting_method\tn\tseeds\ttest_avg_mae_mean\ttest_avg_mae_sample_std\n")
    for (case_name, method), values in sorted(groups.items()):
        maes = [v[1] for v in values]
        seeds = ",".join(str(v[0]) for v in sorted(values))
        mean = statistics.mean(maes)
        std = statistics.stdev(maes) if len(maes) > 1 else 0.0
        f.write(f"{case_name}\t{method}\t{len(maes)}\t{seeds}\t{mean}\t{std}\n")
print(f"[ROUTE_EVAL_SUMMARY] {out_path}")
PY
}

echo "[INFO] RUN_PREFIX=$RUN_PREFIX"
echo "[INFO] cases=${CASES} seeds=${SEEDS} gpu_ids=${GPU_IDS} max_parallel=${MAX_PARALLEL} dry_run=${DRY_RUN}"
echo "[INFO] pretrained_inv=${FPEM_USE_PRETRAINED_INV_AGCRN} ckpt=${FPEM_PRETRAINED_INV_AGCRN_PATH}"
echo "[INFO] run_route_eval=${RUN_ROUTE_EVAL} route_eval_only=${ROUTE_EVAL_ONLY} route_eval_checkpoint_prefix=${ROUTE_EVAL_CHECKPOINT_PREFIX} route_eval_cases=${ROUTE_EVAL_CASES:-$ROUTE_EVAL_CASE}"

if truthy "$ROUTE_EVAL_ONLY"; then
  run_route_eval_all
  write_route_eval_summary
  route_eval_ok_count=$( (grep -Rhs '^ROUTE_EVAL_OK$' "$ROUTE_EVAL_STATUS_DIR" || true) | wc -l | tr -d ' ')
  route_eval_fail_count=$( (grep -Rhs '^ROUTE_EVAL_FAIL$' "$ROUTE_EVAL_STATUS_DIR" || true) | wc -l | tr -d ' ')
  echo "[ROUTE_EVAL_DONE] OK=${route_eval_ok_count} FAIL=${route_eval_fail_count}"
  if [ "$route_eval_fail_count" -gt 0 ]; then
    exit 1
  fi
  exit 0
fi

job_index=0
running=0
for case_name in "${ALL_CASES[@]}"; do
  case_enabled "$case_name" || continue
  for seed in "${SEED_LIST[@]}"; do
    gpu="${GPU_POOL[$((job_index % ${#GPU_POOL[@]}))]}"
    if ! truthy "$DRY_RUN" && ! truthy "$PLAN_ONLY"; then
      while ! gpu_is_available "$gpu"; do
        echo "[WAIT_GPU] gpu=$gpu memory used above ${GPU_MAX_USED_MB}MB"
        sleep "$GPU_POLL_SECONDS"
      done
    fi
    if truthy "$DRY_RUN" || truthy "$PLAN_ONLY"; then
      run_one "$gpu" "$case_name" "$seed"
    else
      while [ "$running" -ge "$MAX_PARALLEL" ]; do
        if ! wait -n; then
          true
        fi
        running=$((running - 1))
      done
      run_one "$gpu" "$case_name" "$seed" &
      running=$((running + 1))
    fi
    job_index=$((job_index + 1))
  done
done

while [ "$running" -gt 0 ]; do
  if ! wait -n; then
    true
  fi
  running=$((running - 1))
done

if truthy "$RUN_ROUTE_EVAL"; then
  run_route_eval_all
  write_route_eval_summary
fi

ok_count=$( (grep -Rhs '^OK$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
skip_count=$( (grep -Rhs '^SKIP_DONE$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
dry_count=$( (grep -Rhs '^DRY_RUN$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
fail_count=$( (grep -Rhs '^FAIL$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
route_eval_ok_count=$( (grep -Rhs '^ROUTE_EVAL_OK$' "$ROUTE_EVAL_STATUS_DIR" || true) | wc -l | tr -d ' ')
route_eval_fail_count=$( (grep -Rhs '^ROUTE_EVAL_FAIL$' "$ROUTE_EVAL_STATUS_DIR" || true) | wc -l | tr -d ' ')
echo "[DONE] OK=${ok_count} SKIP_DONE=${skip_count} DRY_RUN=${dry_count} FAIL=${fail_count} ROUTE_EVAL_OK=${route_eval_ok_count} ROUTE_EVAL_FAIL=${route_eval_fail_count}"
if [ "$fail_count" -gt 0 ] || [ "$route_eval_fail_count" -gt 0 ]; then
  exit 1
fi
