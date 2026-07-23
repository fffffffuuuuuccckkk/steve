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
export RUN_PREFIX=${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
export MAX_PARALLEL=${MAX_PARALLEL:-auto}
export SEEDS=${SEEDS:-2024,2025,2026}
export CASES=${CASES:-all}
export REVERSE_CONFIGS=${REVERSE_CONFIGS:-false}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-16}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-16}
export RESUME=${RESUME:-true}
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
export RUN_CASE_STUDY=${RUN_CASE_STUDY:-true}
export CASE_STUDY_MAX_BATCHES=${CASE_STUDY_MAX_BATCHES:--1}
export CASE_STUDY_SPLIT=${CASE_STUDY_SPLIT:-test}
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

summary_metric_values() {
  local summary_json="${1:-}"
  if [ -z "$summary_json" ] || [ ! -f "$summary_json" ]; then
    printf 'NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA'
    return
  fi
  "$PYTHON" - "$summary_json" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

aliases = {
    "test_avg_mae": ["test_avg_mae"],
    "test_workday_mae": ["test_workday_mae"],
    "test_holiday_mae": ["test_holiday_mae"],
    "env_day_acc": ["env_day_acc", "fpem/env_day_acc"],
    "env_hour_acc": ["env_hour_acc", "fpem/env_hour_acc"],
    "env_rush_acc": ["env_rush_acc", "fpem/env_rush_acc"],
    "inv_day_acc": ["inv_day_acc", "fpem/inv_day_acc"],
    "effective_expert_number": ["effective_expert_number", "fpem/effective_expert_number"],
    "expert0_soft_usage": ["expert0_soft_usage", "route_soft_mean_expert_0", "fpem/route_soft_mean_expert_0"],
    "expert1_soft_usage": ["expert1_soft_usage", "route_soft_mean_expert_1", "fpem/route_soft_mean_expert_1"],
    "expert2_soft_usage": ["expert2_soft_usage", "route_soft_mean_expert_2", "fpem/route_soft_mean_expert_2"],
    "expert0_hard_count": ["expert0_hard_count", "route_hard_count_expert_0", "fpem/route_hard_count_expert_0"],
    "expert1_hard_count": ["expert1_hard_count", "route_hard_count_expert_1", "fpem/route_hard_count_expert_1"],
    "expert2_hard_count": ["expert2_hard_count", "route_hard_count_expert_2", "fpem/route_hard_count_expert_2"],
    "max_expert_usage_ratio": ["max_expert_usage_ratio", "fpem/max_expert_usage_ratio"],
    "fpem_env_use_exogenous": ["fpem_env_use_exogenous", "fpem/fpem_env_use_exogenous"],
    "env_exogenous_available": ["env_exogenous_available", "fpem/env_exogenous_available"],
    "env_exogenous_time_available": ["env_exogenous_time_available", "fpem/env_exogenous_time_available"],
    "env_exogenous_load_available": ["env_exogenous_load_available", "fpem/env_exogenous_load_available"],
    "env_exogenous_feature_dim": ["env_exogenous_feature_dim", "fpem/env_exogenous_feature_dim"],
    "env_exogenous_embedding_norm": ["env_exogenous_embedding_norm", "fpem/env_exogenous_embedding_norm"],
    "env_exogenous_load_embedding_norm": ["env_exogenous_load_embedding_norm", "fpem/env_exogenous_load_embedding_norm"],
}

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None

def fmt(value):
    if value is None:
        return "NA"
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.8g}"
    return str(value).replace("\t", " ").replace("\n", " ")

out = []
for column, names in aliases.items():
    found = None
    for name in names:
        found = find_key(data, name)
        if found is not None:
            break
    out.append(fmt(found))
print("\t".join(out), end="")
PY
}

record_summary() {
  local name="$1"
  local seed="$2"
  local status="$3"
  local run_name="$4"
  local detail="${5:-}"
  local summary_json="${6:-}"
  local tmp_file metrics
  detail="$(printf '%s' "$detail" | tr '\t\r\n' '   ')"
  metrics="$(summary_metric_values "$summary_json")"
  tmp_file="${SUMMARY_FILE}.tmp.$$.${RANDOM}"
  (
    flock -x 8
    if [ -s "$SUMMARY_FILE" ]; then
      awk -F '\t' -v name="$name" -v seed="$seed" \
        'NR == 1 || !(($1 == name) && ($2 == seed))' \
        "$SUMMARY_FILE" > "$tmp_file"
    else
      printf 'name\tseed\tstatus\trun_name\ttest_avg_mae\ttest_workday_mae\ttest_holiday_mae\tenv_day_acc\tenv_hour_acc\tenv_rush_acc\tinv_day_acc\teffective_expert_number\texpert0_soft_usage\texpert1_soft_usage\texpert2_soft_usage\texpert0_hard_count\texpert1_hard_count\texpert2_hard_count\tmax_expert_usage_ratio\tfpem_env_use_exogenous\tenv_exogenous_available\tenv_exogenous_time_available\tenv_exogenous_load_available\tenv_exogenous_feature_dim\tenv_exogenous_embedding_norm\tenv_exogenous_load_embedding_norm\tdetail\n' > "$tmp_file"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$seed" "$status" "$run_name" "$metrics" "$detail" >> "$tmp_file"
    mv "$tmp_file" "$SUMMARY_FILE"
  ) 8>"${SUMMARY_FILE}.lock"
}

# Shared current recipe. Deletion/removal switches are kept exactly the same
# across cases unless a case explicitly re-enables that module.
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

  --fpem_use_env_route true
  --fpem_use_env_fusion true
  --fpem_env_route_k 3
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
  --fpem_env_route_lambda_balance 0.01
  --fpem_env_route_lambda_diverse 0.01
  --fpem_env_route_lambda_entropy 0.0

  --fpem_use_confounder_extractor false
  --fpem_use_env_mask false
  --fpem_confounder_use_mask false
  --fpem_lambda_mask_sparse 0.0
  --fpem_lambda_mask_entropy 0.0

  --fpem_lambda_inv_pred 0.0
  --fpem_hyper_alpha_mode sample_gate
  --fpem_lambda_hyper_delta_norm 0.0001

  --fpem_use_club_mi false
  --fpem_lambda_club_mi 0.0

  --fpem_use_future_mi true
  --fpem_lambda_future_mi 0.02
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
  --fpem_swap_fallback_warmup_epochs 30

  --fpem_use_env_supervision false
  --fpem_lambda_env_day_cls 0.0
  --fpem_lambda_env_hour_cls 0.0
  --fpem_lambda_env_rush_cls 0.0
  --fpem_env_use_exogenous true
  --fpem_use_env_supcon false
  --fpem_lambda_env_supcon 0.0
  --fpem_env_supcon_temperature 0.1
  --fpem_use_inv_projector false
  --fpem_use_inv_env_adversarial false
  --fpem_lambda_inv_env_adv 0.0
  --fpem_grl_alpha 1.0
  --fpem_env_route_target_mode prediction_oracle
  --fpem_use_env_prototype_router false
  --fpem_env_prototype_temperature 1.0
  --fpem_env_route_hybrid_alpha 1.0
  --fpem_env_route_hybrid_alpha_start 1.0
  --fpem_env_route_hybrid_alpha_end 0.5
  --fpem_env_route_hybrid_alpha_decay_epochs 30
  --fpem_use_sinkhorn_route false
  --fpem_sinkhorn_iters 3
  --fpem_sinkhorn_epsilon 0.05
  --fpem_expert_uniform_warmup_epochs 0
  --fpem_env_route_balance_warmup_epochs 0
  --fpem_env_route_initial_temperature 1.0
  --fpem_env_route_final_temperature 0.3
  --fpem_use_cross_cov_sep false
  --fpem_lambda_cross_cov_sep 0.0

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

OPTIONAL_ARGS=()
if [ -n "$MAX_TRAIN_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi
if [ -n "$MAX_EVAL_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_eval_batches "$MAX_EVAL_BATCHES")
fi

ALL_CASE_NAMES=(
  env_exogenous_on
  env_exogenous_off
  current_prediction_oracle
  env_supervision
  env_supervision_supcon
  env_supervision_inv_adv
  env_disentangle_full
  prototype_sinkhorn_route
  env_supervision_prototype_route
  env_disentangle_prototype_full
  env_disentangle_hybrid_route
)

write_launch_config() {
  local path="$1"
  local name="$2"
  local seed="$3"
  local gpu="$4"
  local run_name="$5"
  shift 5
  "$PYTHON" - "$path" "$name" "$seed" "$gpu" "$run_name" "$@" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, name, seed, gpu, run_name = sys.argv[1:6]
extra_args = sys.argv[6:]
data = {
    "name": name,
    "seed": int(seed),
    "gpu": gpu,
    "run_name": run_name,
    "run_prefix": os.environ.get("RUN_PREFIX"),
    "resume": os.environ.get("RESUME"),
    "base_recipe": {
        "fpem_use_pretrained_inv_agcrn": os.environ.get("FPEM_USE_PRETRAINED_INV_AGCRN"),
        "fpem_pretrained_inv_agcrn_path": os.environ.get("FPEM_PRETRAINED_INV_AGCRN_PATH"),
        "fpem_use_confounder_extractor": False,
        "fpem_use_env_mask": False,
        "fpem_env_route_k": 3,
        "fpem_use_future_mi": True,
        "fpem_lambda_future_mi": 0.02,
        "fpem_lambda_inv_pred": 0.0,
        "fpem_use_swap": False,
        "fpem_use_club_mi": False,
    },
    "extra_args": extra_args,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY
}

case_study_done() {
  local exp_dir="$1"
  [ -f "${exp_dir}/case_study/case_outputs/manifest.json" ]
}

maybe_run_case_study() {
  local gpu_id="$1"
  local name="$2"
  local seed="$3"
  local run_name="$4"
  local exp_dir="$5"
  local log_file="$6"
  local ckpt="${exp_dir}/best_val_model.pth"

  if [ "$seed" != "2025" ] || ! truthy "$RUN_CASE_STUDY"; then
    return 0
  fi
  if case_study_done "$exp_dir"; then
    echo "[CASE_STUDY_SKIP] existing ${exp_dir}/case_study/case_outputs/manifest.json" | tee -a "$log_file"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[CASE_STUDY_WARN] checkpoint not found: $ckpt" | tee -a "$log_file"
    return 0
  fi

  echo "[CASE_STUDY_START] name=$name seed=$seed run_name=$run_name max_batches=$CASE_STUDY_MAX_BATCHES" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" scripts/analyze_pretrained_case_study.py \
    --ckpt_path "$ckpt" \
    --exp_dir "$exp_dir" \
    --split "$CASE_STUDY_SPLIT" \
    --max_batches "$CASE_STUDY_MAX_BATCHES" \
    --device cuda:0 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "[CASE_STUDY_OK] ${exp_dir}/case_study/case_outputs" | tee -a "$log_file"
  else
    echo "[CASE_STUDY_WARN] failed status=$status for $run_name" | tee -a "$log_file"
  fi
  return 0
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

  mkdir -p "$exp_dir"
  write_launch_config "$config_file" "$name" "$seed" "$gpu_id" "$run_name" "${extra_args[@]}"

  {
    echo "============================================================"
    echo "[EXPERIMENT] name=$name seed=$seed gpu=$gpu_id run_name=$run_name"
    echo "[BASE] pretrained_frozen_inv=$FPEM_USE_PRETRAINED_INV_AGCRN path=$FPEM_PRETRAINED_INV_AGCRN_PATH"
    echo "[BASE] no_confounder no_env_mask K=3 future_mi=true no_swap no_club"
    echo "[EXTRA_ARGS] ${extra_args[*]}"
    echo "[LOG] $log_file"
    echo "============================================================"
  } | tee -a "$log_file"

  if [ -f "${exp_dir}/summary.json" ]; then
    if "$PYTHON" - "${exp_dir}/summary.json" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        summary = json.load(f)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if summary.get("finished") is True else 1)
PY
    then
      echo "[SKIP_DONE] ${run_name}" | tee -a "$log_file"
      maybe_run_case_study "$gpu_id" "$name" "$seed" "$run_name" "$exp_dir" "$log_file"
      record_summary "$name" "$seed" "OK" "$run_name" "existing_summary=${exp_dir}/summary.json" "${exp_dir}/summary.json"
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
      "${OPTIONAL_ARGS[@]}"
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
      maybe_run_case_study "$gpu_id" "$name" "$seed" "$run_name" "$exp_dir" "$log_file"
      record_summary "$name" "$seed" "OK" "$run_name" "" "${exp_dir}/summary.json"
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
      record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code};memory_error=max_retry" "${exp_dir}/summary.json"
      return "$exit_code"
    fi

    record_summary "$name" "$seed" "FAIL" "$run_name" "exit_code=${exit_code}" "${exp_dir}/summary.json"
    return "$exit_code"
  done
}

case_env_supervision_args=(
  --fpem_use_env_supervision true
  --fpem_lambda_env_day_cls 0.05
  --fpem_lambda_env_hour_cls 0.02
  --fpem_lambda_env_rush_cls 0.02
)

case_supcon_args=(
  --fpem_use_env_supcon true
  --fpem_lambda_env_supcon 0.01
  --fpem_env_supcon_temperature 0.1
)

case_inv_adv_args=(
  --fpem_use_inv_projector true
  --fpem_use_inv_env_adversarial true
  --fpem_lambda_inv_env_adv 0.01
  --fpem_grl_alpha 1.0
)

case_cross_cov_args=(
  --fpem_use_cross_cov_sep true
  --fpem_lambda_cross_cov_sep 0.001
)

case_proto_route_args=(
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
  --fpem_env_route_lambda_balance 0.1
  --fpem_env_route_lambda_diverse 0.02
)

case_swap_club_args=(
  --fpem_use_club_mi true
  --fpem_lambda_club_mi 0.01
  --fpem_use_swap true
  --fpem_lambda_swap 0.01
)

case_hybrid_args=(
  --fpem_env_route_target_mode hybrid
  --fpem_env_route_hybrid_alpha 1.0
  --fpem_env_route_hybrid_alpha_start 1.0
  --fpem_env_route_hybrid_alpha_end 0.5
  --fpem_env_route_hybrid_alpha_decay_epochs 30
)

run_named_ablation() {
  local gpu_id="$1" name="$2" seed="$3"
  case "$name" in
    env_exogenous_on) run_one "$gpu_id" "$name" "$seed" \
      --fpem_env_use_exogenous true ;;
    env_exogenous_off) run_one "$gpu_id" "$name" "$seed" \
      --fpem_env_use_exogenous false ;;
    current_prediction_oracle) run_one "$gpu_id" "$name" "$seed" \
      --fpem_use_env_supervision false \
      --fpem_use_env_supcon false \
      --fpem_use_inv_projector false \
      --fpem_use_inv_env_adversarial false \
      --fpem_use_env_prototype_router false \
      --fpem_env_route_target_mode prediction_oracle \
      --fpem_use_sinkhorn_route false \
      --fpem_use_cross_cov_sep false ;;
    env_supervision) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" ;;
    env_supervision_supcon) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_supcon_args[@]}" ;;
    env_supervision_inv_adv) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_inv_adv_args[@]}" ;;
    env_disentangle_full) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_supcon_args[@]}" \
      "${case_inv_adv_args[@]}" \
      "${case_cross_cov_args[@]}" ;;
    prototype_sinkhorn_route) run_one "$gpu_id" "$name" "$seed" \
      "${case_proto_route_args[@]}" ;;
    env_supervision_prototype_route) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_supcon_args[@]}" \
      "${case_proto_route_args[@]}" ;;
    env_disentangle_prototype_full) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_supcon_args[@]}" \
      "${case_inv_adv_args[@]}" \
      "${case_cross_cov_args[@]}" \
      "${case_proto_route_args[@]}" \
      "${case_swap_club_args[@]}" ;;
    env_disentangle_hybrid_route) run_one "$gpu_id" "$name" "$seed" \
      "${case_env_supervision_args[@]}" \
      "${case_supcon_args[@]}" \
      "${case_inv_adv_args[@]}" \
      "${case_cross_cov_args[@]}" \
      "${case_proto_route_args[@]}" \
      "${case_swap_club_args[@]}" \
      "${case_hybrid_args[@]}" ;;
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

ENABLED_NAMES=()
if [ "$CASES" = "all" ]; then
  ENABLED_NAMES=("${ALL_CASE_NAMES[@]}")
else
  IFS=',' read -r -a ENABLED_NAMES <<< "$CASES"
fi

case_is_known() {
  local requested="$1" known
  for known in "${ALL_CASE_NAMES[@]}"; do
    if [ "$requested" = "$known" ]; then
      return 0
    fi
  done
  return 1
}

for name in "${ENABLED_NAMES[@]}"; do
  if ! case_is_known "$name"; then
    echo "[ERROR] unknown CASES entry: $name" >&2
    echo "[ERROR] valid names: $(IFS=,; printf '%s' "${ALL_CASE_NAMES[*]}")" >&2
    exit 2
  fi
done




# 反转实验配置顺序，但保持每个配置内部的 seed 顺序不变
if truthy "$REVERSE_CONFIGS"; then
  REVERSED_NAMES=()

  for ((i = ${#ENABLED_NAMES[@]} - 1; i >= 0; i--)); do
    REVERSED_NAMES+=("${ENABLED_NAMES[$i]}")
  done

  ENABLED_NAMES=("${REVERSED_NAMES[@]}")
fi




JOB_NAMES=()
JOB_SEEDS=()
for name in "${ENABLED_NAMES[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    JOB_NAMES+=("$name")
    JOB_SEEDS+=("$seed")
  done
done

TOTAL_JOBS=${#JOB_NAMES[@]}
if [ "$TOTAL_JOBS" -eq 0 ]; then
  echo "[ERROR] no experiment is enabled" >&2
  exit 2
fi

# if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
#   echo "[ERROR] MAX_PARALLEL must be a positive integer" >&2
#   exit 2
# fi

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




# MAX_PARALLEL=auto 时，自动使用检测到的 GPU 数量
if [ "$MAX_PARALLEL" = "auto" ]; then
  MAX_PARALLEL=${#GPU_POOL[@]}
elif ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_PARALLEL must be 'auto' or a positive integer" >&2
  exit 2
fi

# 防止手动设置的并行数超过 GPU 池大小
if [ "$MAX_PARALLEL" -gt "${#GPU_POOL[@]}" ]; then
  echo "[WARN] MAX_PARALLEL=$MAX_PARALLEL exceeds detected GPU count=${#GPU_POOL[@]}; clamping."
  MAX_PARALLEL=${#GPU_POOL[@]}
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
echo "CASES=$CASES"
echo "MAX_EPOCH=$MAX_EPOCH"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
echo "RESUME=$RESUME"
echo "MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-NA}"
echo "MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-NA}"
echo "RUN_CASE_STUDY=$RUN_CASE_STUDY"
echo "CASE_STUDY_MAX_BATCHES=$CASE_STUDY_MAX_BATCHES"
echo "FPEM_USE_GRAD_CONSENSUS=$FPEM_USE_GRAD_CONSENSUS"
echo "FPEM_USE_PRETRAINED_INV_AGCRN=$FPEM_USE_PRETRAINED_INV_AGCRN"
echo "FPEM_PRETRAINED_INV_AGCRN_PATH=$FPEM_PRETRAINED_INV_AGCRN_PATH"
echo "EXPERIMENT_NAMES=$(IFS=,; printf '%s' "${ENABLED_NAMES[*]}")"
echo "TOTAL_JOBS=$TOTAL_JOBS"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "============================================================"

if truthy "$PLAN_ONLY"; then
  echo "[PLAN_ONLY] no training will be started"
  for ((job_index = 0; job_index < TOTAL_JOBS; job_index++)); do
    echo "[PLAN] job=$job_index name=${JOB_NAMES[$job_index]} seed=${JOB_SEEDS[$job_index]}"
  done
  exit 0
fi

if [ ! -s "$SUMMARY_FILE" ]; then
  printf 'name\tseed\tstatus\trun_name\ttest_avg_mae\ttest_workday_mae\ttest_holiday_mae\tenv_day_acc\tenv_hour_acc\tenv_rush_acc\tinv_day_acc\teffective_expert_number\texpert0_soft_usage\texpert1_soft_usage\texpert2_soft_usage\texpert0_hard_count\texpert1_hard_count\texpert2_hard_count\tmax_expert_usage_ratio\tfpem_env_use_exogenous\tenv_exogenous_available\tenv_exogenous_time_available\tenv_exogenous_load_available\tenv_exogenous_feature_dim\tenv_exogenous_embedding_norm\tenv_exogenous_load_embedding_norm\tdetail\n' > "$SUMMARY_FILE"
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

echo "============================================================"
echo "[PARTIAL_STATUS]"
cat "$SUMMARY_FILE"
echo "============================================================"
ok_count="$(awk -F '\t' 'NR > 1 && $3 == "OK" {c++} END {print c+0}' "$SUMMARY_FILE")"
fail_count="$(awk -F '\t' 'NR > 1 && $3 == "FAIL" {c++} END {print c+0}' "$SUMMARY_FILE")"
echo "[DONE] OK=$ok_count FAIL=$fail_count summary=$SUMMARY_FILE"
exit "$FAILED"
