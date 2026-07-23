#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="/data/OuXiaoyu/mystg/baselines/STEVE_CODE/STEVE"

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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON=${PYTHON:-python}
export SEED=${SEED:-2024}
export MAX_RETRY=${MAX_RETRY:-2}
export RETRY_SLEEP=${RETRY_SLEEP:-180}
export MAX_EPOCH=${MAX_EPOCH:-100}
export FRESH_START=${FRESH_START:-false}
export RERUN_FAILED=${RERUN_FAILED:-true}


LOG_ROOT=${LOG_ROOT:-experiments/NYCTaxi_TDS/fpem_hyper_ablation_logs}
mkdir -p "$LOG_ROOT"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
EXP_ROOT=${EXP_ROOT:-experiments/NYCTaxi}

BASE_RUN_PREFIX=${BASE_RUN_PREFIX:-fpem_hyper_ablation}

# Keep this in sync with scripts/run_tds_nyctaxi_fpem_core_agcrn.sh.
BASE_ARGS=(
  --mode train
  --config_filename configs/NYCTaxi.yaml
  --model steve
  --dataset NYCTaxi
  --bs 64
  --seed "$SEED"
  --resume true
  --device cuda:0
  --max_epoch "$MAX_EPOCH"

  --fpem_backbone agcrn
  --agcrn_embed_dim 10
  --agcrn_num_layers 2
  --agcrn_cheb_k 2

  --fpem_use_env_mask true
  --fpem_env_mask_hidden_dim 64
  --fpem_env_mask_temperature 1.0
  --fpem_env_mask_warmup_epochs 0
  --fpem_lambda_mask_sparse 0.0005
  --fpem_lambda_mask_entropy 0.0005

  --fpem_use_env_route true
  --fpem_env_route_k 3
  --fpem_env_route_head_mode hyper_inv_film
  --fpem_env_route_use_inv_fallback_expert true
  --fpem_env_route_tau 1.0
  --fpem_env_route_oracle_tau 0.3
  --fpem_env_route_train_mode soft_oracle
  --fpem_env_route_hidden_dim 64
  --fpem_env_route_warmup_epochs 0
  --fpem_env_route_lambda_final 1.0
  --fpem_env_route_lambda_global 0.0
  --fpem_env_route_lambda_route_soft 0.5
  --fpem_env_route_lambda_expert 0.2
  --fpem_env_route_lambda_router_oracle 1.0
  --fpem_env_route_lambda_balance 0.01
  --fpem_env_route_lambda_diverse 0.01
  --fpem_env_route_lambda_entropy 0.0

  --fpem_use_env_fusion true
  --fpem_lambda_inv_pred 0.2
  --fpem_hyper_alpha_mode sample_gate
  --fpem_lambda_hyper_delta_norm 0.0001

  --fpem_use_future_mi true
  --fpem_lambda_future_mi 0.02
  --fpem_future_mi_warmup_epochs 0
  --fpem_future_mi_hidden_dim 64
  --fpem_future_mi_detach_target true

  --fpem_use_swap true
  --fpem_lambda_swap 0.01
  --fpem_swap_warmup_epochs 0
  --fpem_swap_margin 0.01
  --fpem_swap_gain_eta 0.0
  --fpem_swap_gain_tau 0.05
  --fpem_lambda_swap_diff 1.0
  --fpem_lambda_swap_same 0.05
  --fpem_swap_only_diff_route true
  --fpem_swap_detach_inv true
  --fpem_swap_detach_env false
  --fpem_use_swap_fallback_router_loss true
  --fpem_lambda_swap_fallback_router 0.005
  --fpem_swap_fallback_warmup_epochs 10

  --fpem_use_grad_consensus true
  --fpem_gc_pred_loss_only true
  --fpem_gc_inv_rho 0.3
  --fpem_gc_env_rho 0.3
  --fpem_gc_tau 0.5
  --fpem_gc_temp 0.1
  --fpem_gc_min_keep 0.2
  --fpem_gc_warmup_epochs 0
  --fpem_gc_route_min_samples 2

  --fpem_use_gradcompat_aux false
  --fpem_lambda_gradcompat_aux 0.0
)

cleanup_memory() {
  echo "[CLEANUP] date=$(date)"
  nvidia-smi || true

  # The failed Python process should already have exited. Do not kill other
  # users' or unrelated processes here; just do a gentle CUDA cache cleanup.
  "$PYTHON" - <<'PY' || true
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception as e:
    print("cuda cleanup skipped:", e)
PY

  nvidia-smi || true
}

is_memory_error() {
  local log_file="$1"
  local exit_code="${2:-0}"
  if [ "$exit_code" -eq 132 ] || [ "$exit_code" -eq 134 ] || [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 139 ]; then
    return 0
  fi
  grep -Eqi "CUDA out of memory|out of memory|CUDNN_STATUS_ALLOC_FAILED|cublas.*alloc|NCCL|Killed|bus error|illegal instruction|SIGILL|aborted|SIGABRT|illegal memory access|cuda runtime error|segmentation fault|segfault|SIGSEGV" "$log_file"
}

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

summary_status() {
  local name="$1"
  local run_name="$2"
  if [ ! -f "$SUMMARY_FILE" ]; then
    return 0
  fi
  awk -F '\t' -v name="$name" -v run="$run_name" '
    $1 == name && $3 == run { status = $2 }
    END { if (status != "") print status }
  ' "$SUMMARY_FILE"
}

record_summary() {
  local name="$1"
  local status="$2"
  local run_name="$3"
  local detail="${4:-}"
  local tmp_file="${SUMMARY_FILE}.tmp.$$"
  if [ -f "$SUMMARY_FILE" ]; then
    awk -F '\t' -v name="$name" -v run="$run_name" \
      '!(($1 == name) && ($3 == run))' "$SUMMARY_FILE" > "$tmp_file"
  else
    : > "$tmp_file"
  fi
  if [ -n "$detail" ]; then
    echo -e "${name}\t${status}\t${run_name}\t${detail}" >> "$tmp_file"
  else
    echo -e "${name}\t${status}\t${run_name}" >> "$tmp_file"
  fi
  mv "$tmp_file" "$SUMMARY_FILE"
}

checkpoint_progress() {
  local dir="$1"
  local progress=0
  local file
  local base
  local epoch

  if [ -f "${dir}/last_epoch.txt" ]; then
    epoch="$(tr -cd '0-9' < "${dir}/last_epoch.txt")"
    if [ -n "$epoch" ] && [ "$epoch" -gt "$progress" ]; then
      progress="$epoch"
    fi
  fi

  shopt -s nullglob
  for file in "${dir}"/epoch*.pth; do
    base="$(basename "$file")"
    epoch="${base#epoch}"
    epoch="${epoch%.pth}"
    if [[ "$epoch" =~ ^[0-9]+$ ]] && [ "$epoch" -gt "$progress" ]; then
      progress="$epoch"
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$progress"
}

find_resume_dir() {
  local run_name="$1"
  local best_dir=""
  local best_progress=-1
  local best_mtime=0
  local dir
  local progress
  local mtime

  shopt -s nullglob
  for dir in "${EXP_ROOT}/${run_name}_"*; do
    [ -d "$dir" ] || continue
    if [ -f "${dir}/last_model.pth" ] || [ -f "${dir}/best_model.pth" ] || compgen -G "${dir}/epoch*.pth" >/dev/null; then
      progress="$(checkpoint_progress "$dir")"
      mtime="$(stat -c %Y "$dir" 2>/dev/null || echo 0)"
      if [ "$progress" -gt "$best_progress" ] || { [ "$progress" -eq "$best_progress" ] && [ "$mtime" -ge "$best_mtime" ]; }; then
        best_dir="$dir"
        best_progress="$progress"
        best_mtime="$mtime"
      fi
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$best_dir"
}

run_one() {
  local name="$1"
  shift
  local extra_args=("$@")

  local run_name="${BASE_RUN_PREFIX}_${name}_seed${SEED}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local attempt=0
  local status
  local resume_dir

  status="$(summary_status "$name" "$run_name")"
  resume_dir="$(find_resume_dir "$run_name")"
  if [ -n "$resume_dir" ] && [ -f "${resume_dir}/result.npz" ]; then
    echo "[SKIP_DONE] $name result exists: $resume_dir"
    if [ "$status" != "OK" ]; then
      record_summary "$name" "OK" "$run_name" "existing_result=${resume_dir}"
    fi
    return 0
  fi
  if [ "$status" = "OK" ]; then
    echo "[SKIP_SUMMARY_OK] $name"
    return 0
  fi
  if [ "$status" = "FAIL" ] && ! truthy "$RERUN_FAILED"; then
    echo "[SKIP_SUMMARY_FAIL] $name (set RERUN_FAILED=true to retry)"
    return 0
  fi

  echo "============================================================"
  echo "[RUN] $name"
  echo "[RUN_NAME] $run_name"
  echo "[LOG] $log_file"
  echo "============================================================"
  {
    echo "============================================================"
    echo "[RUN] $name"
    echo "[RUN_NAME] $run_name"
    echo "[DATE] $(date)"
    echo "============================================================"
  } >> "$log_file"

  while [ "$attempt" -le "$MAX_RETRY" ]; do
    local resume_args=()
    resume_dir="$(find_resume_dir "$run_name")"
    if [ -n "$resume_dir" ]; then
      resume_args=(--resume_dir "$resume_dir")
      echo "[RESUME_DIR] $resume_dir" | tee -a "$log_file"
    fi

    echo "[ATTEMPT] $name attempt=$attempt / $MAX_RETRY" | tee -a "$log_file"

    "$PYTHON" run.py \
      "${BASE_ARGS[@]}" \
      --exp_name "$run_name" \
      "${resume_args[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee -a "$log_file"

    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
      echo "[OK] $name" | tee -a "$log_file"
      record_summary "$name" "OK" "$run_name"
      return 0
    fi

    echo "[FAIL] $name exit_code=$exit_code" | tee -a "$log_file"

    if is_memory_error "$log_file" "$exit_code"; then
      if [ "$attempt" -lt "$MAX_RETRY" ]; then
        echo "[MEMORY_ERROR] wait ${RETRY_SLEEP}s, cleanup memory, then retry" | tee -a "$log_file"
        sleep "$RETRY_SLEEP"
        cleanup_memory 2>&1 | tee -a "$log_file"
        attempt=$((attempt + 1))
        continue
      else
        echo "[MEMORY_ERROR] max retry reached" | tee -a "$log_file"
      fi
    else
      echo "[NON_MEMORY_ERROR] no retry" | tee -a "$log_file"
    fi

    record_summary "$name" "FAIL" "$run_name" "exit_code=${exit_code}"
    return "$exit_code"
  done
}

if truthy "$FRESH_START"; then
  : > "$SUMMARY_FILE"
else
  touch "$SUMMARY_FILE"
fi

FAILED=0

# 1. Full: complete method.
run_one full || FAILED=1

# 2. Inv Only: only y_inv; tests whether environment modulation helps.
run_one inv_only \
  --fpem_use_env_route false \
  --fpem_use_env_mask false \
  --fpem_use_swap_fallback_router_loss false \
  --fpem_use_future_mi false \
  --fpem_lambda_future_mi 0.0 \
  --fpem_lambda_hyper_delta_norm 0.0 || FAILED=1

# 3. No Inv Loss: no always-on y_inv supervision; tests whether fallback base is necessary.
run_one no_inv_loss \
  --fpem_lambda_inv_pred 0.0 \
  --fpem_env_route_lambda_global 0.0 || FAILED=1

# 4. No Env Mask: no environment filtering; tests whether mask is necessary.
run_one no_env_mask \
  --fpem_use_env_mask false || FAILED=1

# 5. No Swap Fallback: do not teach router to fall back on mismatched swapped env.
run_one no_swap_fallback \
  --fpem_use_swap_fallback_router_loss false \
  --fpem_lambda_swap_fallback_router 0.0 || FAILED=1

# 6. No Future MI: remove future-environment consistency/MI constraint.
run_one no_future_mi \
  --fpem_use_future_mi false \
  --fpem_lambda_future_mi 0.0 || FAILED=1

# 7. No Hyper Reg: no gamma/beta magnitude constraint; tests conservative modulation.
run_one no_hyper_reg \
  --fpem_lambda_hyper_delta_norm 0.0 || FAILED=1

# 8. No Alpha Gate: fixed environment modulation strength; tests adaptive alpha.
run_one no_alpha_gate \
  --fpem_hyper_alpha_mode fixed_one || FAILED=1

# 9. K=1: one environment-modulated candidate; tests whether multiple candidates help.
run_one k1 \
  --fpem_env_route_k 1 || FAILED=1

echo "============================================================"
echo "[SUMMARY]"
cat "$SUMMARY_FILE"
echo "============================================================"

exit "$FAILED"
