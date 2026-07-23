#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_DIR="/data/OuXiaoyu/STEVE_CODE/STEVE"
export CUDA_VISIBLE_DEVICES=0,1,2,3
source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
conda activate basicts

cd "$PROJECT_DIR" || exit 1

export PYTHON=${PYTHON:-python}
export SEED=${SEED:-2024}
export MAX_EPOCH=${MAX_EPOCH:-100}
export MAX_RETRY=${MAX_RETRY:-2}
export RETRY_SLEEP=${RETRY_SLEEP:-180}
export FORCE=${FORCE:-false}
export MEMORY_RETRY_FOREVER=${MEMORY_RETRY_FOREVER:-true}
export EXIT_ON_MEMORY_FAIL=${EXIT_ON_MEMORY_FAIL:-true}
export FPEM_USE_GRAD_CONSENSUS=${FPEM_USE_GRAD_CONSENSUS:-false}
export RUN_PREFIX=${RUN_PREFIX:-fpem_agcrn_aligned_k1_noinv}
export RUN_HYPER_ABLATIONS=${RUN_HYPER_ABLATIONS:-false}
export FPEM_ENV_ROUTE_K=${FPEM_ENV_ROUTE_K:-1}
export FPEM_LAMBDA_INV_PRED=${FPEM_LAMBDA_INV_PRED:-0.0}
export FPEM_USE_FUTURE_MI=${FPEM_USE_FUTURE_MI:-true}
export FPEM_USE_SWAP=${FPEM_USE_SWAP:-true}
export FPEM_USE_CLUB_MI=${FPEM_USE_CLUB_MI:-true}
export FPEM_USE_CONFOUNDER_EXTRACTOR=${FPEM_USE_CONFOUNDER_EXTRACTOR:-true}
export GPU_MAX_USED_MB=${GPU_MAX_USED_MB:-1024}
export GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}

RESULT_ROOT=${RESULT_ROOT:-experiments/NYCTaxi_TDS}
LOG_ROOT=${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_logs}
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
mkdir -p "$LOG_ROOT"

if [ ! -e data/NYCTaxi_TDS ]; then
  ln -s NYCTaxi data/NYCTaxi_TDS
fi

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

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
except Exception as e:
    print("cuda cleanup skipped:", e)
PY
  nvidia-smi -i "$gpu_id" || true
}

is_memory_error() {
  local log_file="$1"
  local exit_code="${2:-0}"
  if [ "$exit_code" -eq 132 ] || [ "$exit_code" -eq 134 ] || [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 139 ]; then
    return 0
  fi
  grep -Eqi "CUDA out of memory|out of memory|CUDNN_STATUS_ALLOC_FAILED|cublas.*alloc|NCCL|Killed|bus error|illegal instruction|SIGILL|aborted|SIGABRT|illegal memory access|cuda runtime error|segmentation fault|segfault|SIGSEGV" "$log_file"
}

record_summary() {
  local name="$1"
  local status="$2"
  local run_name="$3"
  local detail="${4:-}"
  local tmp_file="${SUMMARY_FILE}.tmp.$$.${RANDOM}"
  (
    flock -x 9
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
  ) 9>"${SUMMARY_FILE}.lock"
}

# Aligned full: concat_input route head, K=1, no invariant auxiliary loss.
# Mask/hyper/fallback toggles are not in the main table because they are
# inactive under concat_input + confounder extractor.
BASE_ARGS=(
  --config_filename configs/NYCTaxi.yaml
  --dataset NYCTaxi_TDS
  --data_dir data
  --graph_file data/NYCTaxi_TDS/adj_mx.npz
  --model steve
  --seed "$SEED"
  --epochs "$MAX_EPOCH"
  --batch_size 16
  --test_batch_size 64
  --device cuda:0
  --train_work_per_holiday 2.5
  --result_root "$RESULT_ROOT"
  --resume true
  --early_stop_test_avg_mae_epoch 40
  --early_stop_test_avg_mae_threshold 12

  --fpem_backbone agcrn
  --agcrn_embed_dim 10
  --agcrn_num_layers 2
  --agcrn_cheb_k 2

  --fpem_use_env_mask true
  --fpem_env_mask_hidden_dim 64
  --fpem_env_mask_temperature 1.0
  --fpem_env_mask_warmup_epochs 5
  --fpem_lambda_mask_sparse 0.0005
  --fpem_lambda_mask_entropy 0.0005

  --fpem_use_env_route true
  --fpem_env_route_k "$FPEM_ENV_ROUTE_K"
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

  --fpem_use_env_fusion true
  --fpem_lambda_inv_pred "$FPEM_LAMBDA_INV_PRED"
  --fpem_hyper_alpha_mode sample_gate
  --fpem_lambda_hyper_delta_norm 0.0001

  --fpem_use_future_mi "$FPEM_USE_FUTURE_MI"
  --fpem_lambda_future_mi 0.02
  --fpem_future_mi_target_mode env_encoder
  --fpem_future_mi_warmup_epochs 5
  --fpem_future_mi_hidden_dim 64
  --fpem_future_mi_detach_target true

  --fpem_use_swap "$FPEM_USE_SWAP"
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
  --fpem_use_swap_fallback_router_loss true
  --fpem_lambda_swap_fallback_router 0.005
  --fpem_swap_fallback_warmup_epochs 30

  --fpem_use_club_mi "$FPEM_USE_CLUB_MI"
  --fpem_lambda_club_mi 0.01
  --fpem_use_confounder_extractor "$FPEM_USE_CONFOUNDER_EXTRACTOR"

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

run_one() {
  local gpu_id="$1"
  local name="$2"
  shift 2
  local extra_args=("$@")
  local run_name="${RUN_PREFIX}_${name}_seed${SEED}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local exp_dir="${RESULT_ROOT}/${run_name}"
  local attempt=0

  echo "============================================================"
  echo "[RUN] $name"
  echo "[RUN_NAME] $run_name"
  echo "[GPU] $gpu_id"
  echo "[LOG] $log_file"
  echo "============================================================"

  if [ -f "${exp_dir}/summary.json" ] && ! truthy "$FORCE"; then
    if "$PYTHON" - "$exp_dir/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    s = json.load(f)
sys.exit(0 if s.get("finished") is True else 1)
PY
    then
      echo "[SKIP_DONE] $name summary finished: ${exp_dir}/summary.json"
      record_summary "$name" "OK" "$run_name" "existing_summary=${exp_dir}/summary.json"
      return 0
    fi
  fi

  # if truthy "$FORCE" && [ -d "$exp_dir" ]; then
  #   rm -rf "$exp_dir"
  # fi

  while true; do
    {
      echo "============================================================"
      echo "[RUN] $name"
      echo "[RUN_NAME] $run_name"
      echo "[GPU] $gpu_id"
      if truthy "$MEMORY_RETRY_FOREVER"; then
        echo "[ATTEMPT] $attempt / forever(memory errors)"
      else
        echo "[ATTEMPT] $attempt / $MAX_RETRY"
      fi
      echo "[DATE] $(date)"
      echo "============================================================"
    } | tee -a "$log_file"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" run_tds_nyctaxi.py \
      "${BASE_ARGS[@]}" \
      --exp_name "$run_name" \
      --ablation "$name" \
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
      if truthy "$MEMORY_RETRY_FOREVER" || [ "$attempt" -lt "$MAX_RETRY" ]; then
        echo "[MEMORY_ERROR] stay on current experiment; wait ${RETRY_SLEEP}s, cleanup memory, then resume retry" | tee -a "$log_file"
        echo "[RESUME_AFTER_MEMORY_ERROR] exp_dir=${exp_dir} resume=true" | tee -a "$log_file"
        sleep "$RETRY_SLEEP"
        cleanup_memory "$gpu_id" 2>&1 | tee -a "$log_file"
        attempt=$((attempt + 1))
        continue
      fi
      echo "[MEMORY_ERROR] max retry reached for current experiment" | tee -a "$log_file"
      record_summary "$name" "FAIL" "$run_name" "exit_code=${exit_code};memory_error=max_retry"
      if truthy "$EXIT_ON_MEMORY_FAIL"; then
        echo "[EXIT_ON_MEMORY_FAIL] stop script instead of running next ablation" | tee -a "$log_file"
        exit "$exit_code"
      fi
      return "$exit_code"
    fi

    record_summary "$name" "FAIL" "$run_name" "exit_code=${exit_code}"
    return "$exit_code"
  done
}

# Valid main ablations:
# full, inv_only, k3, with_inv_loss, no_future_mi, no_swap, no_club,
# no_confounder_extractor.
# Optional hyper_* jobs are a separate comparison against hyper_full only.
run_ablation() {
  local job_index="$1"
  local gpu_id="$2"
  case "$job_index" in
    # 0) run_one "$gpu_id" full ;;
    # 1) run_one "$gpu_id" inv_only \
    #      --fpem_use_env_route false \
    #      --fpem_use_env_fusion false \
    #      --fpem_use_env_mask false \
    #      --fpem_use_future_mi false \
    #      --fpem_lambda_future_mi 0.0 \
    #      --fpem_use_swap false \
    #      --fpem_lambda_swap 0.0 \
    #      --fpem_use_club_mi false \
    #      --fpem_lambda_club_mi 0.0 ;;
    2) run_one "$gpu_id" k3 \
         --fpem_env_route_k 3 \
         --fpem_env_route_lambda_balance 0.01 \
         --fpem_env_route_lambda_diverse 0.01 ;;
    3) run_one "$gpu_id" with_inv_loss \
         --fpem_lambda_inv_pred 0.2 ;;
    # 4) run_one "$gpu_id" no_future_mi \
        #  --fpem_use_future_mi false \
        #  --fpem_lambda_future_mi 0.0 ;;
    5) run_one "$gpu_id" no_swap \
         --fpem_use_swap false \
         --fpem_lambda_swap 0.0 ;;
    # 6) run_one "$gpu_id" no_club \
    #      --fpem_use_club_mi false \
    #      --fpem_lambda_club_mi 0.0 ;;
    7) run_one "$gpu_id" no_confounder_extractor \
         --fpem_use_confounder_extractor false ;;
    8) run_one "$gpu_id" hyper_full \
         --fpem_env_route_head_mode hyper_inv_film \
         --fpem_env_route_use_inv_fallback_expert true \
         --fpem_use_swap_fallback_router_loss true \
         --fpem_lambda_swap_fallback_router 0.005 \
         --fpem_hyper_alpha_mode sample_gate \
         --fpem_lambda_hyper_delta_norm 0.0001 ;;
    9) run_one "$gpu_id" hyper_no_hyper_reg \
         --fpem_env_route_head_mode hyper_inv_film \
         --fpem_env_route_use_inv_fallback_expert true \
         --fpem_use_swap_fallback_router_loss true \
         --fpem_lambda_swap_fallback_router 0.005 \
         --fpem_hyper_alpha_mode sample_gate \
         --fpem_lambda_hyper_delta_norm 0.0 ;;
    10) run_one "$gpu_id" hyper_no_alpha_gate \
         --fpem_env_route_head_mode hyper_inv_film \
         --fpem_env_route_use_inv_fallback_expert true \
         --fpem_use_swap_fallback_router_loss true \
         --fpem_lambda_swap_fallback_router 0.005 \
         --fpem_hyper_alpha_mode fixed_one \
         --fpem_lambda_hyper_delta_norm 0.0001 ;;
    11) run_one "$gpu_id" hyper_no_swap_fallback \
         --fpem_env_route_head_mode hyper_inv_film \
         --fpem_env_route_use_inv_fallback_expert true \
         --fpem_use_swap_fallback_router_loss false \
         --fpem_lambda_swap_fallback_router 0.0 \
         --fpem_hyper_alpha_mode sample_gate \
         --fpem_lambda_hyper_delta_norm 0.0001 ;;
    *) echo "Unknown ablation job index: $job_index" >&2; return 2 ;;
  esac
}

gpu_memory_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d '[:space:]'
}

gpu_is_available() {
  local used
  used="$(gpu_memory_used_mb "$1")"
  [ -n "$used" ] && [ "$used" -le "$GPU_MAX_USED_MB" ]
}

detect_gpu_pool() {
  local requested="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
  if [ -n "$requested" ]; then
    printf '%s\n' "$requested" | tr ',' '\n' | sed '/^[[:space:]]*$/d;s/[[:space:]]//g'
  else
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  fi
}

mapfile -t GPU_POOL < <(detect_gpu_pool)
if [ "${#GPU_POOL[@]}" -eq 0 ]; then
  echo "[ERROR] no NVIDIA GPU detected" >&2
  exit 1
fi

touch "$SUMMARY_FILE"
FAILED=0
TOTAL_JOBS=8
if truthy "$RUN_HYPER_ABLATIONS"; then
  TOTAL_JOBS=12
fi
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

echo "============================================================"
echo "[EFFECTIVE_CONFIG]"
echo "RUN_PREFIX=$RUN_PREFIX"
echo "SEED=$SEED"
echo "MAX_EPOCH=$MAX_EPOCH"
echo "FPEM_USE_GRAD_CONSENSUS=$FPEM_USE_GRAD_CONSENSUS"
echo "fpem_env_route_k=$FPEM_ENV_ROUTE_K"
echo "fpem_lambda_inv_pred=$FPEM_LAMBDA_INV_PRED"
echo "fpem_use_future_mi=$FPEM_USE_FUTURE_MI"
echo "fpem_use_swap=$FPEM_USE_SWAP"
echo "fpem_use_club_mi=$FPEM_USE_CLUB_MI"
echo "fpem_use_confounder_extractor=$FPEM_USE_CONFOUNDER_EXTRACTOR"
echo "RUN_HYPER_ABLATIONS=$RUN_HYPER_ABLATIONS"
echo "============================================================"
echo "[SCHEDULER] detected_gpus=${GPU_POOL[*]} count=${#GPU_POOL[@]} max_used_mb=$GPU_MAX_USED_MB total_jobs=$TOTAL_JOBS"

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
          echo "[SCHEDULER] finished job=$job_index gpu=$gpu_id status=0"
        else
          status=$?
          echo "[SCHEDULER] finished job=$job_index gpu=$gpu_id status=$status"
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
    [ -n "${WORKER_PID[$gpu_id]:-}" ] && continue
    if gpu_is_available "$gpu_id"; then
      job_index="$NEXT_JOB"
      echo "[SCHEDULER] launch job=$job_index gpu=$gpu_id used_mb=$(gpu_memory_used_mb "$gpu_id")"
      run_ablation "$job_index" "$gpu_id" &
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

SUMMARY_ARGS=(--result_root "$RESULT_ROOT" --seed "$SEED" --run_prefix "$RUN_PREFIX")
if truthy "$RUN_HYPER_ABLATIONS"; then
  SUMMARY_ARGS+=(--include_hyper)
fi
"$PYTHON" scripts/summarize_tds_fpem_agcrn_aligned.py "${SUMMARY_ARGS[@]}" || true

echo "============================================================"
echo "[SUMMARY]"
cat "$SUMMARY_FILE"
echo "============================================================"

exit "$FAILED"
