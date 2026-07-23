#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  CONDA_ENV="${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
  conda activate "$CONDA_ENV"
fi
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-2024,2025,2026}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-pca}"
RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_progressive_gmm_0720}"
CASE_NAME="${CASE_NAME:-add_progressive_gmm_kmax3_common020}"
RESULT_ROOT="${RESULT_ROOT:-experiments/NYCTaxi_TDS}"
ROUTE_EVAL_ROOT="${ROUTE_EVAL_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_online_route_eval}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_case_study_logs}"
STATUS_DIR="${LOG_ROOT}/status"
mkdir -p "$LOG_ROOT" "$STATUS_DIR"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_name="${RUN_PREFIX}_${CASE_NAME}_seed${seed}"
  local exp_dir="${PROJECT_DIR}/${RESULT_ROOT}/${run_name}"
  local ckpt="${exp_dir}/best_val_model.pth"
  local route_eval_dir="${PROJECT_DIR}/${ROUTE_EVAL_ROOT}/${run_name}"
  local output_dir="${exp_dir}/case_study/progressive_gmm_case_outputs"
  local log_file="${LOG_ROOT}/${run_name}.case_study.log"
  local status_file="${STATUS_DIR}/${run_name}.status"

  if [ ! -f "$ckpt" ]; then
    echo "[CASE_STUDY_FAIL] missing best_val_model.pth: $ckpt" | tee "$log_file"
    printf 'FAIL\n' > "$status_file"
    return 2
  fi
  if [ ! -f "${route_eval_dir}/cluster_to_expert_mapping.json" ]; then
    echo "[CASE_STUDY_FAIL] missing route mapping: ${route_eval_dir}/cluster_to_expert_mapping.json" | tee "$log_file"
    printf 'FAIL\n' > "$status_file"
    return 2
  fi
  if [ ! -f "${route_eval_dir}/online_route_results.json" ]; then
    echo "[CASE_STUDY_FAIL] missing route results: ${route_eval_dir}/online_route_results.json" | tee "$log_file"
    printf 'FAIL\n' > "$status_file"
    return 2
  fi

  echo "[CASE_STUDY_RUN] seed=$seed gpu=$gpu exp=$exp_dir output=$output_dir"
  printf 'RUNNING\n' > "$status_file"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/analyze_progressive_gmm_case_study.py \
    --exp_dir "$exp_dir" \
    --checkpoint "$ckpt" \
    --route_eval_dir "$route_eval_dir" \
    --output_dir "$output_dir" \
    --seed "$seed" \
    --device cuda:0 \
    --max_samples "$MAX_SAMPLES" \
    --embedding_method "$EMBEDDING_METHOD" \
    > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ -f "${output_dir}/metadata.json" ]; then
    echo "[CASE_STUDY_OK] seed=$seed output=$output_dir"
    printf 'OK\n' > "$status_file"
    return 0
  fi
  echo "[CASE_STUDY_FAIL] seed=$seed rc=$rc log=$log_file"
  printf 'FAIL\n' > "$status_file"
  return "$rc"
}

job=0
running=0
for seed in "${SEED_LIST[@]}"; do
  gpu="${GPU_POOL[$((job % ${#GPU_POOL[@]}))]}"
  while [ "$running" -ge "$MAX_PARALLEL" ]; do
    if ! wait -n; then
      true
    fi
    running=$((running - 1))
  done
  run_one "$gpu" "$seed" &
  running=$((running + 1))
  job=$((job + 1))
done

while [ "$running" -gt 0 ]; do
  if ! wait -n; then
    true
  fi
  running=$((running - 1))
done

"$PYTHON" scripts/analyze_progressive_gmm_case_study.py \
  --run_all \
  --summary_only \
  --seeds "$SEEDS" \
  --device cpu \
  --embedding_method pca \
  > "${LOG_ROOT}/three_seed_summary.log" 2>&1 || true

ok_count=$( (grep -Rhs '^OK$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
fail_count=$( (grep -Rhs '^FAIL$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')
echo "[CASE_STUDY_DONE] OK=${ok_count} FAIL=${fail_count}"
if [ "$fail_count" -gt 0 ]; then
  exit 1
fi
