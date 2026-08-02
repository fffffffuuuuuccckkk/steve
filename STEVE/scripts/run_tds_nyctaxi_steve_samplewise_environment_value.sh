#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
CONDA_ENV="${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-2024,2025,2026}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
FORCE="${FORCE:-false}"
RESUME="${RESUME:-true}"
TIE_THRESHOLD="${TIE_THRESHOLD:-1e-6}"

source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  GPU_LIST=(0)
fi

RESULT_ROOT="${PROJECT_DIR}/experiments/NYCTaxi_TDS"
LOG_ROOT="${RESULT_ROOT}/steve_samplewise_environment_value_logs"
ANALYSIS_DIR="${RESULT_ROOT}/steve_samplewise_environment_analysis"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_DIR}"

run_case() {
  local mode="$1"
  local seed="$2"
  local gpu="$3"
  local run_name="steve_env_value_${mode}_seed${seed}"
  local run_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"

  if [ "${FORCE}" != "true" ] && [ -f "${run_dir}/sample_predictions.npz" ] && [ -f "${run_dir}/sample_metrics.csv" ]; then
    printf '[steve-env-value] skip existing %s\n' "${run_name}"
    return 0
  fi

  mkdir -p "${run_dir}"
  printf '[steve-env-value] start %s on gpu %s\n' "${run_name}" "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" run.py \
    --mode train \
    --config_filename configs/NYCTaxi.yaml \
    --seed "${seed}" \
    --device cuda:0 \
    --dataset NYCTaxi_TDS \
    --graph_file data/NYCTaxi_TDS/adj_mx.npz \
    --log_dir "${run_dir}" \
    --resume "${RESUME}" \
    --model_impl steve_original \
    --steve_prediction_mode "${mode}" \
    --save_samplewise true \
    > "${log_file}" 2>&1
  printf '[steve-env-value] done %s\n' "${run_name}"
}

job_index=0
for seed in "${SEED_LIST[@]}"; do
  for mode in full inv_only; do
    gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
    run_case "${mode}" "${seed}" "${gpu}" &
    job_index=$((job_index + 1))
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
      wait -n || true
    done
  done
done
wait

fail_count=0
for seed in "${SEED_LIST[@]}"; do
  for mode in full inv_only; do
    run_dir="${RESULT_ROOT}/steve_env_value_${mode}_seed${seed}"
    if [ ! -f "${run_dir}/sample_predictions.npz" ] || [ ! -f "${run_dir}/sample_metrics.csv" ]; then
      printf '[steve-env-value] MISSING sample-wise outputs: %s\n' "${run_dir}" >&2
      fail_count=$((fail_count + 1))
    else
      printf '[steve-env-value] OK sample-wise outputs: %s\n' "${run_dir}"
    fi
  done
done

if [ "${fail_count}" -ne 0 ]; then
  printf '[steve-env-value] FAIL: %s runs missing outputs; skip analysis\n' "${fail_count}" >&2
  exit 1
fi

"${PYTHON}" scripts/analyze_steve_samplewise_environment_value.py \
  --seeds "${SEEDS}" \
  --tie_threshold "${TIE_THRESHOLD}" \
  --output_dir "${ANALYSIS_DIR}" \
  | tee "${LOG_ROOT}/analysis.log"

printf '[steve-env-value] ALL DONE. analysis_dir=%s\n' "${ANALYSIS_DIR}"
