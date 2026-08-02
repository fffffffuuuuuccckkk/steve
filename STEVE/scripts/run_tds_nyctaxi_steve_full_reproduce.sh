#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
CONDA_ENV="${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
PYTHON="${PYTHON:-python}"

# Start with one faithful STEVE-Full reproduction.  Extra original-STEVE
# invariant baselines can be enabled with:
#   CASES=full,inv_only_with_disentangle,inv_only_no_env
# Override, for example:
#   SEEDS=2024,2025,2026 GPU_IDS=0,1,2 MAX_PARALLEL=3 bash ...
SEEDS="${SEEDS:-2024}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
CASES="${CASES:-full}"

# The public NYCTaxi config currently contains epochs: 3, which is too small
# for a real reproduction run.  Keep every other original setting and run the
# full schedule by default.  Use EPOCHS=3 if a strict config-file smoke run is
# desired.
EPOCHS="${EPOCHS:-128}"
RESUME="${RESUME:-true}"
FORCE="${FORCE:-false}"
RUN_PREFIX="${RUN_PREFIX:-steve_full_reproduce}"

source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
IFS=',' read -r -a CASE_LIST <<< "${CASES}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  GPU_LIST=(0)
fi

RESULT_ROOT="${PROJECT_DIR}/experiments/NYCTaxi_TDS"
LOG_ROOT="${RESULT_ROOT}/${RUN_PREFIX}_logs"
mkdir -p "${LOG_ROOT}"

case_run_name() {
  local case_name="$1"
  local seed="$2"
  case "${case_name}" in
    full)
      printf 'steve_full_reproduce_seed%s' "${seed}"
      ;;
    inv_only_with_disentangle|inv_only_disentangle)
      printf 'steve_inv_only_with_disentangle_seed%s' "${seed}"
      ;;
    inv_only_no_env|no_env_inv_only|inv_only_single_stream)
      printf 'steve_inv_only_no_env_seed%s' "${seed}"
      ;;
    *)
      printf '[steve-repro] unknown case: %s\n' "${case_name}" >&2
      return 1
      ;;
  esac
}

case_prediction_mode() {
  local case_name="$1"
  case "${case_name}" in
    full)
      printf 'full'
      ;;
    inv_only_with_disentangle|inv_only_disentangle)
      printf 'inv_only_with_disentangle'
      ;;
    inv_only_no_env|no_env_inv_only|inv_only_single_stream)
      printf 'inv_only_no_env'
      ;;
    *)
      printf '[steve-repro] unknown case: %s\n' "${case_name}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local case_name="$1"
  local seed="$2"
  local gpu="$3"
  local run_name
  local prediction_mode
  run_name="$(case_run_name "${case_name}" "${seed}")"
  prediction_mode="$(case_prediction_mode "${case_name}")"
  local run_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"

  if [ "${FORCE}" != "true" ] && [ -f "${run_dir}/best_model.pth" ] && [ -f "${run_dir}/result.npz" ]; then
    printf '[steve-repro] skip existing %s\n' "${run_name}"
    return 0
  fi

  mkdir -p "${run_dir}"
  printf '[steve-repro] start %s case=%s mode=%s seed=%s gpu=%s epochs=%s\n' \
    "${run_name}" "${case_name}" "${prediction_mode}" "${seed}" "${gpu}" "${EPOCHS}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" run.py \
    --mode train \
    --config_filename configs/NYCTaxi.yaml \
    --dataset NYCTaxi_TDS \
    --graph_file data/NYCTaxi_TDS/adj_mx.npz \
    --seed "${seed}" \
    --device cuda:0 \
    --max_epoch "${EPOCHS}" \
    --ablation all \
    --log_dir "${run_dir}" \
    --resume "${RESUME}" \
    --model_impl steve_original \
    --steve_prediction_mode "${prediction_mode}" \
    --save_samplewise true \
    > "${log_file}" 2>&1
  printf '[steve-repro] done %s\n' "${run_name}"
}

job_index=0
for seed in "${SEED_LIST[@]}"; do
  for case_name in "${CASE_LIST[@]}"; do
    gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
    run_one "${case_name}" "${seed}" "${gpu}" &
    job_index=$((job_index + 1))
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
      wait -n || true
    done
  done
done
wait

fail_count=0
for seed in "${SEED_LIST[@]}"; do
  for case_name in "${CASE_LIST[@]}"; do
    run_name="$(case_run_name "${case_name}" "${seed}")"
    run_dir="${RESULT_ROOT}/${run_name}"
    if [ ! -f "${run_dir}/best_model.pth" ] || [ ! -f "${run_dir}/result.npz" ]; then
      printf '[steve-repro] FAIL missing checkpoint/result: %s\n' "${run_dir}" >&2
      fail_count=$((fail_count + 1))
    else
      printf '[steve-repro] OK %s\n' "${run_dir}"
    fi
  done
done

if [ "${fail_count}" -ne 0 ]; then
  exit 1
fi

printf '[steve-repro] ALL DONE. logs=%s\n' "${LOG_ROOT}"
