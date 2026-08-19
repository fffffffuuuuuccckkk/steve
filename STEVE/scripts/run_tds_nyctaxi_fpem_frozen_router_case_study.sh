#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-basicts}"
fi

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2024}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STAGE1_ROOT="${STAGE1_ROOT:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_counterfactual_risk_router_testbest_diagnostic_0802}"
STAGE2_ROOT="${STAGE2_ROOT:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/frozen_router_feature_diagnostic_epoch18_testbest_0802}"
CASE="${CASE:-D_std_regret_bce_loadstats_preddiff}"

echo "[case-study] project=${PROJECT_DIR}"
echo "[case-study] stage1_root=${STAGE1_ROOT}"
echo "[case-study] stage2_root=${STAGE2_ROOT}"
echo "[case-study] case=${CASE} device=${DEVICE}"

"$PYTHON" scripts/analyze_tds_nyctaxi_fpem_frozen_router_case_study.py \
  --stage1_root "$STAGE1_ROOT" \
  --stage2_root "$STAGE2_ROOT" \
  --case "$CASE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE"
