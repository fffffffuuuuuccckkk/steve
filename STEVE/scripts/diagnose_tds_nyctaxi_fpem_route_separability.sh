#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
fi
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_add_k3_hard_prediction_sinkhorn_seed2024/best_val_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CKPT_PATH")/route_diagnosis}"
SPLITS="${SPLITS:-train,val,test_mixed,test_workday,test_holiday}"
MAX_BATCHES="${MAX_BATCHES:--1}"
DEVICE="${DEVICE:-cuda:0}"

"$PYTHON" tools/diagnose_route_separability.py \
  --ckpt_path "$CKPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --splits "$SPLITS" \
  --max_batches "$MAX_BATCHES" \
  --device "$DEVICE"
