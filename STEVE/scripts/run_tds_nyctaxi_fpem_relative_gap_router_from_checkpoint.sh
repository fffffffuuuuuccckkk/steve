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
BASE_EXP_DIR="${BASE_EXP_DIR:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_confounder_dep_norm_align_obs_k1_counterfactual_risk_router_period_context_0813_conf_gci_seed2024}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_val_model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/frozen_relative_gap_router_period_context_0814_conf_gci}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ROUTER_DROPOUT="${ROUTER_DROPOUT:-0.0}"
EMA_BETA="${EMA_BETA:-0.99}"
GAP_EPS="${GAP_EPS:-1.0e-6}"
CASE="${CASE:-relative_gap_ema_huber_loadstats_preddiff}"

mkdir -p "$OUTPUT_ROOT"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] BASE_EXP_DIR=${BASE_EXP_DIR}"
echo "[INFO] CHECKPOINT_NAME=${CHECKPOINT_NAME}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] CASE=${CASE} GPU_ID=${GPU_ID} SEED=${SEED}"
echo "[INFO] target=relative_gap_ema_huber route_rule='r>0=>invariant,r<=0=>environment'"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
"$PYTHON" scripts/train_tds_nyctaxi_fpem_relative_gap_router_from_checkpoint.py \
  --base_exp_dir "$BASE_EXP_DIR" \
  --checkpoint_name "$CHECKPOINT_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --case "$CASE" \
  --device cuda:0 \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$ROUTER_DROPOUT" \
  --lr "$LR" \
  --ema_beta "$EMA_BETA" \
  --gap_eps "$GAP_EPS" \
  --include_load_stats true \
  --include_pred_diff true \
  --standardize_features true

echo "[DONE] OK"
