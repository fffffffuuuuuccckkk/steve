#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-data/LargeST-SD_TDS}"
GRAPH_FILE="${GRAPH_FILE:-data/LargeST-SD_TDS/adj_mx.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/LargeST_SD_OOD/AGCRN/seed2024}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
RNN_UNITS="${RNN_UNITS:-32}"
NUM_LAYERS="${NUM_LAYERS:-1}"
CHEB_K="${CHEB_K:-2}"
EMBED_DIM="${EMBED_DIM:-10}"

echo "[baseline] name=AGCRN"
echo "[baseline] paper=Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
echo "[baseline] original_repository=https://github.com/LeiBAI/AGCRN"
echo "[baseline] original_commit=7fbbf2aeb099242098a3cf482b55cd45d7295c28"
echo "[baseline] dataset=LargeST-SD_TDS"
echo "[baseline] train_period=2019"
echo "[baseline] val_period=2020 first half"
echo "[baseline] test_period=2020 second half / OOD"
echo "[baseline] dataset_dir=${DATASET_DIR}"
echo "[baseline] graph_file=${GRAPH_FILE}"
echo "[baseline] batch_size=${BATCH_SIZE} test_batch_size=${TEST_BATCH_SIZE}"
echo "[baseline] seed=${SEED}"
echo "[baseline] scaler=train x only"
echo "[baseline] checkpoint_selection=lowest validation MAE"

"$PYTHON" -m baselines.agcrn.trainer \
  --dataset_dir "$DATASET_DIR" \
  --graph_file "$GRAPH_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --rnn_units "$RNN_UNITS" \
  --num_layers "$NUM_LAYERS" \
  --cheb_k "$CHEB_K" \
  --embed_dim "$EMBED_DIM"
