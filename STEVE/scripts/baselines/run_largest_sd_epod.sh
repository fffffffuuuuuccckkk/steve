#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-data/LargeST-SD_TDS}"
GRAPH_FILE="${GRAPH_FILE:-data/LargeST-SD_TDS/adj_mx.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/LargeST_SD_OOD/EpoD/seed2024}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
NUM_LAYERS="${NUM_LAYERS:-1}"
CHEB_K="${CHEB_K:-2}"
EMBED_DIM="${EMBED_DIM:-10}"
GRAPH_HOPS="${GRAPH_HOPS:-5}"
PROMPT_LOSS_WEIGHT="${PROMPT_LOSS_WEIGHT:-1.0}"
BETA="${BETA:-0.1}"
PROMPT_NOISE_STD="${PROMPT_NOISE_STD:-0.01}"

echo "[baseline] name=EpoD"
echo "[baseline] paper=Improving Generalization of Dynamic Graph Learning via Environment Prompt"
echo "[baseline] original_repository=not found; non-official paper adapter"
echo "[baseline] original_commit=non-official-paper-adapter"
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
echo "[baseline] external_env_labels=false"
echo "[baseline] test_time_adaptation=false"
echo "[baseline] graph_hops=${GRAPH_HOPS}"

"$PYTHON" -m baselines.epod.trainer \
  --dataset_dir "$DATASET_DIR" \
  --graph_file "$GRAPH_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_layers "$NUM_LAYERS" \
  --cheb_k "$CHEB_K" \
  --embed_dim "$EMBED_DIM" \
  --graph_hops "$GRAPH_HOPS" \
  --prompt_loss_weight "$PROMPT_LOSS_WEIGHT" \
  --beta "$BETA" \
  --prompt_noise_std "$PROMPT_NOISE_STD"

