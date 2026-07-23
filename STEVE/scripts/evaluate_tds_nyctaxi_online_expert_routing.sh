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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [ -z "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] CHECKPOINT_PATH is required" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT_PATH")/online_route_eval}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHES="${MAX_BATCHES:--1}"
NUM_RANDOM_TRIALS="${NUM_RANDOM_TRIALS:-20}"
RANDOM_SEED_BASE="${RANDOM_SEED_BASE:-20260721}"

"$PYTHON" tools/evaluate_online_expert_routing.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_batches "$MAX_BATCHES" \
  --num_random_trials "$NUM_RANDOM_TRIALS" \
  --random_seed_base "$RANDOM_SEED_BASE" \
  "$@"
