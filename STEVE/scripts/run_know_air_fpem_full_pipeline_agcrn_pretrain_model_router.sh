#!/usr/bin/env bash
set -euo pipefail

# KnowAir pipeline wrapper around the LargeST full pipeline.
# Defaults: BTHSA, GPU1, 35->1 as prepared, period day/week = 8/56.

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

export DATASET_NAME="${DATASET_NAME:-KnowAir-BTHSA_TDS}"
export CONFIG_FILE="${CONFIG_FILE:-configs/KnowAir_BTHSA_TDS.yaml}"
export GRAPH_FILE="${GRAPH_FILE:-data/KnowAir-BTHSA_TDS/adj_mx.npz}"
export RESULT_ROOT="${RESULT_ROOT:-experiments/KnowAir-BTHSA_TDS}"
export GPU_ID="${GPU_ID:-1}"
export RUN_TAG="${RUN_TAG:-0818_testbest_period8}"

export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
export PRETRAIN_TEST_BATCH_SIZE="${PRETRAIN_TEST_BATCH_SIZE:-16}"
export FPEM_BATCH_SIZE="${FPEM_BATCH_SIZE:-8}"
export FPEM_TEST_BATCH_SIZE="${FPEM_TEST_BATCH_SIZE:-16}"

export D_MODEL="${D_MODEL:-32}"
export AGCRN_RNN_UNITS="${AGCRN_RNN_UNITS:-32}"
export AGCRN_NUM_LAYERS="${AGCRN_NUM_LAYERS:-1}"
export AGCRN_CHEB_K="${AGCRN_CHEB_K:-2}"

export FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS="${FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS:-8}"
export FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS="${FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS:-56}"

echo "[KnowAir] dataset=${DATASET_NAME}"
echo "[KnowAir] config=${CONFIG_FILE}"
echo "[KnowAir] gpu=${GPU_ID}"
echo "[KnowAir] period day/week=${FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS}/${FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS}"
echo "[KnowAir] checkpoint selection follows base pipeline: test"

exec bash scripts/run_largest_sd_fpem_full_pipeline_agcrn_pretrain_model_router.sh
