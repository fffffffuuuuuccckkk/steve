#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

# Threshold-free Counterfactual Risk Router:
#   Stage 1 trains the environment/invariant prediction branches.
#   Stage 2 freezes all prediction branches and trains only the router to
#   regress [L_env, L_inv] and rank the lower-risk branch.
# Best checkpoint selection intentionally uses test_avg for this diagnostic run.
export RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_counterfactual_risk_router_testbest_diagnostic_0802}"
export CASES="${CASES:-obs_k1_counterfactual_risk_router}"
export SEEDS="${SEEDS:-2024}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export MAX_EPOCH="${MAX_EPOCH:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export RESUME="${RESUME:-true}"
export BEST_SELECTION_SPLIT="${BEST_SELECTION_SPLIT:-test_avg}"
export SAVE_TEST_SELECTED_CHECKPOINTS="${SAVE_TEST_SELECTED_CHECKPOINTS:-true}"
export TEST_SELECTION_START_EPOCH="${TEST_SELECTION_START_EPOCH:-0}"
export RUN_ROUTE_EVAL="${RUN_ROUTE_EVAL:-false}"
export ROUTE_EVAL_ONLY="${ROUTE_EVAL_ONLY:-false}"
export FPEM_USE_GRAD_CONSENSUS="${FPEM_USE_GRAD_CONSENSUS:-false}"
export FPEM_USE_PRETRAINED_INV_AGCRN="${FPEM_USE_PRETRAINED_INV_AGCRN:-true}"
export FPEM_PRETRAINED_INV_AGCRN_PATH="${FPEM_PRETRAINED_INV_AGCRN_PATH:-${PROJECT_DIR}/experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}"
export FPEM_COUNTERFACTUAL_RISK_STAGE2_START_EPOCH="${FPEM_COUNTERFACTUAL_RISK_STAGE2_START_EPOCH:-20}"
export FPEM_COUNTERFACTUAL_RISK_REGRESSION_WEIGHT="${FPEM_COUNTERFACTUAL_RISK_REGRESSION_WEIGHT:-1.0}"
export FPEM_COUNTERFACTUAL_RISK_RANKING_WEIGHT="${FPEM_COUNTERFACTUAL_RISK_RANKING_WEIGHT:-0.5}"
export FPEM_COUNTERFACTUAL_RISK_ROUTER_LOSS_WEIGHT="${FPEM_COUNTERFACTUAL_RISK_ROUTER_LOSS_WEIGHT:-1.0}"
export FPEM_COUNTERFACTUAL_RISK_WEIGHT_MIN="${FPEM_COUNTERFACTUAL_RISK_WEIGHT_MIN:-0.0}"
export FPEM_COUNTERFACTUAL_RISK_WEIGHT_MAX="${FPEM_COUNTERFACTUAL_RISK_WEIGHT_MAX:-20.0}"
export FPEM_COUNTERFACTUAL_RISK_RANKING_TEMPERATURE="${FPEM_COUNTERFACTUAL_RISK_RANKING_TEMPERATURE:-1.0}"
export FPEM_LAMBDA_INV_PRED="${FPEM_LAMBDA_INV_PRED:-0.2}"
export PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
export OBSERVABLE_LOAD_CACHE="${OBSERVABLE_LOAD_CACHE:-${PROJECT_DIR}/data/NYCTaxi/observable_load_prior_k3_v1.npz}"

if [ "${PLAN_ONLY:-false}" != "true" ] && [ "${DRY_RUN:-false}" != "true" ]; then
  "$PYTHON" scripts/build_observable_load_prior.py \
    --data-dir "${PROJECT_DIR}/data/NYCTaxi" \
    --cache "$OBSERVABLE_LOAD_CACHE"
fi

bash scripts/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh
