#!/usr/bin/env bash
set -euo pipefail

# Launcher for the GCI graph-linear alignment version.
# It delegates to run_tds_nyctaxi_fpem_confounder_dep_agcrn_aligned.sh, which:
#   - loads pretrained frozen invariant AGCRN by default;
#   - ignores stored future c;
#   - runs none/gci/scd/both confounder cases by default.

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_confounder_dep_norm_align_obs_k1_counterfactual_risk_router_period_context_0813}"
LOG_ROOT="${LOG_ROOT:-experiments/NYCTaxi_TDS/${RUN_PREFIX}_logs}"
mkdir -p "$LOG_ROOT"

exec env \
  GPU_IDS="${GPU_IDS:-0,1}" \
  MAX_PARALLEL="${MAX_PARALLEL:-2}" \
  SEEDS="${SEEDS:-2024}" \
  CASES="${CASES:-conf_gci,conf_scd,conf_both}" \
  RUN_PREFIX="$RUN_PREFIX" \
  GCI_GRAPH_ALIGN_WEIGHT="${GCI_GRAPH_ALIGN_WEIGHT:-0.1}" \
  GCI_EDGE_PRESERVE_BETA="${GCI_EDGE_PRESERVE_BETA:-0.1}" \
  FPEM_USE_PRETRAINED_INV_AGCRN="${FPEM_USE_PRETRAINED_INV_AGCRN:-true}" \
  FPEM_PRETRAINED_INV_AGCRN_PATH="${FPEM_PRETRAINED_INV_AGCRN_PATH:-experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}" \
  FPEM_ENV_PERIOD_CONTEXT_SCALE="${FPEM_ENV_PERIOD_CONTEXT_SCALE:-0.1}" \
  RESUME="${RESUME:-true}" \
  bash scripts/run_tds_nyctaxi_fpem_confounder_dep_agcrn_aligned.sh
