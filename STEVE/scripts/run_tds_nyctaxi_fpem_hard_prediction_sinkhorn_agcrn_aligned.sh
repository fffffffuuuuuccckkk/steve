#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PROJECT_DIR
export RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_hard_prediction_sinkhorn}"
export CASES="${CASES:-add_k3_hard_prediction_sinkhorn}"
export SEEDS="${SEEDS:-2024}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"

exec bash "${SCRIPT_DIR}/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh"
