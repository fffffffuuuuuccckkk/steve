#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

export RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_load_level_expert_gate_0727}"
export CASES="${CASES:-k1_environment_expert,k3_unsupervised_experts,k3_load_level_always_env,k3_load_level_gate,k3_random_load_level_gate}"
export GPU_IDS="${GPU_IDS:-0,1,2}"
export MAX_PARALLEL="${MAX_PARALLEL:-3}"
export MAX_EPOCH="${MAX_EPOCH:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export RESUME="${RESUME:-true}"
export RUN_ROUTE_EVAL=false
export ROUTE_EVAL_ONLY=false

BASE_LAUNCHER="scripts/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh"

if [ "${SEED2024_FIRST:-true}" = "true" ]; then
  echo "[STAGE] seed 2024 sanity-first training"
  SEEDS=2024 bash "$BASE_LAUNCHER"
fi

echo "[STAGE] requested multi-seed training"
SEEDS="${SEEDS:-2024,2025,2026}" bash "$BASE_LAUNCHER"

if [ "${RUN_LOAD_LEVEL_AUDIT:-true}" = "true" ] && [ "${DRY_RUN:-false}" != "true" ] && [ "${PLAN_ONLY:-false}" != "true" ]; then
  IFS=',' read -r -a seed_list <<< "${SEEDS:-2024,2025,2026}"
  IFS=',' read -r -a case_list <<< "$CASES"
  for case_name in "${case_list[@]}"; do
    case "$case_name" in
      k1_environment_expert|k3_load_level_always_env|k3_load_level_gate|k3_random_load_level_gate)
        for seed in "${seed_list[@]}"; do
          checkpoint="experiments/NYCTaxi_TDS/${RUN_PREFIX}_${case_name}_seed${seed}/best_val_model.pth"
          output_dir="experiments/NYCTaxi_TDS/${RUN_PREFIX}_${case_name}_seed${seed}/load_level_gate_audit"
          python scripts/audit_fpem_load_level_expert_gate.py \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --device "${AUDIT_DEVICE:-cpu}"
        done
        ;;
    esac
  done
fi
