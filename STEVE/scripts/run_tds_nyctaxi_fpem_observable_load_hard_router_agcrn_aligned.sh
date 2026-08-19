#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

export RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_observable_load_hard_router_test_selected_0727_v2}"
export CASES="${CASES:-obs_k1_environment_hard_router,obs_k3_original_learned_experts,obs_k3_fixed_always_environment,obs_k3_fixed_hard_router,obs_k3_random_balanced_hard_router}"
export GPU_IDS="${GPU_IDS:-1,2,3}"
export MAX_PARALLEL="${MAX_PARALLEL:-3}"
export SEEDS="${SEEDS:-2024,2025,2026}"
export MAX_EPOCH="${MAX_EPOCH:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export RESUME="${RESUME:-true}"
export SAVE_TEST_SELECTED_CHECKPOINTS="${SAVE_TEST_SELECTED_CHECKPOINTS:-true}"
export TEST_SELECTION_START_EPOCH="${TEST_SELECTION_START_EPOCH:-40}"
export RUN_ROUTE_EVAL=false
export ROUTE_EVAL_ONLY=false
export PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
export OBSERVABLE_LOAD_CACHE="${OBSERVABLE_LOAD_CACHE:-${PROJECT_DIR}/data/NYCTaxi/observable_load_prior_k3_v1.npz}"

BASE_LAUNCHER="scripts/run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh"

if [ "${PLAN_ONLY:-false}" != "true" ]; then
  "$PYTHON" scripts/build_observable_load_prior.py \
    --data-dir "${PROJECT_DIR}/data/NYCTaxi" \
    --cache "$OBSERVABLE_LOAD_CACHE"
fi

bash "$BASE_LAUNCHER"

if [ "${RUN_HARD_ROUTE_AUDIT:-true}" = "true" ] \
  && [ "${DRY_RUN:-false}" != "true" ] \
  && [ "${PLAN_ONLY:-false}" != "true" ]; then
  IFS=',' read -r -a seed_list <<< "$SEEDS"
  IFS=',' read -r -a case_list <<< "$CASES"
  for case_name in "${case_list[@]}"; do
    case "$case_name" in
      obs_k1_environment_hard_router|obs_k3_fixed_hard_router|obs_k3_random_balanced_hard_router)
        for seed in "${seed_list[@]}"; do
          checkpoint="experiments/NYCTaxi_TDS/${RUN_PREFIX}_${case_name}_seed${seed}/best_val_model.pth"
          output_dir="experiments/NYCTaxi_TDS/${RUN_PREFIX}_${case_name}_seed${seed}/observable_hard_route_audit"
          "$PYTHON" scripts/audit_fpem_observable_hard_router.py \
            --checkpoint "$checkpoint" \
            --cache "$OBSERVABLE_LOAD_CACHE" \
            --output-dir "$output_dir" \
            --device "${AUDIT_DEVICE:-cpu}"
        done
        ;;
    esac
  done
fi

if [ "${DRY_RUN:-false}" != "true" ] \
  && [ "${PLAN_ONLY:-false}" != "true" ]; then
  "$PYTHON" scripts/summarize_fpem_load_level_experiments.py \
    --result-root experiments/NYCTaxi_TDS \
    --prefix "$RUN_PREFIX" \
    --cases "$CASES" \
    --seeds "$SEEDS"
fi
