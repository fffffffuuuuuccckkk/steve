#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_pretrained_inv_load_level_expert_gate_0727}"
CURRENT_SEED2024_PID="${CURRENT_SEED2024_PID:-}"
PYTHON_BIN="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,3}"
AUDIT_PHYSICAL_GPU="${AUDIT_PHYSICAL_GPU:-3}"
LOG_PATH="${LOG_PATH:-/tmp/${RUN_PREFIX}_full_acceptance.log}"
DONE_PATH="${DONE_PATH:-/tmp/${RUN_PREFIX}_full_acceptance.done}"
FAILED_PATH="${FAILED_PATH:-/tmp/${RUN_PREFIX}_full_acceptance.failed}"

cd "$PROJECT_DIR"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

exec > >(tee -a "$LOG_PATH") 2>&1
rm -f "$DONE_PATH" "$FAILED_PATH"
trap 'status=$?; if [ "$status" -ne 0 ]; then printf "%s\n" "$status" > "$FAILED_PATH"; fi' EXIT

cases=(
  k1_environment_expert
  k3_unsupervised_experts
  k3_load_level_always_env
  k3_load_level_gate
  k3_random_load_level_gate
)

verify_seed() {
  local seed="$1"
  local case_name summary_path
  for case_name in "${cases[@]}"; do
    summary_path="experiments/NYCTaxi_TDS/${RUN_PREFIX}_${case_name}_seed${seed}/summary.json"
    test -s "$summary_path"
    "$PYTHON_BIN" -c \
      'import json,sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); assert d.get("finished") is True, p' \
      "$summary_path"
  done
}

echo "[SCREEN] start date=$(date -Is) project=$PROJECT_DIR"
echo "[SCREEN] seed2024_pid=${CURRENT_SEED2024_PID:-none} gpu_ids=$GPU_IDS audit_gpu=$AUDIT_PHYSICAL_GPU"

if [ -n "$CURRENT_SEED2024_PID" ]; then
  while kill -0 "$CURRENT_SEED2024_PID" 2>/dev/null; do
    echo "[WAIT] seed 2024 launcher pid=$CURRENT_SEED2024_PID still active date=$(date -Is)"
    sleep 60
  done
fi

verify_seed 2024
echo "[CHECK] all five seed 2024 summaries are complete"

CUDA_VISIBLE_DEVICES="$AUDIT_PHYSICAL_GPU" \
  CASES=k3_load_level_gate \
  GPU_IDS="$GPU_IDS" \
  MAX_PARALLEL=3 \
  SEED2024_FIRST=false \
  SEEDS=2024 \
  RUN_LOAD_LEVEL_AUDIT=true \
  AUDIT_DEVICE=cuda:0 \
  bash scripts/run_tds_nyctaxi_fpem_load_level_expert_gate_agcrn_aligned.sh
echo "[CHECK] seed 2024 main gate audit complete"

GPU_IDS="$GPU_IDS" \
  MAX_PARALLEL=3 \
  SEED2024_FIRST=false \
  SEEDS=2025,2026 \
  RUN_LOAD_LEVEL_AUDIT=false \
  bash scripts/run_tds_nyctaxi_fpem_load_level_expert_gate_agcrn_aligned.sh

verify_seed 2025
verify_seed 2026
echo "[CHECK] all five configurations and all three seeds are complete"

CUDA_VISIBLE_DEVICES="$AUDIT_PHYSICAL_GPU" \
  CASES=k1_environment_expert,k3_load_level_always_env,k3_load_level_gate,k3_random_load_level_gate \
  GPU_IDS="$GPU_IDS" \
  MAX_PARALLEL=3 \
  SEED2024_FIRST=false \
  SEEDS=2024,2025,2026 \
  RUN_LOAD_LEVEL_AUDIT=true \
  AUDIT_DEVICE=cuda:0 \
  bash scripts/run_tds_nyctaxi_fpem_load_level_expert_gate_agcrn_aligned.sh

"$PYTHON_BIN" scripts/summarize_fpem_load_level_experiments.py \
  --result-root experiments/NYCTaxi_TDS \
  --prefix "$RUN_PREFIX" \
  --seeds 2024,2025,2026

date -Is > "$DONE_PATH"
echo "[DONE] full acceptance matrix, audits, and aggregate report complete date=$(date -Is)"
