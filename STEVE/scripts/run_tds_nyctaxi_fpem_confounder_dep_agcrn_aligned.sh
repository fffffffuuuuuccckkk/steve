#!/usr/bin/env bash
set -euo pipefail

# Latent-confounder structural regularization experiments.
# This launcher loads the pretrained invariant AGCRN by default, matching the
# earlier pretrained-inv FPEM protocol.
#
# Default:
#   GPU_IDS=0,1 MAX_PARALLEL=2 SEEDS=2024 \
#     bash scripts/run_tds_nyctaxi_fpem_confounder_dep_agcrn_aligned.sh

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  CONDA_ENV="${CONDA_ENV:-${STEVE_CONDA_ENV:-basicts}}"
  conda activate "$CONDA_ENV"
fi
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
RUN_PREFIX="${RUN_PREFIX:-fpem_agcrn_aligned_confounder_dep_pretrained_inv}"
RESULT_ROOT="${RESULT_ROOT:-experiments/NYCTaxi_TDS}"
LOG_ROOT="${LOG_ROOT:-${RESULT_ROOT}/${RUN_PREFIX}_logs}"
STATUS_DIR="${LOG_ROOT}/status"
SUMMARY_FILE="${LOG_ROOT}/summary.tsv"
GPU_IDS="${GPU_IDS:-0,1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEEDS="${SEEDS:-2024}"
CASES="${CASES:-conf_none,conf_gci,conf_scd,conf_both}"
MAX_EPOCH="${MAX_EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
RESUME="${RESUME:-true}"
BEST_SELECTION_SPLIT="${BEST_SELECTION_SPLIT:-val}"
SAVE_TEST_SELECTED_CHECKPOINTS="${SAVE_TEST_SELECTED_CHECKPOINTS:-true}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
GCI_WEIGHT="${GCI_WEIGHT:-0.001}"
GCI_GRAPH_ALIGN_WEIGHT="${GCI_GRAPH_ALIGN_WEIGHT:-0.1}"
GCI_EDGE_PRESERVE_BETA="${GCI_EDGE_PRESERVE_BETA:-0.1}"
SCD_WEIGHT="${SCD_WEIGHT:-0.001}"
FPEM_USE_PRETRAINED_INV_AGCRN="${FPEM_USE_PRETRAINED_INV_AGCRN:-true}"
FPEM_PRETRAINED_INV_AGCRN_PATH="${FPEM_PRETRAINED_INV_AGCRN_PATH:-experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}"
CONFOUNDER_DIM="${CONFOUNDER_DIM:-8}"
CONFOUNDER_NUM_HEADS="${CONFOUNDER_NUM_HEADS:-4}"
CONFOUNDER_ATTENTION_DROPOUT="${CONFOUNDER_ATTENTION_DROPOUT:-0.0}"
CONFOUNDER_GRAPH_EMBED_DIM="${CONFOUNDER_GRAPH_EMBED_DIM:-16}"
CONFOUNDER_WARMUP_EPOCHS="${CONFOUNDER_WARMUP_EPOCHS:-5}"
CONFOUNDER_RAMP_EPOCHS="${CONFOUNDER_RAMP_EPOCHS:-10}"
DRY_RUN="${DRY_RUN:-false}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

mkdir -p "$LOG_ROOT" "$STATUS_DIR"
if [ ! -f "$SUMMARY_FILE" ]; then
  printf 'case\tseed\tstatus\trun_name\ttest_avg_mae\tbest_epoch\tconfounder_dep_total\tloss_gci\tloss_scd\tloss_conf_kl\tdetail\n' > "$SUMMARY_FILE"
fi

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

completed_summary_valid() {
  local summary_json="$1"
  [ -f "$summary_json" ] || return 1
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    ok = bool(data.get("finished")) and math.isfinite(float(data.get("test_avg_mae")))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

metric_from_summary() {
  local summary_json="$1"
  "$PYTHON" - "$summary_json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
def get(*keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return "NA"
print("\t".join(str(x) for x in [
    get("test_avg_mae"),
    get("best_epoch"),
    get("confounder_dep_total"),
    get("confounder_loss_gci"),
    get("confounder_loss_scd"),
    get("confounder_loss_kl"),
]), end="")
PY
}

record_summary() {
  local case_name="$1"
  local seed="$2"
  local status="$3"
  local run_name="$4"
  local detail="$5"
  local summary_json="${PROJECT_DIR}/${RESULT_ROOT}/${run_name}/summary.json"
  local metrics
  if [ -f "$summary_json" ]; then
    metrics="$(metric_from_summary "$summary_json")"
  else
    metrics=$'NA\tNA\tNA\tNA\tNA\tNA'
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$case_name" "$seed" "$status" "$run_name" "$metrics" "$detail" >> "$SUMMARY_FILE"
}

case_args() {
  local case_name="$1"
  case "$case_name" in
    conf_none)
      printf '%s\n' --confounder_dep_mode none
      ;;
    conf_gci)
      printf '%s\n' --confounder_dep_mode gci --gci_weight "$GCI_WEIGHT" --scd_weight 0.0
      ;;
    conf_scd)
      printf '%s\n' --confounder_dep_mode scd --gci_weight 0.0 --scd_weight "$SCD_WEIGHT"
      ;;
    conf_both)
      printf '%s\n' --confounder_dep_mode both --gci_weight "$GCI_WEIGHT" --scd_weight "$SCD_WEIGHT"
      ;;
    *)
      echo "[ERROR] unknown case: $case_name" >&2
      return 2
      ;;
  esac
}

BASE_ARGS=(
  --config_filename configs/NYCTaxi.yaml
  --dataset NYCTaxi_TDS
  --data_dir data
  --graph_file data/NYCTaxi_TDS/adj_mx.npz
  --model steve
  --epochs "$MAX_EPOCH"
  --batch_size "$BATCH_SIZE"
  --test_batch_size "$TEST_BATCH_SIZE"
  --device cuda:0
  --train_work_per_holiday 2.5
  --result_root "$RESULT_ROOT"
  --resume "$RESUME"
  --resume_reset_patience true
  --best_selection_split "$BEST_SELECTION_SPLIT"
  --early_stop_test_avg_mae_epoch 40
  --early_stop_test_avg_mae_threshold 12
  --save_test_selected_checkpoints "$SAVE_TEST_SELECTED_CHECKPOINTS"

  --fpem_backbone agcrn
  --fpem_use_pretrained_inv_agcrn "$FPEM_USE_PRETRAINED_INV_AGCRN"
  --fpem_pretrained_inv_agcrn_path "$FPEM_PRETRAINED_INV_AGCRN_PATH"
  --agcrn_embed_dim 10
  --agcrn_num_layers 2
  --agcrn_cheb_k 2

  --fpem_use_confounder_extractor false
  --fpem_use_env_mask false
  --fpem_confounder_use_mask false
  --fpem_lambda_mask_sparse 0.0
  --fpem_lambda_mask_entropy 0.0
  --fpem_lambda_inv_pred 0.0

  # Routing/prediction branch protocol: mirror the existing
  # obs_k1_counterfactual_risk_router case from
  # run_tds_nyctaxi_fpem_input_add_module_validity_agcrn_aligned.sh.
  # Stage 1 keeps the top-level prediction on the environment branch; the
  # counterfactual risk router is trained separately and is deployed only for
  # eval/test hard Env-vs-Inv selection.
  --fpem_use_env_route true
  --fpem_env_route_head_mode hyper_inv_film_proto_input_add
  --fpem_env_route_k 1
  --fpem_use_observable_load_prior true
  --fpem_observable_load_prior_cache "${PROJECT_DIR}/data/NYCTaxi/observable_load_prior_k3_v1.npz"
  --fpem_observable_load_random_seed 314159
  --fpem_use_load_level_experts true
  --fpem_load_level_k 1
  --fpem_load_level_mode train_quantile
  --fpem_use_random_balanced_assignment false
  --fpem_ignore_future_c true
  --fpem_hyper_alpha_mode fixed_one
  --fpem_use_environment_gate false
  --fpem_use_hard_environment_router false
  --fpem_conservative_inv_override false
  --fpem_counterfactual_risk_router true
  --fpem_counterfactual_risk_stage2_start_epoch "${FPEM_COUNTERFACTUAL_RISK_STAGE2_START_EPOCH:-20}"
  --fpem_counterfactual_risk_regression_weight "${FPEM_COUNTERFACTUAL_RISK_REGRESSION_WEIGHT:-1.0}"
  --fpem_counterfactual_risk_ranking_weight "${FPEM_COUNTERFACTUAL_RISK_RANKING_WEIGHT:-0.5}"
  --fpem_counterfactual_risk_router_loss_weight "${FPEM_COUNTERFACTUAL_RISK_ROUTER_LOSS_WEIGHT:-1.0}"
  --fpem_counterfactual_risk_weight_min "${FPEM_COUNTERFACTUAL_RISK_WEIGHT_MIN:-0.0}"
  --fpem_counterfactual_risk_weight_max "${FPEM_COUNTERFACTUAL_RISK_WEIGHT_MAX:-20.0}"
  --fpem_counterfactual_risk_ranking_temperature "${FPEM_COUNTERFACTUAL_RISK_RANKING_TEMPERATURE:-1.0}"
  --fpem_hard_router_hidden_dim 64
  --fpem_hard_router_warmup_epochs 0
  --fpem_lambda_hard_router 0.0
  --fpem_lambda_load_expert 0.2
  --fpem_lambda_inv_pred "${FPEM_LAMBDA_INV_PRED:-0.2}"
  --fpem_env_route_use_inv_fallback_expert false
  --fpem_use_env_prototype_router false
  --fpem_use_sinkhorn_route false
  --fpem_env_route_lambda_balance 0.0
  --fpem_env_route_lambda_diverse 0.0
  --fpem_env_route_lambda_proto_align 0.0
  --fpem_env_route_lambda_entropy 0.0
  --fpem_env_route_lambda_route_soft 0.0
  --fpem_use_future_mi false
  --fpem_lambda_future_mi 0.0
  --fpem_use_swap false
  --fpem_lambda_swap 0.0
  --fpem_use_club_mi false
  --fpem_lambda_club_mi 0.0
  --fpem_use_env_fusion false
  --early_stop_test_avg_mae_epoch 0
  --save_test_selected_checkpoints false

  --fpem_env_use_exogenous true
  --fpem_env_use_period_context true
  --fpem_env_period_context_day_steps "${FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS:-0}"
  --fpem_env_period_context_week_steps "${FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS:-0}"
  --fpem_env_period_context_scale "${FPEM_ENV_PERIOD_CONTEXT_SCALE:-0.1}"
  --fpem_ignore_future_c true
  --fpem_use_env_supervision false
  --fpem_lambda_env_day_cls 0.0
  --fpem_lambda_env_hour_cls 0.0
  --fpem_lambda_env_rush_cls 0.0
  --fpem_use_env_supcon false
  --fpem_lambda_env_supcon 0.0
  --fpem_use_inv_projector false
  --fpem_use_inv_env_adversarial false
  --fpem_use_cross_cov_sep false
  --fpem_use_club_mi false
  --fpem_lambda_club_mi 0.0

  --fpem_use_grad_consensus false
  --fpem_gc_pred_loss_only true

  --confounder_extractor token_attention
  --confounder_dim "$CONFOUNDER_DIM"
  --confounder_num_heads "$CONFOUNDER_NUM_HEADS"
  --confounder_attention_dropout "$CONFOUNDER_ATTENTION_DROPOUT"
  --confounder_graph_embed_dim "$CONFOUNDER_GRAPH_EMBED_DIM"
  --confounder_variational true
  --confounder_kl_weight 0.0001
  --confounder_dep_warmup_epochs "$CONFOUNDER_WARMUP_EPOCHS"
  --confounder_dep_ramp_epochs "$CONFOUNDER_RAMP_EPOCHS"
  --confounder_projection_ridge 0.001
  --confounder_dep_detach_target true
  --gci_graph_align_weight "$GCI_GRAPH_ALIGN_WEIGHT"
  --gci_edge_preserve_beta "$GCI_EDGE_PRESERVE_BETA"
  --gci_graph_hops 1
  --gci_symmetrize_adj true
  --gci_add_self_loops true
  --dep_eps 1.0e-6
  --confounder_injection none
)

OPTIONAL_ARGS=()
if [ -n "$MAX_TRAIN_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_train_batches "$MAX_TRAIN_BATCHES")
fi
if [ -n "$MAX_EVAL_BATCHES" ]; then
  OPTIONAL_ARGS+=(--max_eval_batches "$MAX_EVAL_BATCHES")
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a CASE_LIST <<< "$CASES"
if [ "${#GPU_POOL[@]}" -eq 0 ]; then
  GPU_POOL=(0)
fi

run_one() {
  local gpu="$1"
  local case_name="$2"
  local seed="$3"
  local run_name="${RUN_PREFIX}_${case_name}_seed${seed}"
  local run_dir="${PROJECT_DIR}/${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local status_file="${STATUS_DIR}/${run_name}.status"
  local summary_json="${run_dir}/summary.json"
  local -a extra_args=()
  mapfile -t extra_args < <(case_args "$case_name")

  if completed_summary_valid "$summary_json"; then
    echo "[SKIP] completed $run_name"
    record_summary "$case_name" "$seed" "SKIP_DONE" "$run_name" "already finished"
    printf 'SKIP_DONE\n' > "$status_file"
    return 0
  fi

  local -a cmd=(
    "$PYTHON" run_tds_nyctaxi.py
    "${BASE_ARGS[@]}"
    "${OPTIONAL_ARGS[@]}"
    --seed "$seed"
    --exp_name "$run_name"
    --ablation "$case_name"
    "${extra_args[@]}"
  )
  printf '[LAUNCH] gpu=%s case=%s seed=%s run=%s pretrained_inv_agcrn=%s path=%s\n' "$gpu" "$case_name" "$seed" "$run_name" "$FPEM_USE_PRETRAINED_INV_AGCRN" "$FPEM_PRETRAINED_INV_AGCRN_PATH"
  printf '%q ' "${cmd[@]}" > "${LOG_ROOT}/${run_name}.cmd"
  printf '\n' >> "${LOG_ROOT}/${run_name}.cmd"
  if truthy "$DRY_RUN"; then
    cat "${LOG_ROOT}/${run_name}.cmd"
    record_summary "$case_name" "$seed" "DRY_RUN" "$run_name" "not launched"
    return 0
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] && completed_summary_valid "$summary_json"; then
    record_summary "$case_name" "$seed" "OK" "$run_name" "finished"
    printf 'OK\n' > "$status_file"
    return 0
  fi
  record_summary "$case_name" "$seed" "FAIL" "$run_name" "rc=$rc log=$log_file"
  printf 'FAIL\n' > "$status_file"
  return "$rc"
}

echo "[INFO] RUN_PREFIX=$RUN_PREFIX cases=$CASES seeds=$SEEDS gpu_ids=$GPU_IDS max_parallel=$MAX_PARALLEL"
echo "[INFO] pretrained invariant AGCRN=$FPEM_USE_PRETRAINED_INV_AGCRN path=$FPEM_PRETRAINED_INV_AGCRN_PATH"
echo "[INFO] GCI zero-margin relative nonedge-mass loss, edge_preserve_beta=$GCI_EDGE_PRESERVE_BETA"

job_index=0
running=0
for case_name in "${CASE_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    gpu="${GPU_POOL[$((job_index % ${#GPU_POOL[@]}))]}"
    while [ "$running" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
      running=$((running - 1))
    done
    run_one "$gpu" "$case_name" "$seed" &
    running=$((running + 1))
    job_index=$((job_index + 1))
  done
done
while [ "$running" -gt 0 ]; do
  wait -n || true
  running=$((running - 1))
done

ok_count="$( (grep -Rhs '^OK$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')"
fail_count="$( (grep -Rhs '^FAIL$' "$STATUS_DIR" || true) | wc -l | tr -d ' ')"
echo "[DONE] OK=$ok_count FAIL=$fail_count summary=$SUMMARY_FILE logs=$LOG_ROOT"
if [ "$fail_count" -gt 0 ]; then
  exit 1
fi
