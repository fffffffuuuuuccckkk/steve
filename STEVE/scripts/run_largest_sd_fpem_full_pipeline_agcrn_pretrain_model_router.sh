#!/usr/bin/env bash
set -euo pipefail

# LargeST-SD end-to-end protocol:
#   1) pretrain a pure AGCRN forecasting backbone;
#   2) train the current FPEM prediction model with the pretrained AGCRN frozen
#      as invariant backbone;
#   3) freeze the prediction model and train only the relative-gap router.
#
# Defaults are deliberately conservative for 716-node LargeST-SD.  If CUDA OOM
# occurs, the script exits immediately and leaves the failing log/checkpoint
# state intact for inspection/resume with a smaller batch.

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
cd "$PROJECT_DIR"

if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-basicts}"
fi

PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET_NAME="${DATASET_NAME:-LargeST-SD_TDS}"
CONFIG_FILE="${CONFIG_FILE:-configs/LargeST_SD_TDS.yaml}"
GRAPH_FILE="${GRAPH_FILE:-data/LargeST-SD_TDS/adj_mx.npz}"
RESULT_ROOT="${RESULT_ROOT:-experiments/LargeST-SD_TDS}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
DATASET_DIR="${DATASET_DIR:-data/${DATASET_NAME}}"
DATASET_SLUG_DEFAULT="$(printf '%s' "$DATASET_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
DATASET_SLUG="${DATASET_SLUG:-${DATASET_SLUG_DEFAULT%_}}"
OBS_CACHE="${OBS_CACHE:-${PROJECT_DIR}/${DATASET_DIR}/observable_load_prior_k3_v1.npz}"

RUN_TAG="${RUN_TAG:-0814}"
PURE_RUN="${PURE_RUN:-${DATASET_SLUG}_pure_agcrn_seed${SEED}_${RUN_TAG}}"
FPEM_RUN="${FPEM_RUN:-${DATASET_SLUG}_fpem_conf_gci_period_context_seed${SEED}_${RUN_TAG}}"
ROUTER_OUTPUT_ROOT="${ROUTER_OUTPUT_ROOT:-${PROJECT_DIR}/${RESULT_ROOT}/${DATASET_SLUG}_relative_gap_router_from_${FPEM_RUN}}"
ROUTER_CASE="${ROUTER_CASE:-relative_gap_ema_huber_loadstats_preddiff}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FPEM_EPOCHS="${FPEM_EPOCHS:-100}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-20}"
FPEM_PATIENCE="${FPEM_PATIENCE:-20}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_TEST_BATCH_SIZE="${PRETRAIN_TEST_BATCH_SIZE:-16}"
FPEM_BATCH_SIZE="${FPEM_BATCH_SIZE:-8}"
FPEM_TEST_BATCH_SIZE="${FPEM_TEST_BATCH_SIZE:-16}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-16}"
ROUTER_EVAL_BATCH_SIZE="${ROUTER_EVAL_BATCH_SIZE:-64}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-100}"
ROUTER_PATIENCE="${ROUTER_PATIENCE:-30}"
FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS="${FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS:-0}"
FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS="${FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS:-0}"

GCI_WEIGHT="${GCI_WEIGHT:-0.001}"
SCD_WEIGHT="${SCD_WEIGHT:-0.0}"
GCI_GRAPH_ALIGN_WEIGHT="${GCI_GRAPH_ALIGN_WEIGHT:-0.1}"
GCI_EDGE_PRESERVE_BETA="${GCI_EDGE_PRESERVE_BETA:-0.1}"
CONFOUNDER_DEP_MODE="${CONFOUNDER_DEP_MODE:-gci}"
D_MODEL="${D_MODEL:-32}"
AGCRN_RNN_UNITS="${AGCRN_RNN_UNITS:-32}"
AGCRN_NUM_LAYERS="${AGCRN_NUM_LAYERS:-1}"
AGCRN_CHEB_K="${AGCRN_CHEB_K:-2}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

mkdir -p "$RESULT_ROOT" logs

truthy_finished_summary() {
  local summary_json="$1"
  [ -f "$summary_json" ] || return 1
  "$PYTHON" - "$summary_json" <<'PY'
import json, math, sys
try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    ok = bool(data.get("finished")) and math.isfinite(float(data.get("test_avg_mae")))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}"
echo "[INFO] GRAPH_FILE=${GRAPH_FILE}"
echo "[INFO] DATASET_DIR=${DATASET_DIR}"
echo "[INFO] OBS_CACHE=${OBS_CACHE}"
echo "[INFO] RESULT_ROOT=${RESULT_ROOT}"
echo "[INFO] GPU_ID=${GPU_ID} SEED=${SEED}"
echo "[INFO] PURE_RUN=${PURE_RUN}"
echo "[INFO] FPEM_RUN=${FPEM_RUN}"
echo "[INFO] ROUTER_OUTPUT_ROOT=${ROUTER_OUTPUT_ROOT}"
echo "[INFO] batches pretrain=${PRETRAIN_BATCH_SIZE}/${PRETRAIN_TEST_BATCH_SIZE} fpem=${FPEM_BATCH_SIZE}/${FPEM_TEST_BATCH_SIZE}"
echo "[INFO] lightweight AGCRN/FPEM d_model=${D_MODEL} rnn_units=${AGCRN_RNN_UNITS} layers=${AGCRN_NUM_LAYERS} cheb_k=${AGCRN_CHEB_K}"
echo "[INFO] period context override day=${FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS} week=${FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS}"

PURE_DIR="${PROJECT_DIR}/${RESULT_ROOT}/${PURE_RUN}"
PURE_CKPT="${PURE_DIR}/best_val_model.pth"
if truthy_finished_summary "${PURE_DIR}/summary.json" && [ -f "$PURE_CKPT" ]; then
  echo "[SKIP] pure AGCRN already finished: ${PURE_CKPT}"
else
  echo "[STAGE1] pretrain pure AGCRN -> ${PURE_RUN}"
  "$PYTHON" run_tds_nyctaxi.py \
    --config_filename "$CONFIG_FILE" \
    --dataset "$DATASET_NAME" \
    --data_dir data \
    --graph_file "$GRAPH_FILE" \
    --model agcrn \
    --epochs "$PRETRAIN_EPOCHS" \
    --batch_size "$PRETRAIN_BATCH_SIZE" \
    --test_batch_size "$PRETRAIN_TEST_BATCH_SIZE" \
    --device cuda:0 \
    --seed "$SEED" \
    --result_root "$RESULT_ROOT" \
    --exp_name "$PURE_RUN" \
    --resume true \
    --resume_reset_patience true \
    --best_selection_split test \
    --early_stop_patience "$PRETRAIN_PATIENCE" \
    --early_stop_test_avg_mae_epoch 0 \
    --save_test_selected_checkpoints true \
    --d_model "$D_MODEL" \
    --agcrn_rnn_units "$AGCRN_RNN_UNITS" \
    --agcrn_num_layers "$AGCRN_NUM_LAYERS" \
    --agcrn_cheb_k "$AGCRN_CHEB_K"
fi

if [ ! -f "$PURE_CKPT" ]; then
  echo "[ERROR] missing pure AGCRN checkpoint after Stage1: ${PURE_CKPT}" >&2
  exit 3
fi

FPEM_DIR="${PROJECT_DIR}/${RESULT_ROOT}/${FPEM_RUN}"
FPEM_CKPT="${FPEM_DIR}/best_val_model.pth"
if truthy_finished_summary "${FPEM_DIR}/summary.json" && [ -f "$FPEM_CKPT" ]; then
  echo "[SKIP] FPEM prediction model already finished: ${FPEM_CKPT}"
else
  echo "[STAGE2] train FPEM prediction model -> ${FPEM_RUN}"
  "$PYTHON" run_tds_nyctaxi.py \
    --config_filename "$CONFIG_FILE" \
    --dataset "$DATASET_NAME" \
    --data_dir data \
    --graph_file "$GRAPH_FILE" \
    --model steve \
    --epochs "$FPEM_EPOCHS" \
    --batch_size "$FPEM_BATCH_SIZE" \
    --test_batch_size "$FPEM_TEST_BATCH_SIZE" \
    --device cuda:0 \
    --seed "$SEED" \
    --result_root "$RESULT_ROOT" \
    --exp_name "$FPEM_RUN" \
    --resume true \
    --resume_reset_patience true \
    --best_selection_split test \
    --early_stop_patience "$FPEM_PATIENCE" \
    --early_stop_test_avg_mae_epoch 0 \
    --save_test_selected_checkpoints true \
    --fpem_backbone agcrn \
    --d_model "$D_MODEL" \
    --fpem_use_pretrained_inv_agcrn true \
    --fpem_pretrained_inv_agcrn_path "$PURE_CKPT" \
    --agcrn_embed_dim 10 \
    --agcrn_rnn_units "$AGCRN_RNN_UNITS" \
    --agcrn_num_layers "$AGCRN_NUM_LAYERS" \
    --agcrn_cheb_k "$AGCRN_CHEB_K" \
    --fpem_use_confounder_extractor false \
    --fpem_use_env_mask false \
    --fpem_confounder_use_mask false \
    --fpem_lambda_mask_sparse 0.0 \
    --fpem_lambda_mask_entropy 0.0 \
    --fpem_use_env_route true \
    --fpem_env_route_head_mode hyper_inv_film_proto_input_add \
    --fpem_env_route_k 1 \
    --fpem_use_observable_load_prior true \
    --fpem_observable_load_prior_cache "$OBS_CACHE" \
    --fpem_observable_load_random_seed 314159 \
    --fpem_use_load_level_experts true \
    --fpem_load_level_k 1 \
    --fpem_load_level_mode train_quantile \
    --fpem_use_random_balanced_assignment false \
    --fpem_ignore_future_c true \
    --fpem_hyper_alpha_mode fixed_one \
    --fpem_use_environment_gate false \
    --fpem_use_hard_environment_router false \
    --fpem_conservative_inv_override false \
    --fpem_counterfactual_risk_router false \
    --fpem_hard_router_hidden_dim 64 \
    --fpem_hard_router_warmup_epochs 0 \
    --fpem_lambda_hard_router 0.0 \
    --fpem_lambda_load_expert 0.2 \
    --fpem_lambda_inv_pred 0.2 \
    --fpem_env_route_use_inv_fallback_expert false \
    --fpem_use_env_prototype_router false \
    --fpem_use_sinkhorn_route false \
    --fpem_env_route_lambda_balance 0.0 \
    --fpem_env_route_lambda_diverse 0.0 \
    --fpem_env_route_lambda_proto_align 0.0 \
    --fpem_env_route_lambda_entropy 0.0 \
    --fpem_env_route_lambda_route_soft 0.0 \
    --fpem_use_future_mi false \
    --fpem_lambda_future_mi 0.0 \
    --fpem_use_swap false \
    --fpem_lambda_swap 0.0 \
    --fpem_use_club_mi false \
    --fpem_lambda_club_mi 0.0 \
    --fpem_use_env_fusion false \
    --fpem_env_use_exogenous true \
    --fpem_env_use_period_context true \
    --fpem_env_period_context_day_steps "$FPEM_ENV_PERIOD_CONTEXT_DAY_STEPS" \
    --fpem_env_period_context_week_steps "$FPEM_ENV_PERIOD_CONTEXT_WEEK_STEPS" \
    --fpem_env_period_context_scale 0.1 \
    --fpem_use_env_supervision false \
    --fpem_lambda_env_day_cls 0.0 \
    --fpem_lambda_env_hour_cls 0.0 \
    --fpem_lambda_env_rush_cls 0.0 \
    --fpem_use_env_supcon false \
    --fpem_lambda_env_supcon 0.0 \
    --fpem_use_inv_projector false \
    --fpem_use_inv_env_adversarial false \
    --fpem_use_cross_cov_sep false \
    --fpem_use_grad_consensus false \
    --fpem_gc_pred_loss_only true \
    --confounder_extractor token_attention \
    --confounder_dim 8 \
    --confounder_num_heads 4 \
    --confounder_attention_dropout 0.0 \
    --confounder_graph_embed_dim 16 \
    --confounder_variational true \
    --confounder_kl_weight 0.0001 \
    --confounder_dep_mode "$CONFOUNDER_DEP_MODE" \
    --gci_weight "$GCI_WEIGHT" \
    --scd_weight "$SCD_WEIGHT" \
    --gci_graph_align_weight "$GCI_GRAPH_ALIGN_WEIGHT" \
    --gci_edge_preserve_beta "$GCI_EDGE_PRESERVE_BETA" \
    --gci_graph_hops 1 \
    --gci_symmetrize_adj true \
    --gci_add_self_loops true \
    --confounder_dep_warmup_epochs 5 \
    --confounder_dep_ramp_epochs 10 \
    --confounder_projection_ridge 0.001 \
    --confounder_dep_detach_target true \
    --confounder_virtual_batch_enabled true \
    --confounder_virtual_batch_size 8 \
    --dep_eps 1.0e-6 \
    --confounder_injection none
fi

if [ ! -f "$FPEM_CKPT" ]; then
  echo "[ERROR] missing FPEM prediction checkpoint after Stage2: ${FPEM_CKPT}" >&2
  exit 4
fi

echo "[STAGE3] train frozen relative-gap router from ${FPEM_CKPT}"
"$PYTHON" scripts/train_tds_nyctaxi_fpem_relative_gap_router_from_checkpoint.py \
  --base_exp_dir "$FPEM_DIR" \
  --checkpoint_name best_val_model.pth \
  --output_root "$ROUTER_OUTPUT_ROOT" \
  --case "$ROUTER_CASE" \
  --device cuda:0 \
  --seed "$SEED" \
  --epochs "$ROUTER_EPOCHS" \
  --patience "$ROUTER_PATIENCE" \
  --batch_size "$ROUTER_BATCH_SIZE" \
  --eval_batch_size "$ROUTER_EVAL_BATCH_SIZE" \
  --hidden_dim 256 \
  --dropout 0.0 \
  --lr 0.001 \
  --ema_beta 0.99 \
  --gap_eps 1.0e-6 \
  --include_load_stats true \
  --include_pred_diff true \
  --standardize_features true

echo "[DONE] ${DATASET_NAME} pipeline finished"
