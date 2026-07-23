#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/OuXiaoyu/mystg/baselines/STEVE_CODE/STEVE"

source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
conda activate basicts

cd "${PROJECT_DIR}"

python run.py \
  --mode train \
  --config_filename configs/NYCTaxi.yaml \
  --bs 64 \
  --seed 2024 \
  --resume false \
  --device cuda:0 \
  --fpem_backbone agcrn \
  --agcrn_embed_dim 10 \
  --agcrn_num_layers 2 \
  --agcrn_cheb_k 2 \
  --fpem_use_env_mask true \
  --fpem_env_mask_hidden_dim 64 \
  --fpem_env_mask_temperature 1.0 \
  --fpem_env_mask_warmup_epochs 0 \
  --fpem_lambda_mask_sparse 0.0005 \
  --fpem_lambda_mask_entropy 0.0005 \
  --fpem_use_env_route true \
  --fpem_env_route_k 3 \
  --fpem_env_route_head_mode hyper_inv_film \
  --fpem_env_route_use_inv_fallback_expert true \
  --fpem_env_route_tau 1.0 \
  --fpem_env_route_oracle_tau 0.3 \
  --fpem_env_route_train_mode soft_oracle \
  --fpem_env_route_hidden_dim 64 \
  --fpem_env_route_warmup_epochs 0 \
  --fpem_env_route_lambda_final 1.0 \
  --fpem_env_route_lambda_global 0.0 \
  --fpem_env_route_lambda_route_soft 0.5 \
  --fpem_env_route_lambda_expert 0.2 \
  --fpem_env_route_lambda_router_oracle 1.0 \
  --fpem_env_route_lambda_balance 0.01 \
  --fpem_env_route_lambda_diverse 0.01 \
  --fpem_env_route_lambda_entropy 0.0 \
  --fpem_use_env_fusion true \
  --fpem_lambda_inv_pred 0.2 \
  --fpem_hyper_alpha_mode sample_gate \
  --fpem_lambda_hyper_delta_norm 0.0001 \
  --fpem_use_future_mi true \
  --fpem_lambda_future_mi 0.02 \
  --fpem_future_mi_warmup_epochs 0 \
  --fpem_future_mi_hidden_dim 64 \
  --fpem_future_mi_detach_target true \
  --fpem_use_swap true \
  --fpem_lambda_swap 0.01 \
  --fpem_swap_warmup_epochs 0 \
  --fpem_swap_margin 0.01 \
  --fpem_swap_gain_eta 0.0 \
  --fpem_swap_gain_tau 0.05 \
  --fpem_lambda_swap_diff 1.0 \
  --fpem_lambda_swap_same 0.05 \
  --fpem_swap_only_diff_route true \
  --fpem_swap_detach_inv true \
  --fpem_swap_detach_env false \
  --fpem_use_swap_fallback_router_loss true \
  --fpem_lambda_swap_fallback_router 0.005 \
  --fpem_swap_fallback_warmup_epochs 10 \
  --fpem_use_grad_consensus true \
  --fpem_gc_pred_loss_only true \
  --fpem_gc_inv_rho 0.3 \
  --fpem_gc_env_rho 0.3 \
  --fpem_gc_tau 0.5 \
  --fpem_gc_temp 0.1 \
  --fpem_gc_min_keep 0.2 \
  --fpem_gc_warmup_epochs 0 \
  --fpem_gc_route_min_samples 2 \
  --fpem_use_gradcompat_aux false \
  --fpem_lambda_gradcompat_aux 0.0
