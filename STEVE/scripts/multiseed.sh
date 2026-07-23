for SEED in 2024 2025 2026; do
  FORCE=false SEED=$SEED bash scripts/run_tds_nyctaxi_fpem_module_build_ablation_agcrn_aligned.sh
done