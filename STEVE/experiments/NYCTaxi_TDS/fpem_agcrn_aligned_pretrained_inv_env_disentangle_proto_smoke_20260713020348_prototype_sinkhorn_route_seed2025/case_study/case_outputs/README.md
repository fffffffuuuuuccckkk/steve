# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/best_val_model.pth`
- run_name: `fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025`
- split: `test`
- samples: `4`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `117.197510`
- learned-route test MAPE: `1.660383`

## Key probe numbers


Top residual probes by residual MAE:
- h_inv_plus_e_env / ridge: mae=9.3801, r2=-10660.9053
- h_inv_plus_e_env / linear_regression: mae=9.3806, r2=-10659.6699
- e_env / ridge: mae=12.5219, r2=-13266.0625
- e_env / linear_regression: mae=12.6014, r2=-13348.2285
- h_inv / ridge: mae=13.0831, r2=-17753.9277

Route intervention overall MAE:
- uniform: 116.624802
- learned: 117.197510
- shuffled: 117.197510
- remove_2: 140.510941
- remove_1: 146.518021
- remove_0: 154.240906

Environment swap summary:
- same_env: mean_abs_change=0.000149, delta_mae=0.000015
- cross_env: mean_abs_change=0.000000, delta_mae=0.000000

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/probe_env_results.csv`
- inv_raw_vs_projected_probe: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/inv_raw_vs_projected_probe.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/expert_env_crosstab.csv`
- route_usage_summary: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/route_usage_summary.csv`
- prototype_similarity: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/prototype_similarity.csv`
- prototype_env_distribution: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/prototype_env_distribution.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_smoke_20260713020348_prototype_sinkhorn_route_seed2025/case_study/case_outputs/umap_e_env_by_hour.png`
