# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/best_val_model.pth`
- run_name: `fpem_env_exogenous_cpu_smoke_20260713033411`
- split: `test`
- samples: `4`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `113.076485`
- learned-route test MAPE: `1.532420`

## Key probe numbers


Top residual probes by residual MAE:
- h_inv_plus_e_env / ridge: mae=8.3670, r2=-53446.1016
- h_inv_plus_e_env / linear_regression: mae=8.3736, r2=-53399.6562
- z_inv_raw / ridge: mae=9.1484, r2=-88337.6406
- h_inv / ridge: mae=9.1484, r2=-88337.6406
- h_inv / linear_regression: mae=9.1508, r2=-88381.2422

Route intervention overall MAE:
- remove_2: 108.147690
- remove_0: 108.397598
- shuffled: 113.076485
- learned: 113.076485
- uniform: 120.601212
- remove_1: 145.462891

Environment swap summary:
- same_env: mean_abs_change=0.000410, delta_mae=0.000275
- cross_env: mean_abs_change=0.000000, delta_mae=0.000000

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/probe_env_results.csv`
- inv_raw_vs_projected_probe: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/inv_raw_vs_projected_probe.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/expert_env_crosstab.csv`
- route_usage_summary: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/route_usage_summary.csv`
- prototype_similarity: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/prototype_similarity.csv`
- prototype_env_distribution: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/prototype_env_distribution.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_env_exogenous_cpu_smoke_20260713033411/case_study/case_outputs/umap_e_env_by_hour.png`
