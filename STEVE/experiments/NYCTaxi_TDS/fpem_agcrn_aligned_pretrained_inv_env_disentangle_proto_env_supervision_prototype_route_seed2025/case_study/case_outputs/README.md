# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/best_val_model.pth`
- run_name: `fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025`
- split: `test`
- samples: `546`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `7.920070`
- learned-route test MAPE: `0.134479`

## Key probe numbers

Top environment classification probes by AUC:
- rush_hour / h_inv_plus_e_env / logistic_regression: acc=0.9573, auc=0.9897, f1=0.9114
- rush_hour / z_inv_raw / logistic_regression: acc=0.9390, auc=0.9831, f1=0.8810
- rush_hour / h_inv / logistic_regression: acc=0.9390, auc=0.9831, f1=0.8810
- rush_hour / e_env / logistic_regression: acc=0.9207, auc=0.9774, f1=0.8539
- workday / h_inv_plus_e_env / linear_svm: acc=0.9146, auc=0.9722, f1=0.9352

Top residual probes by residual MAE:
- z_inv_raw / ridge: mae=4.2891, r2=0.5436
- h_inv / ridge: mae=4.2891, r2=0.5436
- h_inv_plus_e_env / ridge: mae=4.3193, r2=0.5331
- z_inv_raw / linear_regression: mae=4.5223, r2=0.5051
- h_inv / linear_regression: mae=4.5223, r2=0.5051

Route intervention overall MAE:
- uniform: 7.915822
- learned: 7.920070
- shuffled: 7.920072
- remove_1: 7.921830
- remove_0: 7.936577
- remove_2: 7.965742

Environment swap summary:
- same_env: mean_abs_change=10.936858, delta_mae=16.027470
- cross_env: mean_abs_change=10.916641, delta_mae=15.816735

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/probe_env_results.csv`
- inv_raw_vs_projected_probe: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/inv_raw_vs_projected_probe.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/expert_env_crosstab.csv`
- route_usage_summary: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/route_usage_summary.csv`
- prototype_similarity: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/prototype_similarity.csv`
- prototype_env_distribution: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/prototype_env_distribution.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_prototype_route_seed2025/case_study/case_outputs/umap_e_env_by_hour.png`
