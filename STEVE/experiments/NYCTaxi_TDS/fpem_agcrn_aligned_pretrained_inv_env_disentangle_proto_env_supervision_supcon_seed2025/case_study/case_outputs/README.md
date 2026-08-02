# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/best_val_model.pth`
- run_name: `fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025`
- split: `test`
- samples: `546`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `10.035289`
- learned-route test MAPE: `0.176088`

## Key probe numbers

Top environment classification probes by AUC:
- workday / h_inv_plus_e_env / logistic_regression: acc=0.9878, auc=1.0000, f1=0.9905
- workday / h_inv_plus_e_env / linear_svm: acc=0.9939, auc=1.0000, f1=0.9953
- workday / e_env / linear_svm: acc=0.9939, auc=0.9998, f1=0.9953
- rush_hour / h_inv_plus_e_env / logistic_regression: acc=0.9817, auc=0.9994, f1=0.9639
- rush_hour / e_env / logistic_regression: acc=0.9817, auc=0.9994, f1=0.9647

Top residual probes by residual MAE:
- z_inv_raw / ridge: mae=4.2823, r2=0.5367
- h_inv / ridge: mae=4.2823, r2=0.5367
- h_inv_plus_e_env / ridge: mae=4.3627, r2=0.5143
- z_inv_raw / linear_regression: mae=4.5175, r2=0.4976
- h_inv / linear_regression: mae=4.5175, r2=0.4976

Route intervention overall MAE:
- learned: 10.035289
- shuffled: 10.036738
- remove_2: 10.044151
- remove_1: 10.055190
- remove_0: 10.062231
- uniform: 10.191676

Environment swap summary:
- same_env: mean_abs_change=1.331771, delta_mae=0.286811
- cross_env: mean_abs_change=1.384832, delta_mae=0.281952

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/probe_env_results.csv`
- inv_raw_vs_projected_probe: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/inv_raw_vs_projected_probe.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/expert_env_crosstab.csv`
- route_usage_summary: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/route_usage_summary.csv`
- prototype_similarity: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/prototype_similarity.csv`
- prototype_env_distribution: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/prototype_env_distribution.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_env_supervision_supcon_seed2025/case_study/case_outputs/umap_e_env_by_hour.png`
