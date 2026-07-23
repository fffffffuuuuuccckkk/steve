# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/best_val_model.pth`
- run_name: `fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025`
- split: `test`
- samples: `546`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `10.116356`
- learned-route test MAPE: `0.180696`

## Key probe numbers

Top environment classification probes by AUC:
- rush_hour / h_inv_plus_e_env / logistic_regression: acc=0.9817, auc=0.9986, f1=0.9630
- rush_hour / e_env / logistic_regression: acc=0.9756, auc=0.9974, f1=0.9512
- workday / h_inv_plus_e_env / linear_svm: acc=0.9390, auc=0.9854, f1=0.9541
- rush_hour / h_inv / logistic_regression: acc=0.9390, auc=0.9829, f1=0.8810
- rush_hour / z_inv_raw / logistic_regression: acc=0.9390, auc=0.9829, f1=0.8810

Top residual probes by residual MAE:
- z_inv_raw / ridge: mae=4.2852, r2=0.5377
- h_inv / ridge: mae=4.2852, r2=0.5377
- h_inv_plus_e_env / ridge: mae=4.4868, r2=0.4981
- z_inv_raw / linear_regression: mae=4.5204, r2=0.4987
- h_inv / linear_regression: mae=4.5204, r2=0.4987

Route intervention overall MAE:
- learned: 10.116356
- shuffled: 10.116428
- remove_2: 10.118194
- remove_0: 10.133645
- remove_1: 10.142840
- uniform: 10.345635

Environment swap summary:
- same_env: mean_abs_change=0.632477, delta_mae=0.115311
- cross_env: mean_abs_change=0.635731, delta_mae=0.111616

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/probe_env_results.csv`
- inv_raw_vs_projected_probe: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/inv_raw_vs_projected_probe.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/expert_env_crosstab.csv`
- route_usage_summary: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/route_usage_summary.csv`
- prototype_similarity: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/prototype_similarity.csv`
- prototype_env_distribution: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/prototype_env_distribution.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_env_disentangle_proto_current_prediction_oracle_seed2025/case_study/case_outputs/umap_e_env_by_hour.png`
