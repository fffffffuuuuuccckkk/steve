# Pretrained frozen invariant AGCRN case study

- checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/best_val_model.pth`
- run_name: `fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025`
- split: `test`
- samples: `546`
- route K: `3`
- fpem_use_pretrained_inv_agcrn: `True`
- frozen params: `750800`

## Evaluation

- learned-route test MAE: `10.061665`
- learned-route test MAPE: `0.181402`

## Key probe numbers

Top environment classification probes by AUC:
- rush_hour / z_inv_plus_e_env / logistic_regression: acc=0.9512, auc=0.9921, f1=0.9000
- rush_hour / e_env / logistic_regression: acc=0.9451, auc=0.9841, f1=0.8916
- rush_hour / z_inv / logistic_regression: acc=0.9390, auc=0.9826, f1=0.8810
- workday / z_inv_plus_e_env / linear_svm: acc=0.9024, auc=0.9463, f1=0.9259
- workday / z_inv_plus_e_env / logistic_regression: acc=0.8354, auc=0.9201, f1=0.8643

Top residual probes by residual MAE:
- z_inv / ridge: mae=4.2814, r2=0.5365
- z_inv_plus_e_env / ridge: mae=4.3932, r2=0.5078
- z_inv / linear_regression: mae=4.5172, r2=0.4972
- e_env / ridge: mae=4.9672, r2=0.4927
- z_inv_plus_e_env / linear_regression: mae=5.0168, r2=0.3799

Route intervention overall MAE:
- learned: 10.061665
- shuffled: 10.064802
- remove_1: 10.070719
- remove_0: 10.094234
- remove_2: 10.114122
- uniform: 10.247080

Environment swap summary:
- same_env: mean_abs_change=1.701828, delta_mae=0.258555
- cross_env: mean_abs_change=1.764200, delta_mae=0.250469

## Files

- features: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/features.npz`
- per_sample: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/per_sample.csv`
- probe_env_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/probe_env_results.csv`
- probe_residual_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/probe_residual_results.csv`
- expert_env_crosstab: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/expert_env_crosstab.csv`
- expert_by_workday_holiday: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/expert_by_workday_holiday.png`
- expert_by_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/expert_by_hour.png`
- expert_by_rush_hour: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/expert_by_rush_hour.png`
- expert_per_env_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/expert_per_env_mae.csv`
- route_intervention_mae: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/route_intervention_mae.csv`
- env_swap_results: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/env_swap_results.csv`
- env_swap_boxplot: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/env_swap_boxplot.png`
- umap_z_inv_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/umap_z_inv_by_env.png`
- umap_e_env_by_env.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/umap_e_env_by_env.png`
- umap_e_env_by_expert.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/umap_e_env_by_expert.png`
- umap_z_inv_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/umap_z_inv_by_hour.png`
- umap_e_env_by_hour.png: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_best_recipe_no_conf_k3_no_mask_no_swap_no_club_seed2025/case_study/case_outputs/umap_e_env_by_hour.png`
