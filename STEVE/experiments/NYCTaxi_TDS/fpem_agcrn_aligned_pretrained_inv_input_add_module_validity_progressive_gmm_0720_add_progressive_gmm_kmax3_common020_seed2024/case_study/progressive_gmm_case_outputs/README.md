# Progressive GMM case study

- Checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_progressive_gmm_0720_add_progressive_gmm_kmax3_common020_seed2024/best_val_model.pth`
- Experiment: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/fpem_agcrn_aligned_pretrained_inv_input_add_module_validity_progressive_gmm_0720_add_progressive_gmm_kmax3_common020_seed2024`
- Seed: `2024`
- Git commit: `1b38ed8ad03fdc47198d62e51385df04248f4751`
- Dataset: `NYCTaxi_TDS`
- Active GMM K: `3`
- Progressive common-loss weight: `0.2`
- Pretrained invariant checkpoint: `/data/OuXiaoyu/STEVE_CODE/STEVE/experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth`
- Validation Hungarian mapping: `{0: 2, 1: 1, 2: 0}`
- Best-fixed expert from validation: `0`

## Routing summary
- `best_fixed`: test_avg_mae=8.223312377929688
- `random_uniform_top1`: test_avg_mae=8.228968834877014
- `random_prior_top1`: test_avg_mae=8.228847312927247
- `shuffled_gmm_route`: test_avg_mae=8.227465295791626
- `gmm_hard_identity`: test_avg_mae=8.2356858253479
- `gmm_hard_val_hungarian`: test_avg_mae=8.220052719116211
- `gmm_hard_val_independent`: test_avg_mae=8.221039772033691
- `uniform_all_experts`: test_avg_mae=8.227644920349121
- `oracle_top1`: test_avg_mae=8.208650588989258

## Cross-MAE interpretation
- Identity same as Hungarian: `False`
- Identity minus Hungarian validation MAE sum: `0.014313180257069291`
- Independent and Hungarian agree: `False`
- Per-cluster second-best margin: `[0.008661365020525835, 0.0008535674797105486, 0.0013526298296913097]`

## Generated files
- `arrays.npz`
- `cluster_profiles/cluster_by_hour.png`
- `cluster_profiles/cluster_by_rush_hour.png`
- `cluster_profiles/cluster_by_workday_holiday.png`
- `cluster_profiles/cluster_hour_distribution.csv`
- `cluster_profiles/cluster_profile.csv`
- `cluster_profiles/cluster_profile_heatmap.png`
- `cluster_profiles/cluster_rush_distribution.csv`
- `cluster_profiles/cluster_size.png`
- `cluster_profiles/cluster_workday_distribution.csv`
- `confidence_bin_summary.csv`
- `confidence_vs_gain.png`
- `corrections/cluster_correction_magnitude.png`
- `corrections/cluster_correction_summary.csv`
- `corrections/cluster_horizon_correction_heatmap.png`
- `corrections/correction_arrays.npz`
- `corrections/selected_expert_correction_distribution.png`
- `corrections/top_nodes_by_correction.csv`
- `embeddings/embedding_coordinates.csv`
- `embeddings/linear_probe_results.csv`
- `embeddings/umap_e_env_by_gmm_cluster.png`
- `embeddings/umap_e_env_by_gmm_confidence.png`
- `embeddings/umap_e_env_by_hour.png`
- `embeddings/umap_e_env_by_hungarian_expert.png`
- `embeddings/umap_e_env_by_workday_holiday.png`
- `embeddings/umap_z_inv_by_gmm_cluster.png`
- `embeddings/umap_z_inv_by_hour.png`
- `embeddings/umap_z_inv_by_workday_holiday.png`
- `entropy_vs_gain.png`
- `mappings/cluster_to_expert_mapping.json`
- `mappings/validation_cross_mae.tsv`
- `mappings/validation_cross_mae_heatmap.png`
- `mappings/validation_cross_mae_interpretation.json`
- `metadata.json`
- `per_sample_metrics.csv`
- `routing/routing_gain_summary.csv`
- `routing/routing_method_comparison.png`
- `temporal/cluster_duration_statistics.csv`
- `temporal/cluster_timeline.png`
- `temporal/cluster_transition_heatmap.png`
- `temporal/cluster_transition_matrix.csv`
- `temporal/temporal_summary.json`
- `test_cluster_expert_mae.csv`
- `test_cluster_expert_mae_heatmap.png`
- `test_mapping_generalization.csv`
- `typical_samples/cluster_0_failure.png`
- `typical_samples/cluster_0_neutral.png`
- `typical_samples/cluster_0_positive.png`
- `typical_samples/cluster_1_failure.png`
- `typical_samples/cluster_1_neutral.png`
- `typical_samples/cluster_1_positive.png`
- `typical_samples/cluster_2_failure.png`
- `typical_samples/cluster_2_neutral.png`
- `typical_samples/cluster_2_positive.png`
- `typical_samples/selected_case_metadata.csv`
- `validation_cluster_expert_mae.csv`

## Required limitations
- GMM environment discovery does not use target values.
- Cluster-to-expert mapping uses validation prediction errors.
- Test labels are used only for final evaluation and oracle diagnostics.
- Oracle is not deployable.
- Three seeds share the same seed-2024 invariant backbone.
- Two-dimensional UMAP/PCA plots do not prove strict disentanglement.
- Seed 2025 may exhibit limited expert differentiation.
- Case examples are selected using deterministic rules.
