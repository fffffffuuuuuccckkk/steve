| experiment | model | ablation | seed | best_epoch | best_val_loss | test_mixed_mae | test_workday_mae | test_holiday_mae | test_avg_mae | finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AGCRN baseline | agcrn | baseline |  |  |  |  |  |  |  | false |
| FPEM full | steve | full | 2024 | 37 | 12.220444 | 11.971105 | 11.879971 | 12.129142 | 12.004556 | true |
| FPEM inv_only | steve | inv_only | 2024 | 28 | 13.502338 | 13.132832 | 13.141795 | 13.117289 | 13.129542 | true |
| FPEM no_env_mask | steve | no_env_mask | 2024 | 37 | 12.220444 | 11.971105 | 11.879971 | 12.129142 | 12.004556 | true |
| FPEM no_inv_loss | steve | no_inv_loss | 2024 | 39 | 12.194759 | 11.978074 | 11.928058 | 12.064811 | 11.996434 | true |
| FPEM no_swap_fallback | steve | no_swap_fallback | 2024 | 37 | 12.220444 | 11.971105 | 11.879971 | 12.129142 | 12.004556 | true |
| FPEM no_future_mi | steve | no_future_mi | 2024 | 40 | 12.548731 | 11.878107 | 11.815965 | 11.985867 | 11.900916 | true |
| FPEM no_hyper_reg | steve | no_hyper_reg | 2024 | 37 | 12.220444 | 11.971105 | 11.879971 | 12.129142 | 12.004556 | true |
| FPEM no_alpha_gate | steve | no_alpha_gate | 2024 | 37 | 12.220444 | 11.971105 | 11.879971 | 12.129142 | 12.004556 | true |
| FPEM k1 | steve | k1 | 2024 | 87 | 11.751267 | 10.835111 | 10.761775 | 10.962283 | 10.862029 | true |
