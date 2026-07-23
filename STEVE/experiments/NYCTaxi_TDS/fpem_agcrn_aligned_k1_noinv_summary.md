| name | model | seed | finished | best_epoch | best_val_loss | test_mixed_mae | test_workday_mae | test_holiday_mae | test_avg_mae | fpem_env_route_k | fpem_lambda_inv_pred | fpem_use_future_mi | fpem_use_swap | fpem_use_club_mi | fpem_use_confounder_extractor | fpem_env_route_head_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | agcrn | 2026 | false |  |  |  |  |  |  |  |  |  |  |  |  |  |
| full | steve | 2026 | false |  |  |  |  |  |  |  |  |  |  |  |  |  |
| inv_only | steve | 2026 | false |  |  |  |  |  |  |  |  |  |  |  |  |  |
| k3 | steve | 2026 | true | 37 | 12.307529 | 11.922884 | 11.835243 | 12.074862 | 11.955053 | 3 | 0.000000 | true | true | true | true | concat_input |
| with_inv_loss | steve | 2026 | true | 100 | 11.606656 | 10.605498 | 10.554064 | 10.694689 | 10.624376 | 1 | 0.200000 | true | true | true | true | concat_input |
| no_future_mi | steve | 2026 | false |  |  |  |  |  |  |  |  |  |  |  |  |  |
| no_swap | steve | 2026 | true | 100 | 11.673691 | 10.707557 | 10.696112 | 10.727403 | 10.711757 | 1 | 0.000000 | true | false | true | true | concat_input |
| no_club | steve | 2026 | false |  |  |  |  |  |  |  |  |  |  |  |  |  |
| no_confounder_extractor | steve | 2026 | true | 100 | 11.610465 | 10.549623 | 10.533999 | 10.576718 | 10.555359 | 1 | 0.000000 | true | true | true | false | concat_input |
