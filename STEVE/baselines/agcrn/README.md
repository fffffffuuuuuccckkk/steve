# AGCRN baseline

This directory contains a baseline-local AGCRN implementation for the unified
LargeST/KnowAir OOD benchmark protocol.

The code is intentionally separated from `models/our_model.py` and
`models/fpem/*`.  It does not use FPEM representations, confounder losses,
routers, or other STEVE-specific mechanisms.

## Run

```bash
bash scripts/baselines/run_largest_sd_agcrn.sh
```

## Outputs

Default output directory:

```text
experiments/LargeST_SD_OOD/AGCRN/seed2024
```

Files:

```text
protocol.json
best_val_model.pth
training_curve.csv
summary.json
```

