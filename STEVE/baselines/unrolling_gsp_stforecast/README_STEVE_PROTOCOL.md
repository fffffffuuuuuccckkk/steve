# Unrolling-GSP-STForecast in the STEVE protocol

Source repository copied from:

```text
OuXiaoyu@211.71.76.187:/data/OuXiaoyu/Unrolling-GSP-STForecast
```

This directory is a standalone baseline copy.  It is intentionally not imported
by FPEM/STEVE model code.  The STEVE protocol runner lives at:

```text
scripts/run_tds_nyctaxi_unrolling_gsp_protocol.py
scripts/run_tds_nyctaxi_unrolling_gsp_protocol.sh
```

The runner reads STEVE's pre-windowed NYCTaxi_TDS files:

```text
data/NYCTaxi_TDS/train.npz
data/NYCTaxi_TDS/val.npz
data/NYCTaxi_TDS/test.npz
data/NYCTaxi_TDS/adj_mx.npz
```

It maps each sample to the paper model as:

```text
history x:       [B, 35, 200, C]
full sequence:   concat(x, y) -> [B, 36, 200, C]
future target:   y -> [B, 1, 200, C]
```

By default `C=2` to match STEVE/FPEM evaluation.  Set
`UNROLLING_USE_ONE_CHANNEL=true` for the paper-style one-channel setting.

Metrics use the NYCTaxi_TDS/STEVE masked flow MAE convention:

```text
channel-weighted masked MAE, target > 5, yita=0.5
```

Outputs are saved under:

```text
experiments/NYCTaxi_TDS/<RUN_PREFIX>_seed<seed>/
```

with `summary.json`, `best_val_model.pth`, `last_model.pth`, and launcher
`summary.tsv` under `<RUN_PREFIX>_logs/`.
