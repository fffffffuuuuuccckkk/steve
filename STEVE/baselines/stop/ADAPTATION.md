# STOP adaptation record

Paper:

```text
STOP: Spatial-Temporal Out-of-Distribution Generalization for Traffic Prediction
ICML 2025
```

Official repository:

```text
https://github.com/PoorOtterBob/STOP
```

Pinned commit:

```text
8babb610ece36a4215b2f66e1ef4a154f0c4f440
```

Upstream files:

```text
third_party/STOP_upstream/
```

Original dataset / split / horizon:

```text
The official repository includes LargeST, KnowAir, and TrafficStream paths. Its
LargeST code constructs temporal features and STOP residual/prompt modules in
its own trainer.
```

Our benchmark target:

```text
Dataset: LargeST-SD_TDS
Nodes: fixed 716-node SD subset
Graph: fixed official road-distance graph
Train: 2019
Val: 2020 first half
OOD Test: 2020 second half
Input length / horizon: read from data/LargeST-SD_TDS/meta.json
Scaler: fitted on train x only
```

Necessary adaptation planned:

1. Preserve `third_party/STOP_upstream/LargeST/src/models/stop.py`.
2. Add a thin dataset adapter that adds legal time-of-day/day-of-week channels
   from `time_label` for STOP's LargeST MLP backbone.
3. Keep STOP's residual/prompt modules intact.
4. Save unified summary fields under `experiments/LargeST_SD_OOD/STOP/`.

Extra environment labels / external variables:

```text
STOP uses time features in its official LargeST backbone. These are legal
test-observable calendar features, but must be reported as external/time
features in comparison tables.
```

Current status:

```text
Official snapshot pinned; runnable unified LargeST-SD launcher pending.
```

Relationship to FPEM:

```text
No FPEM-specific imports are allowed in the STOP adapter.
```

