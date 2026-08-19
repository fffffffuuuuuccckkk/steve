# AGCRN adaptation for the unified OOD benchmark

Paper:

```text
Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
```

Official repository:

```text
https://github.com/LeiBAI/AGCRN
```

Pinned commit:

```text
7fbbf2aeb099242098a3cf482b55cd45d7295c28
```

Original dataset / split / horizon:

```text
AGCRN is a standard traffic forecasting baseline and is commonly evaluated on
PeMS-style in-distribution splits.  It is not an OOD-specific method.
```

Our benchmark:

```text
Dataset: LargeST-SD_TDS
Nodes: fixed 716-node SD subset
Graph: not directly consumed by AGCRN; AGCRN learns adaptive adjacency
Train: 2019
Val: 2020 first half
OOD Test: 2020 second half
Input length / horizon: read from data/LargeST-SD_TDS/meta.json
Scaler: fitted on train x only
Metric mask: target > 5.0, matching the current STEVE LargeST protocol
Checkpoint selection: lowest validation MAE
```

Changes made:

1. Kept the official repository snapshot under `third_party/AGCRN_upstream/`.
2. Wrapped the AGCRN model formula in a standalone trainer under `baselines/agcrn/`.
2. Used `baseline_adapters/largest_dataset.py` for the unified split/scaler.
3. Saved unified `summary.json` for paper-table aggregation.

Unchanged core components:

1. Adaptive node embeddings.
2. Row-softmax adaptive adjacency.
3. AVWGCN recurrent cell.
4. Last hidden state forecast head.

Extra environment labels / external variables:

```text
None.
```

Test-time adaptation:

```text
None.
```

Relationship to FPEM:

```text
No dependency on FPEM modules.  This baseline must not import `models.our_model`
or `models/fpem/*`.
```
