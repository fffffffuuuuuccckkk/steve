# STONE adaptation record

Paper:

```text
STONE: A Spatio-temporal OOD Learning Framework Kills Both Structural and Temporal Shifts
KDD 2024
```

Official repository:

```text
https://github.com/PoorOtterBob/STONE-KDD-2024
```

Pinned commit:

```text
aa8e795087cdb14bd0e3ef130715a349fc24ce94
```

Upstream files:

```text
third_party/STONE_upstream/
```

Original dataset / split / horizon:

```text
The official repository supports LargeST SD/GBA generated from CA data and
KnowAir. Its LargeST experiments include temporal and spatial OOD settings,
with observed/unobserved node partitions and Fréchet embedding components.
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

1. Keep official `src/base/stone.py` and its submodules intact.
2. Add an adapter that supplies fixed-node LargeST-SD tensors and graph/side information.
3. Disable structural node expansion only through configuration/adapter where possible, not by deleting STONE modules.
4. Save unified summary fields under `experiments/LargeST_SD_OOD/STONE/`.

Extra environment labels / external variables:

```text
To be audited while wiring the official STONE data path. STONE uses spatial
side information / Fréchet embeddings in its official implementation; this must
be reported separately from methods with no external side information.
```

Current status:

```text
Official snapshot pinned; runnable unified LargeST-SD launcher pending.
```

Relationship to FPEM:

```text
No FPEM-specific imports are allowed in the STONE adapter.
```

