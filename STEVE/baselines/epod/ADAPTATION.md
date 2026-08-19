# EpoD adaptation record

Paper:

```text
Improving Generalization of Dynamic Graph Learning via Environment Prompt
NeurIPS 2024
```

Official repository:

```text
Not found during the initial audit pass.
```

Pinned commit:

```text
N/A
```

Current status:

```text
Runnable non-official LargeST-SD adapter added under `baselines/epod/`.
```

If no official implementation is available:

```text
This implementation follows Eq./Algorithm sections of the paper and is not the
authors' official code.
```

Paper equations/algorithm used:

```text
Eq. 1: node-wise learnable prompt matrix P.
Eq. 3-4: prompt-answer cross attention.
Eq. 5 / Eq. 15: SPL auxiliary loss and KL regularization.
Eq. 8-12: asymmetric environment dependency and graph-support restricted
dynamic subgraph.
Eq. 13-14: concatenate/enhance historical features with prompted environment
representation before prediction.
Algorithm 1: joint environment prompt and utilization training.
```

Original dataset / split / horizon:

```text
The paper reports traffic tasks such as 12 -> 24 on several datasets.  No
official preprocessing code was found during this pass.
```

Our benchmark:

```text
Dataset: LargeST-SD_TDS
Nodes: fixed 716-node SD subset
Graph: fixed official road-distance graph support
Train: 2019
Val: 2020 first half
OOD Test: 2020 second half
Input length / horizon: read from data/LargeST-SD_TDS/meta.json
Scaler: fitted on train x only
```

Changes made:

1. Used a baseline-local AGCRN backbone for both prompt encoding and forecasting.
2. Used recent observed historical signal as the SPL prompt reconstruction target,
   avoiding test-time target/future leakage.
3. Used the fixed LargeST road graph as the support mask for dynamic subgraphs.
4. Saved unified `summary.json` under `experiments/LargeST_SD_OOD/EpoD/`.

Unchanged conceptual components:

1. Self-prompted environment inference.
2. Prompt-answer interaction.
3. KL-style prompt/environment regularization.
4. Environment-induced dynamic subgraph before prediction.

Extra environment labels / external variables:

```text
None.  This adapter does not consume environment labels or external variables.
It uses only historical x and the fixed road graph at inference.
```

Test-time adaptation:

```text
None.
```

Relationship to FPEM:

```text
Any future EpoD implementation must be independent from FPEM-specific modules.
```
