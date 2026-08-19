# EpoD baseline

Paper:

```text
Improving Generalization of Dynamic Graph Learning via Environment Prompt
NeurIPS 2024
```

Official repository:

```text
Not found in the first repository audit pass.
```

Status:

```text
Non-official LargeST-SD adapter implemented.
This implementation follows the paper's self-prompted learning and dynamic
subgraph ideas, but it is not the authors' official code.
```

Implemented components:

```text
1. Node-wise learnable environment prompt tokens P.
2. Prompt-answer cross attention: Q=P W_Q, K=Z W_K, V=Z W_V.
3. SPL auxiliary objective: prompt answer reconstructs recent observed signal.
4. KL regularization between prompt answer distribution and backbone embedding.
5. Environment-induced asymmetric dynamic subgraph from KL(mean||node_i) and
   KL(node_j||mean), restricted by the fixed road graph support.
6. Environment-enhanced AGCRN forecasting backbone.
```

Run:

```bash
bash scripts/baselines/run_largest_sd_epod.sh
```
