## Latent confounder regularization: shared residualization, two measurements

This implementation keeps the default STEVE/FPEM prediction path unchanged and
adds optional latent-confounder dependence/connectivity reduction terms.

### Modes

- `confounder_dep_mode=none`: default, no extractor and no extra loss.
- `gci`: graph-space learned functional connectivity reduction.
- `scd`: sample-space similarity reduction.
- `both`: one extractor call, one latent `C`, one projection, then both GCI/SCD
  branches read the same `Z_before/Z_after`.

The old `ctc` mode is not exposed as an active CLI choice. The lower-level
normalizer still maps `ctc -> scd` with a deprecation warning for old configs.

### Shared part

Both branches share:

```text
Z_V [B,T,N,D]
  -> LatentConfounderExtractor once
  -> C [B,d_c]
  -> project_out_confounder(Z_V, C) once
  -> Z_before [B,T,N,D], Z_after [B,T,N,D]
```

Projection flattens only over non-batch dimensions:

```text
Zf = Z_V.reshape(B, T*N*D)
Zc = Zf - mean_batch(Zf)
Cc = C  - mean_batch(C)
beta = solve(Cc.T @ Cc + ridge I, Cc.T @ Zc)
Z_after = Zc - Cc @ beta
Z_before = Zc
```

No forecasting residual, spatial reconstruction, mean pooling, max pooling, or
adaptive pooling is used in the confounder regularizer.

### SCD branch

SCD uses full-sample cosine similarity:

```text
Z_before/Z_after [B,T,N,D]
  -> reshape [B,T*N*D]
  -> row L2 normalize
  -> S_before/S_after [B,B]
```

The loss is:

```text
mean(S_after[i,j]^2 for i != j)
```

It logs `sample_assoc_before_C`, `sample_assoc_after_C`, and
`sample_assoc_reduction`. Old `sample_dep_*` names are kept as summary aliases.
There is no active CTC, HSIC, Gram-of-Gram, or kernel alignment objective.

### GCI branch

GCI uses a simple shared functional graph learner:

```text
Z [B,T,N,D]
  -> permute/reshape [B,N,T*D]
  -> shared Linear(T*D, graph_embed_dim)
  -> E [B,N,d_e]
  -> softmax(relu(E @ E.T), dim=-1)
  -> A [B,N,N]
```

The same `FunctionalGraphLearner` instance is used for `Z_before` and `Z_after`.
The observed traffic graph is used only to build a non-edge mask; it does not
supervise or generate the learned adjacency.

The graph Linear is trained only by an AGCRN-graph alignment loss:

```text
A_agcrn = softmax(relu(node_embeddings @ node_embeddings.T), dim=1).detach()
A_before = graph_learner(Z_before.detach())
L_graph_align = mse(A_before, A_agcrn)
```

`A_agcrn` is detached, so alignment cannot modify the AGCRN forecasting graph.
`Z_before` is also detached for this loss, so alignment trains only the graph
Linear.

For GCI, `Z_after` is passed through the same numerical Linear parameters, but
the Linear weight/bias are detached:

```text
A_after = graph_learner(Z_after, detach_params=True)
```

Thus `L_GCI` updates the confounder extractor through `C -> Z_after -> A_after`,
but does not update the graph Linear parameters.

The loss is:

```text
mean(A_after[:, nonedge_mask]^2)
```

Graph edges and diagonals are excluded from the penalty. GCI logs non-edge and
edge connectivity before/after C residualization, plus graph embedding and
adjacency mean/std diagnostics to watch for collapse.

### Canonical parameters

Use:

```text
confounder_projection_ridge: 0.001
confounder_dep_detach_target: True
confounder_graph_embed_dim: 16
gci_graph_align_weight: 0.1
```

Deprecated aliases `gci_ridge/scd_ridge/gci_detach_target/scd_detach_target`
are resolved to the canonical values when present; conflicting old values raise
an error. `scd_alignment_weight` is ignored with a deprecation warning.
