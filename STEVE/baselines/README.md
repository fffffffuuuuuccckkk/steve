# OOD / distribution-shift baselines

This directory is for method implementations only.  It is intentionally
separated from STEVE/FPEM code:

```text
baseline_adapters/  -> common benchmark I/O, scaler, graph, metrics
baselines/<method>/ -> method-specific code and adaptation notes
third_party/        -> upstream snapshots pinned by commit
scripts/baselines/ -> one launcher per baseline
```

Rules:

1. Do not import `models.our_model`.
2. Do not import `models/fpem/*`.
3. Do not use FPEM-only variables such as invariant/variant representations,
   latent confounders, GCI/SCD, or counterfactual routers.
4. Keep official upstream code in `third_party/` where possible.
5. Keep benchmark adapters outside upstream code.

Current first-stage status:

| Method | Upstream snapshot | Unified LargeST-SD launcher |
| ------ | ----------------- | --------------------------- |
| AGCRN | `third_party/AGCRN_upstream` | yes |
| STONE | `third_party/STONE_upstream` | pending |
| STOP | `third_party/STOP_upstream` | pending |
| EpoD | official repo not found; non-official adapter | yes |
