# STONE baseline

Paper:

```text
STONE: A Spatio-temporal OOD Learning Framework Kills Both Structural and Temporal Shifts (KDD 2024)
```

Official repository:

```text
https://github.com/PoorOtterBob/STONE-KDD-2024
```

Pinned commit:

```text
aa8e795087cdb14bd0e3ef130715a349fc24ce94
```

The upstream snapshot is kept in:

```text
third_party/STONE_upstream/
```

Status:

```text
Upstream snapshot pinned. LargeST-SD fixed-node temporal-OOD adapter is not yet
enabled as a training launcher in this first patch, because STONE's official
LargeST path is designed around both observed/unobserved node partitions and
Fréchet/spatial side information. The next step is to add a thin adapter that
maps our fixed-node LargeST-SD split to the official STONE inputs without
rewriting the core `src/base/stone.py` implementation.
```

