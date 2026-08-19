from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def load_graph_npz_or_npy(path: Optional[str], device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "adj" in data.files:
            arr = data["adj"]
        elif "adj_mx" in data.files:
            arr = data["adj_mx"]
        else:
            arr = data[data.files[0]]
    else:
        arr = np.load(path, allow_pickle=True)
    tensor = torch.as_tensor(arr, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def count_nodes_from_graph(path: Optional[str]) -> Optional[int]:
    graph = load_graph_npz_or_npy(path, device=None)
    if graph is None:
        return None
    return int(graph.shape[0])

