import inspect
from typing import Dict, Tuple

import torch
from torch import nn


class BatchFirstMultiheadAttention(nn.Module):
    """Version-compatible MultiheadAttention with a batch-first public API.

    PyTorch 1.8 does not support ``batch_first`` in ``nn.MultiheadAttention``
    and older releases also do not accept ``average_attn_weights`` in forward.
    This wrapper keeps the rest of FPEM using ``[B, L, H]`` tensors while
    adapting the call to the installed torch version.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        init_params = inspect.signature(nn.MultiheadAttention.__init__).parameters
        self.supports_batch_first = "batch_first" in init_params
        kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        if self.supports_batch_first:
            kwargs["batch_first"] = True
        self.attn = nn.MultiheadAttention(**kwargs)

        forward_params = inspect.signature(self.attn.forward).parameters
        self.supports_average_attn_weights = "average_attn_weights" in forward_params

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.supports_batch_first:
            q, k, v = query, key, value
        else:
            q = query.transpose(0, 1).contiguous()
            k = key.transpose(0, 1).contiguous()
            v = value.transpose(0, 1).contiguous()

        kwargs = {"need_weights": need_weights}
        if self.supports_average_attn_weights:
            kwargs["average_attn_weights"] = False
        out, weights = self.attn(q, k, v, **kwargs)

        if not self.supports_batch_first:
            out = out.transpose(0, 1).contiguous()
        return out, weights


class NodeTemporalAttentionPool(nn.Module):
    """Pool ``[B,T,N,H]`` into node-wise queries ``[B,N,H]``."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        mid = max(hidden_dim // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(z_seq)
        alpha = torch.softmax(logits, dim=1)
        pooled = (alpha * z_seq).sum(dim=1)
        return pooled, alpha


class EnvConfounderExtractor(nn.Module):
    """Extract node-wise confounders from variable-length environment sequences."""

    def __init__(
        self,
        hidden_dim: int,
        num_basis: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_temporal_attn_pool: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.basis_queries = nn.Parameter(torch.randn(num_basis, hidden_dim) * 0.02)

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.basis_attn = BatchFirstMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.use_temporal_attn_pool = use_temporal_attn_pool
        self.node_pool = (
            NodeTemporalAttentionPool(hidden_dim, dropout=dropout)
            if use_temporal_attn_pool
            else None
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.basis_norm = nn.LayerNorm(hidden_dim)
        self.node_to_basis_attn = BatchFirstMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z_seq.dim() != 4:
            raise ValueError(
                f"EnvConfounderExtractor expects [B,T,N,H], got {tuple(z_seq.shape)}"
            )

        batch_size, time_steps, num_nodes, hidden_dim = z_seq.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: extractor={self.hidden_dim}, input={hidden_dim}"
            )

        tokens = self.token_norm(
            z_seq.reshape(batch_size, time_steps * num_nodes, hidden_dim)
        )
        basis_q = self.basis_queries.unsqueeze(0).expand(batch_size, -1, -1)
        basis, basis_attn = self.basis_attn(
            query=basis_q,
            key=tokens,
            value=tokens,
            need_weights=True,
        )
        basis = self.basis_norm(basis)

        if self.node_pool is not None:
            node_q, temporal_alpha = self.node_pool(z_seq)
        else:
            node_q = z_seq.mean(dim=1)
            temporal_alpha = None
        node_q = self.node_norm(node_q)

        confounder, node_basis_attn = self.node_to_basis_attn(
            query=node_q,
            key=basis,
            value=basis,
            need_weights=True,
        )
        confounder = self.out(confounder)

        logs = {
            "conf_basis_mean": basis.detach().mean(),
            "conf_basis_std": basis.detach().std(unbiased=False),
            "conf_repr_mean": confounder.detach().mean(),
            "conf_repr_std": confounder.detach().std(unbiased=False),
        }
        if temporal_alpha is not None:
            prob = temporal_alpha.detach().clamp_min(1e-8)
            logs["conf_temporal_attn_entropy"] = (
                -(prob * prob.log()).sum(dim=1).mean()
            )
        if basis_attn is not None:
            prob = basis_attn.detach().clamp_min(1e-8)
            logs["conf_basis_attn_entropy"] = (
                -(prob * prob.log()).sum(dim=-1).mean()
            )
        if node_basis_attn is not None:
            prob = node_basis_attn.detach().clamp_min(1e-8)
            logs["conf_node_basis_attn_entropy"] = (
                -(prob * prob.log()).sum(dim=-1).mean()
            )

        return confounder, logs
