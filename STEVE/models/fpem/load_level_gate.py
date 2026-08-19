import torch
import torch.nn as nn


def assign_load_levels(load_score, thresholds):
    """Assign detached sample-level scores to monotonically ordered bins."""
    score = load_score.detach().reshape(-1)
    if thresholds is None or thresholds.numel() == 0:
        return torch.zeros_like(score, dtype=torch.long)
    boundaries = thresholds.detach().to(device=score.device, dtype=score.dtype).reshape(-1)
    return torch.bucketize(score.contiguous(), boundaries.contiguous(), right=True).long().detach()


def select_load_expert(y_env_heads, load_level):
    """Select exactly one environment expert for every sample."""
    if y_env_heads.dim() != 5:
        raise ValueError(
            "y_env_heads must have shape [B,K,T,N,D], got {}".format(tuple(y_env_heads.shape))
        )
    level = load_level.detach().to(device=y_env_heads.device, dtype=torch.long).reshape(-1)
    if level.shape[0] != y_env_heads.shape[0]:
        raise ValueError(
            "load_level batch size mismatch: {} vs {}".format(level.shape[0], y_env_heads.shape[0])
        )
    if bool((level < 0).any()) or bool((level >= y_env_heads.shape[1]).any()):
        raise ValueError("load_level contains an expert index outside [0, K)")
    batch = torch.arange(y_env_heads.shape[0], device=y_env_heads.device)
    return y_env_heads[batch, level]


def _pool_route_representation(rep):
    if rep.dim() == 3:
        return rep.mean(dim=1)
    if rep.dim() == 4:
        return rep.mean(dim=(1, 2))
    raise ValueError(
        "router representation must be [B,N,H] or [B,T,N,H], got {}".format(
            tuple(rep.shape)
        )
    )


class EnvironmentUseGate(nn.Module):
    """Independent soft decision between invariant and selected environment predictions."""

    def __init__(self, representation_dim, num_load_levels, hidden_dim, dropout=0.0):
        super().__init__()
        self.num_load_levels = int(num_load_levels)
        self.load_embedding = nn.Embedding(self.num_load_levels, representation_dim)
        self.mlp = nn.Sequential(
            nn.Linear(representation_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_inv, e_useful, load_level):
        level = load_level.detach().to(device=z_inv.device, dtype=torch.long).reshape(-1)
        load_emb = self.load_embedding(level)
        features = torch.cat(
            [_pool_route_representation(z_inv), _pool_route_representation(e_useful), load_emb],
            dim=-1,
        )
        logits = self.mlp(features).squeeze(-1)
        return {
            "env_use_gate_logits": logits,
            "env_use_gate": torch.sigmoid(logits),
            "load_level_embedding": load_emb,
            "env_use_gate_features": features,
        }


class HardEnvironmentUseRouter(nn.Module):
    """Observable two-class router: invariant (0) or selected environment (1)."""

    def __init__(self, representation_dim, num_load_levels, hidden_dim, dropout=0.0):
        super().__init__()
        self.num_load_levels = int(num_load_levels)
        self.load_embedding = nn.Embedding(self.num_load_levels, representation_dim)
        self.mlp = nn.Sequential(
            nn.Linear(representation_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_inv, e_useful, load_level):
        level = load_level.detach().to(
            device=z_inv.device, dtype=torch.long
        ).reshape(-1)
        load_emb = self.load_embedding(level)
        features = torch.cat(
            [_pool_route_representation(z_inv), _pool_route_representation(e_useful), load_emb],
            dim=-1,
        )
        logits = self.mlp(features)
        route_id = torch.argmax(logits, dim=-1).long()
        return {
            "hard_router_logits": logits,
            "hard_router_route_id": route_id,
            "hard_router_probabilities": torch.softmax(logits, dim=-1),
            "hard_router_features": features,
            "load_level_embedding": load_emb,
        }


def hard_select_invariant_or_environment(y_inv, y_env_selected, route_id):
    route = route_id.detach().to(device=y_inv.device, dtype=torch.long).reshape(-1)
    if route.shape[0] != y_inv.shape[0]:
        raise ValueError("hard route batch size does not match predictions")
    if bool((route < 0).any()) or bool((route > 1).any()):
        raise ValueError("hard route IDs must be 0 or 1")
    route_view = route.view(-1, *([1] * (y_inv.dim() - 1)))
    return torch.where(route_view == 0, y_inv, y_env_selected)
