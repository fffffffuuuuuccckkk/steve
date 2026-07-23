import torch
import torch.nn as nn


class EnvRouteHeads(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads, route_hidden_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, route_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(route_hidden_dim, output_dim),
            )
            for _ in range(num_heads)
        ])
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, route_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim, num_heads),
        )

    def forward(self, z_inv, e_useful, tau=1.0):
        # z_inv/e_useful: [B, N, H]
        expert_feat = z_inv + e_useful
        y_heads = torch.stack([head(expert_feat) for head in self.heads], dim=1).unsqueeze(2)
        logits = self.router(e_useful.mean(dim=1))
        q = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)
        q_prob = q.float().clamp(1e-8, 1.0)
        entropy = -(q_prob * q_prob.log()).sum(dim=-1).to(dtype=q.dtype)
        y_route = (q.view(q.shape[0], self.num_heads, 1, 1, 1) * y_heads).sum(dim=1)
        q_max, selected = q.max(dim=-1)
        return {
            "y_route_heads": y_heads,
            "env_route_logits": logits,
            "env_route_q": q,
            "env_route_entropy_per_sample": entropy,
            "env_route_q_max": q_max,
            "env_route_selected_head": selected,
            "y_route": y_route,
        }
