import torch
import torch.nn as nn


class ConvexGatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_heads + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def zero_logs(ref):
        zero = ref.new_zeros(())
        return {
            "fpem/fusion_alpha_mean": zero,
            "fpem/fusion_alpha_std": zero,
            "fpem/fusion_alpha_min": zero,
            "fpem/fusion_alpha_max": zero,
        }

    def forward(self, y_inv, y_route, z_inv, e_useful, q):
        # alpha = sigmoid(Gate(pool(Z_inv), pool(E_useful), q, q_max, entropy(q)))
        z_pool = z_inv.mean(dim=1)
        e_pool = e_useful.mean(dim=1)
        q_max = q.max(dim=-1, keepdim=True).values
        q_prob = q.float().clamp(1e-8, 1.0)
        q_entropy = -(q_prob * q_prob.log()).sum(dim=-1, keepdim=True).to(dtype=q.dtype)
        gate_in = torch.cat([z_pool, e_pool, q, q_max, q_entropy], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in)).view(-1, 1, 1, 1).to(dtype=y_inv.dtype)
        y_final = (1.0 - alpha) * y_inv + alpha * y_route
        logs = {
            "fpem/fusion_alpha_mean": alpha.detach().mean(),
            "fpem/fusion_alpha_std": alpha.detach().std(unbiased=False),
            "fpem/fusion_alpha_min": alpha.detach().min(),
            "fpem/fusion_alpha_max": alpha.detach().max(),
        }
        return y_final, alpha, logs
