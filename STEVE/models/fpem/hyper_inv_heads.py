import torch
import torch.nn as nn


class EnvConditionedInvariantHeads(nn.Module):
    """FiLM-style environment modulation over the invariant hidden state."""

    def __init__(self, hidden_dim, num_heads, hyper_hidden_dim, dropout, alpha_mode="sample_gate"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.alpha_mode = str(alpha_mode).lower()
        ctx_dim = hidden_dim * 2
        self.hypernets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ctx_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hyper_hidden_dim, hidden_dim * 2),
            )
            for _ in range(num_heads)
        ])
        self.alpha_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ctx_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hyper_hidden_dim, 1),
            )
            for _ in range(num_heads)
        ])

    def forward(self, z_inv, e_useful, h_inv, inv_pred):
        pool_z = z_inv.mean(dim=1)
        pool_e = e_useful.mean(dim=1)
        ctx = torch.cat([pool_z, pool_e], dim=-1)

        preds = []
        alphas = []
        deltas = []
        for hypernet, alpha_gate in zip(self.hypernets, self.alpha_gates):
            gamma_beta = hypernet(ctx)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            if self.alpha_mode == "sample_gate":
                alpha = torch.sigmoid(alpha_gate(ctx))
            elif self.alpha_mode == "fixed_zero":
                alpha = torch.zeros(ctx.shape[0], 1, device=ctx.device, dtype=ctx.dtype)
            elif self.alpha_mode == "fixed_one":
                alpha = torch.ones(ctx.shape[0], 1, device=ctx.device, dtype=ctx.dtype)
            else:
                raise ValueError("unsupported fpem_hyper_alpha_mode={}".format(self.alpha_mode))
            gamma_node = gamma.unsqueeze(1)
            beta_node = beta.unsqueeze(1)
            alpha_node = alpha.unsqueeze(1)
            h_mod = h_inv * (1.0 + alpha_node * gamma_node) + alpha_node * beta_node
            preds.append(inv_pred(h_mod).unsqueeze(1))
            alphas.append(alpha.squeeze(-1))
            deltas.append((gamma.float().pow(2) + beta.float().pow(2)).mean())

        y_hyper_heads = torch.stack(preds, dim=1)
        hyper_alpha = torch.stack(alphas, dim=1)
        hyper_delta_norm = torch.stack(deltas).mean().to(dtype=h_inv.dtype)
        return {
            "y_hyper_heads": y_hyper_heads,
            "hyper_alpha": hyper_alpha,
            "hyper_delta_norm": hyper_delta_norm,
            "hyper_ctx": ctx,
        }
