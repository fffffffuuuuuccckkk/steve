import torch
import torch.nn as nn


class EnvConditionedInvariantHeads(nn.Module):
    """FiLM-style environment modulation over a node-wise forecasting state."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        hyper_hidden_dim,
        dropout,
        alpha_mode="sample_gate",
        state_dim=None,
        context_dim=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = int(hidden_dim if state_dim is None else state_dim)
        self.num_heads = num_heads
        self.alpha_mode = str(alpha_mode).lower()
        ctx_dim = int(hidden_dim * 2 if context_dim is None else context_dim)
        self.context_dim = ctx_dim
        self.hypernets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ctx_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hyper_hidden_dim, self.state_dim * 2),
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

    def reset_identity(self, alpha_bias=-2.0, film_bias_std=1.0e-4):
        """Initialize FiLM heads as a near-identity modulation.

        The last hypernet layer emits ``gamma`` and ``beta``.  Its weight is
        zeroed and its bias is initialized at tiny scale, so ``H_k`` is very
        close to ``H_inv`` while the alpha gate still receives a first-step
        gradient.
        The alpha gate is still trainable; a small negative bias keeps early
        modulation gentle without killing gradients.
        """
        for hypernet in self.hypernets:
            last = hypernet[-1]
            nn.init.zeros_(last.weight)
            nn.init.normal_(last.bias, mean=0.0, std=float(film_bias_std))
        for alpha_gate in self.alpha_gates:
            last = alpha_gate[-1]
            nn.init.xavier_uniform_(last.weight)
            nn.init.constant_(last.bias, float(alpha_bias))

    def forward(self, z_inv, e_useful, h_inv, inv_pred):
        pool_z = z_inv.mean(dim=1)
        pool_e = e_useful.mean(dim=1)
        ctx = torch.cat([pool_z, pool_e], dim=-1)
        if ctx.shape[-1] != self.context_dim:
            raise ValueError(
                "FiLM context dim mismatch: expected {}, got {}".format(
                    self.context_dim, ctx.shape[-1]
                )
            )
        if h_inv.shape[-1] != self.state_dim:
            raise ValueError(
                "FiLM state dim mismatch: expected {}, got {}".format(
                    self.state_dim, h_inv.shape[-1]
                )
            )

        preds = []
        alphas = []
        deltas = []
        gamma_norms = []
        beta_norms = []
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
            pred = inv_pred(h_mod)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            elif pred.dim() != 4:
                raise ValueError(f"FiLM head predictor must return [B,N,F] or [B,T,N,F], got {tuple(pred.shape)}")
            preds.append(pred)
            alphas.append(alpha.squeeze(-1))
            gamma_norms.append(gamma.float().norm(dim=-1).mean())
            beta_norms.append(beta.float().norm(dim=-1).mean())
            deltas.append((gamma.float().pow(2) + beta.float().pow(2)).mean())

        y_hyper_heads = torch.stack(preds, dim=1)
        hyper_alpha = torch.stack(alphas, dim=1)
        hyper_delta_norm = torch.stack(deltas).mean().to(dtype=h_inv.dtype)
        hyper_gamma_norm_per_head = torch.stack(gamma_norms).to(dtype=h_inv.dtype)
        hyper_beta_norm_per_head = torch.stack(beta_norms).to(dtype=h_inv.dtype)
        return {
            "y_hyper_heads": y_hyper_heads,
            "hyper_alpha": hyper_alpha,
            "hyper_delta_norm": hyper_delta_norm,
            "hyper_gamma_norm_per_head": hyper_gamma_norm_per_head,
            "hyper_beta_norm_per_head": hyper_beta_norm_per_head,
            "hyper_ctx": ctx,
        }
