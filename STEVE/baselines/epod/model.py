from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.agcrn.model import AGCRNEncoder


def build_support_mask(adj: Optional[torch.Tensor], num_nodes: int, hops: int = 5) -> torch.Tensor:
    """Return an L-hop binary topology support mask.

    The observed graph is used only as the topology support for the EpoD-style
    dynamic subgraph.  The dynamic edge weights are still induced from prompt
    answers.
    """

    if adj is None:
        support = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
    else:
        support = torch.as_tensor(adj, dtype=torch.float32)
        if support.dim() != 2:
            raise ValueError(f"adj must be [N,N], got {tuple(support.shape)}")
        support = support[:num_nodes, :num_nodes] > 0
    support = support | torch.eye(num_nodes, dtype=torch.bool, device=support.device)
    hops = max(int(hops), 1)
    if hops > 1:
        reach = support.clone()
        frontier = support.clone()
        support_f = support.float()
        for _ in range(1, hops):
            frontier = frontier.float().matmul(support_f) > 0
            reach = reach | frontier
        support = reach
    return support


class PromptAnswerDecoder(nn.Module):
    """Self-prompted prompt-answer squeezing module.

    This follows the paper's Eq. 3-4 in a compact form:

    ``Q=P W_Q, K=Z W_K, V=Z W_V, Z_E=softmax(QK^T/sqrt(d))V + eps``.
    """

    def __init__(self, num_nodes: int, hidden_dim: int, noise_std: float = 0.01):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.noise_std = float(noise_std)
        self.prompt = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, z: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B,T,N,D]
        B, T, N, D = z.shape
        if N != self.num_nodes or D != self.hidden_dim:
            raise ValueError(f"expected z [B,T,{self.num_nodes},{self.hidden_dim}], got {tuple(z.shape)}")
        prompt = self.prompt.view(1, 1, N, D).expand(B, T, N, D)
        q = self.q_proj(prompt)
        k = self.k_proj(z)
        v = self.v_proj(z)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(D))
        att = torch.softmax(att, dim=-1)
        answer = torch.matmul(att, v)
        if training and self.noise_std > 0.0:
            answer = answer + torch.randn_like(answer) * self.noise_std
        return answer, att


class DynamicSubgraphEnhancer(nn.Module):
    """EpoD-style environment-induced dynamic subgraph enhancement.

    We compute the asymmetric environment dependency matrix from prompt answers
    using the paper's Eq. 8-12 proxy: KL(mean||node_i) and KL(node_j||mean),
    then restrict it to the observed graph support and aggregate the concatenated
    ``[X, Z_E]`` features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        graph_hops: int = 5,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.graph_hops = int(graph_hops)
        self.eps = float(eps)
        self.fuse = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _dynamic_adj(self, z_env: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        # z_env: [B,T,N,D]
        probs = torch.softmax(z_env.float(), dim=-1).clamp_min(self.eps)
        mean = probs.mean(dim=2, keepdim=True).clamp_min(self.eps)
        # KL(mean || node_i), KL(node_j || mean)
        kl_mean_to_node = (mean * (mean.log() - probs.log())).sum(dim=-1).clamp_min(0.0)  # [B,T,N]
        kl_node_to_mean = (probs * (probs.log() - mean.log())).sum(dim=-1).clamp_min(0.0)  # [B,T,N]
        score = (kl_mean_to_node.unsqueeze(-1) * kl_node_to_mean.unsqueeze(-2)).clamp_min(0.0)  # [B,T,N,N]
        mask = support_mask.to(device=score.device, dtype=torch.bool).view(1, 1, *support_mask.shape)
        score = score.masked_fill(~mask, 0.0)
        base = mask.to(dtype=score.dtype)
        base = base / base.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        denom = score.sum(dim=-1, keepdim=True)
        normalized = score / denom.clamp_min(self.eps)
        return torch.where(denom > self.eps, normalized, base.expand_as(score))

    def forward(self, x: torch.Tensor, z_env: torch.Tensor, support_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_for_x = z_env[:, -x.shape[1] :, :, :]
        if z_for_x.shape[1] != x.shape[1]:
            raise ValueError("z_env time dimension must cover x time dimension")
        augmented = torch.cat([x, z_for_x.to(dtype=x.dtype)], dim=-1)
        dyn_adj = self._dynamic_adj(z_for_x, support_mask)
        message = torch.einsum("btnm,btmc->btnc", dyn_adj.to(dtype=augmented.dtype), augmented)
        enhanced = self.fuse(torch.cat([augmented, message], dim=-1))
        return enhanced, dyn_adj


class EpoDAGCRNForecast(nn.Module):
    """Non-official EpoD + AGCRN adapter for the unified benchmark.

    The architecture keeps EpoD's two main ideas:

    1. self-prompted environment answer extraction;
    2. environment-induced dynamic subgraph enhancement before forecasting.

    It is deliberately standalone and does not depend on STEVE/FPEM internals.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        horizon: int,
        hidden_dim: int = 32,
        embed_dim: int = 10,
        cheb_k: int = 2,
        num_layers: int = 1,
        graph_hops: int = 5,
        prompt_noise_std: float = 0.01,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.prompt_encoder = AGCRNEncoder(
            node_num=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.prompt_decoder = PromptAnswerDecoder(num_nodes, hidden_dim, noise_std=prompt_noise_std)
        self.prompt_answer_head = nn.Linear(hidden_dim, output_dim)
        self.dynamic_enhancer = DynamicSubgraphEnhancer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            graph_hops=graph_hops,
        )
        self.forecast_encoder = AGCRNEncoder(
            node_num=num_nodes,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.horizon * self.output_dim,
            kernel_size=(1, hidden_dim),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "prompt_decoder.prompt" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.Tensor, support_mask: torch.Tensor, return_debug: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.prompt_encoder(x)
        z_env, prompt_att = self.prompt_decoder(z, training=self.training)
        enhanced, dyn_adj = self.dynamic_enhancer(x, z_env, support_mask)
        encoded = self.forecast_encoder(enhanced)
        last = encoded[:, -1:, :, :]
        pred = self.end_conv(last)
        pred = pred.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        pred = pred.permute(0, 1, 3, 2).contiguous()
        prompt_recon = self.prompt_answer_head(z_env[:, -self.horizon :, :, :])
        z_prob = torch.softmax(z.float(), dim=-1).clamp_min(1.0e-6)
        env_log_prob = torch.log_softmax(z_env.float(), dim=-1)
        kl = F.kl_div(env_log_prob, z_prob, reduction="batchmean")
        debug = {
            "prompt_recon": prompt_recon,
            "prompt_kl": kl,
            "prompt_attention_mean": prompt_att.detach().float().mean(),
            "dynamic_adj_mean": dyn_adj.detach().float().mean(),
            "dynamic_adj_std": dyn_adj.detach().float().std(unbiased=False),
        }
        if return_debug:
            debug["dynamic_adj"] = dyn_adj
        return pred, debug
