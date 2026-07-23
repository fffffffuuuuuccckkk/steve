import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adj(adj, node_num, device=None, dtype=None):
    if adj is None:
        return torch.eye(node_num, device=device, dtype=dtype if dtype is not None else torch.float32)
    if not torch.is_tensor(adj):
        adj = torch.as_tensor(adj)
    adj = adj.float()
    if adj.dim() > 2:
        adj = adj.squeeze()
    if adj.shape[0] != node_num or adj.shape[1] != node_num:
        return torch.eye(node_num, device=device or adj.device, dtype=dtype if dtype is not None else adj.dtype)
    adj = adj.to(device=device or adj.device, dtype=dtype if dtype is not None else adj.dtype)
    adj = adj.abs()
    adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class GraphWaveNetEncoder(nn.Module):
    """Lightweight GraphWaveNet-style encoder.

    Input/output contract matches AGCRNEncoder:
      input:  [B, T, N, C]
      output: [B, T, N, H]
    """

    def __init__(
        self,
        node_num,
        input_dim,
        hidden_dim,
        adj=None,
        num_layers=4,
        kernel_size=2,
        dropout=0.1,
    ):
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.register_buffer("support", _normalize_adj(adj, self.node_num))
        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        for layer in range(self.num_layers):
            dilation = 2 ** layer
            self.filter_convs.append(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=(1, self.kernel_size),
                    dilation=(1, dilation),
                )
            )
            self.gate_convs.append(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=(1, self.kernel_size),
                    dilation=(1, dilation),
                )
            )
            # self, forward graph, backward graph
            self.graph_convs.append(nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=(1, 1)))
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _graph_mix(x, support):
        # x: [B, C, N, T], support: [N, N]
        support = support.to(device=x.device, dtype=x.dtype)
        x_forward = torch.einsum("bcnt,nm->bcmt", x, support)
        x_backward = torch.einsum("bcnt,nm->bcmt", x, support.transpose(0, 1))
        return torch.cat([x, x_forward, x_backward], dim=1)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("GraphWaveNetEncoder expects input shaped [B, T, N, C]")
        if x.shape[2] != self.node_num:
            raise ValueError("GraphWaveNetEncoder node dim mismatch: expected {}, got {}".format(self.node_num, x.shape[2]))
        h = x.permute(0, 3, 2, 1)
        h = self.input_proj(h)
        for filter_conv, gate_conv, graph_conv in zip(self.filter_convs, self.gate_convs, self.graph_convs):
            residual = h
            dilation = filter_conv.dilation[1]
            pad = dilation * (self.kernel_size - 1)
            padded = F.pad(h, (pad, 0, 0, 0))
            z = torch.tanh(filter_conv(padded)) * torch.sigmoid(gate_conv(padded))
            z = graph_conv(self._graph_mix(z, self.support))
            z = F.dropout(z, p=self.dropout, training=self.training)
            h = residual + z
        h = h.permute(0, 3, 2, 1)
        return self.output_norm(h)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        # x: [B, L, H]
        bsz, length, _ = x.shape
        qkv = self.qkv(x).view(bsz, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(bsz, length, self.hidden_dim)
        return self.proj(out)


class STAEformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * float(mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        # x: [B, T, N, H]
        bsz, steps, nodes, hidden = x.shape
        t_in = self.temporal_norm(x).permute(0, 2, 1, 3).contiguous().view(bsz * nodes, steps, hidden)
        t_out = self.temporal_attn(t_in).view(bsz, nodes, steps, hidden).permute(0, 2, 1, 3)
        x = x + t_out
        s_in = self.spatial_norm(x).contiguous().view(bsz * steps, nodes, hidden)
        s_out = self.spatial_attn(s_in).view(bsz, steps, nodes, hidden)
        x = x + s_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class STAEformerEncoder(nn.Module):
    """STAEformer-style spatio-temporal attention encoder.

    It keeps the same adapter contract as AGCRN/GraphWaveNet:
      input:  [B, T, N, C]
      output: [B, T, N, H]
    """

    def __init__(
        self,
        node_num,
        input_dim,
        hidden_dim,
        input_length=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_length = int(max(input_length, 1))
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.node_embedding = nn.Parameter(torch.randn(self.node_num, self.hidden_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(self.input_length, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            STAEformerBlock(self.hidden_dim, int(num_heads), dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(int(num_layers))
        ])
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("STAEformerEncoder expects input shaped [B, T, N, C]")
        if x.shape[2] != self.node_num:
            raise ValueError("STAEformerEncoder node dim mismatch: expected {}, got {}".format(self.node_num, x.shape[2]))
        steps = x.shape[1]
        if steps > self.input_length:
            raise ValueError(
                "STAEformerEncoder input length {} exceeds configured max {}; set --staeformer_input_length_max larger".format(
                    steps, self.input_length
                )
            )
        h = self.input_proj(x)
        node_emb = self.node_embedding.to(device=x.device, dtype=h.dtype).view(1, 1, self.node_num, self.hidden_dim)
        time_emb = self.time_embedding[:steps].to(device=x.device, dtype=h.dtype).view(1, steps, 1, self.hidden_dim)
        h = h + node_emb + time_emb
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)
