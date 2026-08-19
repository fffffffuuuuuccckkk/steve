from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """AGCRN adaptive vertex-wise graph convolution.

    This baseline-local copy is intentionally independent from FPEM/STEVE
    modules.  It follows the standard AGCRN adaptive adjacency construction.
    """

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = int(cheb_k)
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, self.cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        node_num = int(node_embeddings.shape[0])
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.t())), dim=1)
        support_set = [torch.eye(node_num, device=supports.device, dtype=supports.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x).permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.node_num = int(node_num)
        self.hidden_dim = int(dim_out)
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1.0 - r) * hc

    def init_hidden_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device, dtype=dtype)


class AGCRNEncoder(nn.Module):
    def __init__(self, node_num: int, input_dim: int, hidden_dim: int, cheb_k: int = 2, embed_dim: int = 10, num_layers: int = 1):
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
        cells = [AGCRNCell(node_num, input_dim, hidden_dim, cheb_k, embed_dim)]
        for _ in range(1, self.num_layers):
            cells.append(AGCRNCell(node_num, hidden_dim, hidden_dim, cheb_k, embed_dim))
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"AGCRNEncoder expects [B,T,N,C], got {tuple(x.shape)}")
        current_inputs = x
        for cell in self.cells:
            state = cell.init_hidden_state(x.shape[0], x.device, x.dtype)
            inner_states = []
            for t in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, t, :, :], state, self.node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs


class AGCRNForecast(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        horizon: int,
        rnn_units: int = 64,
        cheb_k: int = 2,
        embed_dim: int = 10,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.encoder = AGCRNEncoder(
            node_num=num_nodes,
            input_dim=input_dim,
            hidden_dim=rnn_units,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.horizon * self.output_dim,
            kernel_size=(1, int(rnn_units)),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        last = encoded[:, -1:, :, :]
        output = self.end_conv(last)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        return output.permute(0, 1, 3, 2).contiguous()

