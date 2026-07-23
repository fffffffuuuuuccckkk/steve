import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num, device=supports.device, dtype=supports.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x).permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1.0 - r) * hc

    def init_hidden_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device, dtype=dtype)


class AGCRNEncoder(nn.Module):
    """AGCRN sequence encoder that returns node states for every input step."""

    def __init__(self, node_num, input_dim, hidden_dim, cheb_k=2, embed_dim=10, num_layers=2):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
        self.cells = nn.ModuleList([AGCRNCell(node_num, input_dim, hidden_dim, cheb_k, embed_dim)])
        for _ in range(1, num_layers):
            self.cells.append(AGCRNCell(node_num, hidden_dim, hidden_dim, cheb_k, embed_dim))

    def forward(self, x):
        # x: [B, T, N, C]
        if x.dim() != 4:
            raise ValueError("AGCRNEncoder expects input shaped [B, T, N, C]")
        current_inputs = x
        for cell in self.cells:
            state = cell.init_hidden_state(x.shape[0], x.device, x.dtype)
            inner_states = []
            for t in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, t, :, :], state, self.node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs
