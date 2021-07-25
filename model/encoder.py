import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class Encoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation=torch.relu):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ae = [nn.Linear(input_dim, output_dim * 2)]
        self.ae.append(nn.Linear(output_dim * 2, output_dim))
        self.ae = nn.ModuleList(self.ae)
        self.gcn = gcn(input_dim, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index):
        x = self.gcn(x, edge_index)
        return x


class gcn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=torch.relu,
                 base_model=GCNConv, k: int = 2):
        super(gcn, self).__init__()

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            x = self.activation(x)
        return x