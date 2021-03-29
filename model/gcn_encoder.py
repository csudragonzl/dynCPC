import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.rrelu,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.input_dim = in_channels
        self.output_dim = out_channels

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
        return x