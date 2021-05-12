import torch
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=torch.rrelu,
                k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = nn.Linear
        self.input_dim = in_channels
        self.output_dim = out_channels

        assert k >= 2
        self.k = k
        self.ae = [self.base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.ae.append(self.base_model(2 * out_channels, 2 * out_channels))
        self.ae.append(self.base_model(2 * out_channels, out_channels))
        self.ae = nn.ModuleList(self.ae)

        self.activation = activation

    def forward(self, x: torch.Tensor):
        for i in range(self.k):
            x = self.ae[i](x)
            x = self.activation(x)
        return x