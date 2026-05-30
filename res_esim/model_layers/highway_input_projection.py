"""
Srivastava et al. (2015) "Training Very Deep Networks"
Replace the linear input projection with a gated highway network — useful given the heterogeneous 1477-d input.
"""

import torch
import torch.nn as nn


class HighwayProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.W_H = nn.Linear(input_dim, output_dim)  # Transform
        self.W_T = nn.Linear(input_dim, output_dim)  # Gate
        self.W_C = nn.Linear(input_dim, output_dim)  # Carry

    def forward(self, x):
        T = torch.sigmoid(self.W_T(x))
        H = torch.relu(self.W_H(x))
        C = self.W_C(x)

        return T * H + (1 - T) * C
