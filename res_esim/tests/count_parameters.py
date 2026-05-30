"""
Prints the total and trainable parameter count for OracleNet
using the default hyperparameters from train.py.

Run from the project root:
    python -m res_esim.tests.count_parameters
"""

import torch
from res_esim.model_layers.oracle_net import OracleNet


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = OracleNet(
        input_dim=1477,
        hidden_dim=512,
        num_blocks=3,
        num_classes=2,
        dropout_rate=0.2,
        num_heads=8,
    )

    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
