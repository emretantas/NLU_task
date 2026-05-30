"""
OracleNet

Full Stack structure, combining both encoder and classifier into one unified model architecture.
"""

import torch
import torch.nn as nn

from .res_esim_block import ResESIM
from .stock_classifier import StockClassifier


class OracleNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_classes: int,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.encoder = ResESIM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
        )

        self.classifier = StockClassifier(
            hidden_dim=hidden_dim,
            n_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    # --- Unified Forward Pass -----------------------------
    def forward(self, premise_embedding, hyp_embedding, premise_length, hyp_length):
        # Returns Classifier Logits

        h_p, h_h, mask_p, mask_h = self.encoder(
            premise_embedding, hyp_embedding, premise_length, hyp_length
        )

        return self.classifier(h_p, h_h, mask_p, mask_h)
