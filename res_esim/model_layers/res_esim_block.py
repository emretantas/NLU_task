import torch
import torch.nn as nn

from .esim_block import ESIMBlock
from .highway_input_projection import HighwayProjection


class ResESIM(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int,
        input_dim: int,
        dropout_rate: float = 0.2,
        num_heads: int = 4,
        # num_classes         : int = 3
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        # self.num_classes = num_classes

        # --- Input Projection ------------------
        """
        Convert input dimension into internal working dimension.
        """
        self.input_projection = HighwayProjection(input_dim, hidden_dim)

        # --- Input Dropout ---------------------
        self.input_dropout = nn.Dropout(dropout_rate)

        # --- ESIM Blocks -----------------------
        self.esim_blocks = nn.ModuleList(
            [ESIMBlock(hidden_dim, dropout_rate, num_heads) for _ in range(num_blocks)]
        )

    # --- Utilities -----------------------------
    @staticmethod
    def _make_padding_mask(sequence_lengths, max_length):
        """
        Build a boolean mask of shape (batch, max_len) where
        True marks a padding position (index >= actual length).

        Args:
            lengths : (batch,) LongTensor of actual sequence lengths
            max_len : int

        Returns:
            mask : (batch, max_len) BoolTensor
        """
        idx = torch.arange(max_length, device=sequence_lengths.device).unsqueeze(
            0
        )  # (1, max_length)
        mask = idx >= sequence_lengths.unsqueeze(1)  # (batch, max_length)
        return mask

    # --- Forward Pass -------------------------
    def forward(self, premise_embedding, hyp_embedding, premise_length, hyp_length):
        """
        Args:
            premise_embedding       : (batch, len_p, input_dim)
            hypothesis_embedding    : (batch, len_h, input_dim)
            premise_length          : (batch,)  actual (non-padded) lengths for premise
            hypothesis_length       : (batch,)  actual (non-padded) lengths for hypothesis

        Returns:
            h_p    : (batch, len_p, hidden_dim)  refined premise representations
            h_h    : (batch, len_h, hidden_dim)  refined hypothesis representations
            mask_p : (batch, len_p)              padding mask (True = padding)
            mask_h : (batch, len_h)              padding mask (True = padding)
        """

        # --- Generate Padding Masks ----------------------
        mask_p = self._make_padding_mask(
            premise_length, premise_embedding.size(1)
        )  # (batch, len_p)
        mask_h = self._make_padding_mask(
            hyp_length, hyp_embedding.size(1)
        )  # (batch, len_h)

        # --- Input Projection & Dropout ------------------
        h_p = self.input_projection(
            self.input_dropout(premise_embedding)
        )  # (batch, len_p, hidden_dim)
        h_h = self.input_projection(
            self.input_dropout(hyp_embedding)
        )  # (batch, len_h, hidden_dim)

        # --- ESIM Blocks ---------------------------------
        for block in self.esim_blocks:
            h_p, h_h = block(
                h_p, h_h, mask_p, mask_h
            )  # (batch, len_p, hidden_dim), (batch, len_h, hidden_dim)

        # --- Output --------------------------------------
        return h_p, h_h, mask_p, mask_h
