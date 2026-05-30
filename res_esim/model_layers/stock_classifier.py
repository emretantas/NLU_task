"""
classifier.py
-------------
Aggregation and classification module for res-ESIM.

Implements Sections II-B-5 and II-B-6 of:
  Li et al. (2019) "Residual Connected Enhanced Sequential Inference Model
  for Natural Language Inference." IEEE.

Sits downstream of ResESIM, which outputs refined token-level representations.
This module aggregates those representations into sentence-level vectors and
produces class logits.

Pipeline (per the paper):
  h_p, h_h  (from ResESIM)
      │
  Aggregation BiLSTM           ← eq. 20–21: vp = BiLSTM(h_p), vh = BiLSTM(h_h)
      │
      |
  Attention-Weighted Pooling: Bahdanau et al. (2015); dos Santos et al. (2016) "Attentive Pooling Networks"
      |
      |
  Masked Max Pooling            ← eq. 22:   vp_max = max(vp),  vh_max = max(vh)
      │
  Concatenation                 ← eq. 23:   v = [vp_max; vh_max; vp_max−vh_max; vp_max*vh_max]
      │
  FFN: ReLU(vW4 + b4)W5 + b5   ← eq. 24
      │
  Softmax → class probabilities
"""

import torch
import torch.nn as nn


class StockClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_classes: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Aggregation BILSTM (Independent for Premise & Hypothesis) ---------
        self._bilstm_premise = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self._bilstm_hyp = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Classification FFN ------------------------------------------------
        self.dropout = nn.Dropout(dropout_rate)

        self.W4_linear = nn.Linear(8 * hidden_dim, hidden_dim)
        self.W5_linear = nn.Linear(hidden_dim, n_classes)

        # --- Attention-Weighted Pooling ----------------------------------------
        self.attn_premise = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_hyp = nn.Linear(hidden_dim, 1, bias=False)

    # --- Utilities ---------------------------------------------------------------

    @staticmethod
    def _masked_attn_pool(tensor, mask, attn_layer):
        scores = attn_layer(tensor).squeeze(-1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * tensor).sum(dim=1)

    @staticmethod
    def _masked_max_pool(tensor, mask):
        """
        Max Pool over sequence dimension. Implements eq. 22 in the paper.
        """
        tensor = tensor.masked_fill(mask.unsqueeze(-1), -1e9)
        return tensor.max(dim=1).values

    @staticmethod
    def _masked_mean_pool(tensor, mask):
        """
        Mean Pool over sequence dimension, ignoring padding positions.

        Args:
            tensor : (batch, seq_len, hidden_dim)
            mask   : (batch, seq_len)  True = padding

        Returns:
            (batch, hidden_dim)
        """
        # Zero out padding positions before summing
        tensor = tensor.masked_fill(mask.unsqueeze(-1), 0.0)
        # Count actual (non-padding) tokens per sequence, clamp to avoid /0
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1).float()  # (batch, 1)
        return tensor.sum(dim=1) / lengths  # (batch, hidden_dim)

    # --- Forward Pass -----------------------------------------------------------
    def forward(self, h_p, h_h, mask_p, mask_h):
        # Returns: LOGITS: (batch, n_classes)

        # --- Aggregation BILSTM ------------------------------------------------
        v_p, _ = self._bilstm_premise(h_p)  # (batch, seq_len, hidden_dim)
        v_h, _ = self._bilstm_hyp(h_h)  # (batch, seq_len, hidden_dim)

        # --- Attention Pooling -------------------------------------------------
        v_p_attn = self._masked_attn_pool(v_p, mask_p, self.attn_premise)
        v_h_attn = self._masked_attn_pool(v_h, mask_h, self.attn_hyp)

        # --- Masked Max & Mean Pooling -----------------------------------------
        v_p_max = self._masked_max_pool(v_p, mask_p)  # (batch, hidden_dim)
        v_h_max = self._masked_max_pool(v_h, mask_h)  # (batch, hidden_dim)
        v_p_mean = self._masked_mean_pool(v_p, mask_p)  # (batch, hidden_dim)
        v_h_mean = self._masked_mean_pool(v_h, mask_h)  # (batch, hidden_dim)

        # --- Concatenation -----------------------------------------------------
        # v = [vp_max; vp_mean; vh_max; vh_mean; vp_max − vh_max; vp_max ∗ vh_max]
        # Max captures the most salient features; mean captures the overall context.
        # Difference and product capture alignment between the two sentences.
        v = torch.cat(
            [
                v_p_max,
                v_p_mean,
                v_p_attn,
                v_h_max,
                v_h_mean,
                v_h_attn,
                v_p_max - v_h_max,
                v_p_max * v_h_max,
            ],
            dim=-1,
        )  # (batch, 6*hidden_dim)

        # --- FFN (equation 24) ------------------------------------------------
        # ypred = softmax(ReLU (vW4 + b4)W5) + b5)
        # == softmax(W5[ReLU(W4[v])])
        # Note: The paper's equation 24 omits the final softmax, but in PyTorch we typically return raw logits and apply softmax in the loss function (e.g., CrossEntropyLoss).
        y_pred = self.W5_linear(torch.relu(self.W4_linear(self.dropout(v))))

        # DEBUG: Check final logits
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print(f"[DEBUG] NaN/inf in final logits!")
            print(
                f"        NaN: {torch.isnan(y_pred).any()}, inf: {torch.isinf(y_pred).any()}"
            )

        return y_pred
