import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Added Intra-Sentence Self-Attention, sitting between the BiLSTM and Cross Attention Modules.

Parikh et al. (2016) "A Decomposable Attention Model for NLI"; Wang & Jiang (2017)

Add a self-attention pass within each sentence before cross-attention, so each token attends to its own sentence first.
"""


class ESIMBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Shared BI-LSTM for Premise & Hypothesis as per paper.
        # Use //2 for hidden dimension - doubles for both directions.
        self.shared_bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Intra-Sentence Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # Multi-Head attention, to replace manual sigle-dot attention.
        # Shared for both directions (hyp & premise)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # FFN: 4*hidden_dim -> hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # LayerNorm per stream
        self.layer_norm_p = nn.LayerNorm(hidden_dim)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    # ── Enhancement ───────────────────────────────────────────────────────────
    @staticmethod
    def _enhance(h, h_att):
        """
        Compute the enhancement vector (equations 11–12 in paper):
            m = [h, h_att, h − h_att, h * h_att]

        Difference captures what is mismatched; product captures similarity.

        Args:
            h     : (batch, len, hidden_dim)   original BiLSTM output
            h_att : (batch, len, hidden_dim)   attended representation

        Returns:
            m : (batch, len, 4 * hidden_dim)
        """
        return torch.cat([h, h_att, h - h_att, h * h_att], dim=-1)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, h_p, h_h, mask_p=None, mask_h=None):
        """
        Args:
            h_p    : (batch, len_p, hidden_dim)
            h_h    : (batch, len_h, hidden_dim)
            mask_p : (batch, len_p)  True = padding
            mask_h : (batch, len_h)  True = padding

        Returns:
            out_p : (batch, len_p, hidden_dim)  refined premise representations
            out_h : (batch, len_h, hidden_dim)  refined hypothesis representations
        """
        # Save inputs for residual connection
        residual_p = h_p
        residual_h = h_h

        # ── 1. BiLSTM encoding ───────────────────────────────────────────────
        enc_p, _ = self.shared_bilstm(self.dropout(h_p))  # (batch, len_p, hidden_dim)
        enc_h, _ = self.shared_bilstm(self.dropout(h_h))  # (batch, len_h, hidden_dim)

        # --- 2. Intra-Sentence Attention -------------------------------------
        enc_p, _ = self.self_attn(
            enc_p, enc_p, enc_p, key_padding_mask=mask_p, need_weights=False
        )
        enc_h, _ = self.self_attn(
            enc_h, enc_h, enc_h, key_padding_mask=mask_h, need_weights=False
        )

        # ── 3. Multi-Head attention ──────────────────────────────────────----
        att_p, _ = self.cross_attn(
            query=enc_p,
            key=enc_h,
            value=enc_h,
            key_padding_mask=mask_h,
            need_weights=False,
        )

        att_h, _ = self.cross_attn(
            query=enc_h,
            key=enc_p,
            value=enc_p,
            key_padding_mask=mask_p,
            need_weights=False,
        )

        # ── 4. Enhancement ───────────────────────────────────────────────────
        m_p = self._enhance(enc_p, att_p)  # (batch, len_p, 4*hidden_dim)
        m_h = self._enhance(enc_h, att_h)  # (batch, len_h, 4*hidden_dim)

        # ── 5. FFN ───────────────────────────────────────────────────────────
        n_p = self.ffn(m_p)  # (batch, len_p, hidden_dim)
        n_h = self.ffn(m_h)  # (batch, len_h, hidden_dim)

        # ── 6. Residual connection + LayerNorm (equation 19 in paper) ────────
        out_p = self.layer_norm_p(residual_p + n_p)
        out_h = self.layer_norm_h(residual_h + n_h)

        return out_p, out_h
