"""Hybrid 4: Transformer + LSTM.

A Transformer encoder (with sinusoidal positional encoding) attends to
global dependencies across the entire input window. The resulting
sequence is then fed to an LSTM for sequential refinement before
being mapped to regression and classification heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (fixed, not learnable)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                     # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)                                  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)."""
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerLSTM(nn.Module):
    """Transformer encoder (global attention) -> LSTM (sequential) -> FC heads.

    Args:
        input_size: number of features per timestep.
        d_model: embedding dimension used by the Transformer encoder.
            The input features are projected to this dimension. Must be
            divisible by ``nhead``.
        nhead: number of attention heads.
        num_layers: number of stacked Transformer encoder layers.
        lstm_hidden: hidden size of the follow-up LSTM. Defaults to
            ``d_model`` if not specified.
        lstm_layers: number of LSTM layers.
        output_size: regression output dimension.
        num_classes: number of AQI bucket classes or None.
        dropout: global dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lstm_hidden: int | None = None,
        lstm_layers: int = 1,
        output_size: int = 1,
        num_classes: int | None = None,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes
        lstm_hidden = lstm_hidden or d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_reg = nn.Linear(lstm_hidden, output_size)
        self.fc_cls = (
            nn.Linear(lstm_hidden, num_classes) if num_classes else None
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            ``(reg_out, cls_out)``.
        """
        x = self.input_proj(x)                          # (B, T, D)
        x = self.pos_enc(x)
        x = self.encoder(x)                             # (B, T, D)
        lstm_out, (h_n, _) = self.lstm(x)               # (B, T, L)
        feats = h_n[-1]                                 # (B, L)

        feats = self.head(feats)
        reg_out = self.fc_reg(feats)
        cls_out = self.fc_cls(feats) if self.fc_cls is not None else None
        return reg_out, cls_out