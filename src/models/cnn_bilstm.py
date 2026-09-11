"""Hybrid 2: CNN + BiLSTM.

A 1D CNN first extracts local patterns across the time axis from the raw
pollutant features; the resulting feature map is fed into a bidirectional
LSTM that captures temporal dependencies in both directions; the fused
hidden state is passed to FC heads for regression and classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """1D CNN local features -> BiLSTM bidirectional temporal -> FC heads.

    Args:
        input_size: number of features per timestep.
        hidden_size: hidden dimension of the BiLSTM (split across
            forward/backward directions, so each direction sees
            ``hidden_size // 2`` units).
        num_layers: number of stacked BiLSTM layers.
        output_size: regression output dimension.
        num_classes: number of AQI classes or None to skip classification.
        dropout: dropout rate.
        kernel_size: 1D convolution kernel width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        num_classes: int | None = None,
        dropout: float = 0.2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        assert hidden_size % 2 == 0, "hidden_size must be even for BiLSTM"

        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_size,
                hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc_reg = nn.Linear(hidden_size, output_size)
        self.fc_cls = (
            nn.Linear(hidden_size, num_classes) if num_classes else None
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            ``(reg_out, cls_out)``.
        """
        x_t = x.transpose(1, 2)                     # (B, F, T)
        cnn_out = self.cnn(x_t).transpose(1, 2)     # (B, T, H)

        bilstm_out, (h_n, _) = self.bilstm(cnn_out)  # (B, T, H), (L*2, B, H/2)
        # Concatenate the final forward and backward hidden states.
        h_fwd = h_n[-2]                             # (B, H/2)
        h_bwd = h_n[-1]                             # (B, H/2)
        feats = torch.cat([h_fwd, h_bwd], dim=1)    # (B, H)

        feats = self.head(feats)
        reg_out = self.fc_reg(feats)
        cls_out = self.fc_cls(feats) if self.fc_cls is not None else None
        return reg_out, cls_out