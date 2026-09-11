"""Hybrid 1: LSTM + CNN.

LSTM extracts sequential/temporal features from the input window, the
per-timestep LSTM hidden states are then passed through a 1D CNN that
captures local patterns, and a fully-connected head produces both
regression (AQI forecast) and classification (AQI bucket) outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCNN(nn.Module):
    """LSTM temporal feature extractor -> 1D CNN local patterns -> FC heads.

    Args:
        input_size: number of features per timestep (pollutants/weather).
        hidden_size: LSTM hidden dimension.
        num_layers: number of stacked LSTM layers.
        output_size: regression output dimension (default 1 -> AQI).
        num_classes: number of AQI bucket classes. ``None`` disables the
            classification head.
        dropout: dropout applied inside LSTM and before the heads.
        kernel_size: 1D convolution kernel width over the time axis.
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

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                hidden_size * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size * 2,
                hidden_size * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )

        fused_dim = hidden_size * 2
        self.pool = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_size),
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
            ``(reg_out, cls_out)``. ``cls_out`` is ``None`` when the
            classification head is disabled.
        """
        lstm_out, _ = self.lstm(x)                  # (B, T, H)
        cnn_in = lstm_out.transpose(1, 2)           # (B, H, T)
        cnn_feat = self.cnn(cnn_in)                 # (B, 2H, 1)
        cnn_feat = cnn_feat.flatten(1)              # (B, 2H)
        feats = self.pool(cnn_feat)                 # (B, H)

        reg_out = self.fc_reg(feats)
        cls_out = self.fc_cls(feats) if self.fc_cls is not None else None
        return reg_out, cls_out