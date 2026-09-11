"""Hybrid 5: ConvLSTM + Attention.

Stacked 1D convolutions extract multi-scale local patterns from the raw
input window, an LSTM then encodes the resulting sequence, and a scaled
dot-product temporal attention layer learns to weight each timestep to
form a context vector fed to the FC heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMAttention(nn.Module):
    """Conv1D local features -> LSTM -> temporal attention -> FC heads.

    Args:
        input_size: number of features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: number of LSTM layers.
        conv_channels: channel count for the 1D conv stack.
        kernel_size: 1D convolution kernel width.
        output_size: regression output dimension.
        num_classes: number of AQI bucket classes or None.
        dropout: dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        conv_channels: int = 32,
        kernel_size: int = 3,
        output_size: int = 1,
        num_classes: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv1d(
                input_size,
                conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                conv_channels,
                conv_channels * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                conv_channels * 2,
                hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Temporal self-attention via scaled dot product.
        self.attn_q = nn.Linear(hidden_size, hidden_size)
        self.attn_k = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc_reg = nn.Linear(hidden_size, output_size)
        self.fc_cls = (
            nn.Linear(hidden_size, num_classes) if num_classes else None
        )

    def attention_weights(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product self-attention over the time axis.

        Args:
            lstm_out: ``(batch, seq_len, hidden_size)``.

        Returns:
            Attention weights ``(batch, seq_len)`` normalized over time.
        """
        q = self.attn_q(lstm_out)                      # (B, T, H)
        k = self.attn_k(lstm_out)                      # (B, T, H)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        return F.softmax(scores, dim=-1)               # (B, T, T)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            ``(reg_out, cls_out)``.
        """
        x_t = x.transpose(1, 2)                        # (B, F, T)
        conv_out = self.conv(x_t).transpose(1, 2)      # (B, T, H)

        lstm_out, _ = self.lstm(conv_out)              # (B, T, H)

        attn_w = self.attention_weights(lstm_out)      # (B, T, T)
        context = torch.matmul(attn_w, lstm_out)       # (B, T, H)
        pooled, _ = context.max(dim=1)                 # self-attention pooling

        feats = self.head(pooled)
        reg_out = self.fc_reg(feats)
        cls_out = self.fc_cls(feats) if self.fc_cls is not None else None
        return reg_out, cls_out