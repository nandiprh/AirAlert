"""Hybrid 3: LSTM + Attention.

An LSTM encodes the input window and produces per-timestep hidden states.
A self-attention (additive / Bahdanau-style) mechanism computes a weight
for every hidden state and aggregates them into a single context vector,
allowing the model to focus on the most informative timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttention(nn.Module):
    """LSTM encoder -> additive self-attention -> FC heads.

    Args:
        input_size: number of features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: number of LSTM layers.
        output_size: regression output dimension.
        num_classes: number of AQI classes or None to skip classification.
        dropout: dropout rate.
        attention_size: dimension of the attention projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        num_classes: int | None = None,
        dropout: float = 0.2,
        attention_size: int = 32,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if hidden_size % 2 != 0:
            # Keep hidden_size odd-friendly by padding the projection below.
            pass
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

        # Bahdanau-style attention: score = W_tanh(W_a * h + b).squeeze
        self.attn_W = nn.Linear(hidden_size, attention_size, bias=False)
        self.attn_v = nn.Linear(attention_size, 1, bias=False)

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
        """Compute softmax attention scores over the time axis.

        Args:
            lstm_out: ``(batch, seq_len, hidden_size)``.

        Returns:
            Attention weights ``(batch, seq_len, 1)`` summing to one.
        """
        energies = torch.tanh(self.attn_W(lstm_out))       # (B, T, A)
        scores = self.attn_v(energies).squeeze(-1)         # (B, T)
        return F.softmax(scores, dim=1).unsqueeze(-1)      # (B, T, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            ``(reg_out, cls_out)``. ``cls_out`` is ``None`` when
            classification is disabled.
        """
        lstm_out, _ = self.lstm(x)                         # (B, T, H)
        attn_w = self.attention_weights(lstm_out)          # (B, T, 1)
        context = torch.sum(lstm_out * attn_w, dim=1)      # (B, H)

        feats = self.head(context)
        reg_out = self.fc_reg(feats)
        cls_out = self.fc_cls(feats) if self.fc_cls is not None else None
        return reg_out, cls_out