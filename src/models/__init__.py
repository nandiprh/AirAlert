"""Model package: 5 hybrid deep learning architectures for AQI prediction.

Registry maps model names (used by ``train.py`` / ``evaluate.py``) to
their ``nn.Module`` classes.
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from .base import SequenceDataset, compute_metrics, deterministic_seed
from .cnn_bilstm import CNNBiLSTM
from .convlstm_attention import ConvLSTMAttention
from .lstm_attention import LSTMAttention
from .lstm_cnn import LSTMCNN
from .transformer_lstm import TransformerLSTM

__all__ = [
    "MODEL_REGISTRY",
    "LSTMCNN",
    "CNNBiLSTM",
    "LSTMAttention",
    "TransformerLSTM",
    "ConvLSTMAttention",
    "SequenceDataset",
    "compute_metrics",
    "deterministic_seed",
]

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "lstm_cnn": LSTMCNN,
    "cnn_bilstm": CNNBiLSTM,
    "lstm_attention": LSTMAttention,
    "transformer_lstm": TransformerLSTM,
    "convlstm_attention": ConvLSTMAttention,
}


def build_model(
    model_name: str,
    input_size: int,
    seq_len: int | None = None,
    output_size: int = 1,
    num_classes: int | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
) -> nn.Module:
    """Instantiate a model from the registry with shared hyperparameters.

    Args:
        model_name: one of ``MODEL_REGISTRY`` keys.
        input_size: number of input features per timestep.
        seq_len: window length (used by TransformerLSTM positional encoding).
        output_size: regression output dimension.
        num_classes: number of AQI bucket classes or None.
        hidden_size: base hidden dimension / d_model.
        num_layers: number of recurrent/Transformer layers.
        dropout: dropout rate.
        seed: RNG seed.

    Returns:
        An ``nn.Module`` whose ``forward(x)`` returns ``(reg, cls)``.
    """
    deterministic_seed(seed)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )

    cls = MODEL_REGISTRY[model_name]
    common = dict(input_size=input_size, output_size=output_size,
                  num_classes=num_classes, dropout=dropout)
    if model_name == "transformer_lstm":
        kwargs = dict(
            d_model=hidden_size,
            nhead=max(1, hidden_size // 16),
            num_layers=num_layers,
            max_len=(seq_len + 64) if seq_len is not None else 1024,
            **common,
        )
    else:
        kwargs = dict(hidden_size=hidden_size, num_layers=num_layers, **common)
    return cls(**kwargs)


if __name__ == "__main__":
    import torch

    for name in MODEL_REGISTRY:
        model = build_model(name, input_size=9, seq_len=24, num_classes=4)
        x = torch.randn(4, 24, 9)
        reg, cls = model(x)
        print(f"{name:22s} reg shape {tuple(reg.shape)} cls shape", end=" ")
        print(tuple(cls.shape) if cls is not None else None)