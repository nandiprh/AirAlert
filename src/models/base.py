"""Shared utilities and helpers for all air quality models.

Includes:
- SequenceDataset: PyTorch Dataset over sliding time-series windows.
- compute_metrics: RMSE/MAE/R2 + classification metrics.
- normalize helpers that work on 2D (n_samples, n_features) and
  3D (n_samples, seq_len, n_features) windows.
- deterministic_seed: reproducible training.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# numpy/torch helpers
# ---------------------------------------------------------------------------

def as_numpy(x) -> np.ndarray:
    """Convert torch tensor / list / pandas to numpy float array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def deterministic_seed(seed: int = 42) -> None:
    """Set seeds for numpy, torch and python random for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score_py(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Manual R2 to avoid sklearn count-parameter warning for small sets."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float(np.nan)
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def fit_normalizer(
    X: np.ndarray,
    method: str = "standard",
    feature_range=(0.0, 1.0),
):
    """Fit a scaler on a windowed array.

    Works for both 2D ``(n_samples, n_features)`` and 3D
    ``(n_samples, seq_len, n_features)`` input by collapsing the
    window dimension.

    Returns a fitted sklearn scaler (StandardScaler or MinMaxScaler).
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 3:
        n, seq, feat = arr.shape
        flat = arr.reshape(n * seq, feat)
    else:
        flat = arr
    if method == "minmax":
        return MinMaxScaler(feature_range=feature_range).fit(flat)
    return StandardScaler().fit(flat)


def apply_normalizer(X: np.ndarray, scaler) -> np.ndarray:
    """Transform array using a pre-fitted scaler (2D or 3D)."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 3:
        n, seq, feat = arr.shape
        flat = scaler.transform(arr.reshape(n * seq, feat))
        return flat.reshape(n, seq, feat)
    return scaler.transform(arr)


def inverse_normalize_target(values: np.ndarray, scaler, feature_index: int = 0) -> np.ndarray:
    """Inverse-transform a single target column using the fitted scaler.

    ``values`` may be shape ``(n,)`` or ``(n, 1)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    orig_shape = arr.shape
    arr = arr.reshape(-1, 1)
    col = arr[:, 0]
    # Build a fake row of the scaler width and fill one feature column.
    n_feat = len(scaler.scale_) if hasattr(scaler, "scale_") else len(scaler.data_max_)
    n = col.shape[0]
    feats = np.zeros((n, n_feat), dtype=np.float64)
    feats[:, feature_index] = col
    inv = scaler.inverse_transform(feats)[:, feature_index]
    return inv.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """PyTorch Dataset of sliding time-series windows.

    Args:
        X: shape ``(n_samples, seq_len, n_features)``.
        y_reg: regression targets ``(n_samples,)`` or None.
        y_cls: classification targets ``(n_samples,)`` integer codes or None.
        use_reg: bool, whether to load regression target tensor.
        use_cls: bool, whether to load classification target tensor.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_reg: Optional[np.ndarray] = None,
        y_cls: Optional[np.ndarray] = None,
        use_reg: bool = True,
        use_cls: bool = True,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.use_reg = bool(use_reg and y_reg is not None)
        self.use_cls = bool(use_cls and y_cls is not None)
        self.y_reg = (
            np.asarray(y_reg, dtype=np.float32) if self.use_reg else None
        )
        self.y_cls = (
            np.asarray(y_cls, dtype=np.int64) if self.use_cls else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        if self.use_reg and self.use_cls:
            return (
                x,
                torch.tensor(self.y_reg[idx], dtype=torch.float32),
                torch.tensor(self.y_cls[idx], dtype=torch.long),
            )
        if self.use_reg:
            return x, torch.tensor(self.y_reg[idx], dtype=torch.float32)
        if self.use_cls:
            return x, torch.tensor(self.y_cls[idx], dtype=torch.long)
        return x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "test",
    classification: bool = False,
    labels: Optional[list] = None,
) -> Dict:
    """Compute regression and/or classification metrics.

    Regression: RMSE, MAE, R2.
    Classification (when ``classification=True``): accuracy, macro/weighted
    precision/recall/F1 and confusion matrix.

    ``y_pred`` may be a flat numpy array (regression) or predicted class
    indices (classification).

    Returns a dict with keys prefixed by ``prefix``.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    metrics: Dict = {}

    reg_mask = ~np.isnan(as_numpy(y_true))
    if reg_mask.sum() > 0 and not classification:
        yt = y_true[reg_mask]
        yp = y_pred[reg_mask]
        metrics[f"{prefix}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"{prefix}_mae"] = float(mean_absolute_error(yt, yp))
        metrics[f"{prefix}_r2"] = float(r2_score_py(yt, yp))
        metrics[f"{prefix}_n"] = int(reg_mask.sum())

    if classification:
        valid = (y_true >= 0) & ~np.isnan(y_true) & (y_pred >= 0)
        if valid.sum() > 0:
            yt = y_true[valid].astype(int)
            yp = y_pred[valid].astype(int)
            metrics[f"{prefix}_accuracy"] = float(accuracy_score(yt, yp))
            metrics[f"{prefix}_precision"] = float(
                precision_score(yt, yp, average="weighted", zero_division=0)
            )
            metrics[f"{prefix}_recall"] = float(
                recall_score(yt, yp, average="weighted", zero_division=0)
            )
            metrics[f"{prefix}_f1"] = float(
                f1_score(yt, yp, average="weighted", zero_division=0)
            )
            metrics[f"{prefix}_confusion_matrix"] = (
                confusion_matrix(yt, yp, labels=labels).tolist()
            )
            metrics[f"{prefix}_cls_n"] = int(valid.sum())

    return metrics


def summarize_metrics(metrics: Dict, prefix: str = "test") -> str:
    """Human-readable one-line summary of a metrics dict."""
    parts = []
    for key in ("rmse", "mae", "r2"):
        k = f"{prefix}_{key}"
        if k in metrics:
            parts.append(f"{key.upper()}={metrics[k]:.4f}")
    for key in ("accuracy", "precision", "recall", "f1"):
        k = f"{prefix}_{key}"
        if k in metrics:
            parts.append(f"{key}={metrics[k]:.4f}")
    return "  ".join(parts)