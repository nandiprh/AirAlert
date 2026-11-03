"""Learning-curve diagnostics: overfitting vs underfitting.

This module turns a training history (train/validation loss per epoch) into
an explicit, reproducible diagnosis based on the bias-variance trade-off:

* **Underfitting (high bias)** - the model is too simple / too regularised.
  Both curves stay high, the train loss barely falls and the validation
  loss does not improve much either. The gap is small but the absolute
  error is large.
* **Good fit** - train and validation loss fall together and the
  generalisation gap stays small and stable.
* **Overfitting (high variance)** - the train loss keeps falling while the
  validation loss has bottomed out (and usually rises). The generalisation
  gap widens with epoch count.

The diagnosis is consumed by ``src/train.py`` (auto-correction) and by the
benchmarking suite (``benchmarking/benchmark_generalization.py``).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

DEFAULTS = {
    "overfit_budget": 8,
    "underfit_budget": 8,
    "overfit_min_gap": 0.02,
    "overfit_gap_ratio": 0.2,
    "overfit_val_tolerance": 1.02,
    "underfit_train_drop": 0.20,
    "underfit_val_improvement": 0.10,
}


def _cfg_value(cfg: Optional[Mapping], key: str):
    if cfg is None:
        return DEFAULTS[key]
    try:
        value = cfg.get(key, DEFAULTS[key])
    except AttributeError:
        value = getattr(cfg, key, DEFAULTS[key])
    return DEFAULTS[key] if value is None else value


def _curves(history: List[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray([float(h["train_loss"]) for h in history], dtype=np.float64)
    val = np.asarray([float(h["val_loss"]) for h in history], dtype=np.float64)
    return train, val


def diagnose_training(history: List[Mapping], cfg: Optional[Mapping] = None) -> Dict:
    """Diagnose overfitting/underfitting from a train/val loss history.

    Args:
        history: list of dicts with ``epoch``, ``train_loss`` and ``val_loss``.
        cfg: optional config mapping with the thresholds in :data:`DEFAULTS`.

    Returns:
        A JSON-serialisable dictionary with the loss/gap summary, the
        ``diagnosis`` label (``overfit`` / ``underfit`` / ``good_fit`` /
        ``insufficient_evidence``), boolean flags, the triggering signals and
        recommended corrective actions.
    """
    if not history:
        return {
            "diagnosis": "insufficient_evidence",
            "is_overfitting": False,
            "is_underfitting": False,
            "n_epochs": 0,
            "reason": "no training history",
            "signals": {},
            "recommendations": ["train for more epochs before diagnosing"],
        }

    train, val = _curves(history)
    n = len(history)
    best_idx = int(np.argmin(val))
    first_train = float(train[0])
    final_train = float(train[-1])
    final_val = float(val[-1])
    best_val = float(val[best_idx])
    epochs_since_best = int(n - 1 - best_idx)
    gap = float(final_val - final_train)
    gap_ratio = float(gap / max(abs(final_train), 1e-12))
    train_drop = float(first_train - final_train)
    train_drop_frac = float(train_drop / max(abs(first_train), 1e-12))
    val_improvement = float((float(val[0]) - best_val) / max(abs(float(val[0])), 1e-12))
    gap_trend = float((final_val - final_train) - (float(val[0]) - float(train[0])))

    min_gap = float(_cfg_value(cfg, "overfit_min_gap"))
    ratio = float(_cfg_value(cfg, "overfit_gap_ratio"))
    tol = float(_cfg_value(cfg, "overfit_val_tolerance"))
    over_budget = int(_cfg_value(cfg, "overfit_budget"))
    under_budget = int(_cfg_value(cfg, "underfit_budget"))
    under_train_drop = float(_cfg_value(cfg, "underfit_train_drop"))
    under_val_improvement = float(_cfg_value(cfg, "underfit_val_improvement"))

    signals = {
        "first_train_loss": round(first_train, 6),
        "final_train_loss": round(final_train, 6),
        "final_val_loss": round(final_val, 6),
        "best_val_loss": round(best_val, 6),
        "best_epoch": int(history[best_idx].get("epoch", best_idx + 1)),
        "epochs_since_best": epochs_since_best,
        "generalization_gap": round(gap, 6),
        "generalization_gap_ratio": round(gap_ratio, 6),
        "generalization_gap_trend": round(gap_trend, 6),
        "train_drop": round(train_drop, 6),
        "train_drop_fraction": round(train_drop_frac, 6),
        "val_improvement_fraction": round(val_improvement, 6),
    }

    # Not enough epochs to distinguish a real trend from noise.
    if n < 4:
        return {
            "diagnosis": "insufficient_evidence",
            "is_overfitting": False,
            "is_underfitting": False,
            "n_epochs": n,
            "reason": f"only {n} epochs logged (need >= 4)",
            "signals": signals,
            "recommendations": ["train for more epochs before diagnosing"],
        }

    overfit_threshold = max(min_gap, ratio * abs(final_train))
    is_overfitting = bool(
        gap > overfit_threshold
        and final_val >= best_val * tol
        and epochs_since_best >= over_budget
    )

    # High bias: the model cannot even fit the training data (train loss
    # barely drops) and validation does not improve either. A large
    # train->val gap would instead be high variance (overfitting), so that
    # case is excluded by the overfit check above.
    is_underfitting = bool(
        train_drop_frac < under_train_drop
        and val_improvement < under_val_improvement
        and n >= under_budget
    )

    if is_overfitting:
        diagnosis = "overfit"
        reason = (
            f"val loss {final_val:.4f} is {gap:+.4f} above train "
            f"({gap_ratio:.0%}) and has not improved for "
            f"{epochs_since_best} epochs"
        )
        recommendations = [
            "increase dropout / weight decay",
            "enable early stopping at the validation minimum",
            "reduce model capacity (hidden_size / layers)",
            "add more training data or augmentation if available",
        ]
    elif is_underfitting:
        diagnosis = "underfit"
        reason = (
            f"train loss only dropped {train_drop_frac:.0%} and validation "
            f"only improved {val_improvement:.0%}; the model cannot fit the "
            f"training signal (final train {final_train:.4f})"
        )
        recommendations = [
            "increase model capacity (hidden_size / layers)",
            "train longer or raise the learning rate",
            "lower dropout / weight decay",
            "add richer input features",
        ]
    else:
        diagnosis = "good_fit"
        reason = (
            f"train and val losses are moving together "
            f"(gap {gap:+.4f}, val improved {val_improvement:.0%})"
        )
        recommendations = ["keep the current capacity and regularisation"]

    return {
        "diagnosis": diagnosis,
        "is_overfitting": is_overfitting,
        "is_underfitting": is_underfitting,
        "n_epochs": n,
        "reason": reason,
        "signals": signals,
        "recommendations": recommendations,
    }


def summarize_diagnosis(diag: Mapping) -> str:
    """One-line human-readable summary of a diagnosis dict."""
    if not diag:
        return "diagnosis=n/a"
    signals = diag.get("signals", {})
    return (
        f"diagnosis={diag.get('diagnosis', 'n/a')} "
        f"gap={signals.get('generalization_gap', float('nan')):+.4f} "
        f"best_epoch={signals.get('best_epoch', 'n/a')}"
    )
