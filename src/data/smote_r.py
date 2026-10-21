"""SMOTE-R — SMOTE for regression targets (Torgo et al., 2013).

A standalone, dependency-light implementation of the SMOTE-R algorithm
described in *"SMOTE for Regression"* (L. Torgo, R. P. Ribeiro, B. Pfahringer,
P. Branco, ECML PKDD 2013). It is written from scratch here (numpy +
scikit-learn nearest neighbours only, **no** ``imbalanced-learn``), so it is
completely independent of the two SMOTE passes that live in ``src/train.py``:

* ``apply_smote`` (classification) — balances the AQI *buckets*;
* ``apply_smote_regression`` (opt-in ``--smote_reg``) — binarises the
  regression target into "rare / normal" and oversamples the rare half.

SMOTE-R instead keeps the target **continuous** and adds the piece the other
two lack: a *relevance function* on the target and a **distance-weighted
target assignment** for every synthetic sample.

Algorithm (one synthetic case per draw)
--------------------------------------
1. **Relevance.** A relevance function ``phi(y) in [0, 1]`` scores how
   *interesting* (extreme) a target value is. ``D_r = {i : phi(y_i) >= t_E}``
   is the rare set (``t_E`` = relevance threshold).
2. **Split.** ``D_r`` is split into an extreme-high and an extreme-low group
   (around the median of the rare targets) so the spatial coherence of each
   tail is preserved.
3. **Interpolate.** For a case ``c`` drawn from a group and a randomly chosen
   k-NN ``x`` of the *same* group::

       new = c + lambda * (x - c),          lambda ~ U(0, 1)

4. **Assign the target.** The synthetic target is the inverse-distance
   weighted average of the two seed targets, with the distances measured in
   the *feature* space::

       d1 = ||new - c||,   d2 = ||new - x||
       y_new = (d2 * y_c + d1 * y_x) / (d1 + d2)

   The closer seed gets the larger weight, so a synthetic point that sits
   near an extreme window keeps an extreme target instead of being dragged
   towards the distribution centre.
5. **Optionally under-sample** the normal set ``D_i``.

Everything is affine-invariant in ``y`` (the weights are ratios of distances),
so SMOTE-R works unchanged on the z-scored AQI targets used during training.

Reference
---------
Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).
*SMOTE for Regression*. European Conference on Machine Learning and
Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2013),
Springer, LNCS 7819, 185-197.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "relevance_extremes",
    "smote_r",
    "apply_smote_r",
    "SmoteRResult",
]


# ---------------------------------------------------------------------------
# Relevance function
# ---------------------------------------------------------------------------

def _midrank_quantiles(y: np.ndarray) -> np.ndarray:
    """Empirical quantile position of every value (ties share the mid-rank).

    Returns values in ``[0, 1]``: ``p_i`` is the fraction of the sample that
    lies at or below ``y_i`` (ties averaged).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if n == 0:
        return y
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_y = y[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_y[j + 1] == sorted_y[i]:
            j += 1
        mid_rank = 0.5 * (i + j) + 1.0        # 1-based average rank of the tie block
        ranks[order[i:j + 1]] = mid_rank
        i = j + 1
    return (ranks - 0.5) / n


def relevance_extremes(y: np.ndarray, rare_quantile: float = 0.1) -> np.ndarray:
    """Piecewise-linear "extremes" relevance function on a continuous target.

    The relevance is 1.0 at the two tails of the target distribution, 0.5 at
    ``rare_quantile`` from each end and 0.0 in the central band, so the rare
    set grows/shrinks smoothly with the relevance threshold ``t_E``:

    ==========  ==========
    quantile    relevance
    ==========  ==========
    0.00        1.00
    q/2         0.50
    q           0.00
    1 - q       0.00
    1 - q/2     0.50
    1.00        1.00
    ==========  ==========

    With the default ``q = 0.1`` and the default threshold ``t_E = 0.5`` the
    rare set is the bottom 10% **and** the top 10% of the targets.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return y
    q = float(np.clip(rare_quantile, 0.01, 0.25))
    ctrl_q = np.array([0.0, q / 2.0, q, 1.0 - q, 1.0 - q / 2.0, 1.0])
    ctrl_r = np.array([1.0, 0.5, 0.0, 0.0, 0.5, 1.0])
    p = _midrank_quantiles(y)
    return np.interp(p, ctrl_q, ctrl_r)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

@dataclass
class SmoteRResult:
    """Outcome of one SMOTE-R run.

    Attributes:
        X: resampled feature matrix (originals first, synthetic rows last).
        y: resampled continuous target.
        parent: index (into the original arrays) of the case each synthetic
            row was generated from; ``-1`` for original rows.
        is_synthetic: boolean mask, ``True`` for the synthetic rows.
        stats: summary counters (rare/normal sizes, thresholds, ...).
    """

    X: np.ndarray
    y: np.ndarray
    parent: np.ndarray
    is_synthetic: np.ndarray
    stats: Dict = field(default_factory=dict)


def smote_r(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    threshold: float = 0.5,
    rare_quantile: float = 0.1,
    over_sampling_ratio: float = 1.0,
    under_sampling_ratio: float = 0.0,
    random_state: int = 42,
) -> SmoteRResult:
    """Generate SMOTE-R synthetic samples for a continuous target.

    Args:
        X: feature matrix ``(n_samples, n_features)``.
        y: continuous target ``(n_samples,)``.
        k_neighbors: size of the neighbourhood the seed neighbour is drawn
            from (clamped to the group size).
        threshold: relevance threshold ``t_E``; the rare set is
            ``{i : phi(y_i) >= t_E}``.
        rare_quantile: tail width ``q`` of the relevance function.
        over_sampling_ratio: number of synthetic cases per rare case
            (``1.0`` doubles the rare set).
        under_sampling_ratio: fraction of the *normal* cases to drop
            (``0.0`` keeps all of them).
        random_state: seed for every random draw.

    Returns:
        A :class:`SmoteRResult` with the resampled arrays and statistics.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    n = len(X)
    stats: Dict = {
        "n_input": n,
        "k_neighbors": int(k_neighbors),
        "threshold": float(threshold),
        "rare_quantile": float(rare_quantile),
        "over_sampling_ratio": float(over_sampling_ratio),
        "under_sampling_ratio": float(under_sampling_ratio),
        "seed": int(random_state),
    }
    if n < 3:
        stats["skipped"] = "not enough samples"
        return SmoteRResult(X.copy(), y.copy(), np.full(n, -1, dtype=np.int64),
                            np.zeros(n, dtype=bool), stats)

    rng = np.random.default_rng(random_state)

    # --- 1. relevance + rare set ------------------------------------------
    phi = relevance_extremes(y, rare_quantile)
    rare_idx = np.flatnonzero(phi >= float(threshold))
    stats["n_rare"] = int(rare_idx.size)
    stats["n_normal"] = int(n - rare_idx.size)
    stats["y_min_rare"] = float(y[rare_idx].min()) if rare_idx.size else float("nan")
    stats["y_max_rare"] = float(y[rare_idx].max()) if rare_idx.size else float("nan")

    if rare_idx.size < 2:
        stats["skipped"] = "rare set too small"
        return SmoteRResult(X.copy(), y.copy(), np.full(n, -1, dtype=np.int64),
                            np.zeros(n, dtype=bool), stats)

    # --- 2. split the rare set into the two extreme tails -----------------
    split = float(np.median(y[rare_idx]))
    groups = {
        "high": rare_idx[y[rare_idx] >= split],
        "low": rare_idx[y[rare_idx] < split],
    }
    groups = {name: idx for name, idx in groups.items() if idx.size >= 2}
    stats["groups"] = {name: int(idx.size) for name, idx in groups.items()}
    if not groups:
        stats["skipped"] = "no group with at least 2 rare cases"
        return SmoteRResult(X.copy(), y.copy(), np.full(n, -1, dtype=np.int64),
                            np.zeros(n, dtype=bool), stats)

    # --- 3./4. interpolate + inverse-distance target assignment ----------
    X_new: list = []
    y_new: list = []
    parent: list = []
    for name, idx in groups.items():
        Xg = X[idx]
        yg = y[idx]
        k = int(np.clip(int(k_neighbors), 1, len(idx) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xg)
        _, nbrs = nn.kneighbors(Xg)          # column 0 is the point itself
        n_new = int(round(float(over_sampling_ratio) * len(idx)))
        if n_new <= 0:
            continue
        sources = rng.integers(0, len(idx), size=n_new)
        for s in sources:
            cand = nbrs[s, 1:]                # k neighbours, self excluded
            j = int(cand[rng.integers(0, len(cand))])
            lam = float(rng.random())
            new = Xg[s] + lam * (Xg[j] - Xg[s])
            d1 = float(np.linalg.norm(new - Xg[s]))
            d2 = float(np.linalg.norm(new - Xg[j]))
            denom = d1 + d2
            y_syn = float((d2 * yg[s] + d1 * yg[j]) / denom) if denom > 0 else float(yg[s])
            X_new.append(new)
            y_new.append(y_syn)
            parent.append(int(idx[s]))

    # --- 5. optional under-sampling of the normal set ---------------------
    # The rare cases are always kept (they are the seeds of the synthetic
    # ones); only normal cases may be dropped.
    keep = np.arange(n)
    normal = np.setdiff1d(keep, rare_idx)
    n_drop = int(round(float(under_sampling_ratio) * normal.size))
    if n_drop > 0 and normal.size > 1:
        drop = rng.choice(normal, size=min(n_drop, normal.size - 1), replace=False)
        keep = np.setdiff1d(keep, drop)
    stats["n_kept_normal"] = int(np.setdiff1d(keep, rare_idx).size)

    X_res = np.vstack([X[keep], np.asarray(X_new, dtype=np.float64)]) if X_new else X[keep]
    y_res = np.concatenate([y[keep], np.asarray(y_new, dtype=np.float64)]) if y_new else y[keep]
    parent_res = np.concatenate(
        [np.full(keep.size, -1, dtype=np.int64), np.asarray(parent, dtype=np.int64)]
    ) if parent else np.full(len(X_res), -1, dtype=np.int64)
    is_syn = np.zeros(len(X_res), dtype=bool)
    is_syn[len(keep):] = True

    stats["n_synthetic"] = int(len(X_new))
    stats["n_output"] = int(len(X_res))
    stats["skipped"] = None
    return SmoteRResult(X_res, y_res, parent_res, is_syn, stats)


# ---------------------------------------------------------------------------
# Window-aware wrapper (used by src/train.py)
# ---------------------------------------------------------------------------

def apply_smote_r(
    X_train: np.ndarray,
    y_reg_train: np.ndarray,
    seq_len: int,
    n_features: int,
    y_cls_train: Optional[np.ndarray] = None,
    k_neighbors: int = 5,
    threshold: float = 0.5,
    rare_quantile: float = 0.1,
    over_sampling_ratio: float = 1.0,
    under_sampling_ratio: float = 0.0,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply SMOTE-R to a stack of training windows.

    The windows are flattened to 2D for the neighbourhood search and reshaped
    back afterwards. Classification labels (when given) are propagated to the
    synthetic windows by copying the label of the **parent** case (the window
    the synthetic one was generated from), which keeps ``X``/``y_reg``/``y_cls``
    aligned.

    Args:
        X_train: training windows ``(n_samples, seq_len, n_features)``.
        y_reg_train: continuous targets ``(n_samples,)``.
        seq_len: window length (for the reshape back).
        n_features: features per timestep (for the reshape back).
        y_cls_train: optional integer class labels ``(n_samples,)``.
        k_neighbors: neighbourhood size used to pick the seed neighbour.
        threshold: relevance threshold ``t_E``.
        rare_quantile: tail width ``q`` of the relevance function.
        over_sampling_ratio: synthetic cases generated per rare case.
        under_sampling_ratio: fraction of normal (non-rare) cases dropped.
        random_state: seed for every random draw.
        verbose: print a one-line summary.

    Returns:
        ``(X_res, y_reg_res, y_cls_res)``; ``y_cls_res`` is ``None`` when
        ``y_cls_train`` is ``None``.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_reg_train = np.asarray(y_reg_train, dtype=np.float64).reshape(-1)
    if len(X_train) != len(y_reg_train):
        raise ValueError(
            f"X_train and y_reg_train length mismatch: {len(X_train)} vs {len(y_reg_train)}"
        )

    flat = X_train.reshape(len(X_train), -1)
    result = smote_r(
        flat,
        y_reg_train,
        k_neighbors=k_neighbors,
        threshold=threshold,
        rare_quantile=rare_quantile,
        over_sampling_ratio=over_sampling_ratio,
        under_sampling_ratio=under_sampling_ratio,
        random_state=random_state,
    )

    X_res = result.X.reshape(-1, seq_len, n_features).astype(np.float32)
    y_reg_res = result.y.astype(np.float32)

    y_cls_res = None
    if y_cls_train is not None:
        y_cls_train = np.asarray(y_cls_train).reshape(-1)
        parents = result.parent
        synth = result.is_synthetic
        if synth.any() and (parents[synth] >= 0).all():
            y_cls_res = np.concatenate([y_cls_train, y_cls_train[parents[synth]]])
        else:
            y_cls_res = y_cls_train

    if verbose:
        s = result.stats
        if s.get("skipped"):
            print(f"[smote-r] skipped ({s['skipped']}); training windows unchanged")
        else:
            print(
                f"[smote-r] training windows {s['n_input']} -> {s['n_output']} "
                f"(rare={s['n_rare']}, synthetic={s['n_synthetic']}, "
                f"t_E={s['threshold']}, q={s['rare_quantile']})"
            )
    return X_res, y_reg_res, y_cls_res
