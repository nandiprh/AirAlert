# SMOTE-R — SMOTE for Regression

A standalone implementation of **SMOTE-R** (SMOTE for Regression) for the
air-quality models in this project. It oversamples the **rare / extreme
regression targets** (severe-pollution windows) while keeping the target
**continuous** — the plain SMOTE pass used for the AQI *buckets* cannot do
that.

| | |
|---|---|
| **Where the code lives** | [`src/data/smote_r.py`](../src/data/smote_r.py) |
| **Reference** | Torgo, L., Ribeiro, R. P., Pfahringer, B., Branco, P. (2013). *SMOTE for Regression*. ECML PKDD 2013, LNCS 7819, 185–197. |
| **Dependencies** | numpy + scikit-learn only (**no** `imbalanced-learn`) |
| **CLI switch** | `--smote_r` (opt-in, off by default) |
| **Applied to** | **training windows only** (never val/test) |

---

## 1. Why a separate algorithm?

The repository already contains two SMOTE-style passes. SMOTE-R is a
**third, independent** pass — it does not modify or replace them:

| Pass | Function | Driven by | Target handling |
|------|----------|-----------|-----------------|
| Classification SMOTE | `apply_smote` (`src/train.py:333`) | AQI **bucket** labels | Interpolated as an extra column |
| Regression SMOTE (`--smote_reg`) | `apply_smote_regression` (`src/train.py:374`) | A **binary** "rare / normal" split of `y_reg` (`y_reg ≥ quantile`) | Interpolated as an extra column |
| **SMOTE-R (`--smote_r`)** | `apply_smote_r` (`src/data/smote_r.py:288`) | A **relevance function** `φ(y) ∈ [0,1]` on the continuous target | **Inverse-distance weighted** between the two seeds |

The two differences that matter:

1. **A relevance function instead of a hard threshold.** SMOTE-R scores
   *how extreme* each target is (`φ = 1` at the tails, `φ = 0` in the
   middle) and puts everything with `φ ≥ t_E` into the rare set `D_r`.
   Ties, gaps and the exact shape of the target distribution are handled by
   the relevance curve, not by a single quantile cut.
2. **The synthetic target is *assigned*, not blindly interpolated.** Plain
   SMOTE has no target at all (a class label is constant inside a class).
   SMOTE-R computes the target of a synthetic point as an
   **inverse-distance-weighted average of the two seed targets**, using the
   distances in the *feature* space:

   ```
   d1 = ‖new − c‖,   d2 = ‖new − x‖
   y_new = (d2·y_c + d1·y_x) / (d1 + d2)
   ```

   The seed that is **closer** to the synthetic point gets the **larger**
   weight, so a synthetic window generated next to a severe-pollution
   window keeps a severe target instead of being dragged toward the middle
   of the distribution.

The pass is **affine-invariant in `y`** (the weights are ratios of
distances), so it works unchanged on the z-scored AQI targets used during
training.

---

## 2. The algorithm

```
input : X (n, d) features, y (n,) continuous target
        k          neighbourhood size            (default 5)
        t_E        relevance threshold           (default 0.5)
        q          tail width of φ               (default 0.1)
        r_over     synthetic cases per rare case (default 1.0)
        r_under    normal cases to drop          (default 0.0)

1. relevance      φ(y_i)  = piecewise-linear curve over the target quantiles
                            (1.0 at the tails → 0.0 in the central band)
2. rare set       D_r = { i : φ(y_i) ≥ t_E }
3. split tails    D_r^high = { i ∈ D_r : y_i ≥ median(D_r) }
                  D_r^low  = { i ∈ D_r : y_i <  median(D_r) }
4. for each group, for each of round(r_over · |group|) synthetic cases:
        c   ← random case of the group
        x   ← random k-NN of c *inside the same group*
        λ   ~ U(0, 1)
        new = c + λ·(x − c)                       ← feature interpolation
        d1  = ‖new − c‖,  d2 = ‖new − x‖
        y_new = (d2·y_c + d1·y_x) / (d1 + d2)      ← inverse-distance target
5. under-sample   drop round(r_under · |D_i|) normal cases (random)
6. output         original cases + synthetic cases
```

### Relevance function

`φ` is defined on the **quantile position** of the target (mid-rank, so ties
share the same score) and interpolated linearly between the control points:

| quantile | 0.00 | q/2 | q | 1−q | 1−q/2 | 1.00 |
|----------|------|-----|-----|-----|-------|------|
| `φ` | 1.00 | 0.50 | 0.00 | 0.00 | 0.50 | 1.00 |

With the defaults (`q = 0.1`, `t_E = 0.5`) the rare set is the **bottom 10 %
and the top 10 %** of the training targets — i.e. the calmest and the most
severe windows. Lowering `t_E` widens the rare set, raising it narrows it to
the very extremes.

### Why the target stays extreme

Because the synthetic point lies on the segment between the two seeds, the
inverse-distance rule reduces to the convex combination

```
y_new = (1 − λ)·y_c + λ·y_x
```

which is exactly the target of the vanilla interpolation — but it is
computed from the *distances*, so it stays correct when the metric is not
the plain Euclidean segment distance, and it never overshoots the two seed
targets. The extreme windows are therefore reproduced *in extremis* instead
of being averaged away.

---

## 3. How to use it

### 3.1 Command line — training

```bash
# SMOTE-R on the UCI dataset (regression head only)
python src/train.py --dataset uci --model lstm_cnn --epochs 50 \
                    --seq_len 24 --task regression --smote_r

# SMOTE-R together with the other two passes (they are independent)
python src/train.py --dataset uci --model lstm_cnn --epochs 50 \
                    --seq_len 24 --smote_r --smote_reg

# India city AQI (multi-task: regression + bucket classification)
python src/train.py --dataset india --city Delhi --model cnn_bilstm \
                    --epochs 50 --seq_len 7 --smote_r
```

Console output of the pass (one line, per model):

```
[smote-r] training windows 571 -> 633 (rare=62, synthetic=62, t_E=0.5, q=0.1)
```

### 3.2 Command line — city-wise forecasting

```bash
python src/predict.py --city Delhi --model lstm_cnn --epochs 30 \
                      --seq_len 7 --smote_r
```

### 3.3 CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--smote_r` | off | Enable the SMOTE-R pass (requires a regression head). |
| `--smote_r_k` | `5` | Neighbourhood size `k` used to pick the seed neighbour. |
| `--smote_r_threshold` | `0.5` | Relevance threshold `t_E` (higher → rarer set). |
| `--smote_r_quantile` | `0.1` | Tail width `q` of the relevance function. |
| `--smote_r_over` | `1.0` | Synthetic cases generated per rare case. |
| `--smote_r_under` | `0.0` | Fraction of *normal* cases to under-sample (0 = keep all). |

### 3.4 Python API

```python
from src.data.smote_r import apply_smote_r, smote_r, relevance_extremes

# window-aware wrapper (what src/train.py uses)
X_res, y_res, y_cls_res = apply_smote_r(
    X_train,            # (n, seq_len, n_features) training windows
    y_reg_train,        # (n,) continuous targets
    seq_len=24,
    n_features=13,
    y_cls_train=y_cls_train,   # optional -> labels copied from the parent case
    k_neighbors=5,
    threshold=0.5,
    rare_quantile=0.1,
    over_sampling_ratio=1.0,
    random_state=42,
)

# flat (2D) variant, returns the full result object
res = smote_r(X_flat, y, random_state=42)
res.X, res.y            # resampled arrays
res.is_synthetic        # boolean mask of the synthetic rows
res.parent              # parent (seed) index of every synthetic row
res.stats               # counters: n_rare, n_synthetic, thresholds, ...

# just the relevance curve
phi = relevance_extremes(y, rare_quantile=0.1)
```

---

## 4. Where it sits in the pipeline

```
Temporal split (80:20, before windowing)
        ↓
Windows + z-score (fitted on train only)
        ↓
apply_smote()          ← classification buckets        (default ON)
        ↓
apply_smote_regression()← rare-target binary SMOTE     (--smote_reg)
        ↓
apply_smote_r()        ← SMOTE-R, relevance-weighted   (--smote_r)
        ↓
Training → early stopping → auto overfit/underfit correction
```

* Each pass runs **only on the training windows** — validation and test
  windows are never resampled, so no synthetic sample can leak into the
  evaluation set.
* The passes are **independent**: enabling `--smote_r` does not change
  `apply_smote` or `apply_smote_regression`, and any combination can be
  used.
* Classification labels of the synthetic windows are copied from the
  **parent** case (the window the synthetic one was generated from), so the
  three arrays stay aligned for the multi-task head.

---

## 5. Tuning guide

| Symptom | Knob | Effect |
|---------|------|--------|
| Too few synthetic samples | `--smote_r_over 2.0` | Two synthetic cases per rare case. |
| Rare set too small / too large | `--smote_r_threshold` | ↑ → only the very extremes; ↓ → wider tails. |
| Noisy neighbourhoods | `--smote_r_k 3` | Smaller `k` → seeds stay local (safer for time series). |
| Imbalance persists | `--smote_r_quantile 0.2` | Wider tails → more rare cases (≈40 % of the sample). |
| Class balance still poor | `--smote_r_under 0.3` | Drops 30 % of the normal cases. |

**Caveats**

* `k` too large mixes unrelated windows; for temporal data prefer `k ≤ 5`.
* Under-sampling (`--smote_r_under`) removes training windows at random,
  which thins the *temporal* density of the training set. Keep it small
  (≤ 0.3) and always compare against a run with `= 0.0`.
* SMOTE-R can only rebalance what is *rare*: if the test distribution has no
  extreme days, the pass adds noise without any benefit — compare the test
  RMSE with and without `--smote_r`.

---

## 6. Reproducibility

Every random draw (seed case, neighbour choice, `λ`) comes from a single
`numpy.random.default_rng(random_state)` seeded with the run seed
(`--seed`, default `42`). Two runs with the same seed, the same windows and
the same parameters produce **identical** synthetic samples.

---

## 7. Mathematical summary

```
φ(y)          piecewise-linear relevance on the target quantiles
D_r           = { i : φ(y_i) ≥ t_E }
new           = c + λ(x − c),            λ ~ U(0,1)
y_new         = (d2·y_c + d1·y_x)/(d1+d2),  d1 = ‖new−c‖, d2 = ‖new−x‖
```

See [`docs/MATHEMATICAL_MODELING.md`](MATHEMATICAL_MODELING.md) §3.2 for the
formal derivation and [`docs/MODEL_MATH_AND_RUN.md`](MODEL_MATH_AND_RUN.md)
§2.3.2 for the code walkthrough.
