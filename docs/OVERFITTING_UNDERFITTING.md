# Overfitting and Underfitting

The models in this project are trained with a **leakage-free temporal split**:
the training partition is used to fit weights and scalers, a validation slice
carved from the train tail drives early stopping, and the untouched test
partition is only used for the final report. That setup makes the train →
validation → test relationship an explicit check of **generalisation**, not
just training accuracy.

This document explains the concepts as implemented in
`src/diagnostics.py` and `src/train.py`, shows the rules used by the automatic
correction loop, and reports the dedicated benchmark in
[`benchmarking/overfitting_underfitting.md`](../benchmarking/overfitting_underfitting.md).

---

## 1. The bias-variance trade-off

A model's expected test error can be decomposed into:

```
E[error] = bias² + variance + irreducible noise
```

| Regime | What you see | Cause | Remedy |
|--------|--------------|-------|--------|
| **Underfitting** (high bias) | Train loss stays high; validation loss also stays high and barely improves; both curves are close. The model cannot even fit the training signal. | Model too small, too much dropout/weight decay, too few epochs, weak features. | Increase `hidden_size`/`num_layers`, train longer, lower dropout, add features. |
| **Good fit** | Train and validation loss fall together; the gap is small and stable; validation error plateaus near its minimum. | Capacity and regularisation match the data. | Keep the configuration; monitor the curves. |
| **Overfitting** (high variance) | Train loss keeps falling while validation loss bottoms out and rises; the train→validation gap widens with epochs. | Too much capacity, too little regularisation, too little data, training too long. | Raise dropout/weight decay, early stop, reduce capacity, add data/augmentation. |

The key signal is the **generalisation gap**:

```
gap(epoch) = val_loss(epoch) - train_loss(epoch)
```

Plotting `train_loss` and `val_loss` on the same axes (saved for every run as
`models/reports/<run>_history.png`) makes the regime visible.

---

## 2. How the project detects it

`src.diagnostics.diagnose_training(history, cfg)` turns the loss history into a
JSON-serialisable diagnosis. Defaults (`src/diagnostics.py`):

```python
DEFAULTS = {
    "overfit_budget": 8,          # epochs past the best val epoch
    "underfit_budget": 8,         # minimum epochs before diagnosing
    "overfit_min_gap": 0.02,      # absolute minimum train→val gap
    "overfit_gap_ratio": 0.2,     # gap must exceed 20% of final train loss
    "overfit_val_tolerance": 1.02,# val must be >= 102% of its best value
    "underfit_train_drop": 0.20,  # train loss must drop < 20%
    "underfit_val_improvement": 0.10, # val loss must improve < 10%
}
```

Let `ℓ_t(e)` and `ℓ_v(e)` be the train/validation losses at epoch `e`, `e*` the
epoch of minimum validation loss, and `n` the number of epochs.

**Overfitting** is flagged when all three hold:

```
ℓ_v(n-1) - ℓ_t(n-1) > max(overfit_min_gap, overfit_gap_ratio · |ℓ_t(n-1)|)
ℓ_v(n-1) >= overfit_val_tolerance · min_e ℓ_v(e)
(n - 1) - e* >= overfit_budget
```

In words: the validation loss is materially above the train loss, it has moved
away from its best value, and that regression has persisted for several epochs.

**Underfitting** is flagged when the model never made much progress:

```
(ℓ_t(0) - ℓ_t(n-1)) / |ℓ_t(0)| < underfit_train_drop
(ℓ_v(0) - min_e ℓ_v(e)) / |ℓ_v(0)| < underfit_val_improvement
n >= underfit_budget
```

In words: the train loss dropped less than 20% **and** validation improved less
than 10% — the model is not fitting the training signal. A large train→val gap
is deliberately *not* underfitting; it is handled by the overfitting rule.

Otherwise the diagnosis is `good_fit`. Histories shorter than 4 epochs return
`insufficient_evidence` (too little signal to distinguish a trend from noise).

The function also returns the full evidence:

- `first_train_loss`, `final_train_loss`, `final_val_loss`;
- `best_val_loss` and `best_epoch`;
- `epochs_since_best`;
- `generalization_gap` and `generalization_gap_ratio`;
- `generalization_gap_trend`;
- `train_drop`, `train_drop_fraction`, `val_improvement_fraction`;
- `recommendations` (human-readable corrective actions).

---

## 3. Automatic correction

`src/train.py:detect_issue` delegates to `diagnose_training` and then maps the
diagnosis onto one retraining round (`run_training`, `max_correction_rounds=1`):

| Diagnosis | Correction applied before retraining |
|-----------|--------------------------------------|
| **overfit** | `dropout += 0.15` (capped at 0.6), force `early_stopping=True`, `lr_schedule="plateau"` |
| **underfit** | `hidden_size *= 1.5`, `epochs += 30`, `lr_schedule="plateau"` |
| **good_fit** | no change |

The best-validation weights are always restored before evaluation, so even when
the *history* shows overfitting, the saved checkpoint is the best-val snapshot.
Every run records `detection`, `diagnosis`, `corrections`, `train_rmse`,
`val_rmse`, `test_rmse`, and the gap fields in
`models/reports/train_<dataset>_<stamp>.json`; the diagnosis is also stored in
the `.pt` checkpoint.

---

## 4. Benchmark: deliberately inducing both regimes

`benchmarking/benchmark_generalization.py` trains all five models on the same
UCI windows/scalers/split/seed under three regimes, with automatic correction
**disabled** so the raw curves are observable:

| regime | hidden | layers | dropout | early stop | epochs | intent |
|--------|-------:|-------:|--------:|-----------|-------:|--------|
| underfit | 8 | 1 | 0.5 | yes | 8 | tiny capacity + heavy dropout + short budget (high bias) |
| balanced | 64 | 2 | 0.2 | yes | 20 | moderate capacity |
| overfit | 256 | 3 | 0.0 | no | 20 | large capacity + no regularisation (high variance) |

Run it with:

```bash
python benchmarking/benchmark_generalization.py \
    --dataset uci --epochs 20 --underfit_epochs 8 --seq_len 24
```

Outputs:

- `benchmarking/results/generalization/<model>__<regime>.json` — one JSON per run,
  including the full loss history and diagnosis signals;
- `benchmarking/generalization_summary_<stamp>.json/.csv` — flat roll-up;
- `benchmarking/overfitting_underfitting.md` — the generated write-up;
- `benchmarking/charts/learning_curves_<stamp>.png` — train vs validation loss
  for every model × regime;
- `benchmarking/charts/generalization_gaps_<stamp>.png` — validation-minus-train
  RMSE gap by regime.

### Latest UCI results (seed 42)

| regime | mean train RMSE | mean val RMSE | mean test RMSE | mean gap (val−train) | diagnoses |
|--------|----------------:|--------------:|---------------:|---------------------:|-----------|
| underfit | 1.247 | 1.960 | 1.302 | +0.713 | underfit: 3, good_fit: 2 |
| balanced | 0.837 | 1.538 | 1.000 | +0.701 | overfit: 5 |
| overfit | 0.859 | 1.567 | 0.985 | +0.708 | overfit: 5 |

**Reading the result:** the underfit regime has the highest test error (high
bias) — extra capacity clearly helps. The moderate and large regimes both show
the high-variance signature on this small UCI subset (only ~571 training
windows survive the complete-case filter): train error keeps falling while
validation bottoms out, so the detector flags them as overfitting. The best-val
snapshot keeps those saved checkpoints usable, but the curves are why the
pipeline never evaluates the last epoch. `transformer_lstm` is the most accurate
model overall (test RMSE `0.725`), and its best snapshot is in the overfit
regime; moving to a larger training set would let that capacity pay off without
the gap.

The per-run table and exact numbers are in
[`benchmarking/overfitting_underfitting.md`](../benchmarking/overfitting_underfitting.md).

---

## 5. Related benchmarks

- `benchmarking/run_sweep.py` sweeps `seq_len × hidden_size` and now records
  `train_rmse`, `val_rmse`, `diagnosis`, and `diagnosis_signals` per run, so
  `compare_and_report.py` can render a generalisation-gap section and chart.
- `scripts/benchmark_models.py` (single-configuration head-to-head) also stores
  the diagnosis and train/val/test split metrics.
- `docs/MODEL_COMPARISON.md` covers accuracy vs efficiency; this document covers
  *why* the saved model generalises (or not).
