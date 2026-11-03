# SMOTE / SMOTE-R comparison

Generated 20261021_184642 · dataset `india` (city `Delhi`) · seq_len `7` · epochs `20` · task `both` · seeds `[42, 43]` · models `['lstm_cnn', 'cnn_bilstm']`.

Every arm shares the **same** windows, scalers, split, model family, epoch budget and seed set — only the resampling strategy changes. Metrics are means ± std over all (model × seed) runs.

## Arms

| arm | what it does |
|-----|--------------|
| `none` | no resampling |
| `cls` | classification SMOTE |
| `cls_reg` | cls SMOTE + reg SMOTE |
| `cls_smoter` | cls SMOTE + SMOTE-R |
| `cls_reg_smoter` | cls SMOTE + reg SMOTE + SMOTE-R |
| `smoter` | SMOTE-R only |

## Headline results (all models × seeds pooled)

| arm | test RMSE ↓ | test MAE ↓ | test R² ↑ | accuracy ↑ | F1 ↑ | tail RMSE (top 10 %) ↓ | train time (s) |
|---|---|---|---|---|---|---|---|
| `none` | 36.547 ± 0.143 | 27.235 ± 0.262 | 0.903 ± 0.001 | 0.705 ± 0.038 | 0.688 ± 0.060 | 55.742 ± 2.534 | 26.8 ± 9.2 |
| `cls` | 39.140 ± 2.326 | 30.035 ± 2.293 | 0.888 ± 0.013 | 0.728 ± 0.007 | 0.729 ± 0.006 | 57.664 ± 2.905 | 46.9 ± 10.0 |
| `cls_reg` | 39.237 ± 1.069 | 29.086 ± 1.631 | 0.888 ± 0.006 | 0.692 ± 0.040 | 0.689 ± 0.041 | 59.513 ± 3.927 | 108.0 ± 9.5 |
| `cls_smoter` | 36.598 ± 0.952 | 27.009 ± 1.296 | 0.903 ± 0.005 | 0.705 ± 0.059 | 0.705 ± 0.060 | 56.922 ± 2.432 | 61.6 ± 22.2 |
| `cls_reg_smoter` | 42.995 ± 4.286 | 33.646 ± 4.892 | 0.865 ± 0.028 | 0.649 ± 0.077 | 0.642 ± 0.086 | 62.640 ± 2.042 | 78.9 ± 18.9 |
| `smoter` | 37.370 ± 1.354 | 27.749 ± 1.527 | 0.899 ± 0.007 | 0.673 ± 0.040 | 0.654 ± 0.054 | 58.738 ± 3.794 | 31.5 ± 6.9 |

### Δ RMSE vs `none` (negative = better than no resampling)

| arm | Δ RMSE | Δ tail RMSE (top 10 %) |
|---|---|---|
| `cls` | +2.593 | +1.922 |
| `cls_reg` | +2.690 | +3.771 |
| `cls_smoter` | +0.051 | +1.180 |
| `cls_reg_smoter` | +6.448 | +6.898 |
| `smoter` | +0.823 | +2.995 |

## Target-distribution fidelity

How the **resampled training** target distribution compares with the untouched test distribution (same z-scored space):

| arm | KS distance to test ↓ | mass above test P90 (target 0.10) | mass below test P10 (target 0.10) | train std / test std |
|---|---|---|---|---|
| `none` | 0.407 ± 0.000 | 0.241 ± 0.000 | 0.012 ± 0.000 | 0.938 |
| `cls` | 0.231 ± 0.003 | 0.252 ± 0.000 | 0.088 ± 0.000 | 1.169 |
| `cls_reg` | 0.497 ± 0.000 | 0.584 ± 0.000 | 0.049 ± 0.000 | 1.358 |
| `cls_smoter` | 0.217 ± 0.002 | 0.274 ± 0.000 | 0.125 ± 0.000 | 1.269 |
| `cls_reg_smoter` | 0.490 ± 0.000 | 0.577 ± 0.000 | 0.090 ± 0.000 | 1.470 |
| `smoter` | 0.382 ± 0.000 | 0.265 ± 0.000 | 0.021 ± 0.001 | 1.057 |

## Per-model test RMSE

| model | `none` | `cls` | `cls_reg` | `cls_smoter` | `cls_reg_smoter` | `smoter` |
|---|---|---|---|---|---|---|
| lstm_cnn | 36.474 | 41.116 | 39.064 | 36.311 | 46.208 | 38.661 |
| cnn_bilstm | 36.619 | 37.164 | 39.409 | 36.885 | 39.782 | 36.079 |

## Charts

- `benchmarking/charts/smote_rmse_india_Delhi.png` — test RMSE per arm
- `benchmarking/charts/smote_tail_rmse_india_Delhi.png` — RMSE on the rare (top-decile) test days
- `benchmarking/charts/smote_ks_india_Delhi.png` — KS distance to the test target distribution

## Reproduce

```bash
.venv/bin/python benchmarking/benchmark_smote.py --dataset india --seq-len 7 --epochs 20 --seeds 42 43 --models lstm_cnn cnn_bilstm
```

Raw per-run records: `benchmarking/results/smote/india_Delhi__*.json` · summary: `benchmarking/smote_summary_20261021_184642.json` / `.csv`.
