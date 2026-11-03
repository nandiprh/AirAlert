# SMOTE / SMOTE-R comparison

Generated 20261021_181513 · dataset `uci` · seq_len `24` · epochs `20` · task `both` · seeds `[42, 43, 44]` · models `['lstm_cnn', 'cnn_bilstm', 'transformer_lstm']`.

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
| `none` | 0.833 ± 0.098 | 0.617 ± 0.082 | 0.649 ± 0.086 | 0.613 ± 0.049 | 0.591 ± 0.051 | 1.296 ± 0.163 | 38.6 ± 27.3 |
| `cls` | 0.819 ± 0.070 | 0.604 ± 0.068 | 0.663 ± 0.057 | 0.617 ± 0.043 | 0.588 ± 0.071 | 1.366 ± 0.164 | 53.2 ± 32.9 |
| `cls_reg` | 0.953 ± 0.047 | 0.702 ± 0.050 | 0.545 ± 0.045 | 0.544 ± 0.036 | 0.469 ± 0.072 | 1.466 ± 0.235 | 79.5 ± 46.8 |
| `cls_smoter` | 0.860 ± 0.048 | 0.625 ± 0.045 | 0.629 ± 0.041 | 0.580 ± 0.025 | 0.539 ± 0.042 | 1.321 ± 0.110 | 50.9 ± 26.3 |
| `cls_reg_smoter` | 0.937 ± 0.122 | 0.689 ± 0.104 | 0.554 ± 0.127 | 0.576 ± 0.042 | 0.539 ± 0.067 | 1.557 ± 0.319 | 86.3 ± 52.9 |
| `smoter` | 0.903 ± 0.147 | 0.660 ± 0.137 | 0.582 ± 0.147 | 0.595 ± 0.039 | 0.572 ± 0.050 | 1.363 ± 0.271 | 38.7 ± 25.0 |

### Δ RMSE vs `none` (negative = better than no resampling)

| arm | Δ RMSE | Δ tail RMSE (top 10 %) |
|---|---|---|
| `cls` | -0.014 | +0.070 |
| `cls_reg` | +0.120 | +0.170 |
| `cls_smoter` | +0.027 | +0.025 |
| `cls_reg_smoter` | +0.104 | +0.261 |
| `smoter` | +0.070 | +0.067 |

## Target-distribution fidelity

How the **resampled training** target distribution compares with the untouched test distribution (same z-scored space):

| arm | KS distance to test ↓ | mass above test P90 (target 0.10) | mass below test P10 (target 0.10) | train std / test std |
|---|---|---|---|---|
| `none` | 0.094 ± 0.000 | 0.077 ± 0.000 | 0.119 ± 0.000 | 0.950 |
| `cls` | 0.107 ± 0.000 | 0.085 ± 0.004 | 0.131 ± 0.004 | 0.989 |
| `cls_reg` | 0.372 ± 0.003 | 0.430 ± 0.019 | 0.073 ± 0.002 | 1.341 |
| `cls_smoter` | 0.106 ± 0.003 | 0.120 ± 0.002 | 0.165 ± 0.003 | 1.136 |
| `cls_reg_smoter` | 0.373 ± 0.003 | 0.437 ± 0.017 | 0.112 ± 0.003 | 1.461 |
| `smoter` | 0.091 ± 0.004 | 0.116 ± 0.004 | 0.152 ± 0.003 | 1.113 |

## Per-model test RMSE

| model | `none` | `cls` | `cls_reg` | `cls_smoter` | `cls_reg_smoter` | `smoter` |
|---|---|---|---|---|---|---|
| lstm_cnn | 0.921 | 0.862 | 0.987 | 0.893 | 1.025 | 1.072 |
| cnn_bilstm | 0.844 | 0.849 | 0.925 | 0.869 | 0.893 | 0.818 |
| transformer_lstm | 0.734 | 0.746 | 0.947 | 0.819 | 0.893 | 0.818 |

## Charts

- `benchmarking/charts/smote_rmse.png` — test RMSE per arm
- `benchmarking/charts/smote_tail_rmse.png` — RMSE on the rare (top-decile) test days
- `benchmarking/charts/smote_ks.png` — KS distance to the test target distribution

## Reproduce

```bash
.venv/bin/python benchmarking/benchmark_smote.py --dataset uci --seq-len 24 --epochs 20 --seeds 42 43 44 --models lstm_cnn cnn_bilstm transformer_lstm
```

Raw per-run records: `benchmarking/results/smote/*.json` · summary: `benchmarking/smote_summary_20261021_181513.json` / `.csv`.
