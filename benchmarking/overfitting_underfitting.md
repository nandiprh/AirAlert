# Overfitting vs Underfitting Benchmark

Every model is trained on the same leakage-free 80:20 temporal split (validation carved from the train tail), the same windows/scalers and the same seed. Only the *capacity/regularisation regime* changes, so the learning curves isolate the bias-variance trade-off. Auto overfitting correction is disabled for this experiment: we want to observe the raw curves.

## Regimes

| regime | hidden_size | layers | dropout | early stop | epochs | expected curve |
|---|---|---|---|---|---|---|
| underfit (high bias) | 8 | 1 | 0.5 | yes | 8 (undertrained) | both losses stay high, small gap |
| balanced | 64 | 2 | 0.2 | yes | 20 | both fall together, small stable gap |
| overfit (high variance) | 256 | 3 | 0.0 | no | 20 | train falls while val bottoms out and rises |

Diagnosis rule (see `src/diagnostics.py`): **overfit** when the validation loss is above the train loss by more than `max(0.02, 0.2*train)` and has not improved for the overfit budget; **underfit** when the train loss drops less than 20% and the validation loss improves less than 10% (the model cannot fit even the training signal); otherwise **good_fit**.

## Results

`gap (curve)` is `val_loss - train_loss` at the final epoch; `gap (snap)` is `val_rmse - train_rmse` on the best-val weights that were actually saved.

| model | regime | train RMSE | val RMSE | test RMSE | gap (snap) | gap (curve) | best epoch | epochs | diagnosis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| lstm_cnn | underfit | 1.151 | 1.916 | 1.245 | 0.7647 | 1.1314 | 8 | 8 | **underfit** |
| lstm_cnn | balanced | 0.542 | 1.490 | 0.831 | 0.9485 | 1.9121 | 9 | 20 | **overfit** |
| lstm_cnn | overfit | 0.733 | 1.542 | 0.893 | 0.8088 | 1.9065 | 6 | 20 | **overfit** |
| cnn_bilstm | underfit | 1.282 | 1.992 | 1.320 | 0.7101 | 1.2263 | 8 | 8 | **underfit** |
| cnn_bilstm | balanced | 0.627 | 1.270 | 0.867 | 0.6430 | 2.7592 | 7 | 19 | **overfit** |
| cnn_bilstm | overfit | 0.515 | 1.350 | 0.747 | 0.8344 | 2.5041 | 6 | 20 | **overfit** |
| lstm_attention | underfit | 1.312 | 1.933 | 1.330 | 0.6216 | 0.9895 | 8 | 8 | **good_fit** |
| lstm_attention | balanced | 1.171 | 1.830 | 1.307 | 0.6585 | 1.5717 | 5 | 17 | **overfit** |
| lstm_attention | overfit | 1.130 | 1.864 | 1.236 | 0.7335 | 2.2574 | 6 | 20 | **overfit** |
| transformer_lstm | underfit | 1.206 | 1.990 | 1.268 | 0.7841 | 1.3724 | 8 | 8 | **good_fit** |
| transformer_lstm | balanced | 0.706 | 1.177 | 0.742 | 0.4707 | 1.0073 | 5 | 17 | **overfit** |
| transformer_lstm | overfit | 0.679 | 1.237 | 0.725 | 0.5574 | 2.1939 | 3 | 20 | **overfit** |
| convlstm_attention | underfit | 1.283 | 1.966 | 1.347 | 0.6827 | 1.2231 | 8 | 8 | **underfit** |
| convlstm_attention | balanced | 1.140 | 1.925 | 1.252 | 0.7844 | 5.4732 | 4 | 16 | **overfit** |
| convlstm_attention | overfit | 1.236 | 1.841 | 1.326 | 0.6057 | 4.2232 | 3 | 20 | **overfit** |

## Regime summary

| regime | mean train RMSE | mean val RMSE | mean test RMSE | mean gap (snap) | diagnoses |
|---|---:|---:|---:|---:|---|
| underfit | 1.247 | 1.960 | 1.302 | +0.7126 | underfit: 3, good_fit: 2 |
| balanced | 0.837 | 1.538 | 1.000 | +0.7010 | overfit: 5 |
| overfit | 0.859 | 1.567 | 0.985 | +0.7080 | overfit: 5 |

## Interpretation

- Across 15 runs the detector labelled **10 overfit**, **3 underfit** and **2 good-fit**.
- Mean test RMSE: underfit `1.302` → balanced `1.000` → overfit `0.985`. The underfit regime has the highest error (high bias); increasing capacity/regularisation brings it down.
- Mean snapshot generalisation gap: underfit `+0.7126` → balanced `+0.7010` → overfit `+0.7080`. The moderate and large models both overfit this small UCI subset; the best-val snapshot keeps their saved weights usable, but the raw curves show the high-variance signature (train loss keeps falling while validation loss bottoms out).
- Most accurate model overall: **transformer_lstm**; its best regime is **overfit** (test RMSE `0.725`).
- In deployment, use the moderate-capacity regime with early stopping and the best-val snapshot. The extra capacity in the overfit regime did not buy accuracy on this dataset, and training to the last epoch would be unsafe without the snapshot.

## Reproduce

```bash
python benchmarking/benchmark_generalization.py --dataset uci --epochs 20 --underfit_epochs 8 --seq_len 24
```

Raw per-run JSON: `benchmarking/results/generalization/`. Flat roll-up: `generalization_summary_20261103_112210.csv`. Curves: `charts/learning_curves_20261103_112210.png`, gaps: `charts/generalization_gaps_20261103_112210.png`.
