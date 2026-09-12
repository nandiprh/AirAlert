# Model Math, Code, and How to Run

A single guide covering **how to run the project end-to-end**, the
**mathematics behind every model and algorithm**, and a **line-by-line
code walkthrough**.

Companion docs: `docs/CODE_ARCHITECTURE.md` (files & layout),
`docs/FETCH_AND_RUN.md` (exact reproduction commands),
`docs/DATA_SOURCES.md` (provenance), `docs/MODEL_COMPARISON.md` (results).

---

## Part 1 — How to Run

### 1. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # torch, plotly, folium, kaleido, sklearn, imblearn, geopandas ...
```

### 2. Data cleanup (raw -> processed)

```bash
.venv/bin/python -m src.data.preprocess          # uci, india, delhi -> data/processed/
.venv/bin/python -m src.data.openaq_loader       # enrich worldwide stations (needs data/raw/openaq)
```

### 3. Train models

One model, UCI dataset:

```bash
# 80:20 split, seq_len 24, both regression+classification, seq_len-adaptive
.venv/bin/python src/train.py --dataset uci --model lstm_cnn --seq_len 24 --task both

# All five hybrids with auto overfit/underfit correction:
.venv/bin/python src/train.py --dataset uci --model all --seq_len 24 --task both
```

India dataset, single city:

```bash
.venv/bin/python src/train.py --dataset india --city Delhi --model transformer_lstm --seq_len 24
```

Cross-validation (10-fold time-series) instead of holdout:

```bash
.venv/bin/python src/train.py --dataset india --model all --split_mode cv10 --seq_len 7
```

Outputs: `models/checkpoints/*.pt`, `models/reports/train_*.json` +
`*_history.png`.

### 4. Evaluate a saved checkpoint

```bash
.venv/bin/python src/evaluate.py --model lstm_cnn --dataset uci --seq_len 24
.venv/bin/python src/evaluate.py --checkpoint models/checkpoints/lstm_cnn_uci_both_s24.pt
```

Outputs: test RMSE/MAE/R² + accuracy/F1 + confusion matrix + plots →
`models/reports/`.

### 5. Predict next-day AQI per Indian city

```bash
.venv/bin/python src/predict.py --model lstm_cnn --seq_len 7 --epochs 30
.venv/bin/python src/predict.py --model all --city Delhi        # single city / ensemble
```

Outputs: `data/processed/predictions/city_predictions.csv` +
`models/reports/city_predictions_summary.json`.

### 6. Create hotspot maps

```bash
.venv/bin/python -m src.visualization.map_global      # world + India AQI maps
.venv/bin/python -m src.visualization.map_cities      # city-wise prediction maps
```

Outputs: `output/{world,india}_aqi_map.{html,png}` and
`output/{world,india}_city_predictions.{html,png}`. View in any browser:

```bash
cd output && python3 -m http.server 8000
# -> http://localhost:8000/<file>.html
```

### 7. Reproduce the 5-model benchmark

```bash
.venv/bin/python scripts/benchmark_models.py --seq_len 24
```

Outputs: `models/reports/model_comparison_*.json` (results: `
docs/MODEL_COMPARISON.md`).

---

## Part 2 — Mathematical Model

### 2.1 Notation

Every model consumes a window of the last `T = seq_len` hours/days of
pollution readings and predicts the *next* AQI value and bucket:

```
X = [x_1, x_2, ..., x_T] ∈ R^{T × F},   F = 9 features
    (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3)
ŷ  = predicted AQI (regression)          ∈ R
p  = P(bucket | X) over K classes        ∈ R^K      (K = 4)
```

### 2.2 Preprocessing (sliding windows + train-only z-score)

**Sliding windows.** Each sample is the ordered block
`w_t = (x_{t-s+1}, …, x_t)` and its label `y_t = aqi_{t+1}`. Windowing
happens **after** the temporal split so a window never crosses
train/val/test boundary (no leakage).

**Z-score normalization** (both features and the AQI target; fitted on
the **training partition only** — type-2 leakage prevention):

```
z = (x − μ_train) / σ_train
```

`train.py:286-291` fits `StandardScaler` on `X_train` / `y_train`, applies
to every split (`apply_normalizer`), and stores the scalers in the
checkpoint so forecasts can invert the transform.

### 2.3 SMOTE (minority-bucket oversampling)

Training windows are flattened to 2D and, for each minority sample
`z_i = [flat_window_i, y_reg_i]` (regression target is **appended** so it
gets interpolated too), SMOTE creates a synthetic sample by convex
interpolation between `z_i` and a randomly chosen k-NN `z_j`:

```
z_new = z_i + λ (z_j − z_i),    λ ~ U(0, 1)
```

Applied **only to training windows** (`train.py:329`), so the classifier
no longer collapses onto the dominant AQI bucket.

### 2.4 LSTM cell

The shared building block (PyTorch `nn.LSTM`). For gate vector
`g ∈ {i, f, o, c̃}`:

```
i_t = σ(W_ii x_t + W_hi h_{t−1} + b_i)          input gate
f_t = σ(W_if x_t + W_hf h_{t−1} + b_f)          forget gate
o_t = σ(W_io x_t + W_ho h_{t−1} + b_o)          output gate
c̃_t = tanh(W_ic x_t + W_hc h_{t−1} + b_c)       candidate cell
c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t                cell state
h_t = o_t ⊙ tanh(c_t)                            hidden state
```

The final readout uses the **last hidden state** `h_T`
(`cnn_bilstm`, `transformer_lstm`) or the **full sequence**
`H = [h_1 … h_T]` (`lstm_cnn`, `lstm_attention`, `convlstm_attention`).

### 2.5 1D convolution (local/time-frequency features)

A kernel `W ∈ R^{C_out × C_in × K}` slides over the time axis with `same`
padding, then batch-norm + ReLU:

```
conv(x)_t = ReLU(BN( Σ_{k=0}^{K−1} W[:, :, k] · x_{t + k − ⌊K/2⌋} ))
```

- `lstm_cnn`: conv applied on top of LSTM hidden states `H·W_c` so the CNN
  reads *temporal* features; ends with `AdaptiveMaxPool1d(1)` → one vector
  per window.
- `cnn_bilstm` / `convlstm_attention`: conv applied on the **raw input**
  `X` (transposed to channels-first) to extract multi-scale local patterns
  first.

### 2.6 Attention mechanisms

**(a) Bahdanau / additive attention** (`lstm_attention.py:71`):

```
e_t = v^T tanh(W_a h_t)              energy of timestep t
α_t = softmax_t(e)                   attention weight, Σ_t α_t = 1
c   = Σ_t α_t h_t                    context vector
```

**(b) Scaled dot-product attention** (`convlstm_attention.py:100`):

```
Q = H W_q,   K = H W_k,   V = H W_v                 (all R^{T×H})
A   = softmax( Q K^T / √d_k )                        (T×T attention matrix)
c_t = Σ_{t'} A_{t,t'} v_{t'};   c = max-pool_t(c)    (self-attention pooling)
```

Dividing by `√d_k` keeps softmax logits in a stable range as `H` grows.

### 2.7 Positional encoding + Transformer encoder (`transformer_lstm.py`)

**(a) Sinusoidal positional encoding** (fixed, non-learnable):

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d} )
X' = X W_in + PE
```

**(b) Self-attention head:**

```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
MultiHead(Q,K,V)    = Concat(head_1…head_h) W_O
   head_i = Attention(X W_q^i, X W_k^i, X W_v^i)
```

**(c) Feed-forward + residual + norm per Transformer layer:**

```
Z   = LayerNorm(X + MultiHead(X))
out = LayerNorm(Z + GELU(Z W_1 + b_1) W_2 + b_2)
```

The Transformer encodes global dependencies across the whole day, then a
1-layer LSTM refines the attended sequence (architecture 4).

### 2.8 Output heads

```
f      = Dropout-ReLU(Linear(context))        shared features
ŷ      = W_reg f + b_reg                      AQI scalar
logits = W_cls f + b_cls  →  p_k = softmax_k   AQI bucket probabilities
```

### 2.9 Training objective

Multi-task loss = regression MSE + classification cross-entropy of the
same network:

```
L = MSE(ŷ, y) + γ · CE(p, c) + λ₂ · ‖θ‖²
MSE = (1/N) Σ (ŷᵢ − yᵢ)²
CE  = −(1/N) Σ log p_{i,cᵢ}
```

- Optimizer: **Adam** (`weight_decay = 1e-5`).
- Schedule: **ReduceLROnPlateau** — halves LR when val loss plateaus
  (`patience=8`).
- **Early stopping**: best val-loss weights are restored; training halts
  after `patience=12` epochs without improvement.

### 2.10 Evaluation metrics

```
RMSE = √( (1/N) Σ (yᵢ − ŷᵢ)² )
MAE  = (1/N) Σ |yᵢ − ŷᵢ|
R²   = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²          (1 = perfect)
accuracy = correct / total
F1        = macro-averaged 2·P·R / (P+R) over AQI buckets
```

All computed on **unscaled** AQI values (`inverse_normalize_target`).

### 2.11 Overfit/underfit auto-correction (`detect_issue`, `train.py:451`)

With train loss `ℓ_t`, val loss history `ℓ_v`, best val `ℓ_v*`:

```
overfit  if  (ℓ_v − ℓ_t) > max(0.02, 0.2·ℓ_t)   and  ℓ_v ≥ 1.02·ℓ_v*
            and the val gap persisted ≥ overfit_budget epochs
            → dropout += 0.15  (regularize harder), retrain

underfit if  min(ℓ_v) ≥ 0.93·ℓ_t(0)  and  ℓ_v(end) > ℓ_t(end)
            → hidden_size × 1.5, epochs += 30, retrain
```

`run_training` retrains once with the corrected config, keeps the
best-val-round snapshot, and writes it to the checkpoint.

---

## Part 3 — Code Walkthrough

### 3.1 `src/train.py` — the orchestrator (872 lines)

| Snippet | What happens |
|---------|--------------|
| `UCI_FEATURES` / `INDIA_FEATURES` (:62/:79) | Feature column lists for each dataset. |
| `load_uci_data` / `load_india_data` (:112/:146) | Read processed CSV (raw fallback), optional `city` filter; India also returns the AQI target + `AQI_Bucket` class. |
| `preprocess_dataset` (:187) | **(a)** split the *raw series* into train/val/test (`n_test = n(1−0.8)`); `val=0.0` carves ~10% from the train tail; **(b)** build sliding windows per split; **(c)** fit `StandardScaler` on train only, transform all; returns `(X, y_reg, y_cls)` + scalers + `cls_map`. |
| `apply_smote` (:329) | Flatten train windows → SMOTE k-NN interpolation incl. appended `y_reg` → reshape back to `(B, T, F)`. |
| `Config(dict)` (:438) | Hyperparameter bag with dotted access (`cfg.dropout`). |
| `detect_issue` (:451) | The rules from §2.11; returns a corrected config copy. |
| `train_one_epoch` (:394) | Forward → two losses → backward → Adam step. |
| `evaluate_loader` (:411) | val/test forward pass; returns raw y_true/y_pred in **scaled** space. |
| `to_split_metrics` (:643) | Inverts the target scaler, then computes §2.10 metrics. |
| `run_training` (:501) | Build model → loop (early stop, LR plateau) → restore best → `detect_issue`+retrain → save checkpoint (state_dict, config, metadata, scalers, cls_map) → history plot. |
| `run_cv` (:679) | `TimeSeriesSplit(10)` surrogate — 10-fold, same build/scale/metrics. |
| `main` (:799) | CLI glue written in Part 1. |

### 3.2 `src/models/` — the five architectures

Every forward torch:

```python
lstm_out, _ = self.lstm(x)             # (B,T,H)      lstm_cnn / LSTMMAX
cnn_feat    = self.cnn(...)            # conv+pool    lstm_cnn
context     = ΓΔΣ α_t·h_t or softmax(QKᵀ/√d)·V
feats       = self.head(context)
return fc_reg(feats), fc_cls(feats)    # (B,1) + (B,K)
```

- **`lstm_cnn.py`** — `LSTM` → transpose `(B,H,T)` → 2×(`Conv1d`→`BatchNorm`→`ReLU`) → `AdaptiveMaxPool1d(1)` → heads. Convolution operates on the LSTM's *temporal* states.
- **`cnn_bilstm.py`** — 2×Conv1d on raw input → `nn.LSTM(bidirectional=True, hidden_size=H/2)` → concatenates final fwd+bwd `h` (`h_n[-2]`, `h_n[-1]`) → heads. Requires `H` even.
- **`lstm_attention.py`** — LSTM → `attention_weights` (energy `vᵀtanh(W·h)`, softmax over time) → weighted sum → heads.
- **`transformer_lstm.py`** — `input_proj` → sinusoidal `pos_enc` → `TransformerEncoder` (GELU FFN) → 1×LSTM → `h_n[-1]` → heads. `max_len = seq_len+64` baked into positional buffer.
- **`convlstm_attention.py`** — 3×Conv1d (32→64→H) on raw input → LSTM → scaled dot-product `T×T` attention → **max-pool over time** → heads.

### 3.3 `src/models/base.py` — shared utilities

- `SequenceDataset` — window Dataset; returns `(x, y_reg)` and/or `(x, y_cls)`.
- `fit_normalizer` / `apply_normalizer` — handles 2D and 3D `(B,T,F)` by flattening window axis before scaling then reshaping back.
- `inverse_normalize_target` — fabricates a full-width row, fills one target column, inverts, reads that column back.
- `compute_metrics` — §2.10 formulas.
- `deterministic_seed` — seeds `random/numpy/torch` (+CUDA).

### 3.4 `src/evaluate.py`

`find_checkpoint` by name/pattern → rebuild model from the checkpoint's
`metadata`+`config` → rerun the *identical* `preprocess_dataset` on test →
`compute_metrics` → save `*_evaluation.json` + predictions + confusion
plots.

### 3.5 `src/predict.py` — real forecasting

```python
def last_forecast_window(city, seq_len):   # :72  rebuild the exact cleaned tail
    raw      = load_india_data(city)
    clean    = raw.dropna(subset=INDIA_FEATURES + [INDIA_TARGET])
    return clean.tail(seq_len), last date

def forecast_city(city, ...):              # :132
    data   = preprocess_dataset("india", city=city, train_frac=0.8, val_frac=0.0)
    result = run_training(...)             # SMOTE + auto-correction per city
    window = apply_normalizer(last_window, data["feature_scaler"])
    with torch.no_grad(): ŷ_scaled, _ = model(window)
    ŷ = float(inverse_normalize_target(ŷ_scaled, data["reg_scaler"]))
    return city, last_date, last_date+1d, ŷ, bucket_of(ŷ)
```

Coordinates come from the curated `location_table` (see 3.6), and results
are written to `city_predictions.csv` + a summary JSON.

### 3.6 `src/visualization/location_table.py`

`load_location_table()` reads `data/external/city_locations.csv`
(`code,city,country,state,lat,lon`) into a normalized-name dict;
`resolve(city)` tries table → `FALLBACK_CITY_COORDS` → exact match in
`india_cities.csv` → `None`. `resolve_many` batches it. This hardcoded
table fixes ambiguous/erroneous matches (e.g. "Visakhapatnam"→Bihar).

### 3.7 `scripts/benchmark_models.py` — the §1.7 benchmark

For each of the 5 models with one shared `Config`: `run_training` (timed)
→ `sum(p.numel())` params → `inference_timing` (mean ms over 50 warm
forward passes on `(64, seq_len, F)`) → strip arrays → write
`model_comparison_*.json` and print RMSE + params rankings.

### 3.8 Data layer

- `src/data/preprocess.py` — parsers (`load_uci` handles `;` decimal data),
  median imputation, CPCB bucket recomputation, duplicate-row +
  high-correlation leakage checks, `run_pipeline`.
- `src/data/openaq_loader.py` — station summarization, S3 URL builder,
  `assign_country` via geopandas point-in-polygon.
- `scripts/openaq_scan/select/fetch` — archive discovery, balanced
  country selection, prefetch (only `urllib` + User-Agent works on the
  bucket).

---

## Where math meets code (quick map)

| Concept | Formula | Code |
|---------|---------|------|
| Z-score | `z=(x−μ)/σ` | `src/models/base.py:74`, `train.py:286` |
| Sliding window | `(x_{t−T+1},…,x_t)` | `train.py:187-303` |
| SMOTE | `z_i+λ(z_j−z_i)` | `train.py:329` (`apply_smote`) |
| LSTM | §2.4 gates | `models/*.py` `nn.LSTM` |
| Conv1d | §2.5 | `lstm_cnn.py:54`, `cnn_bilstm.py:47`, `convlstm_attention.py:49` |
| Bahdanau | `Σα_t h_t` | `lstm_attention.py:71-96` |
| Dot-product attn | `softmax(QKᵀ/√d)V` | `convlstm_attention.py:100-131` |
| Positional encoding | sin/cos | `transformer_lstm.py:17-36` |
| Multi-head attn | `Concat(heads)W_O` | `transformer_lstm.py:78-86` |
| Multi-task loss | `MSE + γ·CE + λ₂‖θ‖²` | `train.py:374-393` |
| Early stop / plateau | — | `train.py:576-594` |
| Auto-correction | §2.11 | `train.py:451-489` |
| Metrics | RMSE/MAE/R²/F1 | `models/base.py` `compute_metrics` |