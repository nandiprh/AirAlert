# Mathematical Modeling: Air-Quality Forecasting

The formal mathematical description of every model, algorithm and score in
this project. This is the "why the numbers happen" companion to
`docs/MODEL_COMPARISON.md` (results) and `benchmarking/comparison.md`
(parameter-sweep analysis). A run-through with code pointers lives in
`docs/MODEL_MATH_AND_RUN.md`.

---

## 1. Problem formulation

Given a time series of pollution readings `x_1, …, x_n ∈ R^F` (one vector of
`F` pollutant/gas features per time step) we forecast the **next** AQI value
and its health bucket:

```
X = [x_{t-T+1}, …, x_t]  ∈ R^{T×F}        T = seq_len, F = 9
ŷ = f_reg(X)             ∈ R              predicted AQI
p = f_cls(X)             ∈ Δ^{K-1}        belief/probability over K buckets
```

This is a supervised sequence-to-one problem with a **multi-task** head: a
regression task (exact AQI) and a classification task (bucket label) sharing
one feature extractor.

| Symbol | Meaning | Value in this project |
|--------|---------|----------------------|
| `T` | look-back window | 7 or 24 (sweep) |
| `F` | features per step | 13 (UCI) / 9 (India) |
| `K` | bucket classes | 4 (UCI qcut) / 6 (India CPCB labels) |
| `Hidden` | hidden dimension | 32 / 128 (sweep), 64 default |

---

## 2. Preprocessing

### 2.1 Temporal split + leakage-safe windowing

The raw series is split into train / val / test **before** any windowing, so
no window straddles a boundary (`n_test = (1 − 0.8)·n`, val ≈ 10% of train
carved from its tail):

```
test   = (n_train + n_val, n)
```

Training windows may be **augmented with SMOTE** (§5) but never mixed with
test windows.

### 2.2 Z-score normalization (fitted on train only)

```
z = (x − μ_train) / σ_train          element-wise over features
ŷ_scaled = (y − μ_y) / σ_y
```

- `μ_train, σ_train` are computed **only** on the training partition (type-2
  leakage prevention) and stored in the checkpoint so predictions can invert
  the transform.
- 3D windows `(B, T, F)` are flattened to `(B·T, F)` for fitting then
  reshaped back — one scaler per feature.

### 2.3 AQI index (how PM2.5 becomes AQI)

Used by the OpenAQ world maps. The US-EPA piecewise-linear interpolation
between breakpoints `(BP_low, BP_high, AQI_low, AQI_high)`:

```
AQI = AQI_low + (AQI_high − AQI_low) · (C − BP_low) / (BP_high − BP_low)
```

with `C` the 24-h mean PM2.5 concentration and breakpoints:

| C range (µg/m³) | AQI range |
|---|---|
| 0.0 – 12.0 | 0 – 50 |
| 12.1 – 35.4 | 51 – 100 |
| 35.5 – 55.4 | 101 – 150 |
| 55.5 – 150.4 | 151 – 200 |
| 150.5 – 250.4 | 201 – 300 |
| 250.5 – 350.4 | 301 – 400 |
| 350.5 – 500.4 | 401 – 500 |

### 2.4 CPCB bucket mapping (labels + map colors)

Predictions and map markers are binned by the integer thresholds in
`src/visualization/aqi_spec.py`:

```
bucket(AQI) = first label where AQI ≥ lower bound, scanning high→low:

  [0,51) Good            #2ca02c  (green)
  [51,101) Satisfactory  #f7d448  (yellow)
  [101,201) Moderate     #f49e2a  (orange)
  [201,301) Poor         #d6452e  (red)
  [301,401) Very Poor    #8a1f1f  (dark red)
  [401,∞)  Severe        #8a1f1f
```

This one spec drives the CSV labels, the folium legend and the plotly legend,
so a marker's colour, its bucket name and the "what each color means" legend
can never disagree.

---

## 3. SMOTE (synthetic minority oversampling)

Classification targets are imbalanced (most days are Good/Satisfactory), so
training windows are balanced with SMOTE. A synthetic sample is a convex
combination of a minority sample and one of its k-NN:

```
z_new = z_i + λ·(z_j − z_i),     λ ~ U(0,1)
```

`z_i` packs `[flattened_window_fore_features, y_reg]`, so the regression
target is interpolated **in ℓ2 sense** along with the features
(`y_reg` stays consistent with the synthetic inputs). SMOTE is applied to
**training windows only**.

---

## 4. The five architectures

All five expose the same interface `forward(x) → (reg, cls)` for
`x ∈ R^{B×T×F}`.

### 4.1 LSTM cell (shared recurrent primitive)

For each time step `t`, with gates `i` (input), `f` (forget), `o` (output),
candidate `c̃`:

```
i_t = σ(W_i x_t + U_i h_{t−1} + b_i)
f_t = σ(W_f x_t + U_f h_{t−1} + b_f)
o_t = σ(W_o x_t + U_o h_{t−1} + b_o)
c̃_t = tanh(W_c x_t + U_c h_{t−1} + b_c)
c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t
h_t = o_t ⊙ tanh(c_t)
```

The rest of the pipeline consumes either the **last state** `h_T`
(`cnn_bilstm`, `transformer_lstm`) or the **full state sequence**
`H = [h_1,…,h_T]` (`lstm_cnn`, `lstm_attention`, `convlstm_attention`).

### 4.2 1-D convolution

A kernel `W ∈ R^{C_out × C_in × K}` moves along the (=padded) time axis:

```
conv(x)_t = ReLU(BN( Σ_{k=0}^{K-1} W[:,:,k] · x_{t+k−⌊K/2⌋} ))
```

Two placements:
- **on LSTM outputs** (`lstm_cnn`): conv sees *temporal* features; ends with
  `AdaptiveMaxPool1d(1)` → one vector per window;
- **on raw input** (`cnn_bilstm`, `convlstm_attention`): conv does local
  multi-scale feature extraction first.

### 4.3 BiLSTM fusion (`cnn_bilstm`)

Two LSTMs run the sequence forward and backward; final representation is the
concatenation of the last forward and backward hidden states:

```
h = [h_T^→ ; h_1^←]   ∈ R^{2·(H/2)}
```

(requires `H` even; `hidden_size` is the concatenated width).

### 4.4 Bahdanau / additive attention (`lstm_attention`)

```
e_t     = v^T tanh(W_a h_t)          energy of step t
α_t     = softmax_t(e)               attention weights, Σ_t α_t = 1
c       = Σ_t α_t h_t                context vector (weighted readout)
```

### 4.5 Scaled dot-product self-attention (`convlstm_attention`)

```
Q = H W_q,  K = H W_k,  V = H W_v         ∈ R^{T×H}
A = softmax( Q K^T / √d_k )               T×T attention matrix
c_t = Σ_{t'} A_{t,t'} v_{t'};  c = maxpool_t(c)
```

`√d_k` keeps the logits scale stable as the hidden width grows.

### 4.6 Transformer encoder + LSTM (`transformer_lstm`)

**(a) Sinusoidal positional encoding** (fixed):

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
X' = X W_in + PE
```

**(b) Multi-head self-attention:**

```
head_i  = softmax( X W_q^i (X W_k^i)^T / √d_k ) · X W_v^i
MHA(X)  = Concat(head_1,…,head_h) W_O
```

**(c) Pre-norm Transformer block** (per `TransformerEncoderLayer`):

```
Z   = LayerNorm(X + MHA(X))
out = LayerNorm(Z + GELU(Z W_1 + b_1) W_2 + b_2)
```

A 1-layer LSTM then refines the attended sequence; readout is `h_T`.

### 4.7 Output heads

```
f      = Dropout_ReLU(Linear(context))      shared embedding
ŷ      = W_reg f + b_reg                    AQI scalar
logits = W_cls f + b_cls  →  p = softmax    bucket probabilities
```

---

## 5. Training objective

Multi-task loss with weight decay (all heads trained jointly):

```
L(θ) = MSE(ŷ, y) + γ·CE(p, c) + λ₂·‖θ‖²

MSE(ŷ,y) = (1/N) Σ_i (ŷ_i − y_i)²            regression error
CE(p,c)  = −(1/N) Σ_i log p_{i, c_i}          bucket cross-entropy
```

- **Optimizer:** Adam with `lr = 1e-3`, `weight_decay = 1e-5`.
- **Scheduler:** ReduceLROnPlateau — `lr ← 0.5·lr` when the val metric
  plateaus for `patience = 8` epochs.
- **Early stopping:** weights from the best val-loss epoch are restored;
  training halts after `patience = 12` epochs with no improvement.

### 5.1 Auto overfit/underfit correction

With `ℓ_t` the latest train loss, `ℓ_v` the val-loss history, `ℓ_v*` its min:

```
OVERFIT   if  (ℓ_v − ℓ_t) > max(0.02, 0.2·ℓ_t)
              and ℓ_v ≥ 1.02·ℓ_v* and the gap lasted ≥ 8 epochs
         → dropout += 0.15, retrain (regularize harder)

UNDERFIT  if  min(ℓ_v) ≥ 0.93·ℓ_t(0)  and  ℓ_v(end) > ℓ_t(end)
         → hidden_size × 1.5, epochs += 30, retrain
```

One correction round max; the best-val snapshot is always kept.

---

## 6. Evaluation metrics

All computed on **unscaled** AQI values (`inverse_normalize_target`):

```
RMSE = √( (1/N) Σ_i (y_i − ŷ_i)² )
MAE  = (1/N) Σ_i |y_i − ŷ_i|
R²   = 1 − Σ_i (y_i − ŷ_i)² / Σ_i (y_i − ȳ)²
acc  = correct / N
F1   = macro-averaged 2·P·R / (P + R) over K buckets
```

---

## 7. Benchmark math (`benchmarking/`)

`benchmarking/run_sweep.py` trains every model on the **same windows, scalers
and seed** across a grid `{seq_len} × {hidden_size}`, logging full metrics +
config + history per run. `benchmarking/compare_and_report.py` then quantifies
**how the parameters used change each algorithm**.

### 7.1 Parameter sensitivity

For each model, holding everything else fixed, the effect of raising a
parameter is the difference in test RMSE between the high and low setting:

```
Δ_rmse(param) = RMSE(param_high) − RMSE(param_low)
```

Sign convention (used throughout `benchmarking/comparison.md`):
- `Δ < 0` → the larger setting **improved** accuracy (RMSE dropped);
- `Δ > 0` → it hurt accuracy.

This is computed for `hidden_size` (at each `seq_len`) and for `seq_len`
(at each `hidden_size`), isolating the two swept architectural parameters.

### 7.2 Balance score

To reward both accuracy and cost fairly, each metric is min-max normalized
over the union of all runs, then combined:

```
ñ = (n − n_min) / (n_max − n_min)           ∈ [0,1]

score = 0.5·ṟmse + 0.3·p̂arams + 0.2·t̂rain
```

- `0.5` weight on (normalized) RMSE — accuracy dominates;
- `0.3` on (normalized) parameter count — model size;
- `0.2` on (normalized) CPU training wall-time.

**Lower is better.** This surfaces picks like `lstm_cnn` (India) or
`lstm_attention` (UCI) that are cheap yet still accurate enough, where pure
RMSE would over-privilege a heavyweight architecture.

### 7.3 Reproduce

```bash
.venv/bin/python benchmarking/run_sweep.py --epochs 5        # 5 models × grid
.venv/bin/python benchmarking/compare_and_report.py          # comparison.md + charts
```

---

## Where math meets code (quick map)

| Concept | Formula | Code |
|---------|---------|------|
| Z-score | `z=(x−μ)/σ` | `src/models/base.py` `fit_normalizer`, `src/train.py:286` |
| Windows | `(x_{t−T+1},…,x_t)` | `src/train.py:187` `preprocess_dataset` |
| SMOTE | `z_i+λ(z_j−z_i)` | `src/train.py:329` `apply_smote` |
| EPA AQI | piecewise linear | `map_global.py:99` `pm25_to_aqi` |
| Buckets | `≥` thresholds | `aqi_spec.py:73` `aqi_bucket_name` |
| LSTM gates | §4.1 | `nn.LSTM` in every hybrid |
| Conv1d | §4.2 | `lstm_cnn.py`, `cnn_bilstm.py`, `convlstm_attention.py` |
| Bahdanau attn | `Σα_t h_t` | `lstm_attention.py:71` |
| Dot-product attn | `softmax(QKᵀ/√d)V` | `convlstm_attention.py:100` |
| Positional encoding | sin/cos | `transformer_lstm.py:17` |
| Multi-head attn | `Concat(heads)W_O` | `transformer_lstm.py:78` |
| Multi-task loss | `MSE+γ·CE+λ₂‖θ‖²` | `src/train.py` train loop |
| Correction rules | §5.1 | `src/train.py:451` `detect_issue` |
| Metrics | RMSE/MAE/R²/F1 | `src/models/base.py` `compute_metrics` |
| Sensitivity | `RMSE(high)−RMSE(low)` | `benchmarking/compare_and_report.py` |
| Balance score | `0.5·ṟmse+0.3·p̂arams+0.2·t̂rain` | `benchmarking/compare_and_report.py` |