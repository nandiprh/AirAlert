# Code Architecture & Source Layout

This document explains every file in the repository, how the pieces connect,
and the design decisions behind the AI-based air quality predictor.

Related docs:
- **docs/DATA_SOURCES.md** — where each dataset came from and how it was fetched.
- **docs/FETCH_AND_RUN.md** — the exact commands to reproduce every step.

---

## 1. High-Level Architecture

The project is organized as a **data → model → map** pipeline assembled from
six stages:

```
          FETCH                 CLEAN                  FEATURE ENGINEERING
  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────────┐
  │ scripts/        │    │ src/data/        │    │ src/train.py           │
  │  openaq_*.py    │▶   │  preprocess.py   │▶   │  preprocess_dataset()  │
  │  curl (docs)    │    │  openaq_loader.py│    │  (windows, scaling)    │
  └─────────────────┘    └──────────────────┘    └───────────┬───────────┘
                                                             │
     DETECT & CORRECT      TRAIN              SPLIT + SMOTE   │
  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────▼┐
  │ detect_issue()   │  │ run_training │  │ 80:20 holdout or  │
  │ overfit/underfit │◀─│ + 5 models   │◀─│ 10-fold CV + SMOTE │
  │ retrain w/ new hp│  │ src/models/* │  │ (train only)      │
  └──────────────────┘  └──────┬───────┘  └───────────────────┘
                               │ checkpoints (.pt) + reports/
                               ▼
  ┌──────────────── FORECAST & HOTSPOT MAPS ────────────────┐
  │ src/evaluate.py            test metrics + plots          │
  │ src/predict.py             per-city next-day AQI         │
  │ src/visualization/         world + India + city maps     │
  │   map_global.py            (plotly/folium HTML + PNG)    │
  │   map_cities.py                                          │
  │   location_table.py        hardcoded city → coordinates  │
  └───────────────────────────────────────────────────────────┘
```

### Every model shares one contract

All five architectures output **`(reg_out, cls_out)`** from
`forward(x)` where `x` is `(batch, seq_len, n_features)`:

- `reg_out` — a scalar AQI forecast (regression).
- `cls_out` — logits over AQI bucket classes (`None` if classification is
  disabled).

This single interface is what lets `train.py`, `evaluate.py` and
`predict.py` treat all five models interchangeably through the
`MODEL_REGISTRY`.

---

## 2. Source Layout

```
air-quality-hackathon/
├── README.md                          # project overview + quick start
├── requirements.txt                   # pinned Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                           # downloaded datasets (gitignored)
│   │   ├── airquality_uci.csv         #   UCI hourly, Vigo Italy
│   │   ├── india_city_day.csv         #   CPCB daily AQI, 26 cities
│   │   ├── delhi_cpcb_2024_25.csv     #   Delhi hourly w/ weather
│   │   ├── india_cities.csv           #   city lat/lon census table
│   │   └── openaq/                    #   worldwide stations + day files
│   ├── processed/                     # cleaned CSVs (gitignored)
│   │   ├── uci.csv
│   │   ├── india_city_day.csv
│   │   ├── delhi_cpcb.csv
│   │   ├── openaq_stations.csv
│   │   └── predictions/
│   │       └── city_predictions.csv   #   per-city next-day AQI
│   └── external/                      # curated, versioned
│       ├── geojson/                   #   india_states, world_countries
│       └── city_locations.csv         #   hardcoded code→lat/lon table
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocess.py              # cleaning + leakage checks
│   │   └── openaq_loader.py           # OpenAQ station enrichment
│   ├── models/
│   │   ├── __init__.py                # MODEL_REGISTRY + build_model()
│   │   ├── base.py                    # dataset, metrics, scaling, seeds
│   │   ├── lstm_cnn.py                # hybrid 1
│   │   ├── cnn_bilstm.py              # hybrid 2
│   │   ├── lstm_attention.py          # hybrid 3
│   │   ├── transformer_lstm.py        # hybrid 4
│   │   └── convlstm_attention.py      # hybrid 5
│   ├── train.py                       # training pipeline + CLI
│   ├── evaluate.py                    # checkpoint evaluation + plots
│   ├── predict.py                     # per-city next-day AQI forecast
│   └── visualization/
│       ├── map_global.py              # world + India choropleths
│       ├── map_cities.py              # city-wise prediction maps
│       └── location_table.py          # hardcoded coordinate resolution
│
├── scripts/                           # one-off data-fetch utilities
│   ├── openaq_scan.py                 # discover stations from S3 archive
│   ├── openaq_select.py               # country-balanced station selection
│   └── openaq_fetch.py                # pre-fetch day files to disk
│
├── models/                            # training artifacts (gitignored)
│   ├── checkpoints/                   # *.pt weights + metadata
│   └── reports/                       # metrics JSON, history/eval plots
│
├── output/                            # generated map HTML/PNG (gitignored)
│   ├── world_aqi_map.{html,png}
│   ├── india_aqi_map.{html,png}
│   ├── world_city_predictions.{html,png}
│   └── india_city_predictions.{html,png}
│
└── docs/
    ├── DATA_SOURCES.md
    ├── FETCH_AND_RUN.md
    └── CODE_ARCHITECTURE.md           # this document
```

---

## 3. Data Layer

### `src/data/preprocess.py` — cleaning + leakage checks

Loads raw CSVs and writes cleaned versions to `data/processed/`:

| Function | Purpose |
|----------|---------|
| `load_uci(path)` | UCI file is semicolon-separated with comma decimals and `-200` sentinels → parse into a numeric DataFrame. |
| `load_india(path)` | CPCB `city_day.csv`; reads numeric columns, exposes a clean index. |
| `load_delhi(path)` | Parse timestamps/columns for the Delhi CPCB feed. |
| `_median_impute(df, cols)` | Fill remaining NaN with per-series medians. |
| `_aqi_bucket(aqi)` | Recompute the **CPCB** AQI bucket string (Good/Moderate/Satisfactory/Poor/Very Poor/Severe) from numeric AQI. |
| `flag_duplicate_rows(df)` | Mark exact duplicate rows (leakage check #1). |
| `check_perfect_correlations(df)` | Report pollutant pairs with `|r| ≥ 0.99` (leakage check #2). |
| `run_leakage_checks(df)` | Runs both checks, logs a summary. |
| `save_processed(df, name)` | Write a cleaned CSV to `data/processed/`. |
| `run_pipeline()` | Orchestrates the full clean → impute → leakage-check → save flow. |

### `src/data/openaq_loader.py` — worldwide station enrichment

Turns the S3 scan output into a station table with country labels.

| Function | Purpose |
|----------|---------|
| `load_stations(path)` | Read the initial `location_scan` CSV of discovered stations. |
| `build_s3_url(station, prefix)` | Build the S3 key for a station's daily measurement file. |
| `fetch_station_data(station)` | Download + parse one measurement file (handles the `measurand` vs `parameter` naming used by older archives). |
| `fetch_location_summary(station)` | Reduce a station's measurements to a PM2.5/PM10/NO2 summary row. |
| `assign_country(stations, world)` | **Point-in-polygon** via geopandas: assign ISO3 + country name for each station. |
| `main(geojson_path)` | Runs load → summarize → assign countries → save `openaq_stations.csv`. |

### `scripts/openaq_*.py` — one-off bulk fetch

- `openaq_scan.py` — paginate the public S3 archive
  (`https://openaq-data-archive.s3.amazonaws.com/`, no API key) to enumerate
  every `location_id`, then probe a sample of stations in parallel to learn
  their coordinates and first available file.
- `openaq_select.py` — merge scan outputs, drop any without coordinates,
  assign countries, and pick up to N stations per country so the world map
  has balanced coverage (final: **282 stations / 110 countries**).
- `openaq_fetch.py` — pre-fetch one measurement file per selected station to
  `data/raw/openaq/measurements/location-{id}.csv.gz`. Only `urllib` with a
  User-Agent works here; `requests` is blocked by the bucket.

---

## 4. Model Layer (`src/models/`)

### `base.py` — shared machinery

- `SequenceDataset` — PyTorch `Dataset` of sliding windows; returns
  `(x, y_reg)` and/or `(x, y_cls)` depending on enabled tasks.
- `compute_metrics` — RMSE / MAE / R² for regression; accuracy, precision,
  recall, F1 and a confusion matrix for classification.
- `fit_normalizer` / `apply_normalizer` — fit a StandardScaler/MinMaxScaler
  that operates on both 2D `(n, f)` and 3D `(n, seq, f)` arrays (the model
  sees 3D windows, so these collapse the window axis, fit, then reshape back).
- `inverse_normalize_target` — turn scaled predictions back into real AQI
  units for reporting and forecasting.
- `deterministic_seed` — seeds `random`, `numpy`, `torch` (and CUDA).
- `r2_score_py` — manual R² to avoid sklearn warning on small sets.

### `__init__.py` — the registry

```python
MODEL_REGISTRY = {
    "lstm_cnn":        LSTMCNN,
    "cnn_bilstm":      CNNBiLSTM,
    "lstm_attention":  LSTMAttention,
    "transformer_lstm": TransformerLSTM,
    "convlstm_attention": ConvLSTMAttention,
}
```

`build_model(...)` is the single factory. Its only special case is the
Transformer, which uses `d_model`/`nhead`/`lstm_hidden` instead of
`hidden_size` and needs `seq_len` for its positional encoding. Every other
model receives `input_size, output_size, num_classes, dropout,
hidden_size, num_layers`.

### The five hybrid architectures

All take `x (B, T, F)` and return `(reg, cls)` where `cls` may be `None`.

| # | Model | Encoding pipeline | Temporal attention |
|---|-------|-------------------|--------------------|
| 1 | **LSTMCNN** (`lstm_cnn.py`) | LSTM extracts per-timestep features → 1D CNN (2× Conv1d+BatchNorm+ReLU) reads local patterns across time → adaptive max-pool | none |
| 2 | **CNNBiLSTM** (`cnn_bilstm.py`) | 2× Conv1d+BatchNorm** first → **bidirectional LSTM** (hidden split across fwd/bwd) → concat final hidden states | none |
| 3 | **LSTMAttention** (`lstm_attention.py`) | LSTM encoder → **Bahdanau additive attention** scores each hidden state, softmax → weighted context | yes (additive) |
| 4 | **TransformerLSTM** (`transformer_lstm.py`) | Sinusoidal positional encoding → **Transformer encoder** (multi-head self-attention, GeLU FFN) → LSTM refinement → last hidden state | yes (global) |
| 5 | **ConvLSTMAttention** (`convlstm_attention.py`) | 3× stacked Conv1d multi-scale features → LSTM → **scaled dot-product self-attention** over timesteps → max-pooled context | yes (dot-product) |

Each ends with a dropout + Linear head feeding **two heads**: `fc_reg`
(single AQI value) and optional `fc_cls` (logits over `num_classes` buckets).

---

## 5. Training Pipeline (`src/train.py`)

The largest module. Key flow:

```
parse_args() ──▶ preprocess_dataset() ──▶ apply_smote() ──▶ run_training()
                                                        │
                          detect_issue() ◀── history ────┤
                          (overfit/underfit)             │
                          retrain with new hyperparams   │
                                                         ▼
                                                   checkpoint ✓
```

| Piece | Line | What it does |
|-------|------|--------------|
| `UCI_FEATURES` / `INDIA_FEATURES` | 62/79 | Feature column sets for each dataset. |
| `load_uci_data` / `load_india_data` | 112/146 | Read processed CSV (fall back to raw); India optionally filters a single `city`; the global features/targets for India are PM2.5…O₃ pollutants → `AQI`, class label `AQI_Bucket`. |
| `preprocess_dataset` | 187 | The central piece: builds **leakage-free windows**. |
| `apply_smote` | 329 | Oversamples minority AQI buckets **on the training windows only**; `reg_target` is averaged for synthetic samples. |
| `task_enabled` / `build_loss` | 374/378 | Decides regression/classification/both and the combined MSELoss + CrossEntropyLoss. |
| `train_one_epoch` | 394 | Standard PyTorch training loop. |
| `evaluate_loader` | 411 | Validation/test loop collecting unscaled predictions. |
| `Config(dict)` | 438 | Dict subclass with attribute access; holds all hyperparameters. |
| `detect_issue` | 451 | Overfit rule → raise dropout; underfit rule → raise `hidden_size`/epochs. |
| `run_training` | 501 | Full loop: build model → Adam + ReduceLROnPlateau → early stop → restore best weights → auto-correction retrain → save `.pt` checkpoint → history plot. |
| `run_cv` | 679 | 10-fold TimeSeriesSplit alternative. |
| `parse_args` / `main` | 761/799 | CLI entry point (see below). |

### The 80:20 split (no temporal leakage)

`preprocess_dataset` splits the **raw series** *before* windowing so a window
never straddles a partition boundary:

1. `n_test = n - int(n*train_frac)` → last 20% is test.
2. If `val_frac > 0` → that many rows are taken from the tail as validation.
3. If `val_frac == 0` (default) → a small monitor set is **carved from the
   end of the training block** (≈10% of the 80%), so early stopping and
   overfit detection keep a validation set without touching test.
4. Windows are then built per-partition, and the scaler is **fit on the
   training partition only** (type-2 leakage prevention).

### CLI

```
python src/train.py --dataset uci|india --model [5 hybrids|all]
                    --epochs 50 --seq_len 24 --batch_size 64 --lr 1e-3
                    --task both|regression|classification
                    --split_mode holdout|cv10
                    --train-frac 0.8 --val-frac 0.0
                    --city <name> --seed 42 [--no_smote] [--quiet]
```

Artifacts: checkpoints in `models/checkpoints/{model}_{dataset}_{task}_s{N}.pt`,
per-run metrics JSON + history PNG in `models/reports/`.

---

## 6. Evaluation (`src/evaluate.py`)

- `find_checkpoint` — auto-locate a checkpoint by naming convention.
- `build_model_from_checkpoint` — reconstitute the exact architecture from
  the metadata/config saved inside the `.pt` file.
- `evaluate` — re-runs the SAME `preprocess_dataset` so test windows match
  training exactly, then reports RMSE/MAE/R² and accuracy/F1/confusion
  matrix; writes `{name}_evaluation.json`; plots actual-vs-predicted and the
  confusion matrix.

```
python src/evaluate.py --model lstm_cnn --dataset uci --seq_len 24
python src/evaluate.py --checkpoint models/checkpoints/...pt
```

---

## 7. Forecasting (`src/predict.py`)

Trains one model **per Indian city** and predicts the next-day AQI.

- `list_cities` — the 26 cities in `city_day` data.
- `last_forecast_window` — rebuilds the *exact* cleaned tail of the series
  (same NaN filtering as `preprocess_dataset`) and returns the last
  `seq_len` rows + the last observed date.
- `run_city` — per city:
  1. `preprocess_dataset(dataset="india", city=city, train_frac=0.8, val_frac=0.0)`
  2. `run_training` (SMOTE on train only, overfit/underfit auto-correction)
  3. scale the final window with the training `feature_scaler`
  4. `model(window)` → inverse-scale `reg_scaler` → predicted AQI
  5. map to CPCB bucket (`bucket_of`)
- `forecast_city` — optional ensemble averaging when `--model all`.
- `main` — resolves coordinates through `location_table.resolve_many()`,
  writes `data/processed/predictions/city_predictions.csv` and a summary JSON.

Cities whose series are too sparse after NaN cleaning raise
"Not enough samples" and are skipped with a log line (e.g. Ahmedabad).

```
python src/predict.py --model lstm_cnn --epochs 30 --seq_len 7
python src/predict.py --city Delhi --model all
```

---

## 8. Visualization (`src/visualization/`)

### `map_global.py` — worldwide mosaic + India state map

Constants: the AQI bucket grid `AQI_BUCKETS`/`AQI_COLORS`, the EPA
PM2.5→AQI breakpoints, GeoJSON paths and the OpenAQ S3 base URL.

| Function | Purpose |
|----------|---------|
| `bucket_index` / `bucket_color` / `pm25_to_aqi` | Map a value to a color/label. |
| `load_world_countries` / `load_india_states` | Load GeoJSON; India is dissolved to 36 state polygons. |
| `read_pm25_frame` | Fast mean-of-day PM2.5 (prefers pm25, falls back to pm10/no2/o3/so2). |
| `fetch_station_pm25` | Local first (prefetched dir) then S3, else `station_placeholder_pm25` (seeded random). |
| `load_station_data` | Build the 282-station table, assigning each a representative pollutant. |
| `assign_countries` / `build_country_aggregation` | Point-in-polygon country assignment + per-country mean AQI. |
| `save_world_plotly` | **Plotly choropleth** — countries colored by mean AQI, station dots with tooltips → `world_aqi_map.html`. |
| `save_world_png_writer` / `..._matplotlib` | Static PNG export (kaleido, else matplotlib fallback). |
| `build_india_city_state` / `state_standings` / `save_india_folium` | **Folium** India map — states filled by avg AQI, city markers → `india_aqi_map.html`. |

### `location_table.py` — hardcoded city resolution

Motivation: fuzzy-matching city names against the national `india_cities.csv`
produced wrong results (e.g. "Visakhapatnam" → Bihar). Instead, city
coordinates come from `data/external/city_locations.csv`, a small curated
table:

```
code,city,country,state,lat,lon
DEL,Delhi,India,Delhi,28.6139,77.2090
BLR,Bengaluru,India,Karnataka,12.9716,77.5946
...
```

- `load_location_table()` → normalized-name → entry map.
- `resolve(city)` → lookup order: **table** → `FALLBACK_CITY_COORDS` →
  exact-name match in `india_cities.csv` → `(None, ...)`.
- `resolve_many(cities)` → dict for a whole batch.

To add a new station/city, append one row — no code changes needed.

### `map_cities.py` — city-wise prediction maps

Consumes `city_predictions.csv`, re-resolves coordinates through the
location table (authoritative), and emits four files:

| Output | Renderer | Content |
|--------|----------|---------|
| `india_city_predictions.html` | folium | States colored by **predicted** mean AQI + colored city markers + permanent name labels + legend |
| `india_city_predictions.png` | matplotlib | Choropleth + annotated markers |
| `world_city_predictions.html` | plotly | World choropleth + city markers with hover (name/AQI/bucket/date) |
| `world_city_predictions.png` | kaleido | Static version of the same |

Tiles come from **CartoDB positron** (the default OpenStreetMap tiles return
HTTP 403 and blank the page).

---

## 9. Data Flow (end to end)

```
1. curl/scripts  ─▶ data/raw/                       (documented in DATA_SOURCES.md)
2. preprocess.py ─▶ data/processed/*.csv            (cleaned, imputed, leakage-checked)
3. openaq_loader ─▶ data/processed/openaq_stations.csv
4. train.py      ─▶ models/checkpoints/*.pt         (80:20 + SMOTE, 5 hybrids)
5. evaluate.py   ─▶ models/reports/*_evaluation.json
6. predict.py    ─▶ data/processed/predictions/city_predictions.csv
7. map_global.py ─▶ output/{world,india}_aqi_map.{html,png}
8. map_cities.py ─▶ output/{world,india}_city_predictions.{html,png}
```

Input feature contract (India): `PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3`
→ regression target `AQI`, classification target `AQI_Bucket`.
(UCI: same 9 gas/weather features → `CO(GT)` regression, optional qcut buckets.)

---

## 10. Reproducibility Notes

- `deterministic_seed` + `Config.seed` make training repeatable; auto-correction
  rounds reseed with `seed + round` so each retrain is deterministic.
- Heavy artifacts (`data/raw/*measurements`, `models/`, `output/`) are
  `.gitignore`d and regenerable from `docs/FETCH_AND_RUN.md`.
- OpenAQ S3 archive needs `urllib` with a User-Agent (no API key); `requests`
  is blocked. Older archive files use the `measurand` column name.