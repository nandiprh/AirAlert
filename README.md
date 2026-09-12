# AI-Based Air Quality Predictor & Pollution Hotspot Mapper

Urban air pollution changes based on traffic, weather, industrial activity and seasonal
patterns. AQI monitoring stations provide measurements only at specific locations.
This system **predicts future air quality** and **identifies potential pollution hotspots**
using historical pollutant and meteorological data — with colored maps showing affected
regions worldwide.

---

## Hybrid Deep Learning Models

Five hybrid architectures are implemented in `src/models/` (PyTorch):

| # | Model | File | Description |
|---|-------|------|-------------|
| 1 | LSTM + CNN | `lstm_cnn.py` | LSTM extracts temporal features → CNN captures local pollutant patterns |
| 2 | CNN + BiLSTM | `cnn_bilstm.py` | CNN for local patterns → BiLSTM for bidirectional temporal context |
| 3 | LSTM + Attention | `lstm_attention.py` | LSTM with Bahdanau self-attention over hidden states |
| 4 | Transformer + LSTM | `transformer_lstm.py` | Transformer encoder for global attention → LSTM for sequential refinement |
| 5 | ConvLSTM + Attention | `convlstm_attention.py` | Conv1D stack → LSTM → scaled dot-product temporal attention |

---

## Datasets

| Dataset | Source | Size | Coverage |
|---------|--------|------|----------|
| UCI Air Quality | [UCI ML Repository #360](https://archive.ics.uci.edu/dataset/360/Air+Quality) | 9,358 hourly rows | Single station, Vigo Italy, 2004–2005 |
| India City AQI | [Kaggle CC0 mirror](https://github.com/adityarc19/aqi-india) | 29,531 daily rows | 26 Indian cities, 2015–2020 |
| Delhi CPCB | [opencity.in](https://data.opencity.in) | ~68,984 hourly rows | Delhi stations with weather, 2024–25 |
| OpenAQ Worldwide | [AWS S3 Archive](https://docs.openaq.org/aws/about) | 282 stations · 110 countries | Global PM2.5/PM10/NO₂ measurements |
| India States GeoJSON | [udit-001/india-maps-data](https://github.com/udit-001/india-maps-data) | 36 state polygons | Choropleth boundaries |
| World Countries GeoJSON | [Natural Earth](https://github.com/nvkelso/natural-earth-vector) | 177 countries | Worldwide choropleth |
| Indian Cities Coords | [recurze/IndianCities](https://github.com/recurze/IndianCities) | City lat/lon | For hotspot mapping |

---

## Colored Region Maps (Output)

Run `python src/visualization/map_global.py` to generate:

### World Map
`output/world_aqi_map.html` — interactive Plotly choropleth, countries colored by mean PM2.5
(red–yellow–green AQI scale), OpenAQ station markers with tooltips.

### India Map
`output/india_aqi_map.html` — Folium map, Indian states colored by average AQI,
city-level markers with AQI labels and legend.

### City-Wise AQI Prediction Maps
`src/predict.py` trains a model per Indian city on an **80:20 temporal split**
(SMOTE on training only) and forecasts the **next-day AQI**. The predictions are
rendered by `src/visualization/map_cities.py`, with each city placed at a real
coordinate resolved from the hardcoded lookup table
`data/external/city_locations.csv` (e.g. `DEL` → Delhi, `BLR` → Bengaluru).

- `output/india_city_predictions.html` — Indian states colored by predicted mean
  AQI + city markers colored by predicted bucket (popup: city, AQI, bucket, dates).
- `output/world_city_predictions.html` — world choropleth + predicted city markers.

You can extend `data/external/city_locations.csv` with any station/city code row
(`code,city,country,state,lat,lon`) and the map will resolve it automatically.

| AQI Range | Color | Category |
|-----------|-------|----------|
| 0–50 | 🟩 Green | Good |
| 51–100 | 🟨 Yellow | Moderate |
| 101–150 | 🟧 Orange | Unhealthy for Sensitive Groups |
| 151–200 | 🟥 Red | Poor (Very Unhealthy) |
| 201–300+ | 🟫 Dark Red | Very Poor / Severe |

---

## Project Structure

```
air-quality-hackathon/
├── data/
│   ├── raw/                    ← downloaded raw datasets
│   │   ├── airquality_uci.csv
│   │   ├── india_city_day.csv
│   │   ├── delhi_cpcb_2024_25.csv
│   │   ├── india_cities.csv
│   │   └── openaq/
│   │       ├── location_scan.csv
│   │       ├── location_scan_clean.csv   ← 282 stations, 110 countries
│   │       ├── scan.log, scan2.log
│   │       └── measurements/             ← local OpenAQ day-files
│   ├── processed/              ← cleaned & datetime-indexed
│   │   ├── uci.csv
│   │   ├── india_city_day.csv
│   │   ├── delhi_cpcb.csv
│   │   └── openaq_stations.csv
│   └── external/
│       ├── geojson/            ← map boundaries
│       │   ├── india_states.geojson
│       │   └── world_countries.geojson
│       └── city_locations.csv  ← hardcoded station/city → lat-lon table
├── src/
│   ├── data/
│   │   ├── preprocess.py       ← full preprocessing pipeline
│   │   └── openaq_loader.py    ← OpenAQ station enrichment
│   ├── models/
│   │   ├── base.py             ← shared dataset + metrics
│   │   ├── lstm_cnn.py
│   │   ├── cnn_bilstm.py
│   │   ├── lstm_attention.py
│   │   ├── transformer_lstm.py
│   │   └── convlstm_attention.py
│   ├── train.py                ← training with SMOTE, CV, overfit correction
│   ├── evaluate.py             ← test evaluation + plots
│   ├── predict.py              ← per-city next-day AQI forecast
│   └── visualization/
│       ├── map_global.py       ← world + India colored maps
│       ├── map_cities.py       ← city-wise prediction maps
│       └── location_table.py   ← hardcoded city → coordinate resolution
├── output/                     ← generated map HTML/PNG files
├── models/
│   ├── checkpoints/            ← saved .pt model weights
│   └── reports/                ← metrics + evaluation plots
├── docs/
│   ├── DATA_SOURCES.md         ← full dataset provenance log
│   ├── FETCH_AND_RUN.md        ← step-by-step execution guide
│   └── CODE_ARCHITECTURE.md    ← file-by-file code explanation + layout
│   └── MODEL_COMPARISON.md     ← 5-model accuracy vs efficiency benchmark
│   └── MODEL_MATH_AND_RUN.md   ← how to run + model math + code walkthrough
├── scripts/
│   ├── openaq_scan.py          ← worldwide OpenAQ station discovery (S3 scan)
│   ├── openaq_select.py        ← merge scans + country-balanced selection
│   └── openaq_fetch.py         ← pre-fetch measurement files to local disk
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess data
python -m src.data.preprocess
python -m src.data.openaq_loader

# 3. Train a model
python src/train.py --dataset uci --model lstm_cnn --epochs 50 --seq_len 24

# 4. Evaluate
python src/evaluate.py --checkpoint models/checkpoints/lstm_cnn.pt --dataset uci

# 5. Generate colored region maps
python src/visualization/map_global.py
# → output/world_aqi_map.html  (world choropleth, countries colored by AQI)
# → output/india_aqi_map.html  (India states colored by AQI)

# 6. City-wise next-day AQI prediction + maps
python src/predict.py --model lstm_cnn --epochs 30 --seq_len 7
python src/visualization/map_cities.py
# → output/india_city_predictions.html  (states + city markers by predicted AQI)
# → output/world_city_predictions.html  (predicted city markers on world map)
```

Full execution details are documented in **[docs/FETCH_AND_RUN.md](docs/FETCH_AND_RUN.md)**;
a file-by-file explanation of the code and source layout is in
**[docs/CODE_ARCHITECTURE.md](docs/CODE_ARCHITECTURE.md)**.

---

## Pipeline Flow

```
Load CSV Dataset
↓
Data Preprocessing  (src/data/preprocess.py)
↓
Data Leakage Checks  (duplicates, suspicious correlations)
↓
Train / Test Split  (80:20)  or  10-fold Cross Validation
↓
SMOTE Algorithm  (balancing on TRAINING data only)
↓
Model Training  (one of 5 hybrid architectures)
↓
Overfitting / Underfitting Detection  (train vs val loss)
↓
Auto-Correction  (dropout ↑, LR scheduler, early stopping) → Retrain
↓
Final Test Evaluation  (src/evaluate.py)
```

---

## Output

**AQI Forecast → Pollutant Forecast → Hotspot Map → Public Warning**

The colored maps serve as the hotspot visualization layer:
- World choropleth highlights countries with elevated pollution levels
- India state-level map pinpoints hotspots with city-level AQI markers
- Interactive HTML maps allow hovering for details; PNG exports for reports
