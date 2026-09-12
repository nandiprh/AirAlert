# Fetch and Run — Step-by-Step Guide

This document records every command used to fetch data, install dependencies, and
execute programs in this project.

---

## 1. Environment Setup

```bash
# Python 3.14.6 is pre-installed on the system.

# Create a virtual environment (PEP 668 externally-managed system)
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Key packages (CPU-only PyTorch):
- `torch==2.14.0+cpu` — all 5 hybrid DL models
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `imbalanced-learn` — data/ML
- `matplotlib`, `plotly`, `folium`, `kaleido` — static + interactive maps + PNG export
- `geopandas`, `shapely`, `pyproj` — geospatial point-in-polygon

---

## 2. Data Fetching

All fetch commands were recorded here in the order they were executed.

### 2.1 UCI Air Quality Dataset (Vigo, Italy)

Source: https://archive.ics.uci.edu/dataset/360/Air+Quality

```bash
curl -sL -o data/raw/airquality_uci.zip \
  "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
unzip -o data/raw/airquality_uci.zip -d data/raw
mv data/raw/AirQualityUCI.csv data/raw/airquality_uci.csv
```

Result: `data/raw/airquality_uci.csv` (785 KB, 9,358 hourly rows, 15 features)

### 2.2 India city-level daily AQI (CPCB via Kaggle mirror)

Source: https://github.com/adityarc19/aqi-india (Kaggle CC0 dataset mirrored)

```bash
curl -sL -o data/raw/india_city_day.csv \
  "https://raw.githubusercontent.com/adityarc19/aqi-india/master/city_day.csv"
```

Result: `data/raw/india_city_day.csv` (2.57 MB, 29,531 daily rows, 26 Indian cities)

### 2.3 Delhi CPCB station hourly data (opencity.in)

Source: https://data.opencity.in (CPCB raw feed mirror)

```bash
curl -sL -o data/raw/delhi_cpcb_2024_25.csv \
  "https://data.opencity.in/dataset/0dc7b9fe-9fd4-46ee-a37e-88f0bd6f6362/\
resource/890f786d-fb9f-475e-8516-191bfa1b01ea/download/del-ito-cpcb-2024-25.csv"
```

Result: `data/raw/delhi_cpcb_2024_25.csv` (9.4 MB, ~68,984 hourly rows, 29 columns incl. weather)

### 2.4 OpenAQ worldwide station discovery (S3 archive scan)

Source: https://openaq-data-archive.s3.amazonaws.com/ (Open Data on AWS, no key required)

```bash
# Step A: enumerate all location_id prefixes in the archive (S3 list-type=2, paginated)
# This produced /tmp/opencode/locids.txt with 55,551 unique location ids.

# Step B: geographic scan script (parallel, gzip-aware)
python scripts/openaq_scan.py \
  --ids /tmp/opencode/locids.txt \
  --sample 2500 \
  --out data/raw/openaq/location_scan.csv

# Step C (second, larger scan): 15,000 more location ids -> /tmp/opencode/scan_run2.csv
#   [scan] Wrote 15000 valid rows (14999 with geo)

# Step D: merge scans, assign countries (point-in-polygon), pick up to 3 stations
# per country for balanced worldwide coverage
python scripts/openaq_select.py \
  --scans data/raw/openaq/location_scan_clean.csv /tmp/opencode/scan_run2.csv \
  --world data/external/geojson/world_countries.geojson \
  --max-per-country 3 \
  --out data/raw/openaq/location_scan_clean.csv
#   -> 282 stations across 110 countries

# Step E: pre-fetch one daily measurement file per selected station to local disk
python scripts/openaq_fetch.py \
  --stations data/raw/openaq/location_scan_clean.csv \
  --outdir data/raw/openaq/measurements --max-workers 16
#   -> 282/282 files downloaded into data/raw/openaq/measurements/
```

Result: **282 stations in 110 countries**, each with a local daily measurement file
(PM2.5 / PM10 / NO2 / O3 / SO2 values + coordinates) used directly by the world map.

### 2.5 Indian state boundaries (GeoJSON)

Source: https://github.com/udit-001/india-maps-data via jsDelivr CDN

```bash
curl -sL -o data/external/geojson/india_states.geojson \
  "https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@2884453/geojson/india.geojson"
```

### 2.6 Indian cities coordinates

Source: https://github.com/recurze/IndianCities

```bash
curl -sL -o data/raw/india_cities.csv \
  "https://raw.githubusercontent.com/recurze/IndianCities/master/final_cities.csv"
```

### 2.7 World country boundaries (Natural Earth)

Source: https://github.com/nvkelso/natural-earth-vector

```bash
curl -sL -o data/external/geojson/world_countries.geojson \
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
```

---

## 3. Data Preprocessing

```bash
source .venv/bin/activate

# Run the full preprocessing pipeline (UCI + India city_day + Delhi CPCB + OpenAQ)
python -m src.data.preprocess

# Assign countries to worldwide OpenAQ stations (point-in-polygon)
python -m src.data.openaq_loader
```

Outputs in `data/processed/`:
- `uci.csv` (9,357 rows — cleaned UCI hourly data)
- `india_city_day.csv` (29,531 rows — imputed, AQI_Bucket recomputed)
- `delhi_cpcb.csv` (68,984 rows — parsed timestamps, station-imputed)
- `openaq_stations.csv` (107 rows — worldwide stations with country ISO codes)

---

## 4. Training Models

All five hybrid architectures are in `src/models/`. Training script: `src/train.py`.

```bash
source .venv/bin/activate

# Train LSTM+CNN on UCI hourly data (regression + classification)
python src/train.py --dataset uci --model lstm_cnn --epochs 50 --seq_len 24

# Train CNN+BiLSTM on India city AQI data
python src/train.py --dataset india --model cnn_bilstm --epochs 50 --seq_len 7

# Train all five models on UCI (10-fold CV)
python src/train.py --dataset uci --model all --cv10 --epochs 30 --seq_len 24

# Specific model options:
#   lstm_cnn | cnn_bilstm | lstm_attention | transformer_lstm | convlstm_attention
```

Hyperparameters (CLI args):
- `--dataset`: `uci` or `india`
- `--model`: `lstm_cnn`, `cnn_bilstm`, `lstm_attention`, `transformer_lstm`, `convlstm_attention`, `all`
- `--epochs`, `--batch_size`, `--lr`, `--seq_len`, `--hidden_size`, `--dropout`
- `--cv10`: enable 10-fold cross-validation
- `--smote`: enable SMOTE balancing (on training split only)
- `--train-frac`: train/test split ratio (default `0.8` = 80:20); SMOTE is
  applied to the 80% training split only
- `--val-frac`: explicit validation fraction (default `0.0`, which carves a
  small validation tail from the END of the training block for early stopping)

Checkpoints saved to: `models/checkpoints/`
Metrics saved to: `models/reports/<model_name>_metrics.json`
Training history plots: `models/reports/<model_name>_history.png`

---

## 5. Evaluation

```bash
# Evaluate a trained checkpoint
python src/evaluate.py --checkpoint models/checkpoints/<model_name>.pt --dataset uci

# Outputs:
#   - Confusion matrix plot: models/reports/<model_name>_confusion.png
#   - Actual vs predicted plot: models/reports/<model_name>_actual_vs_pred.png
#   - Metrics printed to stdout
```

---

## 6. Generating Colored Maps (Visualization)

```bash
source .venv/bin/activate

python src/visualization/map_global.py
```

Outputs in `output/`:
- `world_aqi_map.html` — interactive Plotly choropleth (countries colored by mean PM2.5/AQI)
- `world_aqi_map.png` — static matplotlib fallback of the same
- `india_aqi_map.html` — interactive Folium map (Indian states colored by average AQI)
- `india_aqi_map.png` — static matplotlib fallback

Color scale (both maps):
| Color  | AQI Range | Meaning                          |
|--------|-----------|----------------------------------|
| Green  | 0–50      | Good                             |
| Yellow | 51–100    | Moderate                         |
| Orange | 101–150   | Unhealthy for Sensitive Groups   |
| Red    | 151–200   | Poor                             |
| Dark Red | 201+     | Very Poor / Severe              |

---

## 7. City-Wise Prediction (next-day AQI)

Per-city forecast + city-level predicted-AQI maps. Each city is rendered at a
coordinate from the **hardcoded location table** `data/external/city_locations.csv`
(`code,city,country,state,lat,lon`), so a predicted "Delhi" always lands on Delhi
(no fuzzy matching). Fallbacks: `FALLBACK_CITY_COORDS`, then exact match in
`data/raw/india_cities.csv`.

```bash
source .venv/bin/activate

# Train one model per Indian city on 80:20 temporal split (SMOTE on train only)
# and forecast next-day AQI. Defaults: 26 cities, lstm_cnn, 30 epochs, seq_len 7.
python src/predict.py --model lstm_cnn --epochs 30 --seq_len 7

# Options:
#   --city Delhi        forecast only Delhi
#   --model all         ensemble average across all 5 hybrid models
#   --max-cities N      limit city count (dev runs)
#   --no_smote          disable SMOTE balancing

# Draw the city-wise maps
python src/visualization/map_cities.py
```

Bucket labels come from `src/visualization/aqi_spec.py` (CPCB-style: **0–50 Good,
51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401+ Severe**):
`predict.py` writes those labels into the CSV/summary, and every map colours by them.

> Note: earlier builds mapped 101–150 → Moderate etc. (surface-level values shifted
> one band down, so most cities looked "very poor" in the summary). Values were
> always moderate (mean ≈ 90), but the **labels were centred one band too low** and
> the maps had no legend. `aqi_spec.py` fixes the bands and both maps now include a
> **"What each color means"** legend with the AQI range + health advice.

Data written:
- `data/processed/predictions/city_predictions.csv` — city, state, lat, lon,
  forecast_date, predicted_aqi, bucket
- `models/reports/city_predictions_summary.json` — per-city flagged auto-corrections

Maps in `output/`:
- `india_city_predictions.html` / `.png` — states colored by predicted mean AQI +
  city markers colored by predicted bucket (hover shows city, AQI, bucket, dates)
- `world_city_predictions.html` / `.png` — world choropleth + predicted city markers
  **plus the curated global cities** (Sydney, London, Paris, New York, LA, SF, …)
  coloured by nearest OpenAQ station / country-mean AQI. Both maps show a
  color → AQI-range → health legend.

Cities with insufficient valid history (e.g. Ahmedabad, Ernakulam, Jorapokhar,
Lucknow drop almost all rows during NaN cleaning) are skipped with a log message.

### Station number → name lookup (world map hover)

`world_aqi_map.html` shows OpenAQ stations by **name** instead of a raw location id.
`data/external/station_city_lookup.csv` maps `location_id → display_name`, built by
`src/visualization/location_table.py`:

```bash
# Regenerate after re-scanning stations or adding a curated city coordinates
python -m src.visualization.location_table data/raw/openaq/location_scan_clean.csv
```

`display_name` resolves to: the nearest curated city within 150 km
("City (~X km)"), else the station `location` name from the cached measurement file,
else "Station {id}".

---

## 8. Pipeline Overview

```
data/raw/                     ← raw downloaded files
  └── data/processed/         ← cleaned CSVs (datetime-indexed)
        ├── uci.csv
        ├── india_city_day.csv
        ├── delhi_cpcb.csv
        └── openaq_stations.csv

src/data/preprocess.py        ← loads + cleans all CSVs
src/data/openaq_loader.py     ← OpenAQ station enrichment + country assignment
src/models/*.py               ← 5 hybrid DL architectures (PyTorch)
src/train.py                  ← train loop with SMOTE, CV, overfit detection
src/evaluate.py               ← test evaluation + plots
src/predict.py                ← per-city next-day AQI forecast
src/visualization/map_global.py ← colored world + India maps
src/visualization/map_cities.py ← city-wise prediction maps
src/visualization/location_table.py ← hardcoded city → coordinate resolution
src/visualization/aqi_spec.py ← shared AQI buckets/colors + health legend

output/                       ← map HTML + PNG files
models/checkpoints/           ← saved .pt model files
models/reports/               ← metrics JSON + evaluation plots
```

---

_Last updated: data fetching and execution complete, maps confirmed rendering._
