# Data Sources

This document records every dataset used, where it was fetched from, and the exact
commands / steps used to obtain it. It is a living log — every new fetch is appended here.

## 1. UCI Air Quality Dataset (Vigo, Italy) — time-series

- **Source:** UCI Machine Learning Repository, Dataset #360 "Air Quality"
- **Page:** https://archive.ics.uci.edu/dataset/360/Air+Quality
- **Download URL:** https://archive.ics.uci.edu/static/public/360/air+quality.zip
- **Contents:** hourly air-quality readings (CO, NOx, NO2, O3, benzene, temperature,
  humidity, etc.) from a single road-side monitor in an Italian city (Vigo area), Mar 2004 – Feb 2005.
- **Rows:** 9,358 (missing values coded as `-200`)
- **Columns:** Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC),
  NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
- **License:** CC BY 4.0

### Steps used
```bash
# from repo root
curl -sL -o data/raw/airquality_uci.zip \
  "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
unzip -o data/raw/airquality_uci.zip -d data/raw
mv data/raw/AirQualityUCI.csv data/raw/airquality_uci.csv
# -> data/raw/airquality_uci.csv  (785,065 bytes)
```

## 2. OpenAQ — global air-quality measurements (worldwide)

- **Source:** OpenAQ open-data archive on AWS S3 (no API key required for the archive)
- **Docs:** https://docs.openaq.org/aws/about
- **Bucket root:** `https://openaq-data-archive.s3.amazonaws.com/`
- **Layout:** `records/csv.gz/locationid={id}/year={y}/month={m}/location-{id}-{YYYYMMDD}.csv.gz`
- Per daily file: `location_id,sensors_id,location,datetime,lat,lon,parameter,units,value`
  (parameters: pm25, pm10, no2, so2, co, o3, …) — global coverage with real lat/lon.

**Why worldwide:** the OpenAQ archive contains ~55k locations across the whole world
(verified by enumerating every `locationid=` prefix, see step below), so it is used both for
the forecasting pipeline and for the **global colored map**. Two scans (~17.5k samples)
yielded ~11.8k usable stations; a country-balanced subset of **282 stations across
110 countries** is downloaded locally for the map (all with real PM2.5/PM10/NO2/O3/SO2 data).

### Steps used
```bash
# (1) Enumerate every location id in the archive (S3 list, 1000/page)
#      -> 55,551 location ids collected in /tmp/opencode/locids.txt
for page in ...; do
  curl -sL "https://openaq-data-archive.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=records/csv.gz/&max-keys=1000[&continuation-token=...]"
done | extract <Prefix>records/csv.gz/locationid=NNN/</Prefix>

# (2) Geographic scan -> map each sampled location id to (lat, lon)
#      scripts/openaq_scan.py  (parallel HTTP, gzip-aware; 2500 + 15000 samples)
#      scan logs:            data/raw/openaq/scan.log, scan2.log

# (3) Merge + country assignment + country-balanced selection
#      scripts/openaq_select.py
#      -> data/raw/openaq/location_scan_clean.csv  (282 stations, 110 countries)

# (4) Pre-fetch local measurement files per station
#      scripts/openaq_fetch.py
#      -> data/raw/openaq/measurements/location-{id}.csv.gz  (282/282 ok)
```

## 3. India: city-level daily AQI (CPCB via Kaggle mirror)

- **Source:** "Air Quality Data in India (2015–2020)" (Kaggle, CC0), GitHub mirror of `city_day.csv`
- **Mirror URL:** https://raw.githubusercontent.com/adityarc19/aqi-india/master/city_day.csv
  (fallback: https://raw.githubusercontent.com/govindlabhala-cell/India-Air-Quality-EDA/main/dataset/city_day.csv)
- **Rows:** 29,531 daily records · 26 Indian cities · Jan 2015 – Jul 2020
- **Columns:** City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene,
  Xylene, AQI, AQI_Bucket

### Steps used
```bash
curl -sL -o data/raw/india_city_day.csv \
  "https://raw.githubusercontent.com/adityarc19/aqi-india/master/city_day.csv"
```

## 4. India: CPCB station-level hourly data (Delhi, 2024–2025)

- **Source:** opencity.in (open data portal mirroring CPCB)
- **URL:** https://data.opencity.in/dataset/0dc7b9fe-9fd4-46ee-a37e-88f0bd6f6362/resource/890f786d-fb9f-475e-8516-191bfa1b01ea/download/del-ito-cpcb-2024-25.csv
- **Columns:** Station ID, State, City, Station Name, Timestamp, PM2.5, PM10, NO, NO2, NOx,
  NH3, SO2, CO, Ozone, Benzene, Toluene, Xylene, AT, RH, WS, WD, RF
- ~9.4 MB of recent hourly Delhi monitoring-station data.

### Steps used
```bash
curl -sL -o data/raw/delhi_cpcb_2024_25.csv "[URL above]"
```

## 5. India: state boundaries (GeoJSON) — for choropleth map

- **Source:** `udit-001/india-maps-data` via jsDelivr CDN
- **URL:** https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@2884453/geojson/india.geojson
- **Contents:** polygon boundaries for all Indian states/UTs (GeoJSON).

### Steps used
```bash
curl -sL -o data/external/geojson/india_states.geojson "[URL above]"
```

## 6. India: city coordinates (lat/lon)

- **Source:** `recurze/IndianCities` (final_cities.csv)
- **URL:** https://raw.githubusercontent.com/recurze/IndianCities/master/final_cities.csv

### Steps used
```bash
curl -sL -o data/raw/india_cities.csv "[URL above]"
```

## 7. World: country boundaries (GeoJSON) — for worldwide choropleth map

- **Source:** Natural Earth via jsDelivr mirror (`geojson | world-atlas` style)
- **Placeholder URL (to be confirmed on fetch):**
  - https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json
  - https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson

---
_Last updated:_ step-wise with each fetch. See also `docs/FETCH_AND_RUN.md` for the end-to-end
run instructions.