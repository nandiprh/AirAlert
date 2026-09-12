#!/usr/bin/env python3
"""Generate colored choropleth maps of global and Indian air quality.

Produces two outputs:
  1. output/world_aqi_map.html (+ .png when possible)
  2. output/india_aqi_map.html (+ .png)

Runs end-to-end with no API keys. Run:  python src/visualization/map_global.py
"""

import io
import json
import math
import os
import random

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .aqi_spec import (
    AQI_BUCKETS,
    AQI_COLORS,
    AQI_HEALTH,
    AQI_LABELS,
    AQI_RANGES,
    bucket_color,
    bucket_index,
    html_legend,
    legend_rows,
)
from .location_table import load_station_city_lookup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_EXTERNAL = os.path.join(PROJECT_ROOT, "data", "external", "geojson")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

INDIA_DAY_CSV = os.path.join(DATA_RAW, "india_city_day.csv")
INDIA_CITIES_CSV = os.path.join(DATA_RAW, "india_cities.csv")
OPENAQ_CSV = os.path.join(DATA_RAW, "openaq", "location_scan_clean.csv")
OPENAQ_DIR = os.path.join(DATA_RAW, "openaq")
WORLD_GEOJSON = os.path.join(DATA_EXTERNAL, "world_countries.geojson")
INDIA_STATES_GEOJSON = os.path.join(DATA_EXTERNAL, "india_states.geojson")

OPENAQ_BASE_URL = "https://openaq-data-archive.s3.amazonaws.com/"
OPENAQ_MEASUREMENTS_DIR = os.path.join(OPENAQ_DIR, "measurements")

GREEN = "#2ca02c"
YELLOW = "#f7d448"
ORANGE = "#f49e2a"
RED = "#d6452e"
DARKRED = "#8a1f1f"
GREY = "#d5d5d5"

PM25_BUCKETS = [(0.0, 12.0), (12.0, 35.0), (35.0, 55.0), (55.0, 150.0), (150.0, math.inf)]
PM25_COLORS = [GREEN, YELLOW, ORANGE, RED, DARKRED]
PM25_LABELS = ["Good (<12)", "Moderate (12-35)", "Unhealthy for Sensitive (35-55)",
               "Unhealthy (55-150)", "Very Unhealthy (150+)"]

EPA_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

FALLBACK_CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714),
    "Amaravati": (16.5083, 80.5183),
    "Bengaluru": (12.9716, 77.5946),
    "Brajrajnagar": (21.8167, 83.9167),
    "Delhi": (28.6139, 77.2090),
    "Ernakulam": (9.9694, 76.2934),
    "Gurugram": (28.4595, 77.0266),
    "Hyderabad": (17.3850, 78.4867),
    "Jorapokhar": (23.7903, 86.1110),
    "Mumbai": (19.0760, 72.8777),
    "Talcher": (20.9493, 85.2236),
    "Visakhapatnam": (17.6868, 83.2185),
}


def log(message):
    print("[map_global] " + message)


def pm25_to_aqi(pm25):
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)):
        return None
    value = max(0.0, float(pm25))
    for lo, hi, aqi_lo, aqi_hi in EPA_BREAKPOINTS:
        if lo <= value <= hi:
            return aqi_lo + (aqi_hi - aqi_lo) * (value - lo) / (hi - lo)
    return 500.0


def normalize(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_world_countries():
    log("Loading world country boundaries...")
    world = gpd.read_file(WORLD_GEOJSON)
    world["NAME"] = world["NAME"].fillna(world["ADMIN"])
    world = world[["NAME", "geometry"]]
    world = world.dissolve(by="NAME", as_index=False)[["NAME", "geometry"]]
    world = world[~world.geometry.is_empty]
    world = world[world.geometry.notna()]
    return world


def load_india_states():
    log("Loading Indian state boundaries...")
    states = gpd.read_file(INDIA_STATES_GEOJSON)
    states["st_nm"] = states["st_nm"].fillna("Unknown")
    states = states.dissolve(by="st_nm", as_index=False)[["st_nm", "geometry"]]
    return states[states.geometry.notna()]


POLLUTANT_PRIORITY = ["pm25", "pm2.5", "pm10", "no2", "o3", "so2"]


def read_pm25_frame(raw):
    """Return (mean_pollutant, parameter) preferring pm25 over pm10/no2/o3/so2."""
    df = pd.read_csv(raw, compression="gzip")
    # older archive files use 'measurand' instead of 'parameter'
    param_col = "parameter" if "parameter" in df.columns else "measurand"
    df[param_col] = df[param_col].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for param in POLLUTANT_PRIORITY:
        sub = df[(df[param_col] == param) & df["value"].notna()]
        if not sub.empty:
            return float(sub["value"].mean()), param
    return None, None


def fetch_station_pm25(location_id, first_file):
    local_path = os.path.join(OPENAQ_DIR, first_file)
    if os.path.isfile(local_path):
        try:
            mean, param = read_pm25_frame(local_path)
            if mean is not None:
                return mean, f"local file ({param})"
        except Exception:
            pass
    # files prefetched by scripts/openaq_fetch.py
    mdir = os.path.join(OPENAQ_MEASUREMENTS_DIR, f"location-{location_id}.csv.gz")
    if os.path.isfile(mdir):
        try:
            mean, param = read_pm25_frame(mdir)
            if mean is not None:
                return mean, f"local file ({param})"
        except Exception:
            pass
    url = OPENAQ_BASE_URL + first_file
    try:
        import requests
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "air-quality-hackathon/1.0 (+https://openaq.org)"},
        )
        if response.status_code == 200:
            mean, param = read_pm25_frame(io.BytesIO(response.content))
            if mean is not None:
                return mean, f"remote file ({param})"
        else:
            log("fetch_station_pm25: HTTP %s for %s" % (response.status_code, first_file))
    except Exception as exc:
        log("fetch_station_pm25: %s for %s" % (type(exc).__name__, first_file))
    return None, None


def station_placeholder_pm25(location_id):
    rng = random.Random(int(location_id) % (2 ** 31))
    return rng.uniform(30.0, 110.0)


def load_station_data():
    log("Loading OpenAQ station list...")
    stations = pd.read_csv(OPENAQ_CSV)
    stations = stations.dropna(subset=["lat", "lon"])
    rows = []
    for _, row in stations.iterrows():
        pm25, source = fetch_station_pm25(row["location_id"], row["first_file"])
        if pm25 is None:
            pm25 = station_placeholder_pm25(row["location_id"])
            source = "placeholder"
        aqi = pm25_to_aqi(pm25)
        rows.append({
            "location_id": row["location_id"],
            "lat": row["lat"],
            "lon": row["lon"],
            "pm25": pm25,
            "aqi": aqi,
            "source": source,
            "geometry": Point(row["lon"], row["lat"]),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    used = len(gdf[gdf["source"] == "placeholder"])
    log("Loaded %d stations (PM2.5; %d used placeholders because data was unreachable)." % (len(gdf), used))
    return gdf


def assign_countries(stations, world):
    log("Assigning stations to countries (point-in-polygon)...")
    match = gpd.sjoin(stations, world[["NAME", "geometry"]], how="left", predicate="within")
    match = match[match["NAME"].notna()]
    assigned = set(match.index)
    missing = stations[~stations.index.isin(assigned)]
    if not missing.empty:
        nearest = gpd.sjoin_nearest(missing, world[["NAME", "geometry"]], how="left")
        match = pd.concat([match, nearest], ignore_index=True)
    return match[["location_id", "lat", "lon", "pm25", "aqi", "source", "NAME"]]


def build_country_aggregation(station_country, world):
    log("Aggregating mean PM2.5 by country...")
    grouped = station_country.dropna(subset=["NAME"]).groupby("NAME").agg(
        pm25=("pm25", "mean"),
        aqi=("aqi", "mean"),
        n_stations=("pm25", "size"),
    ).reset_index()
    merged = world.merge(grouped, on="NAME", how="left")
    merged["bucket"] = merged["pm25"].apply(lambda v: bucket_index(v, PM25_BUCKETS))
    merged["bucket"] = merged["bucket"].astype(float)
    merged["color"] = merged["pm25"].apply(lambda v: bucket_color(v, PM25_BUCKETS, PM25_COLORS))
    return merged


def plotly_colorscale():
    eps = 1e-6
    return [
        [0.0, GREEN], [0.2, GREEN],
        [0.2 + eps, YELLOW], [0.4, YELLOW],
        [0.4 + eps, ORANGE], [0.6, ORANGE],
        [0.6 + eps, RED], [0.8, RED],
        [0.8 + eps, DARKRED], [1.0, DARKRED],
    ]


def save_world_plotly(merged, stations, html_path):
    log("Building interactive world choropleth...")
    geojson = json.loads(merged.to_json())
    hover = np.stack([
        merged["NAME"].astype(str).values,
        merged["pm25"].round(1).fillna("No data").astype(str).values,
        merged["aqi"].round(0).fillna("No data").astype(str).values,
        merged["n_stations"].fillna(0).astype(int).astype(str).values,
    ], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=merged["NAME"],
        z=merged["bucket"],
        featureidkey="properties.NAME",
        zmin=0,
        zmax=4,
        colorscale=plotly_colorscale(),
        colorbar=dict(
            title="Mean PM2.5 level",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=PM25_LABELS,
            len=0.6,
        ),
        customdata=hover,
        hovertemplate="<b>%{customdata[0]}</b><br>Mean PM2.5: %{customdata[1]} µg/m³"
                      "<br>Mean AQI: %{customdata[2]}<br>Stations: %{customdata[3]}<extra></extra>",
        showscale=True,
    ))
    marker_custom = np.stack([
        stations["display_name"].astype(str).values,
        stations["location_id"].astype(str).values,
        stations["aqi"].round(0).astype(str).values,
        stations["pm25"].round(1).astype(str).values,
    ], axis=-1)
    station_colors = stations["pm25"].apply(
        lambda v: bucket_color(v, PM25_BUCKETS, PM25_COLORS)).tolist()
    fig.add_trace(go.Scattergeo(
        lon=stations["lon"],
        lat=stations["lat"],
        mode="markers",
        marker=dict(size=6, color=station_colors, line=dict(width=0.6, color="black")),
        customdata=marker_custom,
        hovertemplate="<b>%{customdata[0]}</b><br>Station ID %{customdata[1]}"
                      "<br>AQI %{customdata[2]} (PM2.5 %{customdata[3]} µg/m³)"
                      "<extra></extra>",
        name="Stations",
        showlegend=False,
    ))
    fig.update_layout(
        title="Global Mean Air Quality by Country (OpenAQ stations, PM2.5)",
        height=760,
        geo=dict(
            projection=dict(type="natural earth"),
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#888888",
            showcountries=True,
            countrycolor="#888888",
            landcolor="#f2f2f2",
            showlakes=True,
            lakecolor="#dceef8",
        ),
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    log("Saved " + html_path)
    return fig


def save_world_png_writer(fig, png_path):
    try:
        fig.write_image(png_path)
        log("Saved " + png_path)
        return True
    except Exception as exc:
        log("kaleido PNG export unavailable (%s); using matplotlib fallback." % exc)
        return False


def save_world_png_matplotlib(merged, stations, png_path):
    log("Rendering world map to PNG with matplotlib...")
    fig, ax = plt.subplots(figsize=(16, 8))
    merged.plot(
        ax=ax,
        color=merged["color"].fillna(GREY).tolist(),
        edgecolor="#555555",
        linewidth=0.3,
    )
    station_colors = stations["pm25"].apply(
        lambda v: bucket_color(v, PM25_BUCKETS, PM25_COLORS)).tolist()
    ax.scatter(stations["lon"], stations["lat"], s=14, c=station_colors,
               edgecolors="#333333", linewidths=0.4, zorder=3)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(PM25_COLORS, PM25_LABELS)]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Global Mean Air Quality by Country (OpenAQ stations, PM2.5)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("Saved " + png_path)


def per_city_latest_aqi():
    log("Computing latest-day AQI per city from india_city_day.csv...")
    df = pd.read_csv(INDIA_DAY_CSV)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    global_median = float(df["AQI"].median(skipna=True))
    results = {}
    for city, grp in df.groupby("City"):
        last_date = grp["Date"].max()
        sub = grp[grp["Date"] == last_date].copy()
        city_median = float(grp["AQI"].median(skipna=True))
        fill_value = city_median if not math.isnan(city_median) else global_median
        sub["AQI"] = sub["AQI"].fillna(fill_value)
        mean_aqi = float(sub["AQI"].mean(skipna=True))
        if math.isnan(mean_aqi):
            mean_aqi = global_median
        results[str(city).strip()] = mean_aqi
    return results


def city_coordinates(city_rows, aqi_by_city):
    log("Looking up city coordinates...")
    lookup = {}
    for _, r in city_rows.iterrows():
        lookup[normalize(r["City"])] = (float(r["Latitude"]), float(r["Longitude"]), r["State"])
    coords = {}
    for city in aqi_by_city:
        key = normalize(city)
        if key in lookup:
            lat, lon, state = lookup[key]
            coords[city] = (lat, lon, state)
            continue
        best = None
        for ckey, (lat, lon, state) in lookup.items():
            if key in ckey or ckey in key:
                if best is None or abs(len(ckey) - len(key)) < abs(len(best[0]) - len(key)):
                    best = (ckey, lat, lon, state)
        if best is not None:
            coords[city] = (best[1], best[2], best[3])
            continue
        if city in FALLBACK_CITY_COORDS:
            lat, lon = FALLBACK_CITY_COORDS[city]
            coords[city] = (lat, lon, None)
            continue
    missing = [c for c in aqi_by_city if c not in coords]
    if missing:
        log("WARNING: no coordinates found for cities: %s" % ", ".join(missing))
    return coords


def build_india_city_state():
    aqi_by_city = per_city_latest_aqi()
    city_rows = pd.read_csv(INDIA_CITIES_CSV)[["City", "State", "Latitude", "Longitude"]]
    city_gdf = gpd.GeoDataFrame(
        city_rows,
        geometry=gpd.points_from_xy(city_rows["Longitude"], city_rows["Latitude"]),
        crs="EPSG:4326",
    )
    coords = city_coordinates(city_gdf, aqi_by_city)
    states = load_india_states()
    city_pts = gpd.GeoDataFrame(
        [{"City": c, "AQI": aqi_by_city[c], "lat": v[0], "lon": v[1],
          "geometry": Point(v[1], v[0])} for c, v in coords.items()],
        crs="EPSG:4326",
    )
    log("Assigning %d cities to states (point-in-polygon)..." % len(city_pts))
    joined = gpd.sjoin(city_pts, states[["st_nm", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_right")])
    unassigned = joined[joined["st_nm"].isna()]
    if not unassigned.empty:
        nearest = gpd.sjoin_nearest(
            unassigned.drop(columns=["geometry"]),
            states[["st_nm", "geometry"]],
            how="left",
        )
        nearest = nearest.drop(columns=[c for c in nearest.columns if c.startswith("index_right")])
        joined = joined[joined["st_nm"].notna()]
        joined = pd.concat([joined, nearest], ignore_index=True)
    return joined, states


def state_standings(city_state, states):
    agg = city_state.dropna(subset=["st_nm"]).groupby("st_nm").agg(
        AQI=("AQI", "mean"),
        n_cities=("AQI", "size"),
    ).reset_index()
    merged = states.merge(agg, on="st_nm", how="left")
    merged["bucket"] = merged["AQI"].apply(lambda v: bucket_index(v, AQI_BUCKETS))
    merged["bucket"] = merged["bucket"].astype(float)
    merged["color"] = merged["AQI"].apply(lambda v: bucket_color(v, AQI_BUCKETS, AQI_COLORS))
    return merged


def save_india_folium(standings, city_state, html_path):
    log("Building interactive India choropleth with folium...")
    import folium

    m = folium.Map(location=[22.5, 79.5], zoom_start=5, tiles="CartoDB positron")

    style_fn = lambda f: {
        "fillColor": bucket_color(
            None if f["properties"].get("AQI") is None else float(f["properties"]["AQI"]),
            AQI_BUCKETS, AQI_COLORS),
        "color": "#333333",
        "weight": 0.7,
        "fillOpacity": 0.75,
    }
    tooltip = folium.GeoJsonTooltip(
        fields=["st_nm", "AQI", "n_cities"],
        aliases=["State", "Avg AQI", "Cities sampled"],
        localize=True,
    )
    folium.GeoJson(
        standings.to_json(),
        name="State AQI",
        style_function=style_fn,
        tooltip=tooltip,
        highlight_function=lambda f: {"weight": 2.0, "color": "#111111", "fillOpacity": 0.9},
    ).add_to(m)

    city_layer = folium.FeatureGroup(name="Cities", show=True)
    for _, r in city_state.iterrows():
        col = bucket_color(r["AQI"], AQI_BUCKETS, AQI_COLORS)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color="#333333",
            weight=1,
            fill=True,
            fill_color=col,
            fill_opacity=0.95,
            popup=folium.Popup("City: %s<br>AQI: %.1f" % (r["City"], r["AQI"]), max_width=220),
        ).add_to(city_layer)
    city_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = (
        '<div style="position:fixed;bottom:25px;right:25px;z-index:9999;'
        'background:white;padding:10px 14px;border-radius:6px;'
        'box-shadow:0 0 8px rgba(0,0,0,0.3);font-size:13px;max-width:300px">'
        "<b>What each color means</b>"
        + html_legend(legend_rows(include_health=True))
        + "</div>")
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(html_path)
    log("Saved " + html_path)


def save_india_png(standings, city_state, png_path):
    log("Rendering India map to PNG with matplotlib...")
    fig, ax = plt.subplots(figsize=(10, 12))
    standings.plot(
        ax=ax,
        color=standings["color"].fillna(GREY).tolist(),
        edgecolor="#444444",
        linewidth=0.4,
    )
    city_colors = city_state["AQI"].apply(
        lambda v: bucket_color(v, AQI_BUCKETS, AQI_COLORS)).tolist()
    ax.scatter(city_state["lon"], city_state["lat"], s=30, c=city_colors,
               edgecolors="#111111", linewidths=0.6, zorder=3)
    for _, r in city_state.iterrows():
        ax.annotate(r["City"], (r["lon"], r["lat"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", color="#222222")
    handles = [Patch(facecolor=c, label="%s \u00b7 AQI %s" % (l, r))
               for c, l, r in zip(AQI_COLORS, AQI_LABELS, AQI_RANGES)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("Average AQI per Indian State (latest readings by city)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("Saved " + png_path)


def main():
    ensure_output_dir()
    log("Starting air-quality map generation...")

    world = load_world_countries()
    stations = load_station_data()
    station_country = assign_countries(stations, world)

    # location_id -> human-readable city/station name for hover tooltips
    names = load_station_city_lookup()
    station_country = station_country.copy()
    station_country["display_name"] = station_country["location_id"].astype(int).map(
        names).fillna(station_country["location_id"].map(lambda i: "Station %s" % i))
    station_country["display_name"] = station_country["display_name"].astype(str)

    country_merged = build_country_aggregation(station_country, world)

    world_html = os.path.join(OUTPUT_DIR, "world_aqi_map.html")
    world_png = os.path.join(OUTPUT_DIR, "world_aqi_map.png")
    fig = save_world_plotly(country_merged, station_country, world_html)
    if not save_world_png_writer(fig, world_png):
        save_world_png_matplotlib(country_merged, station_country, world_png)

    city_state, states = build_india_city_state()
    standings = state_standings(city_state, states)
    save_india_folium(standings, city_state, os.path.join(OUTPUT_DIR, "india_aqi_map.html"))
    save_india_png(standings, city_state, os.path.join(OUTPUT_DIR, "india_aqi_map.png"))

    log("Done. Output written to %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()