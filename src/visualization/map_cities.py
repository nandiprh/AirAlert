"""City-wise AQI prediction maps.

Draws predicted next-day AQI per city (from ``src/predict.py``) as colored
city markers on top of state/country choropleths.

Outputs (into ``output/``):
    india_city_predictions.html / .png   India: states by predicted mean + city markers
    world_city_predictions.html / .png   World: country choropleth + predicted city markers

Usage:
    python src/predict.py                  # (first) generate predictions
    python src/visualization/map_cities.py # then draw the maps
"""

import os
import sys

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

matplotlib.use("Agg")

from src.visualization.aqi_spec import (  # noqa: E402
    AQI_HEALTH,
    AQI_RANGES,
    html_legend,
    legend_rows,
)
from src.visualization.location_table import (  # noqa: E402
    _haversine_km,
    load_location_table,
    resolve_many,
)
from src.visualization.map_global import (  # noqa: E402
    AQI_BUCKETS,
    AQI_COLORS,
    AQI_LABELS,
    GREY,
    OUTPUT_DIR,
    PROJECT_ROOT,
    WORLD_GEOJSON,
    assign_countries,
    bucket_color,
    bucket_index,
    load_india_states,
    load_station_data,
    load_world_countries,
    log,
)

PREDICTIONS_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed", "predictions", "city_predictions.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"{PREDICTIONS_CSV} not found. Run `python src/predict.py` first.")
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[df["predicted_aqi"].notna()]

    # authoritative coordinates: hardcoded location table (falls back to the
    # nationwide city table), so each city renders at a real coordinate
    resolved = resolve_many(df["city"].tolist())
    coords = pd.DataFrame([
        {"city": c, "lat": v[0], "lon": v[1],
         "state": v[2], "country": v[3]}
        for c, v in resolved.items()
    ])
    df = df.drop(columns=["lat", "lon", "state", "country"],
                 errors="ignore").merge(coords, on="city", how="left")
    df["bucket"] = df["predicted_aqi"].apply(
        lambda v: AQI_LABELS[bucket_index(v, AQI_BUCKETS)])
    df["color"] = df["predicted_aqi"].apply(
        lambda v: bucket_color(v, AQI_BUCKETS, AQI_COLORS))
    return df


def state_predicted_mean(pred):
    g = pred.dropna(subset=["state"]).groupby("state")["predicted_aqi"].mean()
    return g.to_dict()


# ---------------------------------------------------------------------------
# India map (folium)
# ---------------------------------------------------------------------------

def build_india_map(pred, states):
    log("Building India city-prediction map (folium)...")
    m = folium.Map(location=[22.5, 79.5], zoom_start=5,
                   tiles="CartoDB positron")

    state_means = state_predicted_mean(pred)
    def style(feature):
        name = feature["properties"].get("st_nm") or ""
        mean = state_means.get(name)
        color = bucket_color(mean, AQI_BUCKETS, AQI_COLORS) if mean is not None else GREY
        return {"color": "#666", "weight": 1, "fillColor": color,
                "fillOpacity": 0.45}

    name_cols = [c for c in states.columns if c != "geometry"]
    folium.GeoJson(
        states.__geo_interface__,
        name="States (predicted mean)",
        style_function=style,
        highlight_function=lambda f: {"weight": 2, "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(fields=name_cols, aliases=name_cols,
                                      localize=True),
    ).add_to(m)

    for _, r in pred.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7 + min(12.0, r["predicted_aqi"] / 25.0),
            color="#333", weight=1, fill_color=r["color"], fill_opacity=0.9,
            popup=folium.Popup(
                ("<b>%s</b><br>Predicted AQI <b>%.0f</b> (%s)"
                 "<br>Forecast date: %s<br>Last observed: %s")
                % (r["city"], r["predicted_aqi"], r["bucket"],
                   r.get("forecast_date", "?"), r.get("last_date", "?")),
                max_width=260),
            tooltip="%s → AQI %.0f" % (r["city"], r["predicted_aqi"]),
        ).add_to(m)
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.DivIcon(
html=('<div style="font-size:11px;font-weight:bold;color:#333;'
              'text-shadow:-1px 0 #fff,1px 0 #fff,0 -1px #fff,0 1px #fff;'
              'white-space:nowrap;transform:translate(-50%%,-140%%);">'
              "%s</div>") % (r["city"]),
                icon_size=None),
        ).add_to(m)

    for i, label in enumerate(AQI_LABELS):
        folium.Marker(
            location=[0, 0], icon=folium.Icon(color="white"),
            popup=label, opacity=0).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px 14px;border-radius:6px;
                box-shadow:0 0 8px rgba(0,0,0,0.3);font-size:13px;
                max-width:290px">
      <b>What each color means</b>
      {rows}
    </div>
    """.format(rows=html_legend(legend_rows(include_health=True)))
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# India matplotlib
# ---------------------------------------------------------------------------

def plot_india_state_choropleth(pred, states, path):
    log("Rendering India map to PNG (matplotlib)...")
    fig, ax = plt.subplots(figsize=(11, 12))
    state_means = state_predicted_mean(pred)
    colors = [bucket_color(state_means.get(n, None), AQI_BUCKETS, AQI_COLORS)
              for n in states["st_nm"]]
    states.plot(ax=ax, color=colors, edgecolor="#555", linewidth=0.6)

    ax.scatter(pred["lon"], pred["lat"], c=pred["color"], s=60 + pred["predicted_aqi"] * 0.8,
               edgecolors="#333", linewidths=1, zorder=5)
    for _, r in pred.iterrows():
        ax.annotate("%s\n%.0f" % (r["city"], r["predicted_aqi"]),
                    (r["lon"], r["lat"]), fontsize=7,
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#999", lw=0.5))
    legend = [Patch(facecolor=AQI_COLORS[i],
                    label="%s \u00b7 AQI %s" % (AQI_LABELS[i], AQI_RANGES[i]))
              for i in range(len(AQI_LABELS))]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_title("Predicted Next-Day AQI by City (India)", fontsize=14)
    ax.set_xlim(65, 98); ax.set_ylim(6, 37)
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log("Saved %s" % path)


# ---------------------------------------------------------------------------
# World map
# ---------------------------------------------------------------------------

def add_aqi_color_legend(fig):
    """Add scope-independent legend entries: one per AQI bucket with health text."""
    for i, label in enumerate(AQI_LABELS):
        name = "%s \u00b7 AQI %s\n%s" % (label, AQI_RANGES[i], AQI_HEALTH[i])
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode="markers",
            marker=dict(size=13, color=AQI_COLORS[i], symbol="square",
                        line=dict(width=1, color="black")),
            name=name, showlegend=True,
        ))


def load_global_cities():
    """Curated non-India cities (Sydney, London, Paris, NYC, ...) + coords."""
    table = load_location_table()
    return [e for e in table.values() if e["country"].lower() != "india"]


def _iso3_lookup(world):
    """{normalized name/ADMIN: iso3} from the world boundaries GeoJSON."""
    out = {}
    for _, r in world.iterrows():
        iso = r.get("ISO_A3")
        for key in (r.get("NAME"), r.get("ADMIN")):
            if key is not None and not pd.isna(key) and iso:
                out["".join(ch for ch in str(key).lower() if ch.isalnum())] = iso
    return out


def _match_iso3(country, iso3_map):
    """ISO3 for a country string, tolerating name variants (fuzzy containment)."""
    norm = "".join(ch for ch in str(country).lower() if ch.isalnum())
    if norm in iso3_map:
        return iso3_map[norm]
    cands = sorted((len(k), k, v) for k, v in iso3_map.items()
                   if norm in k or k in norm)
    return cands[0][2] if cands else None


def nearest_station_aqi(lat, lon, station_country, max_km=300):
    """(aqi, distance_km) of the nearest OpenAQ station, else None."""
    best = None
    for _, r in station_country.dropna(subset=["aqi", "lat", "lon"]).iterrows():
        dist = _haversine_km(lat, lon, float(r["lat"]), float(r["lon"]))
        if dist <= max_km and (best is None or dist < best[0]):
            best = (float(r["aqi"]), dist)
    return best


def build_world_map(pred, world_geojson, path_html, path_png, station_country=None):
    log("Building world city-prediction map...")
    world = gpd.read_file(WORLD_GEOJSON)
    iso3_map = _iso3_lookup(world)

    g_cities = []
    if station_country is not None and not station_country.empty:
        # country-mean fallback, keyed by ISO3 (robust to name variants)
        sc = station_country.copy()
        sc["iso3"] = sc["NAME"].map(lambda n: _match_iso3(n, iso3_map))
        mean = sc.dropna(subset=["aqi", "iso3"]).groupby("iso3")["aqi"].mean()

        for e in load_global_cities():
            hit = nearest_station_aqi(e["lat"], e["lon"], station_country)
            if hit is not None:
                aqi, dist = hit
                g_cities.append({
                    "city": e["city"], "country": e["country"],
                    "lat": e["lat"], "lon": e["lon"],
                    "aqi": float(aqi), "dist": float(dist),
                    "src": "nearest station",
                })
                continue
            iso3 = _match_iso3(e["country"], iso3_map)
            if iso3 is None or iso3 not in mean:
                log("  global city %s: no station near or in %s; skipping"
                    % (e["city"], e["country"]))
                continue
            g_cities.append({
                "city": e["city"], "country": e["country"],
                "lat": e["lat"], "lon": e["lon"],
                "aqi": float(mean[iso3]), "dist": None,
                "src": "country mean",
            })

    colors = [bucket_color(v, AQI_BUCKETS, AQI_COLORS) for v in pred["predicted_aqi"]]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=world_geojson, locations=world["ISO_A3"], z=[0] * len(world),
        marker_line_color="rgba(128,128,128,0.5)", marker_line_width=0.4,
        colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]], showscale=False,
        hoverinfo="skip", name="countries"))
    fig.add_trace(go.Scattergeo(
        lon=pred["lon"], lat=pred["lat"], mode="markers",
        name="Predicted next-day AQI (India)",
        marker=dict(size=10 + pred["predicted_aqi"] * 0.35, color=colors,
                    line=dict(width=1, color="black")),
        text=pred["city"].tolist(),
        textposition="top center",
        textfont=dict(size=11, color="#111"),
        customdata=pred[["city", "predicted_aqi", "bucket", "forecast_date"]]
                  .to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Predicted AQI: %{customdata[1]} "
            "(%{customdata[2]})<br>Forecast date: %{customdata[3]}"
            "<extra></extra>"),
        showlegend=True))

    if g_cities:
        g_colors = [bucket_color(c["aqi"], AQI_BUCKETS, AQI_COLORS) for c in g_cities]
        fig.add_trace(go.Scattergeo(
            lon=[c["lon"] for c in g_cities],
            lat=[c["lat"] for c in g_cities],
            mode="markers",
            name="Current AQI \u2014 nearest OpenAQ station",
            marker=dict(size=9, color=g_colors,
                        symbol="diamond",
                        line=dict(width=1, color="#111")),
            text=[c["city"] for c in g_cities],
            textposition="bottom center",
            textfont=dict(size=10, color="#333"),
            customdata=[[c["city"], c["country"], round(c["aqi"], 1),
                         ("nearest station \u00b7 %.1f km" % c["dist"]
                          if c["dist"] is not None else "country mean"),
                         c["src"]] for c in g_cities],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "AQI: %{customdata[2]}<br>%{customdata[3]}"
                "<br><i>recent OpenAQ readings</i><extra></extra>"),
            showlegend=True))

    add_aqi_color_legend(fig)

    fig.update_geos(showframe=False, projection_type="natural earth",
                    coastlinecolor="#999", landcolor="#f2f2f2")
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0),
                      title="Predicted Next-Day AQI by City (world view)",
                      legend=dict(
                          y=0.98, x=1.01, traceorder="normal",
                          font=dict(size=12),
                          itemsizing="constant"),
                      )
    fig.write_html(path_html)
    log("Saved %s" % path_html)
    try:
        import plotly.io as pio
        pio.write_image(fig, path_png, width=1800, height=900)
        log("Saved %s" % path_png)
    except Exception as exc:
        log("kaleido PNG unavailable (%s); plot_data not exported as PNG" % exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred = load_predictions()
    log("Loaded %d city predictions from %s" % (len(pred), PREDICTIONS_CSV))

    states = load_india_states()
    m = build_india_map(pred, states)
    hp = os.path.join(OUTPUT_DIR, "india_city_predictions.html")
    m.save(hp); log("Saved %s" % hp)

    png = os.path.join(OUTPUT_DIR, "india_city_predictions.png")
    plot_india_state_choropleth(pred, states, png)

    world = gpd.read_file(WORLD_GEOJSON)
    world_geojson = world.__geo_interface__

    # station_country lets the world map place the curated global cities and
    # colour them by their country's recent OpenAQ average
    station_country = None
    try:
        station_country = assign_countries(load_station_data(), load_world_countries())
    except Exception as exc:
        log("could not load OpenAQ stations for global cities (%s); "
            "world map will show India predictions only" % exc)

    build_world_map(pred, world_geojson,
                    os.path.join(OUTPUT_DIR, "world_city_predictions.html"),
                    os.path.join(OUTPUT_DIR, "world_city_predictions.png"),
                    station_country)
    log("Done. Output written to %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()