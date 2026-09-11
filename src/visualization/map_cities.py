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
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

matplotlib.use("Agg")

from src.visualization.location_table import resolve_many  # noqa: E402
from src.visualization.map_global import (  # noqa: E402
    AQI_BUCKETS,
    AQI_COLORS,
    AQI_LABELS,
    GREY,
    OUTPUT_DIR,
    PROJECT_ROOT,
    WORLD_GEOJSON,
    bucket_color,
    bucket_index,
    load_india_states,
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
                   tiles="OpenStreetMap")

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
                background:white;padding:10px;border-radius:6px;
                box-shadow:0 0 8px rgba(0,0,0,0.3);font-size:13px">
      <b>Predicted AQI (next-day)</b><br>
      {rows}
    </div>
    """.format(rows="<br>".join(
        '<i style="background:%s;display:inline-block;width:14px;height:14px;'
        'border-radius:3px"></i> %s' % (AQI_COLORS[i], label)
        for i in range(len(AQI_LABELS))))
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
    legend = [Patch(facecolor=AQI_COLORS[i], label=AQI_LABELS[i])
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

def build_world_map(pred, world_geojson, path_html, path_png):
    log("Building world city-prediction map...")
    import plotly.graph_objects as go
    world = gpd.read_file(WORLD_GEOJSON)
    colors = [bucket_color(v, AQI_BUCKETS, AQI_COLORS) for v in pred["predicted_aqi"]]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=world_geojson, locations=world["ISO_A3"], z=[0] * len(world),
        marker_line_color="rgba(128,128,128,0.5)", marker_line_width=0.4,
        colorscale=[[0, "#e8e8e8"], [1, "#e8e8e8"]], showscale=False,
        hoverinfo="skip", name="countries"))
    fig.add_trace(go.Scattergeo(
        lon=pred["lon"], lat=pred["lat"], mode="markers",
        name="Predicted city AQI",
        marker=dict(size=10 + pred["predicted_aqi"] * 0.35, color=colors,
                    line=dict(width=1, color="black")),
        text=pred["city"].tolist(),
        textposition="top center",
        textfont=dict(size=11, color="#111"),
        customdata=pred[["predicted_aqi", "bucket", "forecast_date"]]
                  .to_numpy(),
        hovertemplate=(
            "<b>%{text}</b><br>Predicted AQI: %{customdata[0]} "
            "(%{customdata[1]})<br>Forecast date: %{customdata[2]}"
            "<extra></extra>"),
        showlegend=True))
    fig.update_geos(showframe=False, projection_type="natural earth",
                    coastlinecolor="#999", landcolor="#f2f2f2")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0),
                      title="Predicted Next-Day AQI by City (world view)")
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
    build_world_map(pred, world_geojson,
                    os.path.join(OUTPUT_DIR, "world_city_predictions.html"),
                    os.path.join(OUTPUT_DIR, "world_city_predictions.png"))
    log("Done. Output written to %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()