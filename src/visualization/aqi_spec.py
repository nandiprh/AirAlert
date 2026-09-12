"""Single source of truth for the CPCB AQI bucket spec.

All maps and the forecast pipeline share these bucket boundaries, colors,
range labels and health guidance so the colour of a marker always agrees
with the legend and the predicted bucket string.

Used by ``map_global.py``, ``map_cities.py`` and ``predict.py``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

GREEN = "#2ca02c"
YELLOW = "#f7d448"
ORANGE = "#f49e2a"
RED = "#d6452e"
DARKRED = "#8a1f1f"
GREY = "#d5d5d5"

# One row per CPCB AQI band:
#   (short_label, color, min_aqi, max_aqi, range_label, health_guidance)
AQI_BANDS: List[Tuple[str, str, float, float, str, str]] = [
    ("Good", GREEN, 0.0, 50.0, "0-50",
     "Air quality is satisfactory; little or no health risk."),
    ("Satisfactory", YELLOW, 50.0, 100.0, "51-100",
     "Acceptable; minor irritation possible for very sensitive people."),
    ("Moderate", ORANGE, 100.0, 200.0, "101-200",
     "May cause breathing discomfort for prolonged exposure or sensitive groups."),
    ("Poor", RED, 200.0, 300.0, "201-300",
     "Breathing discomfort on prolonged exposure; sensitive groups affected."),
    ("Very Poor / Severe", DARKRED, 300.0, math.inf, "301-400+",
     "Respiratory illness on prolonged exposure; health alert."),
]

AQI_BUCKETS = [(lo, hi) for _, _, lo, hi, _, _ in AQI_BANDS]
AQI_COLORS = [c for _, c, _, _, _, _ in AQI_BANDS]
AQI_LABELS = [label for label, *_ in AQI_BANDS]
AQI_RANGES = [rng for *_, rng, _ in AQI_BANDS]
AQI_HEALTH = [health for *_, health in AQI_BANDS]

# Integer thresholds used by ``aqi_bucket_name`` (CPCB bands):
# 0-50 Good, 51-100 Satisfactory, 101-200 Moderate, 201-300 Poor,
# 301-400 Very Poor, 401+ Severe.
AQI_BUCKET_NAME_THRESHOLDS: List[Tuple[float, str]] = [
    (0.0, "Good"),
    (51.0, "Satisfactory"),
    (101.0, "Moderate"),
    (201.0, "Poor"),
    (301.0, "Very Poor"),
    (401.0, "Severe"),
]


def bucket_index(value, buckets=AQI_BUCKETS):
    """Index of the bucket containing ``value`` (last bucket if out of range)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


def bucket_color(value, buckets=AQI_BUCKETS, colors=AQI_COLORS):
    idx = bucket_index(value, buckets)
    if idx is None:
        return GREY
    return colors[idx]


def aqi_bucket_name(aqi) -> str:
    """Short CPCB bucket name for a numeric AQI value (used by predict.py)."""
    aqi = float(aqi)
    for lo, label in reversed(AQI_BUCKET_NAME_THRESHOLDS):
        if aqi >= lo:
            return label
    return "Good"


def legend_rows(prefix="", include_health=True):
    """Render the colour -> range -> health rows shared by the map legends.

    Returns a list of ``(color, title, subtitle)`` tuples where ``title`` is
    e.g. ``"Good · AQI 0-50"`` and ``subtitle`` is the health guidance.
    """
    rows = []
    for label, color, lo, hi, rng, health in AQI_BANDS:
        title = "%s%s \u00b7 AQI %s" % (prefix, label, rng)
        subtitle = health if include_health else ""
        rows.append((color, title, subtitle))
    return rows


def html_legend(rows, title_text="", width=16, height=16):
    """Render the AQI legend as an HTML string (folium overlays)."""
    title = ("<div style='margin-bottom:6px'><b>%s</b></div>" % title_text
             if title_text else "")
    items = [title]
    for color, title, subtitle in rows:
        item = (
            '<div style="display:flex;align-items:flex-start;margin:3px 0">'
            '<i style="background:%s;display:inline-block;width:%dpx;height:%dpx;'
            'border-radius:3px;flex:0 0 %dpx;margin:2px 8px 0 0"></i>'
            "<div><b>%s</b>%s</div></div>"
            % (color, width, height, width, title,
               ("<br><span style='color:#555;font-size:12px'>%s</span>" % subtitle)
               if subtitle else ""))
        items.append(item)
    return "".join(items)