"""Hardcoded city location table.

The map renders each predicted city at a coordinate from this curated table
(``data/external/city_locations.csv``), so a predicted "Delhi" always lands on
Delhi and never on a fuzzy-matched impostor. Colums:

    code, city, country, state, lat, lon

If a city is missing from the table, resolution falls back to the smaller
hardcoded dict ``FALLBACK_CITY_COORDS`` and finally to the nationwide
``india_cities.csv`` population table (exact match only).
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LOCATION_TABLE_CSV = os.path.join(PROJECT_ROOT, "data", "external",
                                  "city_locations.csv")
INDIA_CITIES_CSV = os.path.join(PROJECT_ROOT, "data", "raw",
                                "india_cities.csv")

FALLBACK_CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Amaravati": (16.5083, 80.5183),
    "Bengaluru": (12.9716, 77.5946), "Brajrajnagar": (21.8167, 83.9167),
    "Delhi": (28.6139, 77.2090), "Ernakulam": (9.9694, 76.2934),
    "Gurugram": (28.4595, 77.0266), "Hyderabad": (17.3850, 78.4867),
    "Jorapokhar": (23.7903, 86.1110), "Mumbai": (19.0760, 72.8777),
    "Talcher": (20.9493, 85.2236), "Visakhapatnam": (17.6868, 83.2185),
}


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_location_table():
    """Return {normalized_city: dict(code, city, country, state, lat, lon)}."""
    table = {}
    if os.path.exists(LOCATION_TABLE_CSV):
        df = pd.read_csv(LOCATION_TABLE_CSV)
        for _, r in df.iterrows():
            key = _norm(r["city"])
            table[key] = {
                "code": str(r.get("code", "")),
                "city": str(r["city"]),
                "country": str(r.get("country", "")),
                "state": str(r.get("state", "")) if not pd.isna(r.get("state")) else "",
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
    return table


def resolve(city, location_table=None):
    """(lat, lon, state, country) for a city via hardcoded table / fallbacks."""
    table = load_location_table() if location_table is None else location_table
    key = _norm(city)
    if key in table:
        ent = table[key]
        return ent["lat"], ent["lon"], ent["state"], ent["country"]

    if city in FALLBACK_CITY_COORDS:
        lat, lon = FALLBACK_CITY_COORDS[city]
        return lat, lon, "", "India"

    if os.path.exists(INDIA_CITIES_CSV):
        df = pd.read_csv(INDIA_CITIES_CSV)
        hit = df[df["City"].map(_norm) == key]
        if not hit.empty:
            r = hit.iloc[0]
            return (float(r["Latitude"]), float(r["Longitude"]),
                    str(r["State"]), "India")

    return None, None, "", ""


def resolve_many(cities, location_table=None):
    """{city: (lat, lon, state, country)} for a list of cities."""
    table = load_location_table() if location_table is None else location_table
    return {c: resolve(c, table) for c in cities}