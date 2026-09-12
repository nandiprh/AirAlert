"""Hardcoded city location table.

The map renders each predicted city at a coordinate from this curated table
(``data/external/city_locations.csv``), so a predicted "Delhi" always lands on
Delhi and never on a fuzzy-matched impostor. Colums:

    code, city, country, state, lat, lon

If a city is missing from the table, resolution falls back to the smaller
hardcoded dict ``FALLBACK_CITY_COORDS`` and finally to the nationwide
``india_cities.csv`` population table (exact match only).
"""

import math
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LOCATION_TABLE_CSV = os.path.join(PROJECT_ROOT, "data", "external",
                                  "city_locations.csv")
STATION_LOOKUP_CSV = os.path.join(PROJECT_ROOT, "data", "external",
                                  "station_city_lookup.csv")
OPENAQ_MEASUREMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "raw",
                                       "openaq", "measurements")
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


# ---------------------------------------------------------------------------
# Station lookups (location_id -> display name / nearest city)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def station_name_from_cache(location_id):
    """Station display name read from a prefetched measurement file."""
    from gzip import open as gzopen
    path = os.path.join(OPENAQ_MEASUREMENTS_DIR, f"location-{location_id}.csv.gz")
    if not os.path.isfile(path):
        return None
    try:
        with gzopen(path, "rt", errors="ignore") as fh:
            header = fh.readline().strip().split(",")
            if "location" not in header:
                return None
            name_col = header.index("location")
            for line in fh:
                fields = line.rstrip("\n").split(",")
                if len(fields) > name_col and fields[name_col].strip():
                    return fields[name_col].strip()
    except Exception:
        return None
    return None


def nearest_city(lat, lon, location_table=None, threshold_km=150):
    """(city, country, distance_km) of the nearest curated city, else None."""
    table = load_location_table() if location_table is None else location_table
    best = None
    for entry in table.values():
        dist = _haversine_km(lat, lon, entry["lat"], entry["lon"])
        if dist <= threshold_km and (best is None or dist < best[0]):
            best = (dist, entry["city"], entry["country"])
    if best is None:
        return None
    return best[1], best[2], round(best[0], 1)


def build_station_city_lookup(stations, location_table=None, threshold_km=150):
    """Map every station to a human-readable display name.

    Lookup priority for each ``location_id``:
      1. nearest curated city (``data/external/city_locations.csv``) within
         ``threshold_km`` -> "\"City (~X km)\""
      2. the station's own ``location`` name from a local measurement file
      3. "\"Station <id>\""

    ``stations`` needs columns ``location_id``, ``lat``, ``lon``.
    """
    table = load_location_table() if location_table is None else location_table
    rows = []
    for _, r in stations.iterrows():
        loc_id = int(r["location_id"])
        lat, lon = float(r["lat"]), float(r["lon"])
        city, country, dist = (nearest_city(lat, lon, table, threshold_km)
                               or (None, None, None))
        st_name = station_name_from_cache(loc_id)
        if city is not None:
            display = f"{city} (~{dist:g} km)"
        elif st_name:
            display = st_name
        else:
            display = f"Station {loc_id}"
        rows.append({
            "location_id": loc_id,
            "city": city or "",
            "country": country or "",
            "distance_km": dist if dist is not None else "",
            "station_name": st_name or "",
            "display_name": display,
        })
    return pd.DataFrame(rows)


def save_station_city_lookup(stations, out_path=None):
    """Build the location_id -> display-name table and write it to CSV."""
    df = build_station_city_lookup(stations)
    out = out_path or STATION_LOOKUP_CSV
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    matched = int((df["city"] != "").sum())
    print(f"[location_table] saved {len(df)} station lookups -> {out} "
          f"({matched} matched a curated city)")
    return out


def load_station_city_lookup(path=None):
    """{location_id: display_name} from the generated CSV (empty if absent)."""
    path = path or STATION_LOOKUP_CSV
    out = {}
    if os.path.isfile(path):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            out[int(r["location_id"])] = str(r["display_name"])
    return out


if __name__ == "__main__":
    import sys
    stations_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PROJECT_ROOT, "data", "raw", "openaq", "location_scan_clean.csv")
    stations = pd.read_csv(stations_csv)
    save_station_city_lookup(stations)