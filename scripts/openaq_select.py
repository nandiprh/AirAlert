#!/usr/bin/env python3
"""Merge OpenAQ scan results, assign countries, and select a diverse subset.

Produces `data/raw/openaq/location_scan_clean.csv` (the file consumed by the
world-map script) containing stations spread across as many countries as possible.

Usage:
    python scripts/openaq_select.py \
        --scans data/raw/openaq/location_scan_clean.csv /tmp/opencode/scan_run2.csv \
        --world data/external/geojson/world_countries.geojson \
        --max-per-country 3 \
        --out data/raw/openaq/location_scan_clean.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--max-per-country", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for p in args.scans:
        if Path(p).exists():
            frames.append(pd.read_csv(p, dtype={"location_id": str}))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["lat", "lon", "first_file"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    valid = df[
        df["lat"].between(-90, 90) & df["lon"].between(-180, 180) &
        df["first_file"].astype(str).str.startswith("records/")
    ]
    valid = valid.drop_duplicates(subset="location_id")
    print(f"[openaq_select] {len(df)} raw -> {len(valid)} valid unique stations")

    world = gpd.read_file(args.world)
    world["iso3"] = world["ISO_A3"].where(world["ISO_A3"] != "-99", world["ADM0_A3"])
    gdf = gpd.GeoDataFrame(
        valid.assign(geometry=gpd.points_from_xy(valid["lon"], valid["lat"])),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf, world[["iso3", "NAME", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined["index_right"].isna()]
    print(f"[openaq_select] {len(joined)} stations assigned to {joined['iso3'].nunique()} countries")

    joined = joined.sort_values(["iso3", "location_id"])
    selected = joined.groupby("iso3").head(args.max_per_country).reset_index(drop=True)
    selected = selected[["location_id", "lat", "lon", "first_file", "iso3", "NAME"]]
    selected = selected.rename(columns={"NAME": "country"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False)
    print(f"[openaq_select] wrote {len(selected)} stations across "
          f"{selected['iso3'].nunique()} countries -> {args.out}")
    print(selected["iso3"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()