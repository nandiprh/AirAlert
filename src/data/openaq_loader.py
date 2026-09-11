"""Loader and geocoder for the OpenAQ worldwide station data.

The project ships a coordinate scan of worldwide OpenAQ stations in
``data/raw/openaq/location_scan_clean.csv``. Each row carries a
``location_id``, ``lat``/``lon`` and the path of the first archived CSV on
OpenAQ's public S3 bucket (``first_file``).

This module provides:

    * :func:`load_stations` - read and sanitise the station coordinate scan,
    * :func:`build_s3_url` - turn a location id / year / month into an S3 URL,
    * :func:`fetch_station_data` - download and parse a station's archived
      CSV (or a whole day of measurements) straight from the S3 archive,
    * :func:`assign_country` - point-in-polygon assignment of stations to
      countries using a GeoJSON world-borders file via ``geopandas``,
    * :func:`main` - end-to-end run that writes
      ``data/processed/openaq_stations.csv``.

Dependencies: pandas, numpy, geopandas (shapely comes with geopandas).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger("air_quality.openaq")

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
GEOJSON_DIR = ROOT_DIR / "data" / "external" / "geojson"

STATION_SCAN_PATH = RAW_DIR / "openaq" / "location_scan_clean.csv"
WORLD_COUNTRIES_PATH = GEOJSON_DIR / "world_countries.geojson"
OUTPUT_PATH = PROCESSED_DIR / "openaq_stations.csv"

#: Public OpenAQ S3 archive base ("records/csv.gz/locationid=..." is relative
#: to this bucket).
OPENAQ_S3_BASE = "https://openaq-data-archive.s3.amazonaws.com"

#: Column in the countries GeoJSON that holds the country name.
COUNTRY_NAME_COL = "ADMIN"


# --------------------------------------------------------------------------- #
#  Station coordinate scan
# --------------------------------------------------------------------------- #


def load_stations(path: Path = STATION_SCAN_PATH) -> pd.DataFrame:
    """Load and sanitise the OpenAQ location scan CSV.

    Args:
        path: Path to ``location_scan_clean.csv``.

    Returns:
        Dataframe with columns ``location_id`` (int), ``lat``, ``lon``,
        ``first_file`` and a clean ``datetime`` index. Rows with unusable
        coordinates are dropped.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Station scan not found at {path}")

    df = pd.read_csv(path)

    if df.empty:
        logger.warning("Station scan %s is empty.", path)
        return df

    expect = {"location_id", "lat", "lon"}
    missing = expect - set(df.columns)
    if missing:
        raise ValueError(f"Station scan missing required columns: {sorted(missing)}")

    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")

    if "first_file" in df.columns:
        df["first_file"] = df["first_file"].replace("", np.nan)

    # Coordinates must be finite and physically plausible.
    coords_ok = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    )
    df = df.loc[coords_ok & df["location_id"].notna()].copy()
    df = df.drop_duplicates(subset=["location_id"], keep="first")

    df.index = pd.RangeIndex(len(df))
    df.index.name = "row"

    logger.info(
        "Loaded %d OpenAQ stations from %s.", len(df), path,
    )
    return df


# --------------------------------------------------------------------------- #
#  S3 archive access
# --------------------------------------------------------------------------- #


def build_s3_url(
    location_id: Optional[int] = None,
    first_file: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    base_url: str = OPENAQ_S3_BASE,
) -> str:
    """Build an S3 URL for a station's archived measurement files.

    The OpenAQ archive mirrors the CSVs under
    ``records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>.csv.gz``.
    If ``first_file`` is provided it is used verbatim; otherwise a path is
    derived from ``location_id`` (and, optionally, ``year``/``month``/``day``).
    When no time frame is given a placeholder pattern is returned.

    Args:
        location_id: OpenAQ location id.
        first_file: S3 object path taken from ``first_file`` of the scan.
        year: Four digit year for the archive folder.
        month: Month (1-12) for the archive folder.
        day: Optional day of the month (1-31) used in the file name.
        base_url: S3 endpoint (defaults to the public OpenAQ archive).

    Returns:
        Fully qualified HTTPS URL.

    Raises:
        ValueError: If neither ``first_file`` nor ``location_id`` is given.
    """
    if first_file:
        obj = first_file.lstrip("/")
    elif location_id is not None:
        prefix = f"records/csv.gz/locationid={location_id}"
        if year is None:
            # No concrete file yet: return the location's archive prefix.
            obj = prefix
        else:
            m = month or 1
            d = day or 1
            obj = (
                f"{prefix}/year={year}/month={m:02d}/"
                f"location-{location_id}-{year}{m:02d}{d:02d}.csv.gz"
            )
    else:
        raise ValueError("Provide either 'first_file' or 'location_id'.")

    return f"{base_url.rstrip('/')}/{obj}"


def fetch_station_data(
    location_id: int,
    coords_df: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    first_file: Optional[str] = None,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Fetch and parse archived measurements for a single OpenAQ station.

    Tries to download the station's measurements from OpenAQ's public S3
    archive. When ``coords_df`` (e.g. the output of :func:`load_stations`) is
    supplied, ``first_file`` is looked up automatically; otherwise a URL is
    built from ``year``/``month``/``day``.

    Args:
        location_id: OpenAQ location id.
        coords_df: Optional station scan / coordinate dataframe used to resolve
            ``first_file``.
        year: Year used if ``first_file`` cannot be resolved.
        month: Month used if ``first_file`` cannot be resolved.
        day: Day used if ``first_file`` cannot be resolved.
        first_file: Explicit S3 object path (overrides the ``coords_df`` lookup).
        **read_csv_kwargs: Extra keyword arguments forwarded to
            ``pandas.read_csv`` (e.g. ``nrows``, ``parse_dates``).

    Returns:
        Dataframe with one row per archived measurement.

    Raises:
        RuntimeError: If every resolution strategy for the archive URL fails.
    """
    url: Optional[str] = None

    if first_file is not None:
        url = build_s3_url(first_file=first_file)
    elif coords_df is not None:
        row = coords_df.loc[coords_df["location_id"] == location_id]
        if not row.empty and pd.notna(row.iloc[0].get("first_file")):
            url = build_s3_url(first_file=str(row.iloc[0]["first_file"]))

    if url is None:
        url = build_s3_url(
            location_id=location_id, year=year, month=month, day=day,
        )
    elif (year is not None) or (month is not None) or (day is not None):
        logger.info(
            "Fetching concrete archive file for location %s (%s), "
            "year=%s month=%s day=%s.",
            location_id, url, year, month, day,
        )

    logger.info("Fetching OpenAQ data for location %s <- %s", location_id, url)

    try:
        df = pd.read_csv(url, compression="gzip", **read_csv_kwargs)
    except Exception as exc:  # noqa: BLE001 - surface a useful message
        raise RuntimeError(
            f"Failed to read OpenAQ measurements for location {location_id} "
            f"from {url}: {exc!r}"
        ) from exc

    if df.empty:
        logger.warning("No measurements returned for location %s.", location_id)

    if "location_id" not in df.columns:
        df.insert(0, "location_id", location_id)

    return df


def fetch_location_summary(
    location_ids: Iterable[int],
    coords_df: Optional[pd.DataFrame] = None,
    nrows: Optional[int] = 5,
) -> pd.DataFrame:
    """Fetch a small preview of several locations (request-level helper).

    Useful for spotting corrupted stations or sanity checking the archive.
    Consecutive failed downloads are skipped with a warning.

    Args:
        location_ids: Iterable of station ids to fetch.
        coords_df: Coordinate/scan dataframe for ``first_file`` resolution.
        nrows: Number of rows per station to keep (``None`` for everything).

    Returns:
        Concatenated dataframe of fetched measurements.
    """
    frames: list[pd.DataFrame] = []
    for loc in location_ids:
        try:
            df = fetch_station_data(loc, coords_df=coords_df, nrows=nrows)
            frames.append(df)
        except RuntimeError as exc:
            logger.warning("%s", exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
#  Country assignment
# --------------------------------------------------------------------------- #


def assign_country(
    stations_df: pd.DataFrame,
    geojson_path: Path = WORLD_COUNTRIES_PATH,
    name_col: str = COUNTRY_NAME_COL,
    predicate: str = "within",
) -> pd.DataFrame:
    """Assign each station a country via point-in-polygon.

    Performs a spatial join (left) between the station points (CRS: EPSG:4326)
    and the world-borders GeoJSON so every station receives an ISO-ish country
    name column. Stations not inside any polygon get ``country = NaN``.

    Args:
        stations_df: Dataframe with ``lat`` and ``lon`` columns.
        geojson_path: Path to the countries GeoJSON file.
        name_col: Property of the GeoJSON features holding the country name.
        predicate: GeoPandas spatial predicate; ``'within'`` (default) or
            ``'intersects'``.

    Returns:
        Copy of ``stations_df`` with an added ``country`` column.

    Raises:
        FileNotFoundError: If ``geojson_path`` does not exist.
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"Countries GeoJSON not found at {geojson_path}")

    if stations_df.empty:
        logger.warning("No stations to assign; returning empty dataframe.")
        return stations_df.copy()

    points = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),
        crs="EPSG:4326",
    )

    countries = gpd.read_file(geojson_path)
    if name_col not in countries.columns:
        available = ", ".join(map(str, countries.columns[:12]))
        raise ValueError(
            f"Country name column '{name_col}' not found in {geojson_path}. "
            f"Available columns include: {available}"
        )

    countries = countries[[name_col, "geometry"]].rename(
        columns={name_col: "country"}
    )

    joined = gpd.sjoin(
        points, countries, how="left", predicate=predicate,
    )
    # Drop the geometry added by the join, keep the original point columns.
    out = joined.drop(columns=["geometry"])

    n_unassigned = int(out["country"].isna().sum())
    if n_unassigned:
        logger.warning(
            "%d station(s) could not be assigned to a country (outside all "
            "polygons).", n_unassigned,
        )

    return out


# --------------------------------------------------------------------------- #
#  End-to-end run
# --------------------------------------------------------------------------- #


def main(geojson_path: Path = WORLD_COUNTRIES_PATH) -> pd.DataFrame:
    """Load stations, assign countries and persist the processed output.

    Args:
        geojson_path: Path to the world countries GeoJSON.

    Returns:
        The enriched stations dataframe that was written to
        ``data/processed/openaq_stations.csv``.
    """
    stations = load_stations()
    enriched = assign_country(stations, geojson_path=geojson_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=True, index_label="row")

    logger.info(
        "Saved %d stations (of which %d with country) -> %s",
        len(enriched),
        int(enriched["country"].notna().sum()),
        OUTPUT_PATH,
    )

    print(f"\nOpenAQ stations with country assignment:")
    for name, count in enriched["country"].value_counts(dropna=False).head(15).items():
        print(f"  {name!r:<30} {count:>6}")
    print("...")
    return enriched


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    main()