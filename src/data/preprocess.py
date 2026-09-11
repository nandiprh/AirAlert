"""Air quality data preprocessing pipeline.

Loads, cleans and saves the three main air quality datasets used in the
hackathon project:

    1. UCI Air Quality (``data/raw/airquality_uci.csv``)
       Semicolon separated, comma decimals, missing values coded as ``-200``.
    2. India daily city AQI (``data/raw/india_city_day.csv``)
       Daily city-level pollutant and AQI readings for 26 cities, 2015-2020.
    3. Delhi CPCB station data (``data/raw/delhi_cpcb_2024_25.csv``)
       Hourly station-level readings from the Central Pollution Control Board.

The pipeline performs:

    * parsing of the idiosyncratic UCI / india / delhi formats,
    * replacement of sentinel missing values with ``NaN``,
    * city-wise / station-wise median imputation,
    * recomputation of the CPCB ``AQI_Bucket`` labels,
    * data leakage checks (duplicate rows, suspicious perfect correlations),
    * saving cleaned dataframes with datetime indexes to ``data/processed``.

Run as a script::

    python src/data/preprocess.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("air_quality.preprocess")

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# --------------------------------------------------------------------------- #
#  Dataset specific constants
# --------------------------------------------------------------------------- #

UCI_PATH = RAW_DIR / "airquality_uci.csv"
INDIA_PATH = RAW_DIR / "india_city_day.csv"
DELHI_PATH = RAW_DIR / "delhi_cpcb_2024_25.csv"

#: Sentinel used by the UCI dataset to code missing values.
UCI_MISSING = -200.0

INDIA_POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]

#: Numeric measurement columns of the Delhi CPCB dataset (everything after the
#: station metadata columns). Purely categorical / metadata columns are kept.
DELHI_NUMERIC_COLS = [
    "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)",
    "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)",
    "Ozone (µg/m³)", "Benzene (µg/m³)", "Toluene (µg/m³)",
    "Xylene (µg/m³)", "O Xylene (µg/m³)", "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)", "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)",
    "RF (mm)", "TOT-RF (mm)", "SR (W/mt2)", "BP (mmHg)", "VWS (m/s)",
]

#: CPCB AQI buckets: lower bound (inclusive) -> bucket label.
CPCB_AQI_BUCKETS: List[Tuple[float, str]] = [
    (0.0, "Good"),
    (51.0, "Moderate"),
    (101.0, "Satisfactory"),
    (151.0, "Poor"),
    (201.0, "Very Poor"),
    (301.0, "Severe"),
]

#: Correlation values above this magnitude are deemed "suspicious".
SUSPICIOUS_CORRELATION = 0.99

# --------------------------------------------------------------------------- #
#  Loading helpers
# --------------------------------------------------------------------------- #


def load_uci(path: Path = UCI_PATH) -> pd.DataFrame:
    """Load and clean the UCI Air Quality dataset.

    Handles the semicolon separator, comma decimals, the ``-200`` missing
    value sentinel, trailing unnamed columns and combined ``Date``/``Time``
    columns.

    Args:
        path: Path to ``airquality_uci.csv``.

    Returns:
        Cleaned dataframe indexed by a monotonic ``DatetimeIndex``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"UCI dataset not found at {path}")

    df = pd.read_csv(path, sep=";", decimal=",", skip_blank_lines=True)

    # Drop unnamed placeholder columns introduced by trailing separators.
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info("Dropped %d unnamed column(s) from UCI data.", len(unnamed))

    # Rows that were only separators (all NaN) are dropped.
    df = df.dropna(how="all")

    # Replace the -200 missing sentinel with NaN on every numeric column.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace(
            UCI_MISSING, np.nan
        )

    # Combine Date (DD/MM/YYYY) and Time (HH.MM.SS) into one datetime index.
    raw_datetime = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    parsed = pd.to_datetime(
        raw_datetime, format="%d/%m/%Y %H.%M.%S", errors="coerce"
    )

    valid = parsed.notna().to_numpy()
    df = df.loc[valid].copy()
    df.index = parsed[valid]
    df.index.name = "datetime"
    df = df.drop(columns=["Date", "Time"])
    df = df.sort_index()

    logger.info(
        "UCI data loaded: %d rows, %d columns, index %s -> %s",
        df.shape[0], df.shape[1], df.index.min(), df.index.max(),
    )
    return df


def load_india(path: Path = INDIA_PATH) -> pd.DataFrame:
    """Load and clean the India daily city AQI dataset.

    Args:
        path: Path to ``india_city_day.csv``.

    Returns:
        Cleaned dataframe indexed by the parsed date. The index intentionally
        contains one row per city (i.e. duplicate timestamps are expected).
    """
    if not path.exists():
        raise FileNotFoundError(f"India city-day dataset not found at {path}")

    df = pd.read_csv(path)

    dates = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["Date"] = dates
    df = df.dropna(subset=["Date"])

    df.index = df["Date"]
    df.index.name = "datetime"
    df = df.drop(columns=["Date"])
    df = df.sort_index()

    # City-wise median imputation for pollutant columns.
    df = _median_impute(df, group_col="City", cols=INDIA_POLLUTANT_COLS)

    # Recompute the AQI_Bucket label straight from the AQI reading.
    if "AQI" in df.columns:
        df["AQI_Bucket"] = _aqi_bucket(df["AQI"])

    logger.info(
        "India city-day data loaded: %d rows, %d columns, index %s -> %s",
        df.shape[0], df.shape[1], df.index.min(), df.index.max(),
    )
    return df


def load_delhi(path: Path = DELHI_PATH) -> pd.DataFrame:
    """Load and clean the Delhi CPCB station-level dataset.

    Args:
        path: Path to ``delhi_cpcb_2024_25.csv``.

    Returns:
        Cleaned dataframe indexed by the parsed ISO timestamp, with station-wise
        median imputation applied to the numeric measurement columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Delhi CPCB dataset not found at {path}")

    df = pd.read_csv(path)

    timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp"] = timestamps
    df = df.dropna(subset=["Timestamp"])

    df.index = df["Timestamp"]
    df.index.name = "datetime"
    df = df.drop(columns=["Timestamp"])
    df = df.sort_index()

    # Station-wise median imputation on the numeric pollutant / meteo columns.
    df = _median_impute(
        df, group_col="Station Name", cols=DELHI_NUMERIC_COLS,
        drop_unselected_numeric=True,
    )

    logger.info(
        "Delhi CPCB data loaded: %d rows, %d columns, index %s -> %s",
        df.shape[0], df.shape[1], df.index.min(), df.index.max(),
    )
    return df


# --------------------------------------------------------------------------- #
#  Transformation helpers
# --------------------------------------------------------------------------- #


def _median_impute(
    df: pd.DataFrame,
    group_col: str,
    cols: List[str],
    drop_unselected_numeric: bool = False,
) -> pd.DataFrame:
    """Impute missing values in ``cols`` with the group-wise column median.

    For each column the median is computed within groups of ``group_col``
    (e.g. city or station). Columns that are missing entirely within a group
    remain missing.

    Args:
        df: Input dataframe.
        group_col: Column whose unique values define the imputation groups.
        cols: Columns to impute. Should be numeric.
        drop_unselected_numeric: If ``True``, numeric columns not listed in
            ``cols`` are untouched but the listed measurement columns are
            moved to the end of the dataframe for readability.

    Returns:
        The imputed dataframe (modified in place and returned).
    """
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        logger.warning(
            "Columns requested for imputation not present in data: %s",
            missing_cols,
        )

    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert any *other* numeric-like object columns listed in `cols` only.
    present = [c for c in cols if c in df.columns]
    for col in present:
        median = df.groupby(group_col)[col].transform("median")
        df[col] = df[col].fillna(median)

    if drop_unselected_numeric:
        # Move the measurement columns to the end for readability.
        other = [c for c in df.columns if c not in present]
        df = df[other + present]

    return df


def _aqi_bucket(aqi: pd.Series) -> pd.Series:
    """Map AQI values to CPCB buckets.

    Args:
        aqi: Series of AQI values (may contain NaN).

    Returns:
        Series of bucket labels; NaN propagates for missing AQI.
    """
    labels = [label for _, label in CPCB_AQI_BUCKETS]
    # 0.0 is the descriptive lower bound; real internal edges start at 51.0.
    edges = [-np.inf] + [b for b, _ in CPCB_AQI_BUCKETS][1:] + [np.inf]
    bucket_labels = pd.cut(
        aqi, bins=edges, labels=labels, right=True, include_lowest=True
    ).astype("object")
    bucket_labels = bucket_labels.mask(aqi.isna(), np.nan)
    return bucket_labels


# --------------------------------------------------------------------------- #
#  Data leakage checks
# --------------------------------------------------------------------------- #


def flag_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Flag fully duplicated rows.

    Args:
        df: Input dataframe.

    Returns:
        Boolean series (``True`` = duplicate) aligned with the input index.
    """
    return df.duplicated(keep="first")


def check_perfect_correlations(df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """Find suspiciously high pairwise feature correlations.

    Features that are near-linearly dependent are a common source of label /
    target leakage, so pairs with ``|r| >= SUSPICIOUS_CORRELATION`` are
    reported. Constant or near-constant columns are skipped (their correlation
    is undefined).

    Args:
        df: Dataframe holding at least two numeric columns.

    Returns:
        List of ``(column_a, column_b, correlation)`` tuples sorted by absolute
        correlation descending.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 2:
        return []

    sample = df[numeric].dropna(how="all")

    # Drop zero-variance columns: their correlation with anything is undefined.
    valid = []
    for col in numeric:
        ser = sample[col]
        if len(ser.dropna()) < 2:
            continue
        if ser.std(ddof=0) == 0 or not np.isfinite(ser.std(ddof=0)):
            continue
        valid.append(col)
    if len(valid) < 2:
        return []

    corr = sample[valid].corr(min_periods=10).abs()

    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if not np.isfinite(r):
                continue
            if r >= SUSPICIOUS_CORRELATION:
                pairs.append((a, b, float(r)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def run_leakage_checks(
    name: str, df: pd.DataFrame
) -> Dict[str, Any]:
    """Run all data leakage checks on a dataframe and report results.

    Args:
        name: Human readable dataset name (used for logging).
        df: Cleaned dataframe.

    Returns:
        Dictionary summary containing the number of duplicate rows and the list
        of suspicious correlations.
    """
    dup_mask = flag_duplicate_rows(df)
    n_dups = int(dup_mask.sum())

    suspicious = check_perfect_correlations(df)

    summary: Dict[str, Any] = {
        "name": name,
        "n_duplicates": n_dups,
        "n_suspicious_correlations": len(suspicious),
        "suspicious_correlations": suspicious,
    }

    if n_dups:
        logger.warning("%s: %d duplicated row(s) flagged.", name, n_dups)
    else:
        logger.info("%s: no duplicate rows.", name)

    if suspicious:
        for a, b, r in suspicious[:10]:
            logger.warning(
                "%s: suspicious correlation %.4f between '%s' and '%s'.",
                name, r, a, b,
            )
    return summary


# --------------------------------------------------------------------------- #
#  Saving helpers
# --------------------------------------------------------------------------- #


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save an index-aligned dataframe to ``data/processed``.

    Args:
        df: Cleaned dataframe (datetime index).
        name: Output file base name (e.g. ``'uci'``).

    Returns:
        Path of the written CSV file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(out, index=True, index_label=df.index.name)
    logger.info("Saved %s -> %s (%d rows)", name, out, df.shape[0])
    return out


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline() -> Dict[str, Any]:
    """Execute the full preprocessing pipeline.

    Returns:
        Summary dictionary keyed by dataset name with shapes, date ranges,
        missing value counts and leakage check results.
    """
    summary: Dict[str, Any] = {}

    for name, loader in [
        ("uci", load_uci),
        ("india_city_day", load_india),
        ("delhi_cpcb", load_delhi),
    ]:
        df = loader()
        path = save_processed(df, name)
        leakage = run_leakage_checks(name, df)

        summary[name] = {
            "path": path,
            "shape": df.shape,
            "index_start": df.index.min(),
            "index_end": df.index.max(),
            "missing_total": int(df.isna().sum().sum()),
            "leakage": leakage,
        }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the pipeline summary produced by :func:`run_pipeline`."""
    width = 90
    print("=" * width)
    print("AIR QUALITY PREPROCESSING SUMMARY")
    print("=" * width)

    for name, info in summary.items():
        print(f"\n[{name}]")
        print(f"  file        : {info['path']}")
        print(f"  rows x cols : {info['shape'][0]} x {info['shape'][1]}")
        print(f"  index range : {info['index_start']}  ->  {info['index_end']}")
        print(f"  missing vals: {info['missing_total']:,}")
        leak = info["leakage"]
        print(f"  duplicates  : {leak['n_duplicates']:,}")
        n_corr = leak["n_suspicious_correlations"]
        print(f"  suspicious correlations: {n_corr}")
        for a, b, r in leak["suspicious_correlations"][:10]:
            print(f"      {a} <-> {b}  (r = {r:.4f})")

    print("\n" + "=" * width)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    summary = run_pipeline()
    print_summary(summary)


if __name__ == "__main__":
    main()