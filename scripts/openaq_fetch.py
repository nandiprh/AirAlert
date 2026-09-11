#!/usr/bin/env python3
"""Download OpenAQ day-file measurements for scanned stations to local disk.

The map script prefers local files before hitting the S3 bucket, so pre-fetching
here makes the visualization fully offline-capable and reproducible.

Usage:
    python scripts/openaq_fetch.py \
        --stations data/raw/openaq/location_scan_clean.csv \
        --outdir data/raw/openaq/measurements \
        --max-per-batch 12
"""

import argparse
import gzip
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# --- pandas is only used for column access; drop to pure python if unavailable ---
try:
    import pandas as pd  # noqa: F811
    _PANDAS = True
except Exception:  # noqa: BLE001
    _PANDAS = False

ARCHIVE = "https://openaq-data-archive.s3.amazonaws.com/"
HEADERS = {"User-Agent": "air-quality-hackathon/1.0 (+https://openaq.org)"}


def fetch_one(args):
    loc_id, key, outdir = args
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"location-{loc_id}.csv.gz"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (loc_id, "cached")
    try:
        req = urllib.request.Request(ARCHIVE + key, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        # sanity: must decompress as gzip
        gzip.decompress(raw)
        out_path.write_bytes(raw)
        return (loc_id, "ok")
    except Exception as exc:
        return (loc_id, f"error:{type(exc).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True,
                    help="CSV with location_id, first_file (location_scan_clean.csv)")
    ap.add_argument("--outdir", default="data/raw/openaq/measurements")
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only fetch N stations (0=all)")
    args = ap.parse_args()

    st = pd.read_csv(args.stations)
    st = st.dropna(subset=["first_file"])
    st = st[st["first_file"].astype(str).str.startswith("records/")]
    if args.limit:
        st = st.head(args.limit)

    jobs = [(int(r.location_id), str(r.first_file), args.outdir) for r in st.itertuples()]
    print(f"[openaq_fetch] downloading {len(jobs)} station files ...")

    stats = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed(pool.submit(fetch_one, j) for j in jobs):
            loc_id, status = fut.result()
            tag = status.split(":")[0]
            stats[tag] = stats.get(tag, 0) + 1
            done = sum(stats.values())
            if done % 20 == 0:
                print(f"[openaq_fetch] {done}/{len(jobs)} {dict(stats)}", file=sys.stderr)

    print(f"[openaq_fetch] done -> {args.outdir}  {dict(stats)}")


if __name__ == "__main__":
    main()