#!/usr/bin/env python3
"""Discover worldwide OpenAQ stations in the public AWS S3 archive.

For each sampled location id we fetch ONE daily measurement file, decompress it and
record the station coordinates so the data can be mapped / filtered later.

Usage:
    python scripts/openaq_scan.py --ids /tmp/opencode/locids.txt \
        --sample 12000 --out data/raw/openaq/location_scan.csv \
        --skip-existing data/raw/openaq/location_scan_clean.csv

Output CSV columns: location_id, lat, lon, first_file
"""

import argparse
import csv
import gzip
import os
import random
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

ARCHIVE = "https://openaq-data-archive.s3.amazonaws.com/"


def s3_list_keys(prefix, max_keys=1, timeout=8):
    url = f"{ARCHIVE}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys={max_keys}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        return [c.findtext("s:Key", default="", namespaces=ns) for c in root.findall("s:Contents", ns)]
    except Exception:  # noqa: BLE001
        return []


def s3_fetch_bytes(key, timeout=8):
    try:
        with urllib.request.urlopen(f"{ARCHIVE}{key}", timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001
        return b""


def probe(location_id):
    keys = s3_list_keys(f"records/csv.gz/locationid={location_id}/")
    if not keys:
        return (location_id, "", "", "")
    first = keys[0]
    raw = s3_fetch_bytes(first)
    try:
        text = gzip.decompress(raw).decode("utf-8", "replace")
        data_line = text.split("\n", 2)[1] if text.count("\n") >= 1 else ""
        parts = data_line.split(",")
        if len(parts) >= 6:
            return (location_id, parts[4].strip(), parts[5].strip(), first)
    except Exception:  # noqa: BLE001
        pass
    return (location_id, "", "", first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default="data/raw/openaq/location_scan.csv")
    ap.add_argument("--skip-existing", default="")
    args = ap.parse_args()

    with open(args.ids, encoding="utf-8") as fh:
        ids = [ln.strip() for ln in fh if ln.strip().isdigit()]

    done_before = set()
    if args.skip_existing and os.path.exists(args.skip_existing):
        with open(args.skip_existing, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["location_id"]:
                    done_before.add(r["location_id"])

    remaining = [i for i in ids if i not in done_before]
    random.seed(42)
    sampled = random.sample(remaining, min(args.sample, len(remaining)))

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, i): i for i in sampled}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"[scan] {done}/{len(sampled)}", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["location_id", "lat", "lon", "first_file"])
        w.writerows(r for r in rows if r[1])
    with_geo = sum(1 for r in rows if r[1])
    print(f"[scan] Wrote {len(rows)} valid rows ({with_geo} with geo) -> {args.out}")


if __name__ == "__main__":
    main()