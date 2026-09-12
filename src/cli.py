"""Single entry point for the packaged ``aqi`` binary (and CLI from source).

Wraps every existing entry point so one executable exposes the whole
project:

    aqi train         -- train/evaluate a model (src/train.py)
    aqi evaluate      -- evaluate a checkpoint (src/evaluate.py)
    aqi predict       -- per-city next-day AQI forecast (src/predict.py)
    aqi map-global    -- world + India AQI maps (map_global.py)
    aqi map-cities    -- city prediction maps (map_cities.py)
    aqi benchmark     -- 5-model single-config benchmark (scripts/benchmark_models.py)
    aqi sweep         -- parameter-sweep benchmark (benchmarking/run_sweep.py)
    aqi version       -- build info

When frozen with PyInstaller the bundled tree lives in ``sys._MEIPASS``, so
the same BASE resolution works from source *and* from the binary.
"""

from __future__ import annotations

import os
import runpy
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys, "_MEIPASS"):
    BASE = sys._MEIPASS  # type: ignore[attr-defined]
sys.path.insert(0, BASE)

USAGE = """aqi — air-quality forecasting toolkit

usage: aqi <command> [options]

commands:
  train        train/evaluate one of the 5 hybrid models (see --help)
  evaluate     evaluate a saved checkpoint
  predict      per-city next-day AQI forecast (needs preprocessed data)
  map-global   world + India AQI maps (geojson boundaries bundled)
  map-cities   India + world city-prediction maps
  benchmark    single-config 5-model comparison (scripts/benchmark_models.py)
  sweep        parameter-sweep benchmark (benchmarking/run_sweep.py)
  version      print build/version info

examples:
  aqi train --dataset uci --model lstm_cnn --seq_len 24 --task both
  aqi predict --model lstm_cnn --seq_len 7 --epochs 5
  aqi map-global
  aqi sweep --epochs 5
"""


def _bundle_path(rel):
    """Path of a bundled-but-shipped file (scripts/, benchmarking/)."""
    return os.path.join(BASE, rel)


def _precreate_dirs():
    for sub in ("models/checkpoints", "models/reports", "output", "data/processed/predictions"):
        try:
            os.makedirs(os.path.join(BASE, sub), exist_ok=True)
        except OSError:
            pass


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if not args else 0
    if args[0] == "help":
        print(USAGE)
        return 0

    cmd, rest = args[0], args[1:]
    sys.argv = ["aqi"] + rest

    _precreate_dirs()

    if cmd == "version":
        import torch
        from src.models import MODEL_REGISTRY
        print("aqi binary")
        print("built     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("python    : %s" % sys.version.split()[0])
        print("pytorch   : %s" % torch.__version__)
        print("models    : %s" % ", ".join(MODEL_REGISTRY))
        print("base      : %s" % BASE)
        return 0

    if cmd == "train":
        from src.train import main as run
    elif cmd == "evaluate":
        from src.evaluate import main as run
    elif cmd == "predict":
        from src.predict import main as run
    elif cmd == "map-global":
        from src.visualization.map_global import main as run
    elif cmd == "map-cities":
        from src.visualization.map_cities import main as run
    elif cmd == "benchmark":
        runpy.run_path(_bundle_path("scripts/benchmark_models.py"), run_name="__main__")
        return 0
    elif cmd == "sweep":
        runpy.run_path(_bundle_path("benchmarking/run_sweep.py"), run_name="__main__")
        return 0
    else:
        print("aqi: unknown command '%s'" % cmd, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())