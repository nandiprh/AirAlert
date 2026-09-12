"""Parameter-sweep benchmark of the 5 hybrid AQI models.

Trains every model in ``MODEL_REGISTRY`` on each (dataset x grid-point)
combination with a fixed training config, varying only the *architectural*
hyper-parameters under test (``seq_len``, ``hidden_size``). Every run records
the full hyper-parameter config actually used (``Config``), the test
regression/classification metrics, parameter count, wall-time, inference
latency and the training loss history, then logs one JSON file per run under
``benchmarking/results/<dataset>/<model>__<gridpoint>.json``.

Usage:
    python benchmarking/run_sweep.py                       # uci + india, full grid
    python benchmarking/run_sweep.py --dataset uci --grid light --epochs 30
    python benchmarking/run_sweep.py --seq-lens 7 24 --hidden-sizes 32 128

``--resume`` (default on) skips runs whose result JSON already exists, so
interrupted sweeps can be continued.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY
from src.train import Config, preprocess_dataset, run_training

BENCH_DIR = Path(__file__).resolve().parent
RESULT_DIR = BENCH_DIR / "results"
CKPT_DIR = BENCH_DIR / "ckpt"
LOG_DIR = BENCH_DIR / "logs"


def clean_metrics(metrics):
    """Drop non-serializable fields, coerce numpy scalars to float."""
    out = {}
    for k, v in metrics.items():
        if "_y_true" in k or "_y_pred" in k:
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        out[k] = v
    return out


def inference_timing(model, input_size, seq_len, batch_size, reps=50):
    import torch

    model.eval()
    x = torch.randn(batch_size, seq_len, input_size)
    with torch.no_grad():
        model(x)  # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t0) / reps * 1000.0


def build_grid(seq_lens, hidden_sizes, epochs, dropout):
    grid = []
    for seq, hid in itertools.product(seq_lens, hidden_sizes):
        grid.append((f"s{seq}_h{hid}", seq, hid, epochs, dropout))
    return grid


def main():
    p = argparse.ArgumentParser(description="Sweep hyper-parameters across all models")
    p.add_argument("--dataset", choices=["uci", "india", "both"], default="both")
    p.add_argument("--grid", choices=["full", "light", "triple", "single"],
                   default="full",
                   help="full = seq_len x hidden_size cross product; "
                        "light = seq_len{7,24} x hidden{64}; "
                        "triple = {s7_h64, s24_h32, s24_h128}; single = baseline")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip runs whose result JSON already exists")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.grid == "single":
        grid = build_grid([24], [64], args.epochs, 0.2)
    elif args.grid == "light":
        grid = build_grid([7, 24], [64], args.epochs, 0.2)
    elif args.grid == "triple":
        grid = [("s7_h64", 7, 64, args.epochs, 0.2),
                ("s24_h32", 24, 32, args.epochs, 0.2),
                ("s24_h128", 24, 128, args.epochs, 0.2)]
    else:  # full
        grid = build_grid([7, 24], [32, 128], args.epochs, 0.2)

    datasets = ["uci", "india"] if args.dataset == "both" else [args.dataset]

    # One shared data blob per (dataset, seq_len) so every model sees the same
    # exact windows/scaling for a fair head-to-head.
    cache = {}
    for ds in datasets:
        cache[ds] = {}
        for cfgid, seq, hid, epochs, drop in grid:
            if seq not in cache[ds]:
                data = preprocess_dataset(ds, seq_len=seq, task="both",
                                          num_classes=4, city=None,
                                          train_frac=0.8, val_frac=0.0)
                meta = data["metadata"]
                print(f"[setup] dataset={ds} seq_len={seq} "
                      f"train={meta['n_train']} val={meta['n_val']} "
                      f"test={meta['n_test']} features={meta['input_size']}",
                      flush=True)
                cache[ds][seq] = data

    log_path = LOG_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.log"
    summary = {}
    failures = []

    for ds in datasets:
        summary[ds] = {}
        for model in MODEL_REGISTRY:
            summary[ds][model] = {}
            for cfgid, seq, hid, epochs, drop in grid:
                out_path = RESULT_DIR / ds / f"{model}__{cfgid}.json"
                if args.resume and out_path.exists():
                    print(f"[skip] {ds}/{model}__{cfgid} (resume)", flush=True)
                    summary[ds][model][cfgid] = json.loads(
                        out_path.read_text())
                    continue

                cfg = Config(
                    epochs=epochs, seq_len=seq, batch_size=args.batch_size,
                    lr=1e-3, hidden_size=hid, num_layers=2, dropout=drop,
                    task="both", patience=12, early_stopping=True,
                    lr_schedule="plateau", city=None, weight_decay=1e-5,
                    overfit_budget=8, underfit_budget=8, max_lr_reductions=2,
                    quiet=args.quiet, no_smote=False,
                )

                ckpt = CKPT_DIR / f"{model}__{cfgid}_{ds}.pt"
                t0 = time.perf_counter()
                try:
                    res = run_training(model, cache[ds][seq], cfg,
                                       checkpoint_path=str(ckpt),
                                       seed=args.seed)
                except Exception as exc:  # noqa: BLE001 - log + keep going
                    failures.append({"dataset": ds, "model": model,
                                     "cfgid": cfgid, "error": str(exc),
                                     "config": dict(cfg)})
                    print(f"[FAIL] {ds}/{model}__{cfgid}: {exc!r}", flush=True)
                    continue
                train_s = time.perf_counter() - t0

                n_params = sum(p.numel() for p in res["model"].parameters())
                meta = cache[ds][seq]["metadata"]
                infer_ms = inference_timing(res["model"], meta["input_size"],
                                            meta["seq_len"], args.batch_size)

                metrics = clean_metrics(
                    {**res["val_metrics"], **res["test_metrics"]})
                metrics.pop("val_loss", None)
                entry = {
                    "dataset": ds,
                    "model": model,
                    "cfgid": cfgid,
                    "config": dict(cfg),
                    "task": "both",
                    "input_size": meta["input_size"],
                    "corrections": res["corrections"],
                    "detection": res["reason"],
                    "epochs": len(res["history"]),
                    "params": int(n_params),
                    "train_time_s": round(train_s, 2),
                    "inference_ms_per_batch": round(infer_ms, 3),
                    "checkpoint_mb": round(os.path.getsize(ckpt) / 1e6, 2),
                    "history": [{k: float(v) for k, v in h.items()}
                                for h in res["history"]],
                    **{k: (round(float(v), 4) if isinstance(v, (int, float))
                           else v) for k, v in metrics.items()},
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(entry, indent=2))
                summary[ds][model][cfgid] = entry
                print(f"[done] {ds}/{model}__{cfgid} "
                      f"rmse={entry.get('test_rmse')} r2={entry.get('test_r2')} "
                      f"acc={entry.get('test_accuracy')} "
                      f"f1={entry.get('test_f1')} "
                      f"params={entry['params']:,} {entry['train_time_s']}s "
                      f"detect={entry['detection']}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (BENCH_DIR / f"summary_{stamp}.json").write_text(
        json.dumps(summary, indent=2))
    (BENCH_DIR / f"summary_{stamp}.csv").write_text(_to_csv(summary))
    with open(log_path, "w") as f:
        json.dump({"time": stamp, "failures": failures}, f, indent=2)
    print(f"\n[save] summary -> benchmarking/summary_{stamp}.json / .csv")
    if failures:
        print(f"[warn] {len(failures)} failed runs -> benchmarking/logs/"
              f"{log_path.name}")


def _to_csv(summary):
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    keys = ["dataset", "model", "cfgid", "epochs", "params", "train_time_s",
            "inference_ms_per_batch", "checkpoint_mb", "test_rmse", "test_mae",
            "test_r2", "test_accuracy", "test_precision", "test_recall",
            "test_f1", "detection", "corrections"]
    w.writerow(keys)
    for ds, models in summary.items():
        for model, runs in models.items():
            for cfgid, e in runs.items():
                w.writerow([e.get(k, "") for k in keys])
    return buf.getvalue()


if __name__ == "__main__":
    main()