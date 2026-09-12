"""Controlled head-to-head benchmark of the 5 hybrid models.

Trains every model in MODEL_REGISTRY on the same UCI dataset with an
identical config (fixed seed, seq_len, epochs, task), then records:

- accuracy: test RMSE/MAE/R2 and classification accuracy/F1
- efficiency: parameter count, training wall-time, inference ms/batch,
  checkpoint size

Writes a comparison report to ``models/reports/model_comparison_*.json``
and prints a ranked summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY
from src.train import CKPT_DIR, REPORT_DIR, Config, preprocess_dataset, run_training


def clean(metrics):
    """Drop non-serializable arrays, coerce numpy scalars to float."""
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
    model.eval()
    x = torch.randn(batch_size, seq_len, input_size)
    with torch.no_grad():
        model(x)  # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    p = argparse.ArgumentParser(description="Benchmark all five hybrid models")
    p.add_argument("--dataset", choices=["uci", "india"], default="uci")
    p.add_argument("--city", default=None)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = Config(
        epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size,
        lr=1e-3, hidden_size=64, num_layers=2, dropout=0.2,
        task="both", patience=12, early_stopping=True, lr_schedule="plateau",
        city=args.city, weight_decay=1e-5, overfit_budget=8,
        underfit_budget=8, max_lr_reductions=2, quiet=args.quiet,
        no_smote=False,
    )

    data = preprocess_dataset(args.dataset, seq_len=args.seq_len, task="both",
                              num_classes=4, city=args.city,
                              train_frac=0.8, val_frac=0.0)
    meta = data["metadata"]
    print(f"[setup] dataset={args.dataset} seq_len={args.seq_len} "
          f"train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
          f"features={meta['input_size']}")

    report = {}
    for name in MODEL_REGISTRY:
        print(f"\n===== Bench: {name} =====")
        ckpt = os.path.join(CKPT_DIR, f"bench_{name}_{args.dataset}_both_s{args.seq_len}.pt")
        t0 = time.perf_counter()
        res = run_training(name, data, cfg, checkpoint_path=ckpt, seed=args.seed)
        train_s = time.perf_counter() - t0

        model = res["model"]  # already moved to cpu
        n_params = sum(p.numel() for p in model.parameters())
        infer_ms = inference_timing(model, meta["input_size"], meta["seq_len"],
                                    args.batch_size)

        metrics = clean({**res["val_metrics"], **res["test_metrics"]})
        metrics.pop("val_loss", None)
        entry = {
            "dataset": args.dataset,
            "task": "both",
            "seq_len": args.seq_len,
            "corrections": res["corrections"],
            "detection": res["reason"],
            "epochs": len(res["history"]),
            "params": int(n_params),
            "train_time_s": round(train_s, 2),
            "inference_ms_per_batch": round(infer_ms, 3),
            "checkpoint_mb": round(os.path.getsize(ckpt) / 1e6, 2),
            **{k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()},
        }
        report[name] = entry
        print(f"  params={entry['params']:,} train={entry['train_time_s']}s "
              f"infer={entry['inference_ms_per_batch']}ms/batch "
              f"test_rmse={entry.get('test_rmse')} "
              f"test_r2={entry.get('test_r2')} "
              f"test_acc={entry.get('test_accuracy')} "
              f"test_f1={entry.get('test_f1')} "
              f"corrections={entry['corrections']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPORT_DIR, f"model_comparison_{ts}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[save] comparison report -> {out}")

    def rank(key, reverse=False):
        return sorted(report.items(), key=lambda kv: kv[1].get(key, float("inf")),
                      reverse=reverse)

    print("\n===== Regression accuracy rank (test RMSE, lower=better) =====")
    for i, (name, m) in enumerate(rank("test_rmse"), 1):
        print(f"  {i}. {name:22s} rmse={m['test_rmse']:.4f} ")
        print(f"  {'':4}mae={m['test_mae']:.4f} r2={m['test_r2']:.4f} "
              f"acc={m['test_accuracy']:.4f} f1={m['test_f1']:.4f} "
              f"params={m['params']:,} infer={m['inference_ms_per_batch']}ms")

    print("\n===== Efficiency rank (params, fewest=best) =====")
    for i, (name, m) in enumerate(rank("params"), 1):
        print(f"  {i}. {name:22s} params={m['params']:,} "
              f"train={m['train_time_s']}s infer={m['inference_ms_per_batch']}ms/batch "
              f"ckpt={m['checkpoint_mb']}MB")


if __name__ == "__main__":
    main()