"""Controlled benchmark of the oversampling strategies (SMOTE family).

Compares the resampling passes of this project on **identical** data
(windows, scalers, split), model, epoch budget and seed, so that the only
thing that changes between arms is the resampling strategy:

    none            no oversampling at all
    cls             classification SMOTE (the project default)
    cls_reg         classification SMOTE + regression SMOTE (--smote_reg)
    cls_smoter      classification SMOTE + SMOTE-R (--smote_r)
    cls_reg_smoter  all three passes
    smoter          SMOTE-R only (no classification SMOTE)

Besides the usual test metrics it reports the two things that actually
matter for an imbalanced *regression* target:

* **tail metrics** - RMSE/MAE restricted to the top and bottom decile of the
  test targets (the rare days SMOTE-R is supposed to help with);
* **distribution fidelity** - how the resampled training target
  distribution compares with the untouched test distribution (KS distance,
  mass above the test 90th percentile, mean/std).

Usage:
    python benchmarking/benchmark_smote.py                       # default grid
    python benchmarking/benchmark_smote.py --models lstm_cnn --seeds 42 43
    python benchmarking/benchmark_smote.py --dataset india --city Delhi

Outputs:
    benchmarking/results/smote/<arm>__<model>__s<seed>.json   one file / run
    benchmarking/smote_summary_<stamp>.json / .csv
    benchmarking/smote_comparison.md                            write-up
    benchmarking/charts/smote_*.png                             figures
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY  # noqa: E402
from src.train import (  # noqa: E402
    Config,
    apply_smote,
    apply_smote_regression,
    preprocess_dataset,
    run_training,
    task_enabled,
)
from src.data.smote_r import apply_smote_r  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
RESULT_DIR = BENCH_DIR / "results" / "smote"
CKPT_DIR = BENCH_DIR / "ckpt" / "smote"
CHART_DIR = BENCH_DIR / "charts"

# --- the benchmark arms --------------------------------------------------
ARMS: Dict[str, Dict[str, bool]] = {
    "none": {"no_smote": True, "smote_reg": False, "smote_r": False},
    "cls": {"no_smote": False, "smote_reg": False, "smote_r": False},
    "cls_reg": {"no_smote": False, "smote_reg": True, "smote_r": False},
    "cls_smoter": {"no_smote": False, "smote_reg": False, "smote_r": True},
    "cls_reg_smoter": {"no_smote": False, "smote_reg": True, "smote_r": True},
    "smoter": {"no_smote": True, "smote_reg": False, "smote_r": True},
}

ARM_LABEL = {
    "none": "no resampling",
    "cls": "classification SMOTE",
    "cls_reg": "cls SMOTE + reg SMOTE",
    "cls_smoter": "cls SMOTE + SMOTE-R",
    "cls_reg_smoter": "cls SMOTE + reg SMOTE + SMOTE-R",
    "smoter": "SMOTE-R only",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def clean_metrics(metrics: Dict) -> Dict:
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


def resample_arm(data: Dict, arm: Dict, seed: int) -> Dict:
    """Apply an arm's resampling passes to the training split.

    Mirrors the sequence executed inside ``src/train.py:run_training`` so the
    distribution audit below sees exactly the arrays the model is trained on.
    """
    meta = data["metadata"]
    use_cls = task_enabled(meta["task"], "classification")
    use_reg = task_enabled(meta["task"], "regression")

    X = data["X_train"].astype(np.float32)
    y_cls = data["y_cls_train"]
    y_reg = data["y_reg_train"]
    stats = {"n_start": int(len(X)), "log": []}

    buf = io.StringIO()
    if use_cls and not arm["no_smote"]:
        with contextlib.redirect_stdout(buf):
            X, y_cls, y_reg = apply_smote(
                X, np.asarray(y_cls, dtype=np.int64), y_reg,
                meta["seq_len"], meta["input_size"], seed)
        stats["after_cls"] = int(len(X))
    if use_reg and arm["smote_reg"]:
        with contextlib.redirect_stdout(buf):
            X, y_reg, y_cls = apply_smote_regression(
                X, y_reg, meta["seq_len"], meta["input_size"],
                y_cls_train=np.asarray(y_cls, dtype=np.int64) if use_cls else None,
                random_state=seed)
        stats["after_reg"] = int(len(X))
    if use_reg and arm["smote_r"]:
        with contextlib.redirect_stdout(buf):
            X, y_reg, y_cls = apply_smote_r(
                X, y_reg, meta["seq_len"], meta["input_size"],
                y_cls_train=np.asarray(y_cls, dtype=np.int64) if use_cls else None,
                random_state=seed)
        stats["after_smoter"] = int(len(X))

    stats["log"] = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    stats["n_final"] = int(len(X))
    return {"X": X, "y_reg": y_reg, "y_cls": y_cls, "stats": stats}


def distribution_audit(y_train: np.ndarray, y_test: np.ndarray) -> Dict:
    """Compare the resampled training target with the untouched test target.

    Both vectors live in the same z-scored space (the train-fitted target
    scaler is applied to every split), so they are directly comparable.
    """
    from scipy.stats import ks_2samp

    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    hi = float(np.quantile(y_test, 0.9))
    lo = float(np.quantile(y_test, 0.1))
    return {
        "ks_to_test": float(ks_2samp(y_train, y_test).statistic),
        "train_hi_decile_mass": float((y_train >= hi).mean()),
        "test_hi_decile_mass": 0.10,
        "train_lo_decile_mass": float((y_train <= lo).mean()),
        "test_lo_decile_mass": 0.10,
        "train_mean": float(y_train.mean()),
        "test_mean": float(y_test.mean()),
        "train_std": float(y_train.std()),
        "test_std": float(y_test.std()),
        "train_max": float(y_train.max()),
        "test_max": float(y_test.max()),
    }


def tail_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """RMSE/MAE split by the target deciles of the test set."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    hi_thr = float(np.quantile(y_true, 0.9))
    lo_thr = float(np.quantile(y_true, 0.1))
    masks = {
        "hi": y_true >= hi_thr,
        "lo": y_true <= lo_thr,
        "mid": (y_true < hi_thr) & (y_true > lo_thr),
    }
    out = {}
    for name, m in masks.items():
        if m.sum() == 0:
            out[f"rmse_{name}"] = float("nan")
            out[f"mae_{name}"] = float("nan")
            continue
        err = y_pred[m] - y_true[m]
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"n_{name}"] = int(m.sum())
    out["hi_threshold"] = hi_thr
    out["lo_threshold"] = lo_thr
    return out


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark the SMOTE / SMOTE-R passes")
    p.add_argument("--dataset", choices=["uci", "india"], default="uci")
    p.add_argument("--city", default=None)
    p.add_argument("--models", nargs="*", default=["lstm_cnn", "cnn_bilstm",
                                                  "transformer_lstm"],
                   help="models to benchmark (default: 3)")
    p.add_argument("--arms", nargs="*", default=list(ARMS),
                   help="resampling arms to compare")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    p.add_argument("--seq-len", dest="seq_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--task", choices=["both", "regression", "classification"],
                   default="both")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    models = [m for m in args.models if m in MODEL_REGISTRY] or ["lstm_cnn"]
    arms = [a for a in args.arms if a in ARMS]

    print(f"[setup] dataset={args.dataset} seq_len={args.seq_len} "
          f"epochs={args.epochs} task={args.task} models={models} "
          f"arms={arms} seeds={args.seeds}", flush=True)

    data = preprocess_dataset(args.dataset, seq_len=args.seq_len, task=args.task,
                              num_classes=4, city=args.city,
                              train_frac=0.8, val_frac=0.0)
    meta = data["metadata"]
    print(f"[setup] windows: train={meta['n_train']} val={meta['n_val']} "
          f"test={meta['n_test']} features={meta['input_size']}", flush=True)

    tag = args.dataset + (f"_{args.city}" if args.city else "")
    runs: List[Dict] = []
    for arm in arms:
        for seed in args.seeds:
            # --- distribution audit (no training needed) ------------------
            audit = resample_arm(data, ARMS[arm], seed)
            dist = distribution_audit(audit["y_reg"], data["y_reg_test"])

            for model in models:
                out_path = RESULT_DIR / f"{tag}__{arm}__{model}__s{seed}.json"
                if args.resume and out_path.exists():
                    entry = json.loads(out_path.read_text())
                    print(f"[skip] {arm}/{model}/s{seed}", flush=True)
                    runs.append(entry)
                    continue

                cfg = Config(
                    epochs=args.epochs, seq_len=args.seq_len,
                    batch_size=args.batch_size, lr=1e-3, hidden_size=64,
                    num_layers=2, dropout=0.2, task=args.task, patience=12,
                    early_stopping=True, lr_schedule="plateau", city=args.city,
                    weight_decay=1e-5, overfit_budget=8, underfit_budget=8,
                    max_lr_reductions=2, quiet=True,
                    no_smote=ARMS[arm]["no_smote"],
                    smote_reg=ARMS[arm]["smote_reg"],
                    smote_r=ARMS[arm]["smote_r"],
                )
                ckpt = CKPT_DIR / f"{arm}__{model}__s{seed}.pt"
                t0 = time.perf_counter()
                res = run_training(model, data, cfg, checkpoint_path=str(ckpt),
                                   seed=seed)
                train_s = time.perf_counter() - t0

                metrics = clean_metrics({**res["val_metrics"],
                                         **res["test_metrics"]})
                metrics.pop("val_loss", None)
                tails = tail_metrics(res["test_metrics"]["test_y_true"],
                                     res["test_metrics"]["test_y_pred"])

                entry = {
                    "dataset": args.dataset,
                    "city": args.city,
                    "model": model,
                    "arm": arm,
                    "arm_label": ARM_LABEL[arm],
                    "seed": seed,
                    "seq_len": args.seq_len,
                    "epochs_run": len(res["history"]),
                    "corrections": res["corrections"],
                    "detection": res["reason"],
                    "train_time_s": round(train_s, 2),
                    "resample": audit["stats"],
                    "distribution": dist,
                    "tails": tails,
                    # flattened copies so the aggregation helpers can pick
                    # them up directly
                    "rmse_hi": tails["rmse_hi"], "mae_hi": tails["mae_hi"],
                    "rmse_lo": tails["rmse_lo"], "rmse_mid": tails["rmse_mid"],
                    "ks_to_test": dist["ks_to_test"],
                    "train_hi_decile_mass": dist["train_hi_decile_mass"],
                    "train_lo_decile_mass": dist["train_lo_decile_mass"],
                    "n_train_final": audit["stats"]["n_final"],
                    **{k: (round(float(v), 4) if isinstance(v, (int, float))
                           else v) for k, v in metrics.items()},
                }
                out_path.write_text(json.dumps(entry, indent=2, default=float))
                runs.append(entry)
                print(f"[done] {arm:16s} {model:18s} s{seed} "
                       f"rmse={entry.get('test_rmse')} "
                       f"tail_rmse_hi={tails['rmse_hi']:.3f} "
                       f"ks={dist['ks_to_test']:.3f} "
                       f"n_train={audit['stats']['n_final']} "
                       f"{train_s:.0f}s", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (BENCH_DIR / f"smote_summary_{tag}_{stamp}.json").write_text(
        json.dumps(runs, indent=2, default=float))
    (BENCH_DIR / f"smote_summary_{tag}_{stamp}.csv").write_text(_to_csv(runs))
    print(f"\n[save] summary -> benchmarking/smote_summary_{tag}_{stamp}.json / .csv")

    write_report(runs, args, stamp, tag)
    print(f"[save] report -> benchmarking/smote_comparison_{tag}.md")


# ---------------------------------------------------------------------------
# aggregation + report
# ---------------------------------------------------------------------------

def _agg(rows: List[Dict], key: str) -> Dict[str, float]:
    """Mean/std of a metric per arm (over models x seeds)."""
    out = {}
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        vals = [r[key] for r in rows
                if r["arm"] == arm and r.get(key) is not None
                and not (isinstance(r.get(key), float) and np.isnan(r[key]))]
        if vals:
            out[arm] = {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals)}
    return out


def _to_csv(runs: List[Dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    keys = ["dataset", "model", "arm", "seed", "epochs_run", "train_time_s",
            "test_rmse", "test_mae", "test_r2", "test_accuracy", "test_f1"]
    tail_keys = ["rmse_hi", "mae_hi", "rmse_lo", "rmse_mid"]
    dist_keys = ["ks_to_test", "train_hi_decile_mass", "train_lo_decile_mass",
                 "train_std", "test_std"]
    w.writerow(keys + tail_keys + dist_keys + ["n_train_final"])
    for r in runs:
        row = [r.get(k, "") for k in keys]
        row += [r["tails"].get(k, "") for k in tail_keys]
        row += [r["distribution"].get(k, "") for k in dist_keys]
        row += [r["resample"].get("n_final", "")]
        w.writerow(row)
    return buf.getvalue()


def _fmt(entry: Dict, digits: int = 3) -> str:
    if not entry:
        return "—"
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f}"


def write_report(runs: List[Dict], args, stamp: str, tag: str = "uci") -> None:
    """Write benchmarking/smote_comparison.md + the charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ARMS if a in {r["arm"] for r in runs}]
    rmse = _agg(runs, "test_rmse")
    mae = _agg(runs, "test_mae")
    r2 = _agg(runs, "test_r2")
    acc = _agg(runs, "test_accuracy")
    f1 = _agg(runs, "test_f1")
    tail_hi = _agg(runs, "rmse_hi")
    tail_lo = _agg(runs, "rmse_lo")
    tail_mid = _agg(runs, "rmse_mid")
    ks = _agg(runs, "ks_to_test")
    hi_mass = _agg(runs, "train_hi_decile_mass")
    lo_mass = _agg(runs, "train_lo_decile_mass")
    train_s = _agg(runs, "train_time_s")

    base = "none"
    lines = [
        "# SMOTE / SMOTE-R comparison",
        "",
        f"Generated {stamp} · dataset `{args.dataset}`"
        + (f" (city `{args.city}`)" if args.city else "")
        + f" · seq_len `{args.seq_len}` "
        f"· epochs `{args.epochs}` · task `{args.task}` · "
        f"seeds `{args.seeds}` · models `{args.models}`.",
        "",
        "Every arm shares the **same** windows, scalers, split, model family, "
        "epoch budget and seed set — only the resampling strategy changes. "
        "Metrics are means ± std over all (model × seed) runs.",
        "",
        "## Arms",
        "",
        "| arm | what it does |",
        "|-----|--------------|",
    ]
    for a in arms:
        lines.append(f"| `{a}` | {ARM_LABEL[a]} |")
    lines += [
        "",
        "## Headline results (all models × seeds pooled)",
        "",
        "| arm | test RMSE ↓ | test MAE ↓ | test R² ↑ | accuracy ↑ | F1 ↑ | "
        "tail RMSE (top 10 %) ↓ | train time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(
            f"| `{a}` | {_fmt(rmse.get(a))} | {_fmt(mae.get(a))} | "
            f"{_fmt(r2.get(a))} | {_fmt(acc.get(a))} | {_fmt(f1.get(a))} | "
            f"{_fmt(tail_hi.get(a))} | {_fmt(train_s.get(a), 1)} |")

    if base in rmse:
        lines += ["", "### Δ RMSE vs `none` (negative = better than no resampling)",
                  "", "| arm | Δ RMSE | Δ tail RMSE (top 10 %) |", "|---|---|---|"]
        for a in arms:
            if a == base or a not in rmse or base not in rmse:
                continue
            d_rmse = rmse[a]["mean"] - rmse[base]["mean"]
            if a in tail_hi and base in tail_hi:
                d_tail = tail_hi[a]["mean"] - tail_hi[base]["mean"]
                tail_cell = f"{d_tail:+.3f}"
            else:
                tail_cell = "—"
            lines.append(f"| `{a}` | {d_rmse:+.3f} | {tail_cell} |")

    lines += [
        "",
        "## Target-distribution fidelity",
        "",
        "How the **resampled training** target distribution compares with the "
        "untouched test distribution (same z-scored space):",
        "",
        "| arm | KS distance to test ↓ | mass above test P90 (target 0.10) | "
        "mass below test P10 (target 0.10) | train std / test std |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        if a not in ks:
            continue
        std_ratio = ""
        rows_a = [r for r in runs if r["arm"] == a]
        ratios = [r["distribution"]["train_std"] / max(r["distribution"]["test_std"], 1e-9)
                  for r in rows_a]
        if ratios:
            std_ratio = f"{np.mean(ratios):.3f}"
        lines.append(
            f"| `{a}` | {_fmt(ks.get(a))} | {_fmt(hi_mass.get(a))} | "
            f"{_fmt(lo_mass.get(a))} | {std_ratio} |")

    # --- per-model table ------------------------------------------------
    lines += ["", "## Per-model test RMSE", "",
              "| model | " + " | ".join(f"`{a}`" for a in arms) + " |",
              "|---|" + "---|" * len(arms)]
    for m in args.models:
        cells = []
        for a in arms:
            vals = [r["test_rmse"] for r in runs
                    if r["arm"] == a and r["model"] == m
                    and r.get("test_rmse") is not None]
            cells.append(f"{np.mean(vals):.3f}" if vals else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # --- charts ---------------------------------------------------------
    labels = [ARM_LABEL[a] for a in arms]
    colors = ["#9e9e9e", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    colors = colors[:len(arms)] + ["#999999"] * max(0, len(arms) - len(colors))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    means = [rmse.get(a, {}).get("mean", np.nan) for a in arms]
    stds = [rmse.get(a, {}).get("std", 0.0) for a in arms]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel("test RMSE (AQI units)")
    ax.set_title("Resampling strategy vs test RMSE (mean ± std over models × seeds)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = CHART_DIR / f"smote_rmse_{tag}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    means = [tail_hi.get(a, {}).get("mean", np.nan) for a in arms]
    stds = [tail_hi.get(a, {}).get("std", 0.0) for a in arms]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel("test RMSE on the top-decile AQI days")
    ax.set_title("Tail accuracy (rare/extreme test targets)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = CHART_DIR / f"smote_tail_rmse_{tag}.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    means = [ks.get(a, {}).get("mean", np.nan) for a in arms]
    stds = [ks.get(a, {}).get("std", 0.0) for a in arms]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel("KS distance (resampled train targets vs test targets)")
    ax.set_title("Target-distribution fidelity after resampling")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p3 = CHART_DIR / f"smote_ks_{tag}.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    lines += [
        "",
        "## Charts",
        "",
        f"- `benchmarking/charts/{p1.name}` — test RMSE per arm",
        f"- `benchmarking/charts/{p2.name}` — RMSE on the rare (top-decile) test days",
        f"- `benchmarking/charts/{p3.name}` — KS distance to the test target distribution",
        "",
        "## Reproduce",
        "",
        "```bash",
        f".venv/bin/python benchmarking/benchmark_smote.py --dataset {args.dataset} "
        f"--seq-len {args.seq_len} --epochs {args.epochs} "
        f"--seeds {' '.join(str(s) for s in args.seeds)} "
        f"--models {' '.join(args.models)}",
        "```",
        "",
        f"Raw per-run records: `benchmarking/results/smote/{tag}__*.json` · "
        f"summary: `benchmarking/smote_summary_{stamp}.json` / `.csv`.",
    ]
    (BENCH_DIR / f"smote_comparison_{tag}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
