"""Evaluate & compare the 5 hybrid AQI models based on the parameters used.

Reads every per-run result JSON under ``benchmarking/results/`` and produces:

- ``benchmarking/comparison.md``          — human-readable comparison write-up
- ``benchmarking/ranking_<ts>.csv``       — every run, sorted by test RMSE
- ``benchmarking/charts/accuracy_<ts>.png``  — RMSE vs parameter-count scatter
- ``benchmarking/charts/rmse_bars_<ts>.png` — per-model RMSE grouped by grid point
- ``benchmarking/comparison_<ts>.json``   — machine-readable summary

Analysis dimensions ("based on the parameters used"):
  1. per-model, per-grid-point accuracy (RMSE/MAE/R2, acc/F1) and efficiency
     (params, train time, inference latency);
  2. parameter sensitivity for each algorithm: low->high ``hidden_size`` and
     low->high ``seq_len`` deltas;
  3. cross-model ranking at the best grid point per dataset;
  4. a composite balance score = 0.5*norm(RMSE) + 0.3*norm(params)
     + 0.2*norm(train_time) so a winner can be picked fairly.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).resolve().parent
RESULT_DIR = BENCH_DIR / "results"
CHARTS_DIR = BENCH_DIR / "charts"

GRID_ORDER = ["s7_h32", "s7_h128", "s24_h32", "s24_h128"]
MODEL_ORDER = ["lstm_cnn", "cnn_bilstm", "lstm_attention",
               "transformer_lstm", "convlstm_attention"]
METRICS = ["test_rmse", "test_mae", "test_r2", "test_accuracy", "test_f1"]
EFF = ["params", "train_time_s", "inference_ms_per_batch", "checkpoint_mb"]


def load_results():
    runs = []
    for path in glob.glob(str(RESULT_DIR / "*" / "*.json")):
        with open(path) as f:
            run = json.load(f)
        if "test_rmse" not in run:
            continue  # failed / incomplete
        runs.append(run)
    return runs


def gridkey(run):
    return run["cfgid"]


def norm_min_max(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def balance_score(run, scales):
    """Lower is better. Combines accuracy (RMSE) + cost (params, train time)."""
    r = (run["test_rmse"] - scales["rmse_lo"]) / \
        max(scales["rmse_hi"] - scales["rmse_lo"], 1e-9)
    p = (run["params"] - scales["p_lo"]) / \
        max(scales["p_hi"] - scales["p_lo"], 1e-9)
    t = (run["train_time_s"] - scales["t_lo"]) / \
        max(scales["t_hi"] - scales["t_lo"], 1e-9)
    return 0.5 * r + 0.3 * p + 0.2 * t


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main(specific_ts: str | None = None):
    CHARTS_DIR.mkdir(exist_ok=True)
    runs = load_results()
    if not runs:
        print("No results found under benchmarking/results/*/*.json. "
              "Run benchmarking/run_sweep.py first.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets = sorted({r["dataset"] for r in runs})
    out = []

    def head(t):
        out.append(f"\n## {t}\n")

    out.append("# Hybrid Model Benchmark — accuracy vs parameters\n")
    head("What was measured")
    out.append(
        "Every model trains on the identical 80:20 temporal split (SMOTE on "
        "train only, fixed seed 42) of the same windows/scalers, varying only "
        "the two architectural hyper-parameters under test: the input look-back "
        "`seq_len` (7 vs 24 hours) and the hidden capacity `hidden_size` "
        "(32 vs 128 units). Epochs, LR, dropout, layers, scheduler, early "
        "stopping and auto overfit/underfit correction are identical across "
        "runs. Reported: regression RMSE/MAE/R2, bucket accuracy/F1, parameter "
        "count, CPU training wall-clock, inference ms/batch.")

    all_runs = runs  # for charts across datasets/models/gridpoints

    for ds in datasets:
        arr = np.array([r["params"] for r in all_runs])
        scales = {
            "rmse_lo": float(min(r["test_rmse"] for r in all_runs)),
            "rmse_hi": float(max(r["test_rmse"] for r in all_runs)),
            "p_lo": float(arr.min()), "p_hi": float(arr.max()),
            "t_lo": float(min(r["train_time_s"] for r in all_runs)),
            "t_hi": float(max(r["train_time_s"] for r in all_runs)),
        }

        head(f"Dataset: {ds}")
        out.append("### 1. Per-model, per-grid-point results\n")
        for model in MODEL_ORDER:
            mruns = [r for r in runs if r["dataset"] == ds and r["model"] == model]
            mruns.sort(key=lambda r: gridkey(r))
            rows = []
            for r in mruns:
                rows.append([
                    r["cfgid"],
                    f'{r["test_rmse"]:.3f}', f'{r["test_mae"]:.3f}',
                    f'{r["test_r2"]:.3f}', f'{r["test_accuracy"]:.3f}',
                    f'{r["test_f1"]:.3f}',
                    f'{r["params"]:,}', f'{r["train_time_s"]:.0f}s',
                    f'{r["inference_ms_per_batch"]:.2f}ms',
                    r["detection"],
                    f'{r["epochs"]}',
                ])
            if rows:
                out.append(f"**{model}**\n")
                out.append(md_table(
                    ["gridpt", "RMSE", "MAE", "R2", "acc", "F1",
                     "params", "train", "infer", "detect", "epochs"],
                    rows) + "\n")

        head("2. Parameter sensitivity (delta from low -> high)")
        for model in MODEL_ORDER:
            mruns = {r["cfgid"]: r for r in runs
                     if r["dataset"] == ds and r["model"] == model}
            cells = ["**%s**" % model]
            # hidden effect at seq_len 7: h32 -> h128
            if "s7_h32" in mruns and "s7_h128" in mruns:
                cells.append("hidden(7): %+.3f RMSE" % (
                    mruns["s7_h128"]["test_rmse"] - mruns["s7_h32"]["test_rmse"]))
            else:
                cells.append("hidden(7): --")
            if "s24_h32" in mruns and "s24_h128" in mruns:
                cells.append("hidden(24): %+.3f RMSE" % (
                    mruns["s24_h128"]["test_rmse"] - mruns["s24_h32"]["test_rmse"]))
            else:
                cells.append("hidden(24): --")
            # seq effect at h32 and h128
            if "s7_h32" in mruns and "s24_h32" in mruns:
                cells.append("seq(h32): %+.3f RMSE" % (
                    mruns["s24_h32"]["test_rmse"] - mruns["s7_h32"]["test_rmse"]))
            else:
                cells.append("seq(h32): --")
            if "s7_h128" in mruns and "s24_h128" in mruns:
                cells.append("seq(h128): %+.3f RMSE" % (
                    mruns["s24_h128"]["test_rmse"] - mruns["s7_h128"]["test_rmse"]))
            else:
                cells.append("seq(h128): --")
            out.append("| " + " | ".join(cells) + " |")

        head("3. Cross-model ranking (best-accuracy grid point each)")
        if ds in datasets:
            best = {}
            for model in MODEL_ORDER:
                mruns = [r for r in runs if r["dataset"] == ds
                         and r["model"] == model]
                if not mruns:
                    continue
                b = min(mruns, key=lambda r: r["test_rmse"])
                b["_score"] = balance_score(b, scales)
                best[model] = b
            rows = []
            for model in sorted(best, key=lambda m: best[m]["test_rmse"]):
                b = best[model]
                rows.append([
                    model, b["cfgid"], f'{b["test_rmse"]:.3f}',
                    f'{b["test_r2"]:.3f}', f'{b["test_accuracy"]:.3f}',
                    f'{b["params"]:,}', f'{b["train_time_s"]:.0f}s',
                    f'{b["inference_ms_per_batch"]:.2f}ms',
                    f'{b["_score"]:.3f}',
                ])
            out.append(md_table(
                ["model", "gridpt", "RMSE", "R2", "acc", "params",
                 "train", "infer", "balance"],
                rows) + "\n")
            out.append("\n*Sensitivity sign: a negative delta means the larger "
                       "`hidden_size` / `seq_len` setting **improved** RMSE "
                       "(negative is good); a positive delta means it hurt "
                       "accuracy.*\n")

        head("4. Overall verdict (per dataset)")
        best = {}
        for model in MODEL_ORDER:
            mruns = [r for r in runs if r["dataset"] == ds and r["model"] == model]
            if mruns:
                best[model] = min(mruns, key=lambda r: r["test_rmse"])
        if best:
            acc_win = min(best, key=lambda m: best[m]["test_rmse"])
            allr = [r for r in runs if r["dataset"] == ds]
            cheap = min(allr, key=lambda r: r["params"])
            bal_win = min(best, key=lambda m: balance_score(best[m], scales))
            e = best[acc_win]
            out.append(f"- **Most accurate**: `{acc_win}` (grid `{e['cfgid']}`, "
                       f"RMSE {e['test_rmse']:.3f}, R2 {e['test_r2']:.3f}, "
                       f"acc {e['test_accuracy']:.3f}).")
            e = cheap
            out.append(f"- **Fewest parameters overall**: `{cheap['model']}` "
                       f"(grid `{e['cfgid']}`, {e['params']:,} params).")
            e = best[bal_win]
            out.append(f"- **Best balance (accuracy vs cost)**: `{bal_win}` "
                       f"(grid `{e['cfgid']}`, RMSE {e['test_rmse']:.3f}).")

    # ---- charts ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"lstm_cnn": "#1f77b4", "cnn_bilstm": "#ff7f0e",
              "lstm_attention": "#2ca02c", "transformer_lstm": "#d62728",
              "convlstm_attention": "#9467bd"}
    for model in MODEL_ORDER:
        mr = [r for r in all_runs if r["model"] == model]
        if not mr:
            continue
        ax.scatter([r["params"] for r in mr], [r["test_rmse"] for r in mr],
                   s=48, alpha=0.75, label=model, color=colors[model],
                   edgecolors="k", linewidths=0.4)
    for r in all_runs:
        ax.annotate(r["cfgid"], (r["params"], r["test_rmse"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.5,
                    color="#444")
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log)")
    ax.set_ylabel("test RMSE (lower better)")
    ax.set_title("Accuracy vs model size — all datasets / grid points")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p1 = CHARTS_DIR / f"accuracy_{stamp}.png"
    fig.tight_layout(); fig.savefig(p1, dpi=130); plt.close(fig)

    # grouped RMSE bars per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(10, 4.5),
                             squeeze=False)
    x = np.arange(len(GRID_ORDER))
    width = 0.15
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        for k, model in enumerate(MODEL_ORDER):
            vals = []
            for g in GRID_ORDER:
                hits = [r for r in runs
                        if r["model"] == model and r["dataset"] == ds
                        and r["cfgid"] == g]
                vals.append(hits[0]["test_rmse"] if hits else None)
            ax.bar(x + (k - 2) * width, [v if v is not None else 0 for v in vals],
                   width, label=model)
        ax.set_xticks(x); ax.set_xticklabels(GRID_ORDER, rotation=45, fontsize=7)
        ax.set_ylabel("test RMSE")
        ax.set_title(ds)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=6.5, ncol=2)
    p2 = CHARTS_DIR / f"rmse_bars_{stamp}.png"
    fig.tight_layout(); fig.savefig(p2, dpi=130); plt.close(fig)

    out.append("\n## Charts\n")
    out.append(f"![accuracy vs size](charts/accuracy_{stamp}.png)")
    out.append(f"![grouped RMSE](charts/rmse_bars_{stamp}.png)")

    # ---- csv ranking ----
    import csv
    import io
    from collections import OrderedDict

    rows = []
    for r in all_runs:
        rows.append({k: r.get(k, "") for k in
                     ["dataset", "model", "cfgid"] + METRICS + EFF +
                     ["detection", "epochs"]})
    rows.sort(key=lambda r: r["dataset"] + f'{r["test_rmse"]}')  # noqa
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    csv_path = BENCH_DIR / f"ranking_{stamp}.csv"
    csv_path.write_text(buf.getvalue())

    md_path = BENCH_DIR / "comparison.md"
    md_path.write_text("\n".join(out) + "\n")

    # machine-readable verdicts per dataset
    verdicts = {}
    for ds in datasets:
        arr = np.array([r["params"] for r in all_runs])
        scales = {
            "rmse_lo": float(min(r["test_rmse"] for r in all_runs)),
            "rmse_hi": float(max(r["test_rmse"] for r in all_runs)),
            "p_lo": float(arr.min()), "p_hi": float(arr.max()),
            "t_lo": float(min(r["train_time_s"] for r in all_runs)),
            "t_hi": float(max(r["train_time_s"] for r in all_runs)),
        }
        best = {}
        for model in MODEL_ORDER:
            mruns = [r for r in runs if r["dataset"] == ds and r["model"] == model]
            if mruns:
                b = min(mruns, key=lambda r: r["test_rmse"])
                best[model] = {"gridpt": b["cfgid"], "rmse": b["test_rmse"],
                               "r2": b["test_r2"], "acc": b["test_accuracy"],
                               "params": b["params"],
                               "train_time_s": b["train_time_s"],
                               "balance": balance_score(b, scales)}
        best_sorted = sorted(best.items(), key=lambda kv: kv[1]["rmse"])
        allr = [r for r in runs if r["dataset"] == ds]
        cheap = min(allr, key=lambda r: r["params"])
        verdicts[ds] = {
            "ranking": [{"model": m, **d} for m, d in best_sorted],
            "most_accurate": best_sorted[0][0],
            "fewest_params": {"model": cheap["model"], "cfgid": cheap["cfgid"],
                              "params": cheap["params"]},
            "best_balance": min(best.items(),
                                key=lambda kv: kv[1]["balance"])[0],
        }

    summary = {"stamp": stamp, "datasets": datasets, "epochs": 5,
               "verdicts": verdicts}
    (BENCH_DIR / f"comparison_{stamp}.json").write_text(
        json.dumps(OrderedDict(sorted(summary.items())), indent=2))

    print(f"[save] commentary  -> {md_path}")
    print(f"[save] rankings    -> {csv_path}")
    print(f"[save] charts      -> {p1}\n                      {p2}")
    print(f"[save] data        -> benchmarking/comparison_{stamp}.json")


if __name__ == "__main__":
    main()