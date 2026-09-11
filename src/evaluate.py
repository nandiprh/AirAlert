"""Evaluation script for trained hybrid air quality models.

Loads a checkpoint saved by ``train.py``, rebuilds the same model
architecture + data pipeline, runs on the test split and prints
RMSE / MAE / R2 (regression) plus the confusion matrix (classification).
Matplotlib plots (predictions vs actual, confusion matrix) are saved to
``models/reports/``.

Examples:
    python src/evaluate.py --checkpoint models/checkpoints/lstm_cnn_uci_both_s24.pt
    python src/evaluate.py --model lstm_cnn --dataset uci --task both --seq_len 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow direct execution:  python src/evaluate.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.models import MODEL_REGISTRY, SequenceDataset, build_model
from src.models.base import inverse_normalize_target
from src.train import REPORT_DIR, preprocess_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint(model, dataset, task, seq_len):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "checkpoints",
        f"{model}_{dataset}_{task}_s{seq_len}.pt",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train.py first.")
    return path


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Recreate the model from saved metadata + config."""
    meta = ckpt["metadata"]
    cfg = ckpt["config"]
    model = build_model(
        meta["model_name"],
        input_size=meta["input_size"],
        seq_len=meta["seq_len"],
        output_size=1,
        num_classes=meta["num_classes"] if ckpt.get("use_cls") else None,
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        seed=int(cfg.get("seed", 42)),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(DEVICE)


def evaluate(checkpoint_path: str, city: str | None = None, save_plots: bool = True,
             prefix: str | None = None):
    """Load checkpoint, run test evaluation, print + plot results."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    task = ckpt.get("task", meta.get("task", "both"))
    use_cls = bool(ckpt.get("use_cls", task in ("both", "classification")))
    use_reg = bool(ckpt.get("use_reg", task in ("both", "regression")))

    print(f"[evaluate] model={meta['model_name']} dataset={meta['dataset']} "
          f"task={task} seq_len={meta['seq_len']}")

    data = preprocess_dataset(
        meta["dataset"],
        seq_len=meta["seq_len"],
        task=task,
        num_classes=meta["num_classes"],
        city=city,
    )
    model = build_model_from_checkpoint(ckpt)

    ds = SequenceDataset(
        data["X_test"].astype(np.float32),
        y_reg=data["y_reg_test"] if use_reg else None,
        y_cls=data["y_cls_test"] if use_cls else None,
    )
    loader = DataLoader(ds, batch_size=int(ckpt["config"].get("batch_size", 64)),
                        shuffle=False)

    model.eval()
    preds_r, trues_r, preds_c, trues_c = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if use_reg and use_cls:
                x, y_r, y_c = batch
            elif use_reg:
                x, y_r = batch
                y_c = None
            else:
                x, y_c = batch
                y_r = None
            x = x.to(DEVICE)
            reg_out, cls_out = model(x)
            if use_reg and reg_out is not None:
                preds_r.append(reg_out.squeeze(-1).cpu().numpy())
                trues_r.append(y_r.numpy())
            if use_cls and cls_out is not None:
                preds_c.append(cls_out.argmax(dim=1).cpu().numpy())
                trues_c.append(y_c.numpy())

    os.makedirs(REPORT_DIR, exist_ok=True)
    results = {"metadata": meta, "task": task}
    base = prefix or os.path.splitext(os.path.basename(checkpoint_path))[0]

    # ---- Regression metrics + plot ----
    if use_reg:
        sc = data["reg_scaler"]
        yt = inverse_normalize_target(np.concatenate(trues_r), sc)
        yp = inverse_normalize_target(np.concatenate(preds_r), sc)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        results.update(rmse=rmse, mae=mae, r2=r2)
        print("\nRegression (test set)")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R2   = {r2:.4f}")
        if save_plots:
            _plot_predictions(yt, yp, base)

    # ---- Classification metrics + confusion matrix plot ----
    if use_cls:
        yt_c = np.concatenate(trues_c)
        yp_c = np.concatenate(preds_c)
        from sklearn.metrics import accuracy_score, f1_score
        acc = float(accuracy_score(yt_c, yp_c))
        f1 = float(f1_score(yt_c, yp_c, average="weighted", zero_division=0))
        cm = confusion_matrix(yt_c, yp_c).tolist()
        results.update(accuracy=acc, f1=f1, confusion_matrix=cm)
        print("\nClassification (test set)")
        print(f"  Accuracy = {acc:.4f}")
        print(f"  F1       = {f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print(confusion_matrix(yt_c, yp_c))
        if save_plots:
            _plot_confusion(yt_c, yp_c, base)

    report_out = os.path.join(REPORT_DIR, f"{base}_evaluation.json")
    with open(report_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[save] evaluation report -> {report_out}")
    return results


def _plot_predictions(y_true, y_pred, base):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = len(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Time series comparison.
    idx = np.linspace(0, n - 1, min(n, 500)).astype(int)
    axes[0].plot(idx, y_true[idx], "o-", ms=3, lw=1, label="actual")
    axes[0].plot(idx, y_pred[idx], "s-", ms=3, lw=1, label="predicted")
    axes[0].set_xlabel("test sample"); axes[0].set_ylabel(target_name(base))
    axes[0].set_title("Actual vs predicted (test set)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Scatter.
    axes[1].scatter(y_true, y_pred, s=8, alpha=0.5)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_xlabel("actual"); axes[1].set_ylabel("predicted")
    axes[1].set_title("Prediction scatter")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"{base}_predictions.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] actual vs predicted -> {path}")


def _plot_confusion(y_true, y_pred, base):
    labels = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title("AQI bucket confusion matrix (test set)")
    ax.set_xlabel("Predicted bucket"); ax.set_ylabel("True bucket")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"{base}_confusion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] confusion matrix -> {path}")


def target_name(base: str) -> str:
    return "AQI" if "india" in base else "CO (mg/m3)"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained hybrid AQI model")
    p.add_argument("--checkpoint", default=None,
                   help="path to .pt checkpoint (defaults to auto-located name)")
    p.add_argument("--model", choices=list(MODEL_REGISTRY), default="lstm_cnn")
    p.add_argument("--dataset", choices=["uci", "india"], default="uci")
    p.add_argument("--task", choices=["both", "regression", "classification"],
                   default="both")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--city", default=None,
                   help="india city used during training (default: most frequent)")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = find_checkpoint(args.model, args.dataset, args.task,
                                          args.seq_len)
    elif not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    prefix = os.path.splitext(os.path.basename(args.checkpoint))[0]
    evaluate(args.checkpoint, city=args.city,
             save_plots=not args.no_plots, prefix=prefix)


if __name__ == "__main__":
    main()