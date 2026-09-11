"""End-to-end training pipeline for the 5 hybrid air quality models.

Pipeline (mirrors README methodology):
  1. Load UCI or India city AQI data (processed CSV when available).
  2. Preprocess + data-leakage checks (sequential split before windowing).
  3. Train/Validation/Test split (70/15/15) OR 10-fold time-series CV.
  4. SMOTE oversampling on TRAINING data only (classification imbalance).
  5. Train each requested model.
  6. Overfit/underfit detection and automatic correction (dropout bump,
     early stopping, LR scheduling, capacity bump) + retrain.
  7. Save checkpoints -> models/checkpoints/, metrics -> models/reports/.

Example:
    python src/train.py --dataset uci --model lstm_cnn --epochs 50 --seq_len 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Allow direct execution:  python src/train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models import (
    MODEL_REGISTRY,
    SequenceDataset,
    build_model,
    compute_metrics,
    deterministic_seed,
)
from src.models.base import (
    apply_normalizer,
    fit_normalizer,
    inverse_normalize_target,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RAW_DIR = os.path.join(ROOT, "data", "raw")
CKPT_DIR = os.path.join(ROOT, "models", "checkpoints")
REPORT_DIR = os.path.join(ROOT, "models", "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UCI_FEATURES = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
UCI_TARGET = "CO(GT)"

INDIA_FEATURES = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
]
INDIA_TARGET = "AQI"
INDIA_CLASS = "AQI_Bucket"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_european_decimal(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert comma-as-decimal values (UCI) to floats; -200 -> NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("-200", "NaN", regex=False)
                .apply(pd.to_numeric, errors="coerce")
            )
    return df


def load_uci_data() -> pd.DataFrame:
    """Load UCI air quality hourly data (processed CSV preferred)."""
    path = os.path.join(PROCESSED_DIR, "uci_clean.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = _parse_european_decimal(
            df, [c for c in UCI_FEATURES + [UCI_TARGET] if c in df.columns]
        )
        if "datetime" not in df.columns:
            if "Date" in df.columns and "Time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str),
                    dayfirst=True, errors="coerce",
                )
            else:
                df["datetime"] = pd.date_range(start="2004-03-10",
                                               periods=len(df), freq="h")
    else:
        raw_path = os.path.join(RAW_DIR, "airquality_uci.csv")
        df = pd.read_csv(raw_path, sep=";", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
        df = _parse_european_decimal(df, UCI_FEATURES + [UCI_TARGET])
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            dayfirst=True, format="%d/%m/%Y %H.%M.%S", errors="coerce",
        )

    missing = [c for c in UCI_FEATURES + [UCI_TARGET, "datetime"] if c not in df.columns]
    if missing:
        raise ValueError(f"UCI data missing columns: {missing}")
    return df.sort_values("datetime").reset_index(drop=True)


def load_india_data(city: Optional[str] = None) -> pd.DataFrame:
    """Load Indian city daily AQI data (processed CSV preferred)."""
    path = os.path.join(PROCESSED_DIR, "india_city_day_clean.csv")
    raw_path = os.path.join(RAW_DIR, "india_city_day.csv")
    file = path if os.path.exists(path) else raw_path
    df = pd.read_csv(file, low_memory=False)

    if "datetime" not in df.columns:
        date_col = "Date" if "Date" in df.columns else "Timestamp"
        df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")

    if city is None and "City" in df.columns:
        needed_cols = INDIA_FEATURES + [INDIA_TARGET]
        completeness = (
            df.groupby("City")[needed_cols]
            .apply(lambda g: int(g.notna().all(axis=1).sum()), include_groups=False)
            .sort_values(ascending=False)
        )
        city = completeness.index[0]
        print(f"[data] Using most complete city: {city} "
              f"({int(completeness.iloc[0])} complete rows)")
    if city is not None and "City" in df.columns:
        df = df[df["City"] == city]

    for c in INDIA_FEATURES + [INDIA_TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    missing = [c for c in INDIA_FEATURES + [INDIA_TARGET, "datetime"] if c not in df.columns]
    if missing:
        raise ValueError(f"India data missing columns: {missing}")
    return df


def _encode_labels(series: pd.Series):
    """Encode categorical AQI buckets to ints -> (codes, mapping)."""
    uniq = sorted(series.dropna().unique().tolist())
    mapping = {str(v): i for i, v in enumerate(uniq)}
    codes = series.map(lambda v: mapping.get(str(v), -1)).to_numpy()
    return codes, mapping


def preprocess_dataset(
    dataset: str,
    seq_len: int = 24,
    task: str = "both",
    num_classes: int = 4,
    city: Optional[str] = None,
    scale_target: bool = True,
) -> Dict:
    """Load + preprocess data into leakage-free train/val/test windows.

    The raw series is split BEFORE windowing so no window crosses
    partition boundaries (temporal leakage check). Returns arrays,
    fitted scalers, class mapping and metadata.
    """
    use_reg = task_enabled(task, "regression")
    use_cls = task_enabled(task, "classification")
    cls_map = None

    if dataset == "uci":
        df = load_uci_data()
        features, target = UCI_FEATURES, UCI_TARGET
        if use_cls:
            values = df[target].to_numpy()
            mask = ~np.isnan(values)
            edges = pd.qcut(pd.Series(values[mask]), q=num_classes, retbins=True,
                            duplicates="drop")[1]
            buckets = pd.Series(np.nan, index=df.index, dtype=float)
            buckets[mask] = pd.cut(values[mask], bins=edges, labels=False).astype(float)
            y_cls_raw = buckets.to_numpy()
        else:
            y_cls_raw = None
    elif dataset == "india":
        df = load_india_data(city)
        features, target = INDIA_FEATURES, INDIA_TARGET
        if use_cls:
            if INDIA_CLASS not in df.columns:
                raise ValueError("AQI_Bucket column missing in India data")
            y_cls_raw, cls_map = _encode_labels(df[INDIA_CLASS])
        else:
            y_cls_raw, cls_map = None, None
    else:
        raise ValueError(f"--dataset must be 'uci' or 'india', got {dataset}")

    X_raw = df[features].to_numpy(dtype=np.float64)
    y_reg_raw = df[target].to_numpy(dtype=np.float64)

    bad = np.isnan(X_raw).any(axis=1) | np.isnan(y_reg_raw)
    if y_cls_raw is not None:
        bad |= y_cls_raw < 0
    if int(bad.sum()) > 0:
        print(f"[preprocess] Dropping {int(bad.sum())} rows with NaN/invalid values")
        X_raw, y_reg_raw = X_raw[~bad], y_reg_raw[~bad]
        if y_cls_raw is not None:
            y_cls_raw = y_cls_raw[~bad]

    n = len(X_raw)
    if n <= seq_len + 3:
        raise ValueError("Not enough samples after cleaning for the requested seq_len")
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    print(f"[preprocess] temporal split  train={n_train}  val={n_val}  test={n_test}")

    def build_windows(lo: int, hi: int):
        rows = np.arange(lo + seq_len, hi)
        Xw = np.stack([X_raw[i - seq_len:i] for i in rows]).astype(np.float64)
        y_r = y_reg_raw[rows]
        y_c = y_cls_raw[rows] if y_cls_raw is not None else None
        return Xw, y_r, y_c

    Xs, ys_reg, ys_cls = {}, {}, {}
    for fname, (lo, hi) in {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }.items():
        Xs[fname], ys_reg[fname], ys_cls[fname] = build_windows(lo, hi)

    # --- scaling: fit on train only ---
    f_scaler = fit_normalizer(Xs["train"], method="standard")
    Xs = {k: apply_normalizer(v, f_scaler) for k, v in Xs.items()}

    r_scaler = None
    if use_reg and scale_target:
        r_scaler = StandardScaler()
        y_tr = ys_reg["train"].reshape(-1, 1)
        r_scaler.fit(y_tr.astype(np.float64))
        ys_reg = {
            k: r_scaler.transform(ys_reg[k].reshape(-1, 1)).reshape(-1)
            if len(ys_reg[k]) else ys_reg[k]
            for k in ys_reg
        }

    if use_cls and dataset == "uci":
        _, cls_map = _encode_labels(pd.Series(np.concatenate(list(ys_cls.values()))))

    num_classes_eff = len(cls_map) if use_cls and cls_map is not None else num_classes
    metadata = dict(
        dataset=dataset,
        target=target,
        features=list(features),
        seq_len=seq_len,
        task=task,
        num_classes=num_classes_eff,
        input_size=len(features),
        n_train=len(Xs["train"]),
        n_val=len(Xs["val"]),
        n_test=len(Xs["test"]),
    )
    return dict(
        X_train=Xs["train"], X_val=Xs["val"], X_test=Xs["test"],
        y_reg_train=ys_reg["train"], y_reg_val=ys_reg["val"], y_reg_test=ys_reg["test"],
        y_cls_train=ys_cls["train"], y_cls_val=ys_cls["val"], y_cls_test=ys_cls["test"],
        feature_scaler=f_scaler, reg_scaler=r_scaler, cls_map=cls_map,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SMOTE (train split only)
# ---------------------------------------------------------------------------

def apply_smote(X_train: np.ndarray, y_cls_train: np.ndarray, y_reg_train: np.ndarray | None,
                seq_len: int, n_features: int, random_state: int = 42):
    """Oversample minority classes using SMOTE. Windows are flattened to
    2D for SMOTE (synthetic neighbour search) then reshaped back.

    When regression targets are available they are appended as an extra
    column so SMOTE's convex interpolation also produces regression
    targets consistent with the synthetic windows.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_cls_train, y_reg_train

    counts = pd.Series(y_cls_train).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        print("[smote] too few samples, skipping")
        return X_train, y_cls_train, y_reg_train

    X_flat = X_train.reshape(X_train.shape[0], -1)
    if y_reg_train is not None:
        Z = np.hstack([X_flat, y_reg_train.reshape(-1, 1)])
    else:
        Z = X_flat
    sm = SMOTE(random_state=random_state)
    try:
        Z_r, y_r = sm.fit_resample(Z, y_cls_train)
    except Exception as exc:  # pragma: no cover
        print(f"[smote] failed ({exc}); skipping")
        return X_train, y_cls_train, y_reg_train

    X_r = Z_r[:, :X_flat.shape[1]]
    y_reg_r = Z_r[:, X_flat.shape[1]:].ravel() if y_reg_train is not None else y_reg_train
    print(f"[smote] training windows {len(X_flat)} -> {len(X_r)}")
    return (
        X_r.reshape(-1, seq_len, n_features).astype(np.float32),
        y_r,
        y_reg_r.astype(np.float32) if y_reg_r is not None else None,
    )


# ---------------------------------------------------------------------------
# Loss / training helpers
# ---------------------------------------------------------------------------

def task_enabled(task: str, name: str) -> bool:
    return task == "both" or task == name


def build_loss(model_out, y_reg, y_cls, task, crit_reg, crit_cls):
    """Combined regression (MSE) + classification (CE) loss."""
    reg_out, cls_out = model_out
    loss = torch.tensor(0.0, device=reg_out.device)
    terms = {}
    if task_enabled(task, "regression") and reg_out is not None:
        t = crit_reg(reg_out.squeeze(-1), y_reg)
        loss = loss + t
        terms["reg"] = t.item()
    if task_enabled(task, "classification") and cls_out is not None:
        t = crit_cls(cls_out, y_cls)
        loss = loss + t
        terms["cls"] = t.item()
    return loss, terms


def train_one_epoch(model, loader, task, crit_reg, crit_cls, optimizer):
    model.train()
    total, cnt = 0.0, 0
    for batch in loader:
        x, y_reg, y_cls = batch
        optimizer.zero_grad()
        loss, _ = build_loss(model(x.to(DEVICE)), y_reg.to(DEVICE), y_cls.to(DEVICE),
                             task, crit_reg, crit_cls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item()
        cnt += 1
    return total / max(1, cnt)


@torch.no_grad()
def evaluate_loader(model, loader, task, crit_reg, crit_cls):
    """Return loss + raw prediction arrays on a loader."""
    model.eval()
    total, cnt = 0.0, 0
    preds_r, trues_r, preds_c, trues_c = [], [], [], []
    for batch in loader:
        x, y_reg, y_cls = batch
        reg_out, cls_out = model(x.to(DEVICE))
        loss, _ = build_loss((reg_out, cls_out), y_reg.to(DEVICE), y_cls.to(DEVICE),
                             task, crit_reg, crit_cls)
        total += loss.item()
        cnt += 1
        if task_enabled(task, "regression") and reg_out is not None:
            preds_r.append(reg_out.squeeze(-1).cpu().numpy())
            trues_r.append(y_reg.numpy())
        if task_enabled(task, "classification") and cls_out is not None:
            preds_c.append(cls_out.argmax(dim=1).cpu().numpy())
            trues_c.append(y_cls.numpy())
    return dict(
        loss=total / max(1, cnt),
        y_pred_reg=np.concatenate(preds_r) if preds_r else None,
        y_true_reg=np.concatenate(trues_r) if trues_r else None,
        y_pred_cls=np.concatenate(preds_c) if preds_c else None,
        y_true_cls=np.concatenate(trues_c) if trues_c else None,
    )


class Config(dict):
    """dict with attribute access for training config."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def detect_issue(history, cfg):
    """Detect overfit/underfit from training history.

    Returns ``(reason, corrected_cfg)`` where reason in
    ``{'none', 'overfit', 'underfit'}``.
    """
    if len(history) < 4:
        return "none", cfg
    tr = np.array([h["train_loss"] for h in history])
    va = np.array([h["val_loss"] for h in history])
    best_idx = int(np.argmin(va))

    over_budget = int(cfg.get("overfit_budget", 8))
    under_budget = int(cfg.get("underfit_budget", 8))

    # Overfit: training loss small but val loss climbing well above train.
    gap = float(va[-1] - tr[-1])
    if (
        gap > max(0.02, 0.2 * tr[-1])
        and va[-1] >= float(np.min(va)) * 1.02
        and len(history) - 1 - best_idx >= over_budget
    ):
        c = Config(cfg)
        c["dropout"] = round(min(0.6, float(cfg.get("dropout", 0.2)) + 0.15), 3)
        c["early_stopping"] = True
        c["lr_schedule"] = "plateau"
        print("  [correction] OVERFITTING -> dropout up, early stopping, LR plateau")
        return "overfit", c

    # Underfit: val never improves meaningfully (flat / above train).
    if float(np.min(va)) >= float(tr[0]) * 0.93 and float(va[-1]) > float(tr[-1]) * 1.0:
        if len(history) >= under_budget:
            c = Config(cfg)
            c["hidden_size"] = int(float(cfg.get("hidden_size", 64)) * 1.5)
            c["epochs"] = int(cfg.get("epochs", 50)) + 30
            c["lr_schedule"] = "plateau"
            print("  [correction] UNDERFITTING -> capacity up, more epochs, LR plateau")
            return "underfit", c
    return "none", cfg


# ---------------------------------------------------------------------------
# Main training routine for a single model
# ---------------------------------------------------------------------------

def make_loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=shuffle,
                      num_workers=0, drop_last=False)


def run_training(model_name, data, cfg, checkpoint_path, seed=42,
                 max_correction_rounds=1):
    """Train one model with auto overfit/underfit correction + retrain."""
    meta = data["metadata"]
    task = meta["task"]
    use_reg = task_enabled(task, "regression")
    use_cls = task_enabled(task, "classification")

    X_train = data["X_train"].astype(np.float32)
    y_cls_train = data["y_cls_train"]
    y_reg_train = data["y_reg_train"]
    if use_cls and not cfg.get("no_smote"):
        X_train, y_cls_train, y_reg_train = apply_smote(
            X_train, np.asarray(y_cls_train, dtype=np.int64), y_reg_train,
            meta["seq_len"], meta["input_size"], seed,
        )

    train_ds = SequenceDataset(
        X_train,
        y_reg=y_reg_train if use_reg else np.zeros(len(X_train), dtype=np.float32),
        y_cls=np.asarray(y_cls_train, dtype=np.int64) if use_cls else np.zeros(len(X_train), dtype=np.int64),
    )
    val_ds = SequenceDataset(
        data["X_val"].astype(np.float32),
        y_reg=data["y_reg_val"] if use_reg else np.zeros(len(data["X_val"]), dtype=np.float32),
        y_cls=np.asarray(data["y_cls_val"], dtype=np.int64) if use_cls else np.zeros(len(data["X_val"]), dtype=np.int64),
    )
    test_ds = SequenceDataset(
        data["X_test"].astype(np.float32),
        y_reg=data["y_reg_test"] if use_reg else np.zeros(len(data["X_test"]), dtype=np.float32),
        y_cls=np.asarray(data["y_cls_test"], dtype=np.int64) if use_cls else np.zeros(len(data["X_test"]), dtype=np.int64),
    )

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    crit_reg = nn.MSELoss()
    crit_cls = nn.CrossEntropyLoss()

    correction_round = 0
    best_overall = None

    while True:
        deterministic_seed(seed + correction_round)
        model = build_model(
            model_name,
            input_size=meta["input_size"],
            seq_len=meta["seq_len"],
            output_size=1,
            num_classes=meta["num_classes"] if use_cls else None,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            seed=seed + correction_round,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                     weight_decay=float(cfg.get("weight_decay", 1e-5)))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8
        )

        patience = int(cfg.get("patience", 12))
        early_stop = bool(cfg.get("early_stopping", True))
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_val_res = None
        history = []

        for epoch in range(1, int(cfg["epochs"]) + 1):
            tr_loss = train_one_epoch(model, train_loader, task, crit_reg, crit_cls, optimizer)
            val_res = evaluate_loader(model, val_loader, task, crit_reg, crit_cls)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_res["loss"]})
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_res["loss"])

            if val_res["loss"] < best_val_loss:
                best_val_loss = val_res["loss"]
                epochs_no_improve = 0
                best_val_res = val_res
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epoch % 5 == 0 or epoch == 1 or epoch == int(cfg["epochs"]):
                if not cfg.get("quiet"):
                    print(f"    [{model_name}] epoch {epoch:3d}/{cfg['epochs']} "
                          f"train_loss={tr_loss:.4f} val_loss={val_res['loss']:.4f}")

            if early_stop and epochs_no_improve >= patience:
                print(f"    [{model_name}] early stopping @ epoch {epoch} "
                      f"(best_val {best_val_loss:.4f})")
                break

        # Restore best weights.
        model.load_state_dict(best_state)

        reason, corrected = detect_issue(history, cfg)
        if reason != "none" and correction_round < max_correction_rounds:
            correction_round += 1
            cfg = corrected
            continue  # retrain with corrected hyperparameters

        test_res = evaluate_loader(model, test_loader, task, crit_reg, crit_cls)
        val_metrics = to_split_metrics(best_val_res, data, use_reg, use_cls, "val")
        test_metrics = to_split_metrics(test_res, data, use_reg, use_cls, "test")

        result = dict(
            model=model, history=history, reason=reason,
            corrections=correction_round,
            val_loss=val_metrics.get("val_loss", float("inf")),
            val_metrics=val_metrics, test_metrics=test_metrics,
            test_res=test_res, val_res=best_val_res,
        )
        if best_overall is None or result["val_loss"] < best_overall["val_loss"]:
            best_overall = result
        break

    # --- checkpoint ---
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "state_dict": best_overall["model"].state_dict(),
            "config": dict(cfg),
            "metadata": {**meta, "model_name": model_name},
            "feature_scaler": data["feature_scaler"],
            "reg_scaler": data["reg_scaler"],
            "cls_map": data["cls_map"],
            "task": task,
            "use_reg": use_reg,
            "use_cls": use_cls,
        },
        checkpoint_path,
    )
    print(f"[save] checkpoint -> {checkpoint_path}")

    best_overall["model"] = best_overall["model"].cpu()
    _plot_history(best_overall["history"], model_name, checkpoint_path)
    return best_overall


def to_split_metrics(res, data, use_reg, use_cls, split):
    """Turn raw eval results into metrics (unscaled regression values)."""
    out = {f"{split}_loss": res["loss"]}
    if use_reg and res["y_true_reg"] is not None:
        sc = data["reg_scaler"]
        yt = inverse_normalize_target(res["y_true_reg"], sc)
        yp = inverse_normalize_target(res["y_pred_reg"], sc)
        out.update(compute_metrics(yt, yp, prefix=split, classification=False))
        out[f"{split}_y_true"] = yt
        out[f"{split}_y_pred"] = yp
    if use_cls and res["y_true_cls"] is not None:
        out.update(compute_metrics(res["y_true_cls"], res["y_pred_cls"],
                                   prefix=split, classification=True,
                                   labels=list(range(data["metadata"]["num_classes"]))))
    return out


def _plot_history(history, model_name, checkpoint_path):
    os.makedirs(REPORT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", ms=3, label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], "s-", ms=3, label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{model_name} training history (detect: {history[-1].get('note', '') or 'ok'})")
    ax.legend(); ax.grid(alpha=0.3)
    path = os.path.join(REPORT_DIR, f"{base}_history.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"[plot] training history -> {path}")


# ---------------------------------------------------------------------------
# 10-fold time-series cross-validation
# ---------------------------------------------------------------------------

def run_cv(dataset, model_name, cfg, seed=42):
    from sklearn.model_selection import TimeSeriesSplit

    if dataset == "uci":
        df = load_uci_data()
        features, target = UCI_FEATURES, UCI_TARGET
    else:
        df = load_india_data(cfg.get("city"))
        features, target = INDIA_FEATURES, INDIA_TARGET
    df = df.dropna(subset=features + [target])
    if "City" in df.columns:
        completeness = (
            df.groupby("City")[features + [target]]
            .apply(lambda g: int(g.notna().all(axis=1).sum()), include_groups=False)
            .sort_values(ascending=False)
        )
        df = df[df["City"] == completeness.index[0]]
    X = df[features].to_numpy(np.float64)
    y = df[target].to_numpy(np.float64)
    seq = int(cfg["seq_len"])

    fold_metrics = []
    tscv = TimeSeriesSplit(n_splits=10)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        print(f"[cv] fold {fold + 1}/10  train={len(tr_idx)} test={len(te_idx)}")
        if len(tr_idx) <= seq or len(te_idx) <= seq:
            print("[cv] fold too small, skipping"); continue
        f_scaler = fit_normalizer(X[tr_idx], "standard")
        Xtr = apply_normalizer(X[tr_idx], f_scaler)
        Xte = apply_normalizer(X[te_idx], f_scaler)
        r_scaler = StandardScaler().fit(y[tr_idx].reshape(-1, 1))

        rows_tr = np.arange(seq, len(Xtr)); rows_te = np.arange(seq, len(Xte))
        Xtr_w = np.stack([Xtr[i - seq:i] for i in rows_tr])
        ytr_w = r_scaler.transform(y[tr_idx][rows_tr].reshape(-1, 1)).ravel()
        Xte_w = np.stack([Xte[i - seq:i] for i in rows_te])
        yte_w = r_scaler.transform(y[te_idx][rows_te].reshape(-1, 1)).ravel()

        model = build_model(model_name, input_size=X.shape[1], seq_len=seq,
                            output_size=1, num_classes=None,
                            hidden_size=int(cfg["hidden_size"]),
                            num_layers=int(cfg["num_layers"]),
                            dropout=float(cfg["dropout"]), seed=seed + fold).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        crit = nn.MSELoss()
        ds = SequenceDataset(Xtr_w.astype(np.float32), y_reg=ytr_w.astype(np.float32))
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)
        model.train()
        for _ in range(int(cfg["epochs"])):
            for bx, by in loader:
                opt.zero_grad()
                reg, _ = model(bx.to(DEVICE))
                loss = crit(reg.squeeze(-1), by.to(DEVICE))
                loss.backward(); opt.step()

        model.eval()
        ds_te = SequenceDataset(Xte_w.astype(np.float32), y_reg=yte_w.astype(np.float32))
        lte = DataLoader(ds_te, batch_size=int(cfg["batch_size"]))
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in lte:
                preds.append(model(bx.to(DEVICE))[0].squeeze(-1).cpu().numpy())
                trues.append(by.numpy())
        preds = r_scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).ravel()
        trues = r_scaler.inverse_transform(np.concatenate(trues).reshape(-1, 1)).ravel()
        m = compute_metrics(trues, preds, prefix=f"fold{fold + 1}")
        fold_metrics.append(m)
        print(f"[cv] fold {fold + 1} " + "  ".join(
            f"{k}={m[f'fold{fold + 1}_{k}']:.4f}" for k in ("rmse", "mae", "r2")))

    agg = {"cv_folds": len(fold_metrics)}
    for key in ("rmse", "mae", "r2"):
        vals = [m[f"fold{i + 1}_{key}"] for i, m in enumerate(fold_metrics)]
        agg[f"cv_{key}_mean"] = float(np.nanmean(vals))
        agg[f"cv_{key}_std"] = float(np.nanstd(vals))
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train hybrid AQI models")
    p.add_argument("--dataset", choices=["uci", "india"], default="uci")
    p.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"], default="all")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--task", choices=["both", "regression", "classification"],
                   default="both")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--split_mode", choices=["holdout", "cv10"], default="holdout")
    p.add_argument("--city", default=None,
                   help="india city to use (default: most frequent)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_smote", action="store_true",
                   help="disable SMOTE oversampling")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def summarize(metrics, split):
    keys = []
    for k in ("rmse", "mae", "r2", "accuracy", "f1"):
        key = f"{split}_{k}"
        if key in metrics and metrics.get(key) is not None:
            keys.append(f"{k}={float(metrics[key]):.4f}")
    return "  ".join(keys)


def main():
    args = parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    deterministic_seed(args.seed)
    print(f"[setup] device={DEVICE} dataset={args.dataset} split={args.split_mode} "
          f"task={args.task} seed={args.seed}")

    cfg = Config(
        epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size,
        lr=args.lr, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, task=args.task, patience=args.patience,
        early_stopping=True, lr_schedule="plateau", city=args.city,
        weight_decay=1e-5, overfit_budget=8, underfit_budget=8,
        max_lr_reductions=2, quiet=args.quiet, no_smote=args.no_smote,
    )
    models = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    if args.split_mode == "cv10":
        report = {}
        for name in models:
            print(f"\n===== 10-fold CV: {name} =====")
            report[name] = run_cv(args.dataset, name, cfg, seed=args.seed)
        out = os.path.join(REPORT_DIR, f"cv_{args.dataset}_{args.model}_{args.seq_len}.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"[save] CV report -> {out}")
        for name, m in report.items():
            print(f"  {name:22s} rmse={m.get('cv_rmse_mean', float('nan')):.3f}±{m.get('cv_rmse_std', float('nan')):.3f} "
                  f"mae={m.get('cv_mae_mean', float('nan')):.3f} r2={m.get('cv_r2_mean', float('nan')):.3f}")
        return

    data = preprocess_dataset(args.dataset, seq_len=args.seq_len, task=args.task,
                              num_classes=args.num_classes, city=args.city)
    print(f"[data] windows: train={data['metadata']['n_train']} "
          f"val={data['metadata']['n_val']} test={data['metadata']['n_test']} "
          f"features={data['metadata']['input_size']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {}
    for name in models:
        print(f"\n===== Training: {name} (dataset={args.dataset}, task={args.task}) =====")
        ckpt = os.path.join(CKPT_DIR, f"{name}_{args.dataset}_{args.task}_s{args.seq_len}.pt")
        result = run_training(name, data, cfg, checkpoint_path=ckpt, seed=args.seed)
        metrics = {**result["val_metrics"], **result["test_metrics"]}
        metrics.pop("val_y_true", None); metrics.pop("val_y_pred", None)
        metrics.pop("test_y_true", None); metrics.pop("test_y_pred", None)
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
        metrics["model_name"] = name
        metrics["dataset"] = args.dataset
        metrics["task"] = args.task
        metrics["detection"] = result["reason"]
        metrics["corrections"] = result["corrections"]
        report[name] = metrics
        print(f"[eval] {name}\n  val : {summarize(metrics, 'val') or 'n/a'}\n"
              f"  test: {summarize(metrics, 'test') or 'n/a'}")

    out = os.path.join(REPORT_DIR, f"train_{args.dataset}_{stamp}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[save] metrics report -> {out}")
    print("\n===== Summary =====")
    for name, m in report.items():
        print(f"  {name:22s} test_rmse={m.get('test_rmse', float('nan')):.3f} "
              f"test_mae={m.get('test_mae', float('nan')):.3f} "
              f"test_r2={m.get('test_r2', float('nan')):.3f} "
              f"test_f1={m.get('test_f1', float('nan')):.3f}")


if __name__ == "__main__":
    main()