#!/usr/bin/env python3
"""City-wise AQI forecasting.

For every Indian city with enough history, trains a hybrid model on an 80:20
temporal split (SMOTE applied to the 80% training split only) and predicts the
NEXT-DAY AQI. Results are saved to
``data/processed/predictions/city_predictions.csv`` for the map visualisations.

Examples:
    python src/predict.py --model lstm_cnn --epochs 20 --seq_len 7
    python src/predict.py --model all --epochs 30 --forecast_days 3   # ensemble + 3-day
    python src/predict.py --city Delhi --model lstm_attention
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MODEL_REGISTRY, deterministic_seed  # noqa: E402
from src.models.base import apply_normalizer  # noqa: E402
from src.train import (  # noqa: E402
    Config,
    INDIA_FEATURES,
    INDIA_TARGET,
    PROCESSED_DIR,
    load_india_data,
    preprocess_dataset,
    run_training,
)
from src.visualization.location_table import resolve_many  # noqa: E402

try:
    from src.train import CKPT_DIR, REPORT_DIR
except ImportError:  # pragma: no cover
    CKPT_DIR = os.path.join("models", "checkpoints")
    REPORT_DIR = os.path.join("models", "reports")

PRED_DIR = os.path.join(PROCESSED_DIR, "predictions")
REPORT_JSON = os.path.join(REPORT_DIR, "city_predictions_summary.json")

# CPCB AQI bucket ranges (min AQI: label)
CPCB_BUCKETS = [
    (0, "Good"), (51, "Moderate"), (101, "Satisfactory"),
    (151, "Poor"), (201, "Very Poor"), (301, "Severe"),
]


def bucket_of(aqi):
    for lo, label in CPCB_BUCKETS:
        if float(aqi) < lo:
            return label
    return "Severe"


def list_cities():
    processed = os.path.join(PROCESSED_DIR, "india_city_day_clean.csv")
    raw = os.path.join("data", "raw", "india_city_day.csv")
    file = processed if os.path.exists(processed) else raw
    df = pd.read_csv(file, usecols=["City"])
    return sorted(df["City"].dropna().unique().tolist())


def last_forecast_window(city, seq_len):
    """Return the (unscaled) last ``seq_len`` feature rows for a city.

    Mirrors the NaN filtering done inside ``preprocess_dataset`` so the model
    sees exactly the same feature space at prediction time.
    """
    df = load_india_data(city)
    features = [f for f in INDIA_FEATURES]
    X_raw = df[features].to_numpy(dtype=np.float64)
    y_reg = df[INDIA_TARGET].to_numpy(dtype=np.float64)

    bad = np.isnan(X_raw).any(axis=1) | np.isnan(y_reg)
    X_raw = X_raw[~bad]
    dates = df["datetime"].to_numpy()[~bad]
    if len(X_raw) < seq_len:
        return None, None
    window = X_raw[-seq_len:]
    last_date = pd.Timestamp(dates[-1])
    return window, last_date


def run_city(city, model_name, cfg, seq_len, task, num_classes, seed, ret):
    data = preprocess_dataset(
        dataset="india", seq_len=seq_len, task=task,
        num_classes=num_classes, city=city, train_frac=0.8, val_frac=0.0,
    )
    ckpt = os.path.join(CKPT_DIR,
                        f"pred_{model_name}_{city}_s{seq_len}.pt")
    result = run_training(model_name, data, cfg, checkpoint_path=ckpt, seed=seed)

    # --- next-day forecast -------------------------------------------------
    window, last_date = last_forecast_window(city, seq_len)
    if window is None:
        ret["forecastable"] = False
        return data, result

    X = apply_normalizer(window[None, ...], data["feature_scaler"]).astype(np.float32)
    model = result["model"].to("cpu").eval()
    with torch.no_grad():
        reg, cls_logits = model(torch.from_numpy(X))
    pred_scaled = float(reg.reshape(-1)[0])
    pred = float(data["reg_scaler"].inverse_transform([[pred_scaled]])[0, 0])
    pred = max(0.0, pred)

    cls_prob = F.softmax(cls_logits, dim=1)[0]
    cls_idx = int(cls_prob.argmax().item())
    cls_label = None
    if data["cls_map"]:
        inv = {v: k for k, v in data["cls_map"].items()}
        cls_label = inv.get(cls_idx)

    ret["forecastable"] = True
    ret["predicted_aqi"] = pred
    ret["predicted_bucket"] = bucket_of(pred)
    ret["cls_bucket"] = cls_label
    ret["forecast_date"] = (last_date + timedelta(days=1)).date().isoformat()
    ret["last_date"] = last_date.date().isoformat()
    return data, result


def forecast_city(city, model_names, cfg, seq_len, task, num_classes, seed):
    """Ensemble forecast over ``model_names`` and return averaged AQI."""
    preds = []
    per_model = {}
    for name in model_names:
        ret = {}
        data, result = run_city(city, name, cfg, seq_len, task, num_classes,
                                seed, ret=ret)
        per_model[name] = {
            "predicted_aqi": ret.get("predicted_aqi"),
            "test_rmse": result["test_metrics"].get("test_rmse"),
            "test_r2": result["test_metrics"].get("test_r2"),
            "reason": result["reason"],
            "corrections": result["corrections"],
        }
        if ret.get("predicted_aqi") is not None:
            preds.append(ret["predicted_aqi"])
        fc_date = ret.get("forecast_date")
        last_date = ret.get("last_date")

    if not preds:
        return None, per_model, None, None
    return float(np.mean(preds)), per_model, fc_date, last_date


def main():
    ap = argparse.ArgumentParser(description="City-wise AQI forecast")
    ap.add_argument("--dataset", choices=["india"], default="india")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY) + ["all"],
                    default="lstm_cnn")
    ap.add_argument("--city", default=None, help="single city (default: all)")
    ap.add_argument("--max-cities", type=int, default=0, help="limit for dev runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--task", choices=["both", "regression", "classification"],
                    default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_smote", action="store_true")
    args = ap.parse_args()

    deterministic_seed(args.seed)
    os.makedirs(PRED_DIR, exist_ok=True)

    model_names = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    if args.city:
        cities = [args.city]
    else:
        cities = list_cities()
    if args.max_cities:
        cities = cities[: args.max_cities]

    cfg = Config(
        epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size,
        lr=1e-3, hidden_size=64, num_layers=2, dropout=0.2,
        task=args.task, patience=6, early_stopping=True,
        lr_schedule="plateau", weight_decay=1e-5,
        overfit_budget=8, underfit_budget=8, max_lr_reductions=2,
        quiet=True, no_smote=args.no_smote,
    )

    resolved = resolve_many(cities)
    rows = []
    summary = {}
    for city in cities:
        print(f"\n===== City: {city}  models={','.join(model_names)} =====")
        try:
            aqi, per_model, fc_date, last_date = forecast_city(
                city, model_names, cfg, args.seq_len, args.task,
                args.num_classes, args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  !!! {city} failed: {type(exc).__name__}: {exc}")
            continue

        lat, lon, state, country = resolved.get(city, (None, None, "", ""))
        summary[city] = {
            "state": state, "country": country, "n_models": len(per_model),
            "per_model": per_model,
        }
        if aqi is None:
            print(f"  {city}: insufficient history, skipped")
            continue
        print(f"  → predicted AQI for {fc_date}: {aqi:.1f} "
              f"({bucket_of(aqi)}) last={last_date}")
        for name, m in per_model.items():
            print(f"      {name:20s} pred={m['predicted_aqi'] if m['predicted_aqi'] is not None else float('nan'):.1f} "
                  f"test_rmse={m['test_rmse']:.2f} r2={m['test_r2']:.2f} "
                  f"({m['reason']})")
        rows.append({
            "city": city, "state": state, "country": country,
            "lat": lat, "lon": lon,
            "last_date": last_date, "forecast_date": fc_date,
            "predicted_aqi": round(aqi, 2),
            "bucket": bucket_of(aqi),
            "n_models": len(per_model),
            "models": ",".join(model_names),
        })

    if not rows:
        print("No predictions produced.")
        return

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(PRED_DIR, "city_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\n[save] city predictions -> {out_csv}  ({len(out_df)} cities)")

    with open(REPORT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[save] summary -> {REPORT_JSON}")

    bucket_counts = out_df["bucket"].value_counts().to_dict()
    print("\n===== Summary (predicted AQI buckets) =====")
    for b, c in bucket_counts.items():
        print(f"  {b:14s} {c} city/cities")
    print(out_df[["city", "state", "forecast_date", "predicted_aqi", "bucket"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()