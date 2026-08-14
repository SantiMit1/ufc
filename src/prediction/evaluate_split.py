"""
Evaluate a trained model on the same chronological 85/15 train/test split used
by ``train_model.py`` and dump the metrics as JSON.

This lets you compare baselines vs. retrained models on identical data.

Usage:
  .venv/Scripts/python src/prediction/evaluate_split.py [--out models/metrics.json]
  .venv/Scripts/python src/prediction/evaluate_split.py --model models/baseline/ufc_stacking_ensemble.pkl --features models/baseline/ufc_stacking_ensemble_meta.pkl
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)

from config import DATASET_PATH, MODEL_PATH, FEATURE_COLS_PATH
from stats_utils import (
    ece,
    brier_decomposition,
    favorite_underdog_split,
)


def build_test_split():
    """Reproduce train_model.py's feature pipeline and return (X_test, y_test)."""
    df = pd.read_csv(DATASET_PATH)
    df = df[df["winner"].isin(["1", "2", 1, 2])].copy()
    df["winner"] = df["winner"].astype(int)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df.sort_values("event_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    y = (df["winner"] == 1).astype(int)

    meta = joblib.load(FEATURE_COLS_PATH)
    raw_feature_cols = meta["raw_feature_cols"]
    cat_cols = meta["cat_cols"]
    feature_cols_final = meta["feature_cols_final"]

    X_raw = df[raw_feature_cols].copy()
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    for col in feature_cols_final:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_cols_final]

    split_idx = int(len(df) * (1 - 0.15))
    return X_encoded.iloc[split_idx:].copy(), y.iloc[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on the train/test split")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--features", default=str(FEATURE_COLS_PATH))
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    X_test, y_test = build_test_split()
    model = joblib.load(args.model)

    calibrator_name = getattr(model, "calibrator_name", None)
    if calibrator_name:
        model.calibrator_name = calibrator_name
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    brier, uncertainty, reliability, resolution = brier_decomposition(y_test, y_prob)

    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
    table = []
    for i in range(10):
        mask = idx == i
        if mask.sum() == 0:
            continue
        table.append({
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i + 1]),
            "n": int(mask.sum()),
            "avg_prob": float(y_prob[mask].mean()),
            "obs_freq": float(y_test.values[mask].mean()),
        })

    result = {
        "calibrator_name": calibrator_name,
        "n_test": int(len(y_test)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier": brier,
        "brier_uncertainty": uncertainty,
        "brier_reliability": reliability,
        "brier_resolution": resolution,
        "ece": ece(y_test, y_prob),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "sharpness_mean_abs_dev": float(np.mean(np.abs(y_prob - 0.5))),
        "calibration_table": table,
        "favorite_underdog": favorite_underdog_split(y_test, y_prob),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()