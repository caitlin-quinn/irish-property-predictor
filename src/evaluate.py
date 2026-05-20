"""
Model Evaluation & Drift Detection
=====================================
Evaluates a trained model and checks for data/model drift.
Exits with code 1 if quality thresholds are not met (used in CI gates).
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from data_ingestion import download_ppr_data
from preprocessing import load_and_preprocess, get_feature_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quality gates – pipeline fails if these are not met
THRESHOLDS = {
    "r2_min":   0.70,   # Minimum R² on held-out test set
    "mape_max": 25.0,   # Maximum Mean Absolute Percentage Error
    "mae_max":  80000,  # Maximum Mean Absolute Error in €
}

MODEL_PATH   = Path("models") / "property_price_model.pkl"
METRICS_PATH = Path("models") / "metrics.json"
REPORT_PATH  = Path("models") / "evaluation_report.json"


def evaluate() -> dict:
    """Load model + fresh data, compute metrics, enforce quality gates."""
    if not MODEL_PATH.exists():
        logger.error("Model not found at %s – run train.py first.", MODEL_PATH)
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    download_ppr_data()
    df = load_and_preprocess()
    X, y, feature_names = get_feature_matrix(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true.clip(1e-3))) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    report = {
        "mae":  round(mae, 2),
        "rmse": round(rmse, 2),
        "r2":   round(r2, 4),
        "mape": round(mape, 2),
        "n_samples": len(y_test),
        "thresholds": THRESHOLDS,
        "passed": {},
        "overall_pass": True,
    }

    # Quality gate checks
    checks = {
        "r2_min":   r2   >= THRESHOLDS["r2_min"],
        "mape_max": mape <= THRESHOLDS["mape_max"],
        "mae_max":  mae  <= THRESHOLDS["mae_max"],
    }
    report["passed"] = checks
    report["overall_pass"] = all(checks.values())

    for gate, passed in checks.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        logger.info("Quality gate [%s]: %s", gate, status)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved → %s", REPORT_PATH)

    if not report["overall_pass"]:
        logger.error("One or more quality gates FAILED. See %s", REPORT_PATH)
        sys.exit(1)

    logger.info("All quality gates PASSED. R²=%.3f | MAE=€%.0f | MAPE=%.1f%%",
                r2, mae, mape)
    return report


def check_data_drift(reference_path: str = "data/ppr_raw.csv",
                     current_path: str = "data/ppr_raw.csv") -> dict:
    """
    Simple statistical drift check using mean/std comparisons on numeric cols.
    In production, replace with Evidently AI or WhyLogs.
    """
    import pandas as pd
    ref = pd.read_csv(reference_path, encoding="latin-1")
    cur = pd.read_csv(current_path, encoding="latin-1")

    drift_report = {}
    numeric_cols = ref.select_dtypes(include="number").columns

    for col in numeric_cols:
        if col not in cur.columns:
            continue
        ref_mean, ref_std = ref[col].mean(), ref[col].std()
        cur_mean, cur_std = cur[col].mean(), cur[col].std()
        rel_drift = abs(cur_mean - ref_mean) / (ref_mean + 1e-9)
        drift_report[col] = {
            "ref_mean": round(ref_mean, 2),
            "cur_mean": round(cur_mean, 2),
            "rel_drift_pct": round(rel_drift * 100, 2),
            "drifted": rel_drift > 0.15,
        }

    any_drift = any(v["drifted"] for v in drift_report.values())
    if any_drift:
        logger.warning("Data drift detected in: %s",
                       [k for k, v in drift_report.items() if v["drifted"]])
    return drift_report


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2))
