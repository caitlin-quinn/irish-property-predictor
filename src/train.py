"""
Model Training Module
======================
Trains a Random Forest regressor on the Irish PPR dataset.
Logs metrics/artefacts with MLflow and saves the model to /models.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_ingestion import download_ppr_data
from preprocessing import load_and_preprocess, get_feature_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "property_price_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_PATH = MODEL_DIR / "feature_names.json"

# Hyperparameters (can be overridden via env vars in CI/CT)
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def train(params: dict = None) -> dict:
    """Full train → evaluate → persist pipeline."""
    if params is None:
        params = DEFAULT_PARAMS

    MODEL_DIR.mkdir(exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    logger.info("Step 1/4 – Ingesting data …")
    download_ppr_data()

    logger.info("Step 2/4 – Preprocessing …")
    df = load_and_preprocess()
    X, y, feature_names = get_feature_matrix(df)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── 3. Train ─────────────────────────────────────────────────────────────
    logger.info("Step 3/4 – Training Random Forest …")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(**params))
    ])

    mlflow.set_experiment("irish-property-price-prediction")
    with mlflow.start_run():
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        # ── 4. Evaluate ───────────────────────────────────────────────────────
        logger.info("Step 4/4 – Evaluating …")
        y_pred_log = model.predict(X_test)

        # Convert back from log scale
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true.clip(1e-3))) * 100)

        # Cross-val on log-scale
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring="neg_root_mean_squared_error")
        cv_rmse = float(-cv_scores.mean())

        metrics = {
            "mae":     round(mae, 2),
            "rmse":    round(rmse, 2),
            "r2":      round(r2, 4),
            "mape":    round(mape, 2),
            "cv_rmse": round(cv_rmse, 4),
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }

        mlflow.log_metrics({
            "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "cv_rmse": cv_rmse
        })
        mlflow.sklearn.log_model(model, "property_price_model")

        logger.info("Metrics: MAE=€%.0f | RMSE=€%.0f | R²=%.3f | MAPE=%.1f%%",
                    mae, rmse, r2, mape)

    # ── 5. Persist ────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info("Model saved → %s", MODEL_PATH)
    return metrics


def load_model():
    """Load the persisted model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


def load_feature_names():
    if not FEATURE_PATH.exists():
        return []
    with open(FEATURE_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
