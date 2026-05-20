"""
Test Suite – Irish Property Price Predictor
============================================
Covers: data ingestion, preprocessing, model inference, API endpoints.
Run with:  pytest tests/ -v
"""

import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Ensure src/ is on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

# ═══════════════════════════════════════════════════════════════════════════════
# Data Ingestion Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataIngestion:
    def test_synthetic_generation(self, tmp_path, monkeypatch):
        """Synthetic data should produce a DataFrame with expected columns."""
        from data_ingestion import generate_synthetic_ppr
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        df = generate_synthetic_ppr(n=200, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert "County" in df.columns
        assert "Price (€)" in df.columns

    def test_all_irish_counties_present(self, tmp_path, monkeypatch):
        """Dataset should contain records from multiple Irish counties."""
        from data_ingestion import generate_synthetic_ppr
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        df = generate_synthetic_ppr(n=1000, seed=1)
        assert df["County"].nunique() >= 10

    def test_price_range_realistic(self, tmp_path, monkeypatch):
        """Generated prices should be within plausible Irish market range."""
        from data_ingestion import generate_synthetic_ppr
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        df = generate_synthetic_ppr(n=500, seed=2)
        prices = (df["Price (€)"]
                  .str.replace("[€,]", "", regex=True)
                  .astype(float))
        assert prices.min() >= 50_000
        assert prices.max() <= 5_000_000
        assert prices.median() > 100_000


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "models").mkdir()
        from data_ingestion import generate_synthetic_ppr
        generate_synthetic_ppr(n=300, seed=42)

    def test_preprocessing_runs(self):
        from preprocessing import load_and_preprocess
        df = load_and_preprocess()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_log_price_column_exists(self):
        from preprocessing import load_and_preprocess
        df = load_and_preprocess()
        assert "log_price" in df.columns
        assert df["log_price"].isna().sum() == 0

    def test_feature_matrix_shape(self):
        from preprocessing import load_and_preprocess, get_feature_matrix
        df = load_and_preprocess()
        X, y, cols = get_feature_matrix(df)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert len(cols) >= 5

    def test_no_nulls_in_features(self):
        from preprocessing import load_and_preprocess, get_feature_matrix
        df = load_and_preprocess()
        X, y, _ = get_feature_matrix(df)
        assert not np.any(np.isnan(X)), "Feature matrix should contain no NaN values"

    def test_county_encoding_saved(self, tmp_path):
        from preprocessing import load_and_preprocess
        load_and_preprocess()
        encoder_path = tmp_path / "models" / "label_encoders.pkl"
        assert encoder_path.exists(), "Label encoders should be persisted"


# ═══════════════════════════════════════════════════════════════════════════════
# Model Tests (trained model)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:
    @pytest.fixture(scope="class", autouse=True)
    def train_model(self, tmp_path_factory, monkeypatch):
        """Train a small model for testing."""
        tmp = tmp_path_factory.mktemp("model_test")
        monkeypatch.chdir(tmp)
        (tmp / "data").mkdir()
        (tmp / "models").mkdir()
        from train import train
        train(params={
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": 1,
        })

    def test_model_file_created(self, tmp_path):
        assert (Path("models") / "property_price_model.pkl").exists()

    def test_metrics_file_created(self):
        assert (Path("models") / "metrics.json").exists()

    def test_metrics_keys(self):
        with open(Path("models") / "metrics.json") as f:
            metrics = json.load(f)
        for key in ("mae", "rmse", "r2", "mape"):
            assert key in metrics

    def test_r2_positive(self):
        with open(Path("models") / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["r2"] > 0, "R² should be positive for a working model"

    def test_prediction_output(self):
        from train import load_model, load_feature_names
        model = load_model()
        features = load_feature_names()
        X = np.zeros((1, len(features)))
        pred = model.predict(X)
        assert pred.shape == (1,)
        price = float(np.expm1(pred[0]))
        assert price > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Flask API Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlaskAPI:
    SAMPLE_PAYLOAD = {
        "county": "Dublin",
        "property_type": "Semi-detached house",
        "floor_area_sqm": 110,
        "bedrooms": 3,
        "ber_rating": "B2",
        "property_age": 15,
        "sale_year": 2024,
        "sale_month": 6,
    }

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "models").mkdir()
        # Train a tiny model
        from train import train
        train(params={"n_estimators": 5, "max_depth": 3, "min_samples_split": 2,
                      "min_samples_leaf": 1, "random_state": 0, "n_jobs": 1})
        import importlib
        import app as app_module
        importlib.reload(app_module)
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_metrics_ok(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "r2" in resp.get_json()

    def test_predict_valid(self, client):
        resp = client.post("/predict", json=self.SAMPLE_PAYLOAD)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_price_eur" in data
        assert data["predicted_price_eur"] > 0

    def test_predict_missing_fields(self, client):
        resp = client.post("/predict", json={"county": "Cork"})
        assert resp.status_code == 400

    def test_predict_batch(self, client):
        resp = client.post("/predict/batch", json=[self.SAMPLE_PAYLOAD] * 3)
        assert resp.status_code == 200
        results = resp.get_json()
        assert len(results) == 3

    def test_predict_galway(self, client):
        payload = {**self.SAMPLE_PAYLOAD, "county": "Galway", "floor_area_sqm": 90}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_confidence_interval_ordered(self, client):
        resp = client.post("/predict", json=self.SAMPLE_PAYLOAD)
        data = resp.get_json()
        ci = data["confidence_interval"]
        assert ci["lower"] <= data["predicted_price_eur"] <= ci["upper"]
