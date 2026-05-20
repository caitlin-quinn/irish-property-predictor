"""
Irish Property Price Predictor – Flask API
==========================================
Exposes the trained Random Forest model as a REST API.
Endpoints:
  GET  /health          – liveness probe
  GET  /metrics         – last training metrics
  POST /predict         – single property prediction
  POST /predict/batch   – batch predictions (list of properties)
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH    = Path("models/property_price_model.pkl")
METRICS_PATH  = Path("models/metrics.json")
ENCODERS_PATH = Path("models/label_encoders.pkl")
FEATURES_PATH = Path("models/feature_names.json")

# ── Module-level model cache ──────────────────────────────────────────────────
_model    = None
_encoders = None
_features = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


def get_encoders():
    global _encoders
    if _encoders is None:
        if ENCODERS_PATH.exists():
            _encoders = joblib.load(ENCODERS_PATH)
        else:
            _encoders = {}
    return _encoders


def get_features():
    global _features
    if _features is None:
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH) as f:
                _features = json.load(f)
        else:
            _features = []
    return _features


# ── Helpers ───────────────────────────────────────────────────────────────────

BER_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G"]

IRISH_COUNTIES = [
    "Dublin","Cork","Galway","Limerick","Waterford","Clare","Kerry",
    "Kildare","Kilkenny","Laois","Leitrim","Longford","Louth","Mayo",
    "Meath","Monaghan","Offaly","Roscommon","Sligo","Tipperary",
    "Westmeath","Wexford","Wicklow","Carlow","Cavan","Donegal"
]

PROPERTY_TYPES = [
    "Detached house","Semi-detached house","Terraced house",
    "Apartment","End of terrace house"
]


def encode_input(data: dict) -> np.ndarray:
    """Convert a property dict to a feature vector."""
    encoders = get_encoders()
    features = get_features()

    county_enc = 0
    if "county_enc" in features and "county" in data:
        le = encoders.get("county")
        if le is not None:
            try:
                county_enc = int(le.transform([data["county"].strip().title()])[0])
            except ValueError:
                county_enc = 0

    ptype_enc = 0
    if "property_type_enc" in features and "property_type" in data:
        le = encoders.get("property_type")
        if le is not None:
            try:
                ptype_enc = int(le.transform([data["property_type"].strip()])[0])
            except ValueError:
                ptype_enc = 0

    ber_score = BER_ORDER.index(data.get("ber_rating", "C1").upper()) \
        if data.get("ber_rating", "C1").upper() in BER_ORDER else 6

    is_dublin = 1 if data.get("county", "").strip().title() == "Dublin" else 0

    feature_map = {
        "county_enc":        county_enc,
        "property_type_enc": ptype_enc,
        "floor_area_sqm":    float(data.get("floor_area_sqm", 100)),
        "bedrooms":          int(data.get("bedrooms", 3)),
        "ber_score":         ber_score,
        "property_age":      int(data.get("property_age", 20)),
        "is_dublin":         is_dublin,
        "sale_year":         int(data.get("sale_year", 2024)),
        "sale_month":        int(data.get("sale_month", 6)),
    }

    return np.array([[feature_map.get(f, 0) for f in features]])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           counties=IRISH_COUNTIES,
                           property_types=PROPERTY_TYPES,
                           ber_ratings=BER_ORDER)


@app.route("/health")
def health():
    """Liveness/readiness probe for Kubernetes."""
    try:
        get_model()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503


@app.route("/metrics")
def metrics():
    """Return last training metrics."""
    if not METRICS_PATH.exists():
        return jsonify({"error": "No metrics found – train the model first."}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the price of a single Irish residential property.

    Request JSON:
    {
      "county":          "Dublin",
      "property_type":   "Semi-detached house",
      "floor_area_sqm":  105,
      "bedrooms":        3,
      "ber_rating":      "B2",
      "property_age":    15,
      "sale_year":       2024,
      "sale_month":      6
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["county", "property_type", "floor_area_sqm"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        model = get_model()
        X = encode_input(data)
        log_price = model.predict(X)[0]
        price = float(np.expm1(log_price))

        # Confidence interval (± 1 std across trees)
        preds = np.array([est.predict(
            model.named_steps["scaler"].transform(X))[0]
            for est in model.named_steps["rf"].estimators_[:50]])
        price_std = float(np.std(np.expm1(preds)))
        ci_low  = max(0, price - 1.96 * price_std)
        ci_high = price + 1.96 * price_std

        return jsonify({
            "predicted_price_eur": round(price, -2),
            "confidence_interval": {
                "lower": round(ci_low, -2),
                "upper": round(ci_high, -2),
            },
            "input": data,
        })
    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predict prices for a list of properties."""
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array."}), 400

    results = []
    model = get_model()
    for item in data:
        try:
            X = encode_input(item)
            price = float(np.expm1(model.predict(X)[0]))
            results.append({"predicted_price_eur": round(price, -2), "input": item})
        except Exception as exc:
            results.append({"error": str(exc), "input": item})

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
