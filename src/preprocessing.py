"""
Data Preprocessing Module
==========================
Cleans and engineers features from the raw Irish PPR dataset.
Outputs a feature matrix (X) and target vector (y) ready for modelling.
"""

import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_PATH = DATA_DIR / "ppr_processed.csv"
ENCODER_PATH = Path("models") / "label_encoders.pkl"

BER_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G"]


def load_and_preprocess(raw_path: str = "data/ppr_raw.csv") -> pd.DataFrame:
    """Full preprocessing pipeline: clean → engineer → encode."""
    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path, encoding="latin-1")

    df = _clean(df)
    df = _engineer_features(df)
    df = _encode_categoricals(df)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved processed data (%d rows, %d cols) → %s",
                *df.shape, PROCESSED_PATH)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and fix dtypes."""
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
                  .replace("/", "_").replace("€", "eur").replace(",", "")
                  for c in df.columns]

    # Parse price column: "€350,000" → 350000.0
    if "price_eur" in df.columns:
        df["price"] = (df["price_eur"]
                       .astype(str)
                       .str.replace("[€,]", "", regex=True)
                       .str.strip()
                       .pipe(pd.to_numeric, errors="coerce"))
    elif any("price" in c for c in df.columns):
        price_col = [c for c in df.columns if "price" in c][0]
        df["price"] = (df[price_col]
                       .astype(str)
                       .str.replace("[€,]", "", regex=True)
                       .pipe(pd.to_numeric, errors="coerce"))

    # Drop rows with missing price
    before = len(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 10_000]  # remove anomalies
    df = df[df["price"] < 5_000_000]
    logger.info("Removed %d rows with invalid prices.", before - len(df))

    # Parse floor area: "105 sq metres" → 105.0
    if "property_size_description" in df.columns:
        df["floor_area_sqm"] = (df["property_size_description"]
                                .astype(str)
                                .str.extract(r"(\d+\.?\d*)")[0]
                                .pipe(pd.to_numeric, errors="coerce"))

    # Parse sale date
    if "date_of_sale_dd_mm_yyyy" in df.columns:
        df["sale_date"] = pd.to_datetime(
            df["date_of_sale_dd_mm_yyyy"], dayfirst=True, errors="coerce")
        df["sale_year"] = df["sale_date"].dt.year
        df["sale_month"] = df["sale_date"].dt.month

    # Tidy property type
    if "description_of_property" in df.columns:
        df["property_type"] = df["description_of_property"].str.strip()

    # Tidy county
    if "county" in df.columns:
        df["county"] = df["county"].str.strip().str.title()

    # Exclude non-market sales
    if "not_full_market_price" in df.columns:
        df = df[df["not_full_market_price"].str.upper() == "NO"]

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional predictive features."""
    df = df.copy()

    # Property age at time of sale
    ref_year = 2024
    if "year_built" in df.columns:
        df["property_age"] = ref_year - df["year_built"].clip(1800, ref_year)

    # BER numeric score (0 = best, 14 = worst → invert for readability)
    if "ber_rating" in df.columns:
        df["ber_score"] = df["ber_rating"].map(
            {r: i for i, r in enumerate(BER_ORDER)})
        df["ber_score"] = df["ber_score"].fillna(7)  # mid-range default

    # Price per sqm (useful sanity metric, not used as feature)
    if "floor_area_sqm" in df.columns:
        df["price_per_sqm"] = df["price"] / df["floor_area_sqm"].clip(lower=1)

    # Log-transform price for modelling
    df["log_price"] = np.log1p(df["price"])

    # Dublin indicator (commands premium)
    if "county" in df.columns:
        df["is_dublin"] = (df["county"] == "Dublin").astype(int)

    # Fill missing numerics
    for col in ["floor_area_sqm", "bedrooms", "property_age"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and persist encoders."""
    df = df.copy()
    encoders = {}
    cat_cols = [c for c in ["county", "property_type"] if c in df.columns]

    Path("models").mkdir(exist_ok=True)
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded '%s' → %d classes", col, len(le.classes_))

    joblib.dump(encoders, ENCODER_PATH)
    logger.info("Saved label encoders → %s", ENCODER_PATH)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (target) arrays."""
    FEATURE_COLS = [
        "county_enc", "property_type_enc",
        "floor_area_sqm", "bedrooms",
        "ber_score", "property_age",
        "is_dublin", "sale_year", "sale_month",
    ]
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing feature columns: %s", missing)

    X = df[available].values
    y = df["log_price"].values
    return X, y, available


if __name__ == "__main__":
    from data_ingestion import download_ppr_data
    download_ppr_data()
    df = load_and_preprocess()
    X, y, cols = get_feature_matrix(df)
    print(f"Features: {cols}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
