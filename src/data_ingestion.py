"""
Data Ingestion Module
=====================
Downloads and caches the Irish Property Price Register (PPR) dataset.
Source: https://www.propertypriceregister.ie
"""

import os
import io
import logging
import pandas as pd
import numpy as np
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "ppr_raw.csv"

# PPR Open Data endpoint (official Irish government data)
PPR_URL = "https://www.propertypriceregister.ie/website/npsra/pprweb.nsf/Downloads/PPROpenData2024CSV/$FILE/PPROpenData2024.csv"

IRISH_COUNTIES = [
    "Dublin", "Cork", "Galway", "Limerick", "Waterford",
    "Clare", "Kerry", "Kildare", "Kilkenny", "Laois",
    "Leitrim", "Longford", "Louth", "Mayo", "Meath",
    "Monaghan", "Offaly", "Roscommon", "Sligo", "Tipperary",
    "Westmeath", "Wexford", "Wicklow", "Carlow", "Cavan",
    "Donegal"
]

PROPERTY_TYPES = [
    "Detached house",
    "Semi-detached house",
    "Terraced house",
    "Apartment",
    "End of terrace house"
]

BER_RATINGS = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G"]


def download_ppr_data() -> pd.DataFrame:
    """Attempt to download real PPR data; fall back to synthetic if unavailable."""
    DATA_DIR.mkdir(exist_ok=True)

    if RAW_DATA_PATH.exists():
        logger.info("Loading cached PPR data from %s", RAW_DATA_PATH)
        return pd.read_csv(RAW_DATA_PATH, encoding="latin-1")

    logger.info("Attempting to download PPR data from %s", PPR_URL)
    try:
        response = requests.get(PPR_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), encoding="latin-1")
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info("Downloaded %d records from PPR.", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not download PPR data (%s). Generating synthetic dataset.", exc)
        return generate_synthetic_ppr(n=5000)


def generate_synthetic_ppr(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Irish property dataset that mirrors the PPR schema.
    County price distributions are calibrated to approximate real median prices.
    """
    rng = np.random.default_rng(seed)

    # County base prices (€) based on approximate 2023/24 medians
    county_base_price = {
        "Dublin": 420000, "Wicklow": 365000, "Kildare": 340000,
        "Meath": 310000, "Cork": 295000, "Galway": 275000,
        "Limerick": 230000, "Waterford": 210000, "Wexford": 220000,
        "Louth": 225000, "Clare": 215000, "Kerry": 205000,
        "Kilkenny": 210000, "Laois": 185000, "Offaly": 175000,
        "Westmeath": 185000, "Tipperary": 180000, "Cavan": 160000,
        "Monaghan": 155000, "Longford": 140000, "Donegal": 150000,
        "Roscommon": 155000, "Mayo": 155000, "Sligo": 165000,
        "Leitrim": 140000, "Carlow": 190000,
    }

    # Property type multipliers
    type_multiplier = {
        "Detached house": 1.35,
        "Semi-detached house": 1.0,
        "End of terrace house": 0.92,
        "Terraced house": 0.85,
        "Apartment": 0.78,
    }

    counties = rng.choice(IRISH_COUNTIES, size=n, p=_county_weights())
    prop_types = rng.choice(PROPERTY_TYPES, size=n,
                            p=[0.22, 0.30, 0.18, 0.20, 0.10])
    ber = rng.choice(BER_RATINGS, size=n,
                     p=[0.03,0.05,0.07,0.08,0.10,0.10,0.10,0.09,0.08,0.07,0.06,0.05,0.05,0.04,0.03])
    ber_score = [BER_RATINGS.index(b) for b in ber]  # 0 = best, 14 = worst
    floor_area = rng.integers(45, 280, size=n)
    year_built = rng.integers(1900, 2024, size=n)
    bedrooms = rng.integers(1, 7, size=n)
    not_full_market = rng.choice([0, 1], size=n, p=[0.95, 0.05])

    years = rng.integers(2020, 2025, size=n)
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 28, size=n)
    dates = [f"{d:02d}/{m:02d}/{y}" for d, m, y in zip(days, months, years)]

    prices = []
    for i in range(n):
        base = county_base_price[counties[i]]
        mult = type_multiplier[prop_types[i]]
        area_factor = (floor_area[i] / 110) ** 0.6
        ber_factor = 1 - (ber_score[i] * 0.015)
        age_factor = 1 - max(0, (2024 - year_built[i]) / 200)
        noise = rng.normal(1.0, 0.12)
        price = base * mult * area_factor * ber_factor * age_factor * noise
        prices.append(max(50000, round(price, -3)))

    df = pd.DataFrame({
        "Date of Sale (dd/mm/yyyy)": dates,
        "Address": [f"{rng.integers(1,200)} Sample Street, {c}" for c in counties],
        "County": counties,
        "Eircode": [f"D{rng.integers(1,24):02d} {rng.integers(1000,9999)}" for _ in range(n)],
        "Price (€)": [f"€{p:,.0f}" for p in prices],
        "Not Full Market Price": ["No" if v == 0 else "Yes" for v in not_full_market],
        "VAT Exclusive": rng.choice(["No","Yes"], size=n, p=[0.9,0.1]).tolist(),
        "Description of Property": prop_types,
        "Property Size Description": [
            f"{a} sq metres" for a in floor_area],
        "BER Rating": ber,
        "Year Built": year_built,
        "Bedrooms": bedrooms,
    })

    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Generated synthetic PPR dataset with %d records.", n)
    return df


def _county_weights():
    """Population-weighted county probabilities."""
    weights = {
        "Dublin": 0.29, "Cork": 0.11, "Galway": 0.06, "Limerick": 0.05,
        "Waterford": 0.03, "Kildare": 0.05, "Meath": 0.04, "Wicklow": 0.03,
        "Wexford": 0.03, "Louth": 0.03, "Clare": 0.02, "Kerry": 0.03,
        "Kilkenny": 0.02, "Laois": 0.02, "Offaly": 0.02, "Westmeath": 0.02,
        "Tipperary": 0.02, "Cavan": 0.01, "Monaghan": 0.01, "Longford": 0.01,
        "Donegal": 0.03, "Roscommon": 0.01, "Mayo": 0.02, "Sligo": 0.01,
        "Leitrim": 0.005, "Carlow": 0.015,
    }
    vals = list(weights.values())
    total = sum(vals)
    return [v / total for v in vals]


if __name__ == "__main__":
    df = download_ppr_data()
    print(df.head())
    print(f"Shape: {df.shape}")
