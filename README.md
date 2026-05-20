# Irish Property Price Predictor — MLOps Pipeline

[![CI](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/ci.yml)
[![CD](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/cd.yml/badge.svg)](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/cd.yml)
[![CT](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/ct.yml/badge.svg)](https://github.com/caitlin-quinn/irish-property-predictor/actions/workflows/ct.yml)

A production-grade **MLOps pipeline** that predicts Irish residential property prices using data from the [Property Price Register](https://www.propertypriceregister.ie/). Built with GitHub Actions, Docker, Flask, and deployed on a Kind Kubernetes cluster.

---

## Architecture Overview

```
PPR Data Source → Data Ingestion → Preprocessing → Model Training
                                                         ↓
     Kind K8s ← Docker Image ← CD Pipeline ← Quality Gate (CI)
         ↓
    Flask API  →  /predict endpoint
         ↑
    CT Pipeline (weekly retraining)
```

## Pipeline Stages

| Stage | Trigger | Workflow |
|-------|---------|----------|
| CI (lint + test + train) | Push to `feature/*`, `develop` | `ci.yml` |
| CD (build + deploy)       | Push to `main`                 | `cd.yml` |
| CT (retrain + promote)    | Weekly (Mon 02:00 UTC)         | `ct.yml` |

## Branching Strategy

```
main           ←── release branches & hotfixes (production)
  └── develop  ←── feature branches (integration)
        └── feature/xyz  (development)
```

## Quick Start

### Local Development
```bash
# 1. Clone
git clone https://github.com/caitlin-quinn/irish-property-predictor.git
cd irish-property-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python src/train.py

# 4. Start API
python app/app.py
# → http://localhost:5000
```

### Docker
```bash
docker-compose up api
```

### Predict via API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "county": "Dublin",
    "property_type": "Semi-detached house",
    "floor_area_sqm": 110,
    "bedrooms": 3,
    "ber_rating": "B2",
    "property_age": 15
  }'
```

## Dataset

The model uses the **Irish Property Price Register (PPR)** — an official government record of all residential property purchases in Ireland. Features include county, property type, floor area, BER rating, and property age.

County price distributions are calibrated to approximate 2023/24 median sale prices (e.g. Dublin ~€420k, Cork ~€295k, Leitrim ~€140k).

## Model Performance

| Metric | Value |
|--------|-------|
| R²     | > 0.80 |
| MAE    | < €30,000 |
| MAPE   | < 15% |

## Tech Stack

- **ML**: scikit-learn (Random Forest), MLflow
- **API**: Flask + Gunicorn
- **Containerisation**: Docker (multi-stage build)
- **Orchestration**: Kubernetes (Kind)
- **CI/CD/CT**: GitHub Actions
- **Registry**: GitHub Container Registry (GHCR)
