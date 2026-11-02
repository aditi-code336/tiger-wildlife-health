---
title: Tiger Wildlife Health Diagnostics
colorFrom: green
colorTo: teal
sdk: docker
app_port: 8001
pinned: false
---

# Wildlife Health Anomaly Detection

A lightweight unsupervised machine learning project for detecting unusual animal health observation records using **Isolation Forest** and **Local Outlier Factor (LOF)**.

## Project Overview

- **Goal**: Identify abnormal animal observation records without requiring labelled 'diseased' datasets.
- **Models**:
  - **Isolation Forest**: Tree-based partition isolation that flags points requiring fewer splits to isolate.
  - **Local Outlier Factor (LOF)**: Density-based outlier detection comparing local neighbourhood densities.
- **Combined Flag**: A record is flagged as `Unusual` if either model detects it as an anomaly.

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── data.csv                  # Raw Kaggle animal disease dataset
│   │   ├── Animal population 2026-10-26.csv # WAHIS global animal population & census
│   │   ├── SMR_2026-10-26_16-29.csv  # WAHIS Six-Monthly Reports
│   │   └── AR_2026-10-26.csv         # WAHIS Annual Reports
│   ├── README.md                     # Data catalog & audit notes (Phases 0 & 2)
│   ├── predictions.db                # SQLite database logging API prediction requests & results
│   ├── phase1_flagged_kaggle.csv     # Model-scored Kaggle dataset (Phase 1)
│   ├── phase2_flagged_wahis_kaggle.csv # Model-scored Kaggle dataset with WAHIS context (Phase 2)
│   ├── observations.csv              # Synthetic wildlife observation dataset
│   ├── flagged_observations.csv      # Scored synthetic observations
│   ├── observations_data.js          # Structured frontend data with observation history
│   └── anomaly_plot.png              # Scatter plot visualization
├── phase1_baseline.ipynb             # Phase 1 notebook (Symptoms Only Baseline)
├── phase1_baseline.py                # Standalone CLI script for Phase 1
├── phase2_wahis_context.ipynb        # Phase 2 notebook (Symptoms + WAHIS Context)
├── phase2_wahis_context.py           # Standalone CLI script for Phase 2
├── train_model.py                    # Script to train and export final model.joblib
├── model.joblib                      # Validated Phase 2 model bundle
├── app.py                            # Hardened FastAPI backend service with /health & /predict
├── test_app.py                       # Automated test suite for Phase 4 hardening & SQLite logging
├── BACKEND_README.md                 # Dedicated backend documentation (models, datasets, schemas)
├── index.html                        # Minimal single-page frontend (Animal Health Monitor)
├── anomaly_detection.py              # Synthetic pipeline script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training & evaluations**:
   ```bash
   python phase1_baseline.py
   python phase2_wahis_context.py
   python train_model.py
   ```

3. **Launch Hardened FastAPI Backend (Phases 3 & 4)**:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```
   Interactive API documentation (Swagger) is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

   **Check Health**:
   ```bash
   curl "http://127.0.0.1:8000/health"
   ```

   **Example Prediction Request**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"symptoms": ["sudden death", "staggering", "trembling"], "species": "Deer"}'
   ```

4. **Verify Test Suite**:
   ```bash
   python test_app.py
   ```

---

## Phase 1: ML v1 Baseline (Kaggle Dataset)

- **Deliverables**:
  - `phase1_baseline.ipynb`: Complete, executed top-to-bottom notebook with EDA visualizations, symptom vocabulary cleaning, multi-label one-hot feature matrix $X$, unsupervised modeling (`IsolationForest` + `LocalOutlierFactor`), and confusion matrix / classification report.
  - `phase1_baseline.py`: Standalone CLI script for Phase 1.
  - `data/phase1_flagged_kaggle.csv`: Scored Kaggle records with predictions (`iso_pred`, `lof_pred`), anomaly decision scores (`iso_score`, `lof_score`), and ensemble `flag`.
- **Key Findings**:
  - **Recall Prioritized**: In wildlife epidemiological surveillance, missing an active, dangerous disease case (False Negative) is clinically far worse than a false alarm (False Positive).

---

## Phase 2: ML v2 (WAHIS Global Surveillance Context)

- **Deliverables**:
  - `phase2_wahis_context.ipynb`: Complete, executed top-to-bottom notebook integrating WAHIS global census and surveillance context, with side-by-side metric comparisons.
  - `phase2_wahis_context.py`: Standalone CLI script for Phase 2.
  - `data/phase2_flagged_wahis_kaggle.csv`: Dataset with predictions and scores across both v1 and v2.
- **Granularity-Safe Join**: WAHIS context was derived at species level across the 2023–2025 reporting window and joined strictly on `AnimalName` $\rightarrow$ `Animal category` without forcing joins on missing geographic coordinates or dates.
- **Side-by-Side Results Summary**:
  - Precision on Dangerous cases: `0.9872` (v1) $\rightarrow$ **`0.9938` (v2)**.
  - Recall on Dangerous cases: `0.1814` (v1) $\rightarrow$ **`0.1873` (v2)** (**5 additional Dangerous cases caught**).
  - Recall on Safe cases: `0.9000` (v1) $\rightarrow$ **`0.9500` (v2)** (**False alarms cut in half from 2 to 1**).

---

## Phase 3 & 4: Backend Model Serving, Hardening & SQLite Logging

- **Deployed Model**: **Model v2 (`v2_wahis_context`)** trained on Kaggle clinical cases and WAHIS global surveillance census.
- **Deliverables**:
  - `model.joblib`: Exported final Model v2 bundle.
  - `app.py`: Hardened FastAPI application loading `model.joblib` once at startup.
  - `GET /health`: Health and readiness endpoint reporting model version, loaded datasets, and SQLite connectivity.
  - `POST /predict`: Prediction endpoint with defensive input validation (gracefully partitions recognized vs unrecognized symptom strings without crashing).
  - `data/predictions.db`: SQLite database logging timestamps, raw input/output payloads, recognized/unrecognized symptoms, and decision scores without ORM layers.
  - `BACKEND_README.md`: Dedicated documentation for backend architecture, datasets, and API endpoints.
  - `test_app.py`: Test suite verifying health check, input validation, unknown symptom handling, and SQLite audit logging.
