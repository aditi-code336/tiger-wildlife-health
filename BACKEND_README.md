# Wildlife Health Anomaly Detection — Backend API

Hardened **FastAPI** service for unsupervised wildlife health anomaly detection, integrated with the **XAMPP MySQL database** (`wildlife_conservation`) and resilient **SQLite fallback**.

---

## 1. Deployed Model Version

- **Deployed Version**: **Model v2 (`v2_wahis_context`)**
- **Architecture**:
  - **Isolation Forest** (`contamination=0.10`, `random_state=42`)
  - **Local Outlier Factor** (`n_neighbors=20`, `contamination=0.10`, `novelty=True`)
  - Multi-Label Binarizer across 839 clinical symptom tokens
  - Standardized WAHIS host population & surveillance context features
- **Selection Rationale**:
  - Tested superior to Model v1 in Phase 2:
    - **Precision on Dangerous Cases**: `0.9938` vs. `0.9872`
    - **Recall on Dangerous Cases**: `0.1873` vs. `0.1814` (**5 additional dangerous cases caught**)
    - **Recall on Safe Cases**: `0.9500` vs. `0.9000` (**False alarms cut in half**)

---

## 2. Datasets Used in Deployment

1. **Kaggle Animal Condition Classification Dataset** (`data/raw/data.csv`):
   - 871 individual animal clinical observations across 45 species.
2. **WOAH/WAHIS (World Organisation for Animal Health) Global Surveillance Exports**:
   - `Animal population 2026-10-26.csv`, `SMR_2026-10-26_16-29.csv`, `AR_2026-10-26.csv` (2023–2025 global census and surveillance dossiers).

---

## 3. Database Architecture & Schema

Configuration is read dynamically from `.env` using `pymysql` with **plain SQL (strictly zero ORM)**:
```ini
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=wildlife_conservation
```

### Exact Columns Assumed from `animals` Table:
| Column Name | MySQL Data Type | Description |
|---|---|---|
| `id` | `INT(11)` | Primary Key (auto_increment) |
| `name` | `VARCHAR(255)` | Animal's assigned name (e.g. "Tiger", "Elephant") |
| `species` | `VARCHAR(255)` | Biological or common species name (e.g. "Panthera tigris") |
| `age` | `INT(11)` | Age of animal |
| `health_status`| `VARCHAR(255)` | Current recorded health status |
| `location` | `VARCHAR(255)` | Enclosure / park sector |
| `date_recorded`| `TIMESTAMP` | Record creation timestamp |

### New Table Created: `ml_predictions`
Created automatically on startup without touching existing tables:
```sql
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    animal_id INT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    species VARCHAR(255),
    symptoms JSON,
    flag VARCHAR(50) NOT NULL,
    score FLOAT NOT NULL,
    FOREIGN KEY (animal_id) REFERENCES animals(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 4. Endpoints Specification

### A. Health & Database Connectivity: `GET /health`
Reports model status, live database connectivity for MySQL and SQLite, and the NumPy photo model state.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v2_wahis_context",
  "vocabulary_size": 839,
  "database": {
    "primary_mysql": {
      "connected": true,
      "host": "127.0.0.1",
      "port": 3306,
      "database": "wildlife_conservation",
      "user": "root",
      "error": null
    },
    "sqlite_fallback": {
      "connected": true,
      "database_path": "data/predictions.db",
      "total_logged": 10
    }
  },
  "photo_model": {
    "trained": true,
    "gallery_counts": {
      "healthy": 22,
      "injured": 22,
      "total": 44
    },
    "test_accuracy": 1.0,
    "recall_on_injured": 1.0,
    "trained_at": "2026-10-26T12:22:04.522908+00:00"
  }
}
```

---

### B. Animal Prediction: `POST /predict/animal/{animal_id}`
1. Looks up the animal's species from the `animals` table using plain SQL.
2. Runs the unsupervised anomaly detection model.
3. Records the result in `ml_predictions` (or SQLite fallback if MySQL is down).
4. Returns the result:

**Request**:
```bash
POST /predict/animal/2
Content-Type: application/json

{
  "symptoms": ["limping", "swelling on leg", "pain", "reduced appetite"]
}
```

**Response (`200 OK`)**:
```json
{
  "animal_id": 2,
  "species": "Panthera tigris",
  "animal_name": "Tiger",
  "flag": "Normal",
  "score": 0.014,
  "iso_score": 0.014,
  "lof_score": 0.0796,
  "iso_pred": 1,
  "lof_pred": 1,
  "recognized_symptoms": ["limping", "swelling on leg", "pain", "poor appetite"],
  "unrecognized_symptoms": [],
  "warning": null,
  "stored_in": "mysql"
}
```

---

### C. Animal Prediction History: `GET /animal/{animal_id}/predictions`
Returns the historical record of predictions for a specific animal.

**Response (`200 OK`)**:
```json
{
  "animal_id": 2,
  "species": "Panthera tigris",
  "animal_name": "Tiger",
  "total_predictions": 1,
  "source": "mysql",
  "predictions": [
    {
      "id": 1,
      "animal_id": 2,
      "timestamp": "2026-10-26T17:21:26",
      "species": "Panthera tigris",
      "symptoms": ["limping", "swelling on leg", "pain", "reduced appetite"],
      "flag": "Normal",
      "score": 0.014,
      "source": "mysql"
    }
  ]
}
```

---

## 5. Automated Tests

Run the test suite verifying all 7 integration tests:
```bash
python test_app.py
```
Test suite coverage:
- Health check reporting both database states
- Animal prediction resolving species from MySQL
- Non-existent animal handling (404 Not Found)
- History retrieval
- Direct verification of MySQL `ml_predictions` rows
- Automatic SQLite fallback when MySQL is simulated offline

---

## 6. Photo Recognition Module (From-Scratch NumPy Neural Network)

A dedicated, lightweight photo recognition system for wildlife physical condition assessment (**injured = 1, healthy = 0**).

Implemented **strictly from scratch in pure NumPy** adhering to lab techniques (zero PyTorch, TensorFlow, Keras, CLIP, OpenCV, or sklearn):
- **Image Preprocessing** (`preprocess.py`): Pillow opens, converts to grayscale (`'L'`), resizes to $32 \times 32$, divides by 255, and flattens to a $(1024, 1)$ column vector ($X \in \mathbb{R}^{1024 \times m}, Y \in \mathbb{R}^{1 \times m}$).
- **Neural Network Architecture** (`train_photo_nn.py`):
  $$Z_1 = W_1 X + b_1 \quad (W_1 \in \mathbb{R}^{16 \times 1024}, b_1 \in \mathbb{R}^{16 \times 1})$$
  $$A_1 = \tanh(Z_1)$$
  $$Z_2 = W_2 A_1 + b_2 \quad (W_2 \in \mathbb{R}^{1 \times 16}, b_2 \in \mathbb{R}^{1 \times 1})$$
  $$A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}$$
- **Bug Fixes from Lab Code**:
  - Analytical backprop uses exact tanh derivative on $Z_1$: $(1 - A_1^2)$ (or $\frac{1}{\cosh^2(Z_1)}$), not `tanh_grd(A1)`.
  - Hidden layer size $n_h$ is passed as an explicit function argument, never a global variable.
  - Weight matrices initialized as $\mathcal{N}(0, 1) \times 0.01$, biases initialized as zeros.
- **Gallery Storage**:
  - Saved uploads to `data/photos/`.
  - SQLite table `photo_gallery` (`id`, `filename`, `label`, `species`, `notes`, `uploaded_at`).
- **Data Augmentation & Stratification**:
  - Shuffle gallery with fixed seed.
  - 80% train / 20% test split executed **before** adding horizontal flips to prevent data leakage across splits.
  - Hand-computed confusion matrix, accuracy, precision, and recall on the test set. Emphasizes that **recall on injured matters most**.
  - Cost curve saved to `data/photo_cost_plot.png` and parameters saved with `np.savez` to `models/photo_nn.npz`.
  - **Refusal Guard**: Refuses training if fewer than 20 photos exist per label.

### Photo Recognition Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /photos/reference` | Upload reference photo (`.jpg`/`.png` < 5MB) + `label` (0 or 1), `species`, `notes` |
| `GET /photos/gallery` | Retrieve all reference photos and class count breakdown |
| `DELETE /photos/{id}` | Remove mislabeled photo from gallery and disk |
| `POST /photos/train` | Train scratch NN; returns hand-computed test metrics |
| `POST /photos/check` | Upload test photo; returns probability, label, and model test accuracy. If $p \in [0.4, 0.6]$, returns `"uncertain"`. Returns **409 Conflict** if no model is trained yet |
| `POST /assess` | Multimodal assessment: accepts symptoms + optional photo, returns symptom flag and photo result side by side with a 1-line summary |

### Photo Test Suite

Run the full photo module verification:
```bash
python test_photo_module.py
```
Verified features:
1. Shape test `(1024, 1)` and pixel normalization $[0, 1]$
2. Numerical gradient check ($< 10^{-7}$ error)
3. 409 Conflict rejection before training
4. Bad file type and corrupted image rejection
5. Gallery CRUD & training refusal with $< 20$ samples per label
6. Full training run with weight persistence and inference
7. Multimodal `/assess` side-by-side evaluation
