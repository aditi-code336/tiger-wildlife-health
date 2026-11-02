"""
FastAPI Backend Service for Wildlife Health Anomaly Detection.
Integrates with XAMPP MySQL database (wildlife_conservation) with SQLite fallback.
Plain SQL with pymysql — zero ORM abstraction layers.
"""

import os
import re
import io
import json
import uuid
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager

import joblib
import numpy as np
from PIL import Image
import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train_photo_nn import (
    train_model,
    predict_photo,
    load_trained_model,
    DEFAULT_MODEL_FILE,
    DEFAULT_PLOT_PATH,
    PHOTOS_DIR,
    MIN_SAMPLES_PER_CLASS,
)
from tiger_diagnostics import diagnose_tiger_health

# Load configuration from .env file
load_dotenv()

SQLITE_DB_PATH = os.path.join("data", "predictions.db")
MODEL_PATH = "model.joblib"
TIGER_RESERVES_FILE = os.path.join("data", "tiger_reserves_ntca.json")

# Global in-memory model container
model_bundle = {}
tiger_reserves_cache = []


def load_tiger_reserves() -> list:
    global tiger_reserves_cache
    if not tiger_reserves_cache and os.path.exists(TIGER_RESERVES_FILE):
        try:
            with open(TIGER_RESERVES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                tiger_reserves_cache = data.get("reserves", [])
        except Exception as e:
            print(f"Notice: loading tiger reserves JSON failed: {e}")
    return tiger_reserves_cache


def get_mysql_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "wildlife_conservation"),
        "connect_timeout": 3,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def get_mysql_connection():
    config = get_mysql_config()
    return pymysql.connect(**config)


def init_sqlite_db():
    os.makedirs("data", exist_ok=True)
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    # General prediction logs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            species TEXT,
            input_payload TEXT,
            output_payload TEXT,
            flag TEXT NOT NULL,
            score REAL NOT NULL,
            iso_score REAL NOT NULL,
            lof_score REAL NOT NULL,
            recognized_symptoms TEXT,
            unrecognized_symptoms TEXT
        )
        """
    )
    # Fallback animal prediction history table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            species TEXT,
            symptoms TEXT,
            flag TEXT NOT NULL,
            score REAL NOT NULL
        )
        """
    )
    # Photo gallery reference table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS photo_gallery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label INTEGER NOT NULL,
            species TEXT,
            notes TEXT,
            uploaded_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def init_mysql_db():
    """
    Creates ml_predictions table referencing animals(id) if it does not exist.
    Never modifies existing tables.
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
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
                """
            )
            conn.commit()
        conn.close()
        print("XAMPP MySQL database connected and ml_predictions verified.")
        return True
    except Exception as e:
        print(f"Notice: MySQL connection check on startup: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load trained ML model bundle once at startup
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Run python train_model.py first."
        )

    print(f"Loading model bundle from {MODEL_PATH}...")
    global model_bundle
    model_bundle = joblib.load(MODEL_PATH)
    print(
        f"Model {model_bundle.get('version')} loaded! "
        f"Vocabulary size: {model_bundle.get('symptom_vocabulary_size')} tokens."
    )

    # 2. Initialize storage (SQLite fallback + MySQL target)
    init_sqlite_db()
    init_mysql_db()

    yield
    print("Shutting down backend service.")


app = FastAPI(
    title="Wildlife Health Anomaly Detection API",
    description="Connects unsupervised ML model to XAMPP MySQL database with SQLite fallback.",
    version="3.0.0",
    lifespan=lifespan,
)

if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

os.makedirs(PHOTOS_DIR, exist_ok=True)
app.mount("/photos-static", StaticFiles(directory=PHOTOS_DIR), name="photos_static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the static frontend Animal Health Monitor page."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Wildlife Health API is running. Visit /docs for interactive documentation."}


@app.get("/animals")
def list_animals():
    """
    Returns list of tracked animals from XAMPP MySQL database with resilient fallback.
    """
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, species, age, health_status, location FROM animals ORDER BY id ASC")
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Notice: /animals MySQL query fallback: {e}")
        return [
            {"id": 1, "name": "Elephant", "species": "Loxodonta africana", "age": 25, "health_status": "Healthy", "location": "Savannah"},
            {"id": 2, "name": "Tiger", "species": "Panthera tigris", "age": 10, "health_status": "Injured", "location": "Forest"},
            {"id": 3, "name": "Panda", "species": "Ailuropoda melanoleuca", "age": 5, "health_status": "Healthy", "location": "Bamboo Grove"},
            {"id": 4, "name": "Mihir", "species": "Monkey", "age": 21, "health_status": "healthy", "location": "Mumbai"}
        ]


# --- Pydantic Schemas ---

class GenericPredictRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1, example=["fever", "diarrhea", "vomiting"])
    species: Optional[str] = Field(None, example="Deer")


class AnimalPredictRequest(BaseModel):
    symptoms: List[str] = Field(
        ...,
        min_items=1,
        example=["difficulty in breathing", "sudden death", "staggering"],
        description="Observed clinical symptoms for the animal",
    )


class PredictionResult(BaseModel):
    animal_id: Optional[int] = None
    species: str
    animal_name: Optional[str] = None
    flag: str
    score: float
    iso_score: float
    lof_score: float
    iso_pred: int
    lof_pred: int
    recognized_symptoms: List[str]
    unrecognized_symptoms: List[str]
    warning: Optional[str] = None
    stored_in: str  # "mysql" or "sqlite_fallback"


class AnimalHistoryItem(BaseModel):
    id: int
    animal_id: int
    timestamp: str
    species: Optional[str]
    symptoms: Any
    flag: str
    score: float
    source: str


class AnimalHistoryResponse(BaseModel):
    animal_id: int
    species: Optional[str]
    animal_name: Optional[str]
    total_predictions: int
    source: str
    predictions: List[AnimalHistoryItem]


# --- Helper Functions ---

def map_species(species_name: Optional[str]) -> str:
    if not species_name:
        return "Other"
    n = str(species_name).lower().strip()
    if any(b in n for b in ["bird", "chicken", "duck", "fowl", "aves"]):
        return "Birds"
    if any(c in n for c in ["cattle", "cow", "bos"]):
        return "Cattle"
    if any(b in n for b in ["buffalo", "bubalus"]):
        return "Buffaloes"
    if any(s in n for s in ["sheep", "ovis"]):
        return "Sheep"
    if any(g in n for g in ["goat", "capra"]):
        return "Goats"
    if any(p in n for p in ["pig", "swine", "sus"]):
        return "Swine"
    if any(e in n for e in ["horse", "donkey", "mule", "equus"]):
        return "Equidae"
    if any(d in n for d in ["dog", "fox", "wolv", "hyaena", "canis"]):
        return "Dogs"
    if any(f in n for f in ["cat", "lion", "tiger", "panthera", "felis"]):
        return "Cats"
    if any(c in n for c in ["deer", "elk", "wapiti", "moos", "reindeer", "cervus"]):
        return "Cervidae"
    if any(r in n for r in ["rabbit", "lepus", "oryctolagus"]):
        return "Rabbits / hares (mixed group)"
    return "Other"


def run_inference(symptoms: List[str], species_name: Optional[str]) -> Dict[str, Any]:
    raw_tokens = [str(s).strip() for s in symptoms if str(s).strip()]
    if not raw_tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symptoms list contains only empty or whitespace-only strings.",
        )

    # 1. Spelling normalization
    spelling_map = model_bundle.get("spelling_mappings", {})
    cleaned_tokens = []
    for token in raw_tokens:
        norm = re.sub(r"\s+", " ", token.lower())
        mapped = spelling_map.get(norm, norm)
        cleaned_tokens.append(mapped)

    # 2. Vocabulary validation
    mlb = model_bundle["mlb"]
    known_vocab = set(mlb.classes_)
    recognized = [t for t in cleaned_tokens if t in known_vocab]
    unrecognized = [t for t in cleaned_tokens if t not in known_vocab]

    warning_msg = None
    if unrecognized:
        if not recognized:
            warning_msg = (
                f"None of the provided symptoms {unrecognized} were recognized in the clinical training vocabulary. "
                "Inference performed using baseline host species context."
            )
        else:
            warning_msg = (
                f"The following symptoms were unrecognized and ignored during scoring: {unrecognized}."
            )

    # 3. Multi-label binarize symptoms
    X_sym = mlb.transform([recognized])

    # 4. Attach WAHIS species context
    category = map_species(species_name)
    species_dict = model_bundle.get("species_dict", {})
    scaler = model_bundle["scaler"]
    ctx_raw = species_dict.get(category, [0.0, 0.0, 0.0, 0.0, 0.0])
    X_ctx = scaler.transform([ctx_raw])

    # 5. Combined Model Scoring
    X_input = np.hstack([X_sym, X_ctx])
    iso = model_bundle["iso"]
    lof = model_bundle["lof"]

    iso_pred = int(iso.predict(X_input)[0])
    iso_score = float(iso.decision_function(X_input)[0])
    lof_pred = int(lof.predict(X_input)[0])
    lof_score = float(lof.decision_function(X_input)[0])

    flag = "Unusual" if (iso_pred == -1 or lof_pred == -1) else "Normal"
    score = iso_score

    return {
        "flag": flag,
        "score": round(score, 4),
        "iso_score": round(iso_score, 4),
        "lof_score": round(lof_score, 4),
        "iso_pred": iso_pred,
        "lof_pred": lof_pred,
        "recognized_symptoms": recognized,
        "unrecognized_symptoms": unrecognized,
        "warning": warning_msg,
    }


def record_prediction_in_storage(
    animal_id: Optional[int],
    species: str,
    symptoms: List[str],
    flag: str,
    score: float,
    iso_score: float,
    lof_score: float,
    recognized_symptoms: List[str],
    unrecognized_symptoms: List[str],
) -> str:
    """
    Attempts to store in MySQL ml_predictions table.
    Falls back to SQLite if MySQL is unreachable.
    """
    symptoms_json = json.dumps(symptoms)

    # 1. Try MySQL if animal_id is provided
    if animal_id is not None:
        try:
            conn = get_mysql_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ml_predictions (animal_id, species, symptoms, flag, score)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (animal_id, species, symptoms_json, flag, score),
                )
                conn.commit()
            conn.close()
            return "mysql"
        except Exception as e:
            print(f"MySQL unavailable, falling back to SQLite: {e}")

    # 2. Fallback to SQLite
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Log to general prediction_logs
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                timestamp, species, input_payload, output_payload, flag, score,
                iso_score, lof_score, recognized_symptoms, unrecognized_symptoms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                species,
                json.dumps({"animal_id": animal_id, "symptoms": symptoms}),
                json.dumps({"flag": flag, "score": score}),
                flag,
                score,
                iso_score,
                lof_score,
                json.dumps(recognized_symptoms),
                json.dumps(unrecognized_symptoms),
            ),
        )

        # Log to animal fallback table if animal_id provided
        if animal_id is not None:
            cursor.execute(
                """
                INSERT INTO ml_predictions (animal_id, timestamp, species, symptoms, flag, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (animal_id, now_iso, species, symptoms_json, flag, score),
            )

        conn.commit()
        conn.close()
        return "sqlite_fallback"
    except Exception as e:
        print(f"Error logging to SQLite fallback: {e}")
        return "none"


def get_photo_model_health() -> Dict[str, Any]:
    """
    Returns photo model training status, gallery counts per label,
    test accuracy & recall on injured from the last training run,
    and the training timestamp.
    """
    sync_training_photos_folder()
    gallery_counts = {"healthy": 0, "injured": 0, "total": 0}
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT label, COUNT(*) FROM photo_gallery GROUP BY label")
        for lbl, count in cur.fetchall():
            if lbl == 0:
                gallery_counts["healthy"] = count
            elif lbl == 1:
                gallery_counts["injured"] = count
        gallery_counts["total"] = gallery_counts["healthy"] + gallery_counts["injured"]
        conn.close()
    except Exception as e:
        print(f"Notice: SQLite gallery count query error: {e}")

    is_trained = os.path.exists(DEFAULT_MODEL_FILE)
    test_accuracy = None
    recall_on_injured = None
    trained_at = None

    if is_trained:
        try:
            model_info = load_trained_model(DEFAULT_MODEL_FILE)
            meta = model_info.get("metadata", {})
            test_accuracy = meta.get("test_accuracy")
            recall_on_injured = meta.get("test_recall")
            trained_at = meta.get("trained_at")
        except Exception as e:
            print(f"Notice: error reading photo model npz file: {e}")

    return {
        "trained": is_trained,
        "gallery_counts": gallery_counts,
        "test_accuracy": test_accuracy,
        "recall_on_injured": recall_on_injured,
        "trained_at": trained_at,
    }


# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Returns service health, database status, and photo model state.
    """
    is_model_loaded = bool(model_bundle and "iso" in model_bundle)
    mysql_cfg = get_mysql_config()

    # Check MySQL
    mysql_connected = False
    mysql_error = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        mysql_connected = True
    except Exception as e:
        mysql_connected = False
        mysql_error = str(e)

    # Check SQLite
    sqlite_connected = False
    sqlite_log_count = 0
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM prediction_logs")
        sqlite_log_count = cur.fetchone()[0]
        conn.close()
        sqlite_connected = True
    except Exception as e:
        sqlite_connected = False

    overall_status = (
        "healthy"
        if (is_model_loaded and mysql_connected)
        else "degraded"
        if (is_model_loaded and sqlite_connected)
        else "unhealthy"
    )

    return {
        "status": overall_status,
        "model_loaded": is_model_loaded,
        "model_version": model_bundle.get("version", "unknown"),
        "vocabulary_size": model_bundle.get("symptom_vocabulary_size", 0),
        "database": {
            "primary_mysql": {
                "connected": mysql_connected,
                "host": mysql_cfg["host"],
                "port": mysql_cfg["port"],
                "database": mysql_cfg["database"],
                "user": mysql_cfg["user"],
                "error": mysql_error,
            },
            "sqlite_fallback": {
                "connected": sqlite_connected,
                "database_path": SQLITE_DB_PATH,
                "total_logged": sqlite_log_count,
            },
        },
        "photo_model": get_photo_model_health(),
        "tiger_reserves_ntca": {
            "loaded": len(load_tiger_reserves()) > 0,
            "total_reserves": len(load_tiger_reserves()),
            "source": "NTCA & WII 2022-2023 Official Census",
            "national_tiger_estimate_2022": 3682,
            "minimum_tigers_counted_2022": 3167,
        },
    }


@app.get("/reserves")
def get_all_tiger_reserves(state: Optional[str] = None, landscape: Optional[str] = None):
    """
    Returns official list of 53 National Tiger Reserves from NTCA 2022-2023 data.
    Supports filtering by State or Landscape.
    """
    reserves = load_tiger_reserves()
    filtered = reserves
    if state:
        filtered = [r for r in filtered if state.lower() in (r.get("state") or "").lower()]
    if landscape:
        filtered = [r for r in filtered if landscape.lower() in (r.get("landscape") or "").lower()]
    return {
        "total": len(filtered),
        "source": "National Tiger Conservation Authority (NTCA) & Wildlife Institute of India (WII)",
        "national_tiger_estimate_2022": 3682,
        "minimum_tigers_counted_2022": 3167,
        "reserves": filtered,
    }


@app.get("/reserves/{reserve_id}")
def get_tiger_reserve_by_id(reserve_id: int):
    """
    Returns detailed profile and NTCA threat metrics for a specific tiger reserve.
    """
    reserves = load_tiger_reserves()
    for r in reserves:
        if r.get("id") == reserve_id:
            return r
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tiger Reserve with ID {reserve_id} not found.",
    )


@app.get("/api/pathologies")
def get_pathologies_library():
    """
    Returns the comprehensive catalog of tiger health conditions, traumatic injuries,
    and emerging pathologies with clinical indicative signs and veterinary protocols.
    """
    import tiger_diagnostics
    return [
        {
            "id": c["id"],
            "num": c.get("num", 0),
            "name": c["name"],
            "category": c["category"],
            "desc": c.get("whats_wrong", ""),
            "signs": c.get("signs", ", ".join(c.get("keywords", [])[:8])),
            "urgency": c.get("urgency", "Moderate Concern"),
            "anatomical_areas": c.get("anatomical_areas", []),
            "next_steps": c.get("next_steps", {})
        }
        for c in tiger_diagnostics.TIGER_CONDITIONS_DB
    ]


@app.post("/predict/animal/{animal_id}", response_model=PredictionResult)
def predict_animal(animal_id: int, payload: AnimalPredictRequest):
    """
    Look up an animal's species from the animals table in MySQL,
    run unsupervised anomaly detection, record in ml_predictions, and return result.
    """
    # 1. Look up animal from MySQL animals table
    animal_record = None
    mysql_available = True
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, species, age, health_status, location FROM animals WHERE id = %s",
                (animal_id,),
            )
            animal_record = cur.fetchone()
        conn.close()
    except Exception as e:
        mysql_available = False
        print(f"Warning: MySQL query failed for animal {animal_id}: {e}")

    # If MySQL is available but animal not found, return 404
    if mysql_available and animal_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Animal with ID {animal_id} not found in animals table.",
        )

    # Determine species and name (fallback to unknown if MySQL is offline)
    species = animal_record.get("species", "Unknown") if animal_record else "Unknown"
    name = animal_record.get("name") if animal_record else None

    # 2. Run ML inference
    inf = run_inference(payload.symptoms, species)

    # 3. Store result in ml_predictions (with SQLite fallback)
    storage_type = record_prediction_in_storage(
        animal_id=animal_id,
        species=species,
        symptoms=payload.symptoms,
        flag=inf["flag"],
        score=inf["score"],
        iso_score=inf["iso_score"],
        lof_score=inf["lof_score"],
        recognized_symptoms=inf["recognized_symptoms"],
        unrecognized_symptoms=inf["unrecognized_symptoms"],
    )

    return PredictionResult(
        animal_id=animal_id,
        species=species,
        animal_name=name,
        flag=inf["flag"],
        score=inf["score"],
        iso_score=inf["iso_score"],
        lof_score=inf["lof_score"],
        iso_pred=inf["iso_pred"],
        lof_pred=inf["lof_pred"],
        recognized_symptoms=inf["recognized_symptoms"],
        unrecognized_symptoms=inf["unrecognized_symptoms"],
        warning=inf["warning"],
        stored_in=storage_type,
    )


@app.get("/animal/{animal_id}/predictions", response_model=AnimalHistoryResponse)
def get_animal_predictions(animal_id: int):
    """
    Return all past prediction history for a specific animal.
    Queries MySQL ml_predictions with SQLite fallback.
    """
    predictions = []
    source = "mysql"
    species = None
    name = None

    # Try MySQL first
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            # Check if animal exists
            cur.execute("SELECT id, name, species FROM animals WHERE id = %s", (animal_id,))
            animal = cur.fetchone()
            if animal is None:
                conn.close()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Animal with ID {animal_id} not found.",
                )
            name = animal.get("name")
            species = animal.get("species")

            cur.execute(
                """
                SELECT id, animal_id, timestamp, species, symptoms, flag, score
                FROM ml_predictions
                WHERE animal_id = %s
                ORDER BY timestamp DESC, id DESC
                """,
                (animal_id,),
            )
            rows = cur.fetchall()

        conn.close()

        for r in rows:
            syms = r["symptoms"]
            if isinstance(syms, str):
                try:
                    syms = json.loads(syms)
                except Exception:
                    pass

            ts_str = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])

            predictions.append(
                AnimalHistoryItem(
                    id=r["id"],
                    animal_id=r["animal_id"],
                    timestamp=ts_str,
                    species=r["species"],
                    symptoms=syms,
                    flag=r["flag"],
                    score=float(r["score"]),
                    source="mysql",
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        # Fallback to SQLite
        source = "sqlite_fallback"
        print(f"Falling back to SQLite for animal {animal_id} history: {e}")
        try:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, animal_id, timestamp, species, symptoms, flag, score
                FROM ml_predictions
                WHERE animal_id = ?
                ORDER BY timestamp DESC, id DESC
                """,
                (animal_id,),
            )
            rows = cur.fetchall()
            conn.close()

            for r in rows:
                syms = r[4]
                try:
                    syms = json.loads(syms)
                except Exception:
                    pass

                predictions.append(
                    AnimalHistoryItem(
                        id=r[0],
                        animal_id=r[1],
                        timestamp=str(r[2]),
                        species=r[3],
                        symptoms=syms,
                        flag=r[5],
                        score=float(r[6]),
                        source="sqlite_fallback",
                    )
                )
        except Exception as sq_err:
            print(f"SQLite history lookup error: {sq_err}")

    return AnimalHistoryResponse(
        animal_id=animal_id,
        species=species,
        animal_name=name,
        total_predictions=len(predictions),
        source=source,
        predictions=predictions,
    )


@app.post("/predict")
def predict_generic(payload: GenericPredictRequest):
    """
    Original generic prediction endpoint (optional species, logged to SQLite).
    """
    inf = run_inference(payload.symptoms, payload.species)
    storage_type = record_prediction_in_storage(
        animal_id=None,
        species=payload.species or "Unknown",
        symptoms=payload.symptoms,
        flag=inf["flag"],
        score=inf["score"],
        iso_score=inf["iso_score"],
        lof_score=inf["lof_score"],
        recognized_symptoms=inf["recognized_symptoms"],
        unrecognized_symptoms=inf["unrecognized_symptoms"],
    )

    return {
        "flag": inf["flag"],
        "score": inf["score"],
        "iso_score": inf["iso_score"],
        "lof_score": inf["lof_score"],
        "iso_pred": inf["iso_pred"],
        "lof_pred": inf["lof_pred"],
        "species": payload.species or "Unknown",
        "recognized_symptoms": inf["recognized_symptoms"],
        "unrecognized_symptoms": inf["unrecognized_symptoms"],
        "warning": inf["warning"],
        "stored_in": storage_type,
    }


# =============================================================================
# Photo Recognition & Gallery Module (NumPy NN From Scratch)
# =============================================================================

MAX_PHOTO_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_photo_upload(file: UploadFile, contents: bytes):
    """
    Validates uploaded photo for allowed extensions (.jpg, .jpeg, .png),
    size under 5 MB, and readable image data via Pillow.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_PHOTO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Only JPG and PNG files are allowed.",
        )

    if len(contents) > MAX_PHOTO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of 5 MB ({len(contents)} bytes).",
        )

    try:
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is not a valid image: {e}",
        )


class PhotoTrainRequest(BaseModel):
    n_h: int = Field(16, description="Hidden units in tanh layer")
    learning_rate: float = Field(0.05, description="Learning rate for gradient descent")
    num_iterations: int = Field(1500, description="Training iterations")
    seed: int = Field(42, description="Random seed")


@app.post("/photos/reference")
async def add_reference_photo(
    file: UploadFile = File(...),
    label: int = Form(..., description="0 for healthy, 1 for injured"),
    species: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    """
    Upload a reference photo to the training gallery.
    Validates JPG/PNG and < 5 MB.
    """
    if label not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Label must be 0 (healthy) or 1 (injured).",
        )

    contents = await file.read()
    validate_photo_upload(file, contents)

    os.makedirs(PHOTOS_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    unique_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(PHOTOS_DIR, unique_filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO photo_gallery (filename, label, species, notes, uploaded_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (unique_filename, label, species, notes, uploaded_at),
    )
    photo_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": photo_id,
        "filename": unique_filename,
        "label": label,
        "label_name": "injured" if label == 1 else "healthy",
        "species": species,
        "notes": notes,
        "uploaded_at": uploaded_at,
        "size_bytes": len(contents),
    }


TRAINING_PHOTOS_DIR = os.path.join("data", "training_photos")


def sync_training_photos_folder() -> int:
    """
    Scans data/training_photos/healthy and data/training_photos/injured.
    Imports any newly placed photos into the gallery database.
    """
    if not os.path.exists(TRAINING_PHOTOS_DIR):
        return 0

    os.makedirs(PHOTOS_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filename FROM photo_gallery")
    existing = set(r[0] for r in cur.fetchall())

    imported = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    subdirs = [("healthy", 0), ("injured", 1)]

    for subfolder, label in subdirs:
        folder_path = os.path.join(TRAINING_PHOTOS_DIR, subfolder)
        if not os.path.exists(folder_path):
            continue

        # Count current members for this label
        cur.execute("SELECT COUNT(*) FROM photo_gallery WHERE label = ?", (label,))
        current_label_count = cur.fetchone()[0]

        for fname in sorted(os.listdir(folder_path)):
            # Cap healthy images at 150 to maintain balanced training ratio and prevent duplicating 2GB
            if label == 0 and current_label_count >= 150:
                break

            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_PHOTO_EXTENSIONS:
                continue

            src = os.path.join(folder_path, fname)
            if not os.path.isfile(src):
                continue

            target_name = f"tiger_{subfolder}_{fname}"
            if target_name in existing:
                continue

            try:
                with Image.open(src) as img:
                    img.verify()
            except Exception:
                continue

            dst = os.path.join(PHOTOS_DIR, target_name)
            shutil.copy2(src, dst)

            cur.execute(
                """
                INSERT INTO photo_gallery (filename, label, species, notes, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_name, label, "Panthera tigris", f"Training folder ({subfolder})", now_iso),
            )
            existing.add(target_name)
            imported += 1
            current_label_count += 1

    conn.commit()
    conn.close()
    return imported


@app.post("/photos/sync-folder")
def sync_photos_from_folder():
    """
    Scans data/training_photos for newly added photos and imports them.
    """
    count = sync_training_photos_folder()
    return {
        "status": "success",
        "imported": count,
        "folder": TRAINING_PHOTOS_DIR,
    }


@app.get("/photos/gallery")
def get_photo_gallery():
    """
    Retrieve all reference photos in the gallery with summary counts.
    Automatically checks and imports any photos from data/training_photos.
    """
    sync_training_photos_folder()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, filename, label, species, notes, uploaded_at
        FROM photo_gallery
        ORDER BY id DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    photos = []
    healthy_count = 0
    injured_count = 0

    for r in rows:
        photo_id, filename, label, species, notes, uploaded_at = r
        if label == 1:
            injured_count += 1
        else:
            healthy_count += 1

        photos.append({
            "id": photo_id,
            "filename": filename,
            "label": label,
            "label_name": "injured" if label == 1 else "healthy",
            "species": species,
            "notes": notes,
            "uploaded_at": uploaded_at,
            "file_exists": os.path.exists(os.path.join(PHOTOS_DIR, filename)),
        })

    return {
        "total": len(photos),
        "healthy_count": healthy_count,
        "injured_count": injured_count,
        "minimum_required_per_label": MIN_SAMPLES_PER_CLASS,
        "ready_to_train": (healthy_count >= MIN_SAMPLES_PER_CLASS and injured_count >= MIN_SAMPLES_PER_CLASS),
        "photos": photos,
    }


@app.delete("/photos/{photo_id}")
def delete_reference_photo(photo_id: int):
    """
    Delete a reference photo from gallery to fix mislabeled uploads.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filename FROM photo_gallery WHERE id = ?", (photo_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photo with ID {photo_id} not found in gallery.",
        )

    filename = row[0]
    cur.execute("DELETE FROM photo_gallery WHERE id = ?", (photo_id,))
    conn.commit()
    conn.close()

    filepath = os.path.join(PHOTOS_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: could not delete physical file {filepath}: {e}")

    return {
        "status": "success",
        "message": f"Photo {photo_id} ({filename}) deleted successfully.",
    }


@app.post("/photos/train")
def train_photo_model(params: Optional[PhotoTrainRequest] = None):
    """
    Train the from-scratch NumPy neural network on gallery photos.
    Refuses to train if fewer than 20 photos per label.
    """
    n_h = params.n_h if params else 16
    lr = params.learning_rate if params else 0.05
    iters = params.num_iterations if params else 1500
    seed = params.seed if params else 42

    try:
        results = train_model(
            n_h=n_h,
            learning_rate=lr,
            num_iterations=iters,
            seed=seed,
            db_path=SQLITE_DB_PATH,
            photos_dir=PHOTOS_DIR,
            model_save_path=DEFAULT_MODEL_FILE,
            plot_save_path=DEFAULT_PLOT_PATH,
        )
        return results
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {e}",
        )


@app.post("/photos/check")
async def check_photo(file: UploadFile = File(...)):
    """
    Forward propagation on uploaded image with saved NumPy NN weights.
    Returns probability of injured, label (injured, healthy, or uncertain),
    and model test accuracy. Returns 409 Conflict if model not yet trained.
    """
    if not os.path.exists(DEFAULT_MODEL_FILE):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No photo recognition model has been trained yet. Please train the model via POST /photos/train first.",
        )

    contents = await file.read()
    validate_photo_upload(file, contents)

    try:
        pred = predict_photo(contents, model_path=DEFAULT_MODEL_FILE)
        visual_signs = pred.get("visual_findings", [])
        photo_zone = pred.get("anatomical_zone")
        diagnosis = diagnose_tiger_health(
            symptoms_list=visual_signs,
            photo_label=pred["label"],
            photo_prob=pred["probability_injured"],
            photo_zone=photo_zone,
        )
        return {
            "filename": file.filename,
            "label": pred["label"],
            "probability_injured": pred["probability_injured"],
            "anatomical_zone": photo_zone,
            "visual_findings": visual_signs,
            "model_test_accuracy": pred["model_test_accuracy"],
            "model_test_recall": pred["model_test_recall"],
            "threshold_policy": pred["threshold_policy"],
            "status": diagnosis["status"],
            "urgency_level": diagnosis["urgency_level"],
            "top_5_diagnoses": diagnosis.get("top_5_diagnoses", diagnosis["identified_conditions"]),
            "identified_conditions": diagnosis["identified_conditions"],
            "consolidated_next_steps": diagnosis["consolidated_next_steps"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        )


@app.post("/assess")
async def assess_animal_multimodal(
    symptoms: Optional[str] = Form(None, description="JSON array or comma-separated symptoms"),
    species: Optional[str] = Form("Panthera tigris"),
    photo: Optional[UploadFile] = File(None),
    reserve_id: Optional[int] = Form(None, description="Optional NTCA Tiger Reserve ID"),
    reserve_name: Optional[str] = Form(None, description="Optional NTCA Tiger Reserve Name"),
):
    """
    Multi-modal health assessment: runs symptom anomaly detection, photo injury recognition,
    clinical differential diagnosis, and next-step recommendations.
    Accepts photo only, symptoms only, or both side-by-side without merging raw scores.
    Optionally enriches analysis with NTCA Tiger Reserve landscape intelligence.
    """
    # 1. Parse symptoms
    symptoms_list = []
    if symptoms and symptoms.strip():
        try:
            parsed = json.loads(symptoms)
            if isinstance(parsed, list):
                symptoms_list = [str(s).strip() for s in parsed if str(s).strip()]
            else:
                symptoms_list = [str(parsed).strip()]
        except Exception:
            symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    # 2. Check input presence
    has_photo = bool(photo and photo.filename)
    if not symptoms_list and not has_photo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one form of input (photograph or clinical symptoms) must be provided.",
        )

    # 3. Symptom model inference (if symptoms provided)
    symptom_assessment = None
    symptom_flag = None
    symptom_score = None
    if symptoms_list:
        symptom_inf = run_inference(symptoms_list, species)
        symptom_flag = symptom_inf["flag"]
        symptom_score = symptom_inf["score"]
        symptom_assessment = {
            "flag": symptom_flag,
            "score": symptom_score,
            "iso_score": symptom_inf["iso_score"],
            "lof_score": symptom_inf["lof_score"],
            "species": species or "Panthera tigris",
            "recognized_symptoms": symptom_inf["recognized_symptoms"],
            "unrecognized_symptoms": symptom_inf["unrecognized_symptoms"],
            "warning": symptom_inf["warning"],
        }

    # 4. Photo model inference (if photo provided)
    photo_assessment = None
    photo_label = None
    photo_prob = None

    if has_photo:
        photo_bytes = await photo.read()
        if len(photo_bytes) > 0:
            validate_photo_upload(photo, photo_bytes)
            if not os.path.exists(DEFAULT_MODEL_FILE):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Photo recognition model is not loaded or trained yet. Please train the model via the Training Photos tab first.",
                )
            else:
                pred = predict_photo(photo_bytes, model_path=DEFAULT_MODEL_FILE)
                photo_label = pred["label"]
                photo_prob = pred["probability_injured"]
                photo_zone = pred.get("anatomical_zone")
                photo_assessment = {
                    "filename": photo.filename,
                    "label": photo_label,
                    "probability_injured": photo_prob,
                    "anatomical_zone": photo_zone,
                    "visual_findings": pred.get("visual_findings", []),
                    "model_test_accuracy": pred["model_test_accuracy"],
                }

    # 5. Side-by-side one-line summary (do NOT merge into one score)
    if symptom_assessment and photo_assessment and "label" in photo_assessment:
        summary = (
            f"Symptoms flagged as {symptom_flag} (score: {symptom_score:.2f}); "
            f"photo classified as {photo_label} (injury probability: {photo_prob:.2f})."
        )
    elif symptom_assessment and photo_assessment and photo_assessment.get("status") == "model_not_trained":
        summary = (
            f"Symptoms flagged as {symptom_flag} (score: {symptom_score:.2f}); "
            f"photo submitted but recognition model is not yet trained."
        )
    elif symptom_assessment:
        summary = (
            f"Symptoms flagged as {symptom_flag} (score: {symptom_score:.2f}); "
            f"no photo provided."
        )
    elif photo_assessment and "label" in photo_assessment:
        summary = (
            f"Photo classified as {photo_label} (injury probability: {photo_prob:.2f}); "
            f"no symptoms provided."
        )
    else:
        summary = "Assessment completed."

    # Resolve optional NTCA Tiger Reserve context
    selected_reserve = None
    if reserve_id or reserve_name:
        all_res = load_tiger_reserves()
        for r in all_res:
            if reserve_id and r.get("id") == reserve_id:
                selected_reserve = r
                break
            if reserve_name and reserve_name.lower() in (r.get("name") or "").lower():
                selected_reserve = r
                break

    # 6. Merge user symptoms with photographic visual evidence
    effective_symptoms = list(symptoms_list or [])
    if photo_assessment and "visual_findings" in photo_assessment:
        effective_symptoms = effective_symptoms + photo_assessment["visual_findings"]

    photo_zone = photo_assessment.get("anatomical_zone") if photo_assessment else None

    # Clinical differential diagnosis and actionable next steps
    diagnosis = diagnose_tiger_health(
        symptoms_list=effective_symptoms,
        photo_label=photo_label,
        photo_prob=photo_prob,
        photo_zone=photo_zone,
        reserve=selected_reserve,
    )

    return {
        "summary": summary,
        "symptom_assessment": symptom_assessment,
        "photo_assessment": photo_assessment,
        "status": diagnosis["status"],
        "urgency_level": diagnosis["urgency_level"],
        "top_5_diagnoses": diagnosis.get("top_5_diagnoses", diagnosis["identified_conditions"]),
        "identified_conditions": diagnosis["identified_conditions"],
        "consolidated_next_steps": diagnosis["consolidated_next_steps"],
        "reserve_context": diagnosis.get("reserve_context"),
    }

