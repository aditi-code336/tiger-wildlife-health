"""
Comprehensive Test Suite for Wildlife Health Anomaly Detection Service.
Tests MySQL Integration, SQLite Fallback, and Animal Prediction Endpoints.
"""

import os
import json
import sqlite3
import pymysql
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()
from app import app, get_mysql_connection, SQLITE_DB_PATH

def run_tests():
    print("=== RUNNING FULL API & DATABASE TEST SUITE ===")
    
    with TestClient(app) as client:
        # -------------------------------------------------------------
        # Test 1: GET /health (Verify MySQL & SQLite status)
        # -------------------------------------------------------------
        print("\n--- Test 1: GET /health ---")
        res_h = client.get("/health")
        assert res_h.status_code == 200, f"Health failed: {res_h.text}"
        data_h = res_h.json()
        print("Health Status Response:\n", json.dumps(data_h, indent=2))
        assert data_h["model_loaded"] is True
        assert data_h["database"]["primary_mysql"]["connected"] is True
        assert data_h["database"]["sqlite_fallback"]["connected"] is True
        assert "photo_model" in data_h
        pm = data_h["photo_model"]
        assert "trained" in pm
        assert "gallery_counts" in pm
        assert "test_accuracy" in pm
        assert "recall_on_injured" in pm
        assert "trained_at" in pm
        print("[PASS] Test 1: Health check reports MySQL, SQLite, and photo_model status.")

        # -------------------------------------------------------------
        # Test 2: POST /predict/animal/2 (Tiger - Panthera tigris)
        # -------------------------------------------------------------
        print("\n--- Test 2: POST /predict/animal/2 (Tiger) ---")
        req_tiger = {
            "symptoms": ["limping", "swelling on leg", "pain", "reduced appetite"]
        }
        res_tiger = client.post("/predict/animal/2", json=req_tiger)
        assert res_tiger.status_code == 200, f"Predict animal failed: {res_tiger.text}"
        data_tiger = res_tiger.json()
        print("Tiger Prediction Result:\n", json.dumps(data_tiger, indent=2))
        assert data_tiger["animal_id"] == 2
        assert data_tiger["species"] == "Panthera tigris"
        assert data_tiger["animal_name"] == "Tiger"
        assert data_tiger["stored_in"] == "mysql"
        assert data_tiger["flag"] in ["Normal", "Unusual"]
        print("[PASS] Test 2: Tiger species resolved from MySQL animals table and stored in ml_predictions.")

        # -------------------------------------------------------------
        # Test 3: POST /predict/animal/1 (Elephant - Loxodonta africana)
        # -------------------------------------------------------------
        print("\n--- Test 3: POST /predict/animal/1 (Elephant) ---")
        req_elephant = {
            "symptoms": ["fever", "diarrhea", "weight loss"]
        }
        res_elephant = client.post("/predict/animal/1", json=req_elephant)
        assert res_elephant.status_code == 200
        data_elephant = res_elephant.json()
        print("Elephant Prediction Result:\n", json.dumps(data_elephant, indent=2))
        assert data_elephant["species"] == "Loxodonta africana"
        assert data_elephant["stored_in"] == "mysql"
        print("[PASS] Test 3: Elephant prediction succeeded.")

        # -------------------------------------------------------------
        # Test 4: POST /predict/animal/9999 (Non-existent animal -> 404)
        # -------------------------------------------------------------
        print("\n--- Test 4: POST /predict/animal/9999 (Non-existent animal) ---")
        res_404 = client.post("/predict/animal/9999", json={"symptoms": ["fever"]})
        assert res_404.status_code == 404, f"Expected 404, got {res_404.status_code}"
        print("Non-existent animal response (404 as expected):", res_404.json())
        print("[PASS] Test 4: Returned 404 for non-existent animal ID.")

        # -------------------------------------------------------------
        # Test 5: GET /animal/2/predictions (Retrieve history)
        # -------------------------------------------------------------
        print("\n--- Test 5: GET /animal/2/predictions ---")
        res_hist = client.get("/animal/2/predictions")
        assert res_hist.status_code == 200
        data_hist = res_hist.json()
        print("Animal 2 History Response:\n", json.dumps(data_hist, indent=2))
        assert data_hist["animal_id"] == 2
        assert data_hist["species"] == "Panthera tigris"
        assert data_hist["total_predictions"] >= 1
        assert len(data_hist["predictions"]) >= 1
        latest = data_hist["predictions"][0]
        assert latest["animal_id"] == 2
        print(f"[PASS] Test 5: Retrieved {data_hist['total_predictions']} history record(s) for animal 2.")

        # -------------------------------------------------------------
        # Test 6: Verify Direct MySQL Query on ml_predictions
        # -------------------------------------------------------------
        print("\n--- Test 6: Direct MySQL Verification on ml_predictions ---")
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, animal_id, timestamp, species, symptoms, flag, score FROM ml_predictions ORDER BY id DESC LIMIT 2")
            mysql_rows = cur.fetchall()
            print("Recent MySQL ml_predictions rows:\n", json.dumps(mysql_rows, indent=2, default=str))
            assert len(mysql_rows) >= 2
        conn.close()
        print("[PASS] Test 6: Confirmed rows saved directly to XAMPP MySQL ml_predictions.")

        # -------------------------------------------------------------
        # Test 7: SQLite Fallback Verification (Simulate MySQL failure)
        # -------------------------------------------------------------
        print("\n--- Test 7: SQLite Fallback when MySQL is unreachable ---")
        orig_port = os.environ.get("DB_PORT", "3306")
        try:
            # Point to unreachable port
            os.environ["DB_PORT"] = "59999"

            # Check health reports degraded / MySQL offline
            res_degraded = client.get("/health")
            data_deg = res_degraded.json()
            assert data_deg["database"]["primary_mysql"]["connected"] is False
            assert data_deg["database"]["sqlite_fallback"]["connected"] is True
            print("Degraded Health check (MySQL offline):\n", json.dumps(data_deg["database"], indent=2))

            # Make prediction when MySQL is down
            req_fallback = {"symptoms": ["sudden death", "staggering"]}
            res_fb = client.post("/predict/animal/2", json=req_fallback)
            assert res_fb.status_code == 200
            data_fb = res_fb.json()
            print("Fallback Prediction Result:\n", json.dumps(data_fb, indent=2))
            assert data_fb["stored_in"] == "sqlite_fallback"
            print("[PASS] Test 7: Successfully fell back to SQLite when MySQL was offline.")
        finally:
            os.environ["DB_PORT"] = orig_port

    print("\n=======================================================")
    print("[ALL TESTS PASSED]: MySQL Integration, Fallbacks & API Endpoints Verified!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
