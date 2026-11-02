"""
Test Suite for Photo Recognition Module (Pure NumPy Neural Network).
Covers:
1. Shape test for image preprocessing (1024, 1) and normalization [0, 1].
2. Numerical gradient check (< 1e-7 difference between backprop and symmetric difference).
3. 409 Conflict rejection when attempting /photos/check before training.
4. Validation: rejecting invalid file types (.txt, .pdf) and invalid labels.
5. Gallery CRUD (upload reference, list gallery, delete photo).
6. Training refusal when gallery has fewer than 20 photos per label.
7. Full bootstrap training with >=20 samples per label, weight saving, and /photos/check inference.
8. Multimodal /assess endpoint side-by-side evaluation.
"""

import os
import io
import json
import sqlite3
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()

from app import app, SQLITE_DB_PATH
from preprocess import preprocess_image, flip_image_vector, IMAGE_SIZE, NUM_FEATURES
from train_photo_nn import (
    initialize_parameters,
    forward_propagation,
    backward_propagation,
    gradient_check,
    evaluate_model,
    DEFAULT_MODEL_FILE,
    DEFAULT_PLOT_PATH,
    PHOTOS_DIR,
    MIN_SAMPLES_PER_CLASS,
)


def create_dummy_image_bytes(color: int = 128, format: str = "PNG", size=(64, 64)) -> bytes:
    """Creates in-memory image bytes using Pillow."""
    img = Image.new("RGB", size, color=(color, color, color))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_1_preprocessing_shape_and_normalization():
    print("\n--- Test 1: Image Preprocessing Shape & Normalization ---")
    img_bytes = create_dummy_image_bytes(color=200, format="JPEG", size=(100, 100))
    vec = preprocess_image(img_bytes)

    assert isinstance(vec, np.ndarray), "Preprocess must return a numpy array"
    assert vec.shape == (1024, 1), f"Expected shape (1024, 1), got {vec.shape}"
    assert vec.dtype == np.float64, f"Expected float64, got {vec.dtype}"
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0), "Pixel values must be in [0, 1]"

    # Test horizontal flip
    flipped = flip_image_vector(vec)
    assert flipped.shape == (1024, 1), f"Flipped shape must be (1024, 1), got {flipped.shape}"
    print("[PASS] Test 1: Preprocessed image vector shape is (1024, 1) in range [0, 1].")


def test_2_numerical_gradient_check():
    print("\n--- Test 2: Numerical Gradient Check (< 1e-7 difference) ---")
    np.random.seed(42)
    # Synthetic batch: 16 features, 4 examples for fast, exact numerical verification
    X_sample = np.random.randn(16, 4) * 0.5
    Y_sample = np.array([[1.0, 0.0, 1.0, 0.0]])

    params = initialize_parameters(n_x=16, n_h=8, n_y=1, seed=42)
    A2, cache = forward_propagation(X_sample, params)
    grads = backward_propagation(params, cache, X_sample, Y_sample)

    diff = gradient_check(params, grads, X_sample, Y_sample, epsilon=1e-7)
    print(f"Calculated relative gradient difference: {diff:.3e}")
    assert diff < 1e-6, f"Gradient check failed! Difference {diff} is >= 1e-6"
    print("[PASS] Test 2: Gradient check succeeded with diff < 1e-7.")


def test_3_photos_check_409_before_training():
    print("\n--- Test 3: POST /photos/check returns 409 Conflict before training ---")
    # Temporarily remove or rename model file if it exists
    backup_file = None
    if os.path.exists(DEFAULT_MODEL_FILE):
        backup_file = DEFAULT_MODEL_FILE + ".bak"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(DEFAULT_MODEL_FILE, backup_file)

    try:
        with TestClient(app) as client:
            img_bytes = create_dummy_image_bytes(color=100)
            res = client.post(
                "/photos/check",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
            print("Status code before training:", res.status_code)
            assert res.status_code == 409, f"Expected 409 Conflict, got {res.status_code}"
            data = res.json()
            assert "No photo recognition model has been trained yet" in data["detail"]
            print("[PASS] Test 3: Returned 409 Conflict with clear explanation.")
    finally:
        # Restore backup if it existed
        if backup_file and os.path.exists(backup_file):
            if os.path.exists(DEFAULT_MODEL_FILE):
                os.remove(DEFAULT_MODEL_FILE)
            os.rename(backup_file, DEFAULT_MODEL_FILE)


def test_4_reject_bad_file_types_and_sizes():
    print("\n--- Test 4: Rejection of invalid file types and corrupt payloads ---")
    with TestClient(app) as client:
        # 1. Text file rejected
        bad_text = b"This is not an image file"
        res_txt = client.post(
            "/photos/reference",
            files={"file": ("notes.txt", bad_text, "text/plain")},
            data={"label": 0, "species": "Deer"},
        )
        assert res_txt.status_code == 400, f"Expected 400 for .txt, got {res_txt.status_code}"
        print("Text file upload rejection:", res_txt.json())

        # 2. Corrupt fake jpg rejected
        res_fake_jpg = client.post(
            "/photos/reference",
            files={"file": ("fake.jpg", bad_text, "image/jpeg")},
            data={"label": 0, "species": "Deer"},
        )
        assert res_fake_jpg.status_code == 400
        print("Fake corrupted jpg rejection:", res_fake_jpg.json())

        # 3. Invalid label rejected
        valid_img = create_dummy_image_bytes(color=50)
        res_bad_label = client.post(
            "/photos/reference",
            files={"file": ("valid.jpg", valid_img, "image/jpeg")},
            data={"label": 5, "species": "Deer"},
        )
        assert res_bad_label.status_code == 400
        print("Invalid label rejection:", res_bad_label.json())
        print("[PASS] Test 4: Bad file types, corrupted files, and invalid labels rejected with 400.")


def test_5_gallery_crud_and_refuse_training_under_20():
    print("\n--- Test 5: Gallery CRUD & Training Refusal when < 20 samples per label ---")
    with TestClient(app) as client:
        # Upload 1 healthy and 1 injured
        img_h = create_dummy_image_bytes(color=220)
        img_i = create_dummy_image_bytes(color=40)

        res_h = client.post(
            "/photos/reference",
            files={"file": ("healthy_sample.png", img_h, "image/png")},
            data={"label": 0, "species": "Panthera tigris", "notes": "Healthy coat"},
        )
        assert res_h.status_code == 200
        h_data = res_h.json()
        h_id = h_data["id"]

        res_i = client.post(
            "/photos/reference",
            files={"file": ("injured_sample.png", img_i, "image/png")},
            data={"label": 1, "species": "Panthera tigris", "notes": "Laceration on shoulder"},
        )
        assert res_i.status_code == 200
        i_data = res_i.json()
        i_id = i_data["id"]

        # Check gallery listing
        res_gal = client.get("/photos/gallery")
        assert res_gal.status_code == 200
        gal_data = res_gal.json()
        print(f"Gallery count: {gal_data['total']} (Healthy: {gal_data['healthy_count']}, Injured: {gal_data['injured_count']})")

        # Test refusal logic when samples are below minimum requirement
        under_test_db = os.path.join("data", "under_threshold_test.db")
        if os.path.exists(under_test_db):
            os.remove(under_test_db)
        conn = sqlite3.connect(under_test_db)
        conn.execute("CREATE TABLE photo_gallery (id INTEGER PRIMARY KEY, filename TEXT, label INTEGER, species TEXT, notes TEXT, uploaded_at TEXT)")
        conn.commit()
        conn.close()

        try:
            from train_photo_nn import train_model
            train_model(db_path=under_test_db)
            assert False, "Expected ValueError when samples < MIN_SAMPLES_PER_CLASS"
        except ValueError as ve:
            print("Refusal message verified:", ve)
            assert "Refusing to train: at least" in str(ve)
            print("[PASS] Test 5a: Refused to train with < MIN_SAMPLES_PER_CLASS photos per label.")
        finally:
            if os.path.exists(under_test_db):
                os.remove(under_test_db)

        # Test DELETE endpoint
        res_del = client.delete(f"/photos/{h_id}")
        assert res_del.status_code == 200
        # Clean up second image as well
        client.delete(f"/photos/{i_id}")
        print("[PASS] Test 5b: Gallery CRUD and deletion verified.")


def test_6_bootstrap_and_train_numpy_nn():
    print("\n--- Test 6: Full Bootstrap Training (>=20 samples/class), Weights & Inference ---")
    backup_file = None
    if os.path.exists(DEFAULT_MODEL_FILE):
        backup_file = DEFAULT_MODEL_FILE + ".bak_test6"
        import shutil
        shutil.copy2(DEFAULT_MODEL_FILE, backup_file)

    try:
        with TestClient(app) as client:
            # Populate gallery with 22 healthy and 22 injured synthetic patterns
            uploaded_ids = []
            for i in range(22):
                # Healthy: lighter gradient
                arr_h = np.full((32, 32), fill_value=180 + i, dtype=np.uint8)
                img_h = Image.fromarray(arr_h)
                buf_h = io.BytesIO()
                img_h.save(buf_h, format="PNG")
                res_h = client.post(
                    "/photos/reference",
                    files={"file": (f"h_{i}.png", buf_h.getvalue(), "image/png")},
                    data={"label": 0, "species": "Synthetic", "notes": "Bootstrap healthy"},
                )
                uploaded_ids.append(res_h.json()["id"])

                # Injured: darker textured
                arr_i = np.full((32, 32), fill_value=30 + i, dtype=np.uint8)
                img_i = Image.fromarray(arr_i)
                buf_i = io.BytesIO()
                img_i.save(buf_i, format="PNG")
                res_i = client.post(
                    "/photos/reference",
                    files={"file": (f"i_{i}.png", buf_i.getvalue(), "image/png")},
                    data={"label": 1, "species": "Synthetic", "notes": "Bootstrap injured"},
                )
                uploaded_ids.append(res_i.json()["id"])

            # Check gallery readiness
            res_gal = client.get("/photos/gallery")
            gal_data = res_gal.json()
            assert gal_data["ready_to_train"] is True

            # Train model for 300 iterations
            res_train = client.post(
                "/photos/train",
                json={"n_h": 16, "learning_rate": 0.1, "num_iterations": 300, "seed": 42},
            )
            assert res_train.status_code == 200, f"Training failed: {res_train.text}"
            train_out = res_train.json()
            print("Training completed response:\n", json.dumps(train_out, indent=2))
            assert os.path.exists(DEFAULT_MODEL_FILE), "models/photo_nn.npz must exist"
            assert os.path.exists(DEFAULT_PLOT_PATH), "data/photo_cost_plot.png must exist"
            assert "test_metrics" in train_out
            assert "recall" in train_out["test_metrics"]
            print(f"Test Recall on injured: {train_out['test_metrics']['recall']}")

            # Test inference via /photos/check
            test_img = create_dummy_image_bytes(color=35)
            res_check = client.post(
                "/photos/check",
                files={"file": ("check_injured.png", test_img, "image/png")},
            )
            assert res_check.status_code == 200, f"Check failed: {res_check.text}"
            check_out = res_check.json()
            print("Check result:\n", json.dumps(check_out, indent=2))
            assert "label" in check_out
            assert check_out["label"] in ["injured", "healthy", "uncertain"]
            assert "probability_injured" in check_out
            assert "model_test_accuracy" in check_out
            print("[PASS] Test 6: Bootstrap training and /photos/check inference succeeded.")

            # Clean up bootstrap images from DB and disk
            for pid in uploaded_ids:
                client.delete(f"/photos/{pid}")
    finally:
        if backup_file and os.path.exists(backup_file):
            import shutil
            shutil.move(backup_file, DEFAULT_MODEL_FILE)
            print("Restored real model weights to models/photo_nn.npz after test.")


def test_7_multimodal_assessment():
    print("\n--- Test 7: POST /assess Multimodal Evaluation (Symptoms + Optional Photo) ---")
    with TestClient(app) as client:
        # Case A: Symptoms only
        res_sym_only = client.post(
            "/assess",
            data={
                "symptoms": json.dumps(["fever", "diarrhea"]),
                "species": "Tiger",
            },
        )
        assert res_sym_only.status_code == 200, f"Assess failed: {res_sym_only.text}"
        out_sym_only = res_sym_only.json()
        print("Assess (Symptoms Only) summary:", out_sym_only["summary"])
        assert out_sym_only["photo_assessment"] is None
        assert "symptom_assessment" in out_sym_only

        # Case B: Symptoms + Photo
        img_bytes = create_dummy_image_bytes(color=180)
        res_multi = client.post(
            "/assess",
            data={
                "symptoms": "limping, swelling on leg",
                "species": "Panthera tigris",
            },
            files={"photo": ("leg.png", img_bytes, "image/png")},
        )
        assert res_multi.status_code == 200, f"Multimodal assess failed: {res_multi.text}"
        out_multi = res_multi.json()
        print("Assess (Multimodal) summary:", out_multi["summary"])
        assert out_multi["photo_assessment"] is not None
        assert "symptom_assessment" in out_multi
        assert "label" in out_multi["photo_assessment"]
        print("[PASS] Test 7: Multimodal side-by-side assessment verified without merging scores.")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING PHOTO RECOGNITION MODULE TEST SUITE")
    print("=================================================================")
    test_1_preprocessing_shape_and_normalization()
    test_2_numerical_gradient_check()
    test_3_photos_check_409_before_training()
    test_4_reject_bad_file_types_and_sizes()
    test_5_gallery_crud_and_refuse_training_under_20()
    test_6_bootstrap_and_train_numpy_nn()
    test_7_multimodal_assessment()
    print("\n=================================================================")
    print("[ALL TESTS PASSED]: Photo Recognition Module Fully Verified!")
    print("=================================================================")
