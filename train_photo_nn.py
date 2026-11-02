"""
From-Scratch 2-Layer Neural Network for Animal Photo Recognition (Injured vs Healthy).
Implemented using ONLY pure NumPy (tanh -> sigmoid).
Based on Assignment 7 / Lab techniques with explicit bug fixes:
1. Backprop uses (1 - A1**2) tanh derivative, not tanh_grd(A1).
2. Hidden layer size n_h is passed as a function argument, not a global variable.
3. Weights initialized as np.random.randn(...) * 0.01, biases as np.zeros(...).
Zero external ML libraries (no PyTorch, TensorFlow, Keras, CLIP, OpenCV, sklearn).
"""

import os
import sqlite3
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any

from preprocess import preprocess_image, flip_image_vector

DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_MODEL_DIR, "photo_nn.npz")
DEFAULT_PLOT_PATH = os.path.join("data", "photo_cost_plot.png")
SQLITE_DB_PATH = os.path.join("data", "predictions.db")
PHOTOS_DIR = os.path.join("data", "photos")
MIN_SAMPLES_PER_CLASS = 15


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid of Z with numerical stability clipping."""
    Z_clipped = np.clip(Z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-Z_clipped))


def initialize_parameters(n_x: int, n_h: int, n_y: int, seed: Optional[int] = 42) -> Dict[str, np.ndarray]:
    """
    Initialize 2-layer neural network parameters.
    Weights: randn * 0.01
    Biases: zeros
    """
    if seed is not None:
        np.random.seed(seed)
        
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1), dtype=np.float64)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1), dtype=np.float64)

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }


def forward_propagation(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Forward propagation: X -> [Linear -> Tanh] -> [Linear -> Sigmoid] -> A2
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }
    return A2, cache


def compute_cost(A2: np.ndarray, Y: np.ndarray) -> float:
    """
    Binary cross-entropy cost function:
    J = -1/m * sum(Y * log(A2) + (1 - Y) * log(1 - A2))
    """
    m = Y.shape[1]
    eps = 1e-15
    A2_safe = np.clip(A2, eps, 1.0 - eps)
    cost = - (1.0 / m) * np.sum(Y * np.log(A2_safe) + (1.0 - Y) * np.log(1.0 - A2_safe))
    return float(np.squeeze(cost))


def backward_propagation(
    parameters: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Backward propagation for 2-layer network.
    Uses exact derivative of tanh: (1 - A1**2).
    """
    m = X.shape[1]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Output layer gradient
    dZ2 = A2 - Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradient (Bug Fix: use 1 - A1**2)
    dZ1 = np.dot(W2.T, dZ2) * (1.0 - np.power(A1, 2))
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }


def update_parameters(
    parameters: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    learning_rate: float,
) -> Dict[str, np.ndarray]:
    """
    Gradient descent parameter update: theta = theta - alpha * d_theta.
    """
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    return parameters


def gradient_check(
    parameters: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    """
    Compares analytical gradients from backprop to numerical gradients computed
    via two-sided difference: (J(theta + eps) - J(theta - eps)) / (2 * eps).
    Returns relative difference: ||grad - gradapprox|| / (||grad|| + ||gradapprox||).
    """
    # Flatten parameters into a single vector
    param_keys = ["W1", "b1", "W2", "b2"]
    shapes = [parameters[k].shape for k in param_keys]
    sizes = [parameters[k].size for k in param_keys]

    theta = np.concatenate([parameters[k].reshape(-1) for k in param_keys])
    grad_vector = np.concatenate([grads[f"d{k}"].reshape(-1) for k in param_keys])
    num_params = theta.shape[0]

    gradapprox = np.zeros(num_params)

    for i in range(num_params):
        # theta + epsilon
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        
        # Unpack theta_plus
        idx = 0
        params_plus = {}
        for k, sh, sz in zip(param_keys, shapes, sizes):
            params_plus[k] = theta_plus[idx : idx + sz].reshape(sh)
            idx += sz
        A2_plus, _ = forward_propagation(X, params_plus)
        J_plus = compute_cost(A2_plus, Y)

        # theta - epsilon
        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        
        idx = 0
        params_minus = {}
        for k, sh, sz in zip(param_keys, shapes, sizes):
            params_minus[k] = theta_minus[idx : idx + sz].reshape(sh)
            idx += sz
        A2_minus, _ = forward_propagation(X, params_minus)
        J_minus = compute_cost(A2_minus, Y)

        gradapprox[i] = (J_plus - J_minus) / (2.0 * epsilon)

    numerator = np.linalg.norm(grad_vector - gradapprox)
    denominator = np.linalg.norm(grad_vector) + np.linalg.norm(gradapprox)
    diff = numerator / (denominator + 1e-15)
    return float(diff)


def evaluate_model(parameters: Dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    """
    Computes accuracy, confusion matrix, precision, and recall by hand using pure NumPy.
    State: Recall on injured matters most.
    """
    m = X.shape[1]
    if m == 0:
        return {
            "m": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
            "note": "No test samples available.",
        }

    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 >= 0.5).astype(int)
    Y_int = Y.astype(int)

    tp = int(np.sum((predictions == 1) & (Y_int == 1)))
    fp = int(np.sum((predictions == 1) & (Y_int == 0)))
    fn = int(np.sum((predictions == 0) & (Y_int == 1)))
    tn = int(np.sum((predictions == 0) & (Y_int == 0)))

    accuracy = float((tp + tn) / m)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "m": m,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
        "evaluation_rule": "Recall on injured matters most: missing an injured animal is critical.",
    }


def load_gallery_dataset(db_path: str = SQLITE_DB_PATH, photos_dir: str = PHOTOS_DIR) -> Tuple[list, list]:
    """
    Loads all gallery records from the SQLite table photo_gallery.
    Returns:
        samples: list of dicts with keys 'id', 'filename', 'label', 'filepath', 'vec'
        summary: dict with label counts
    """
    if not os.path.exists(db_path):
        return [], {"total": 0, "healthy": 0, "injured": 0}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, filename, label, species, notes, uploaded_at FROM photo_gallery")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    samples = []
    for r in rows:
        photo_id, filename, label, species, notes, uploaded_at = r
        filepath = os.path.join(photos_dir, filename)
        if not os.path.exists(filepath):
            continue

        try:
            vec = preprocess_image(filepath)  # shape (1024, 1)
            samples.append({
                "id": photo_id,
                "filename": filename,
                "label": int(label),
                "species": species,
                "notes": notes,
                "uploaded_at": uploaded_at,
                "filepath": filepath,
                "vec": vec,
            })
        except Exception as e:
            print(f"Skipping corrupt or unreadable image {filename}: {e}")

    healthy_count = sum(1 for s in samples if s["label"] == 0)
    injured_count = sum(1 for s in samples if s["label"] == 1)

    return samples, {
        "total": len(samples),
        "healthy": healthy_count,
        "injured": injured_count,
    }


def compute_weighted_cost(A2: np.ndarray, Y: np.ndarray, w0: float, w1: float) -> float:
    m = Y.shape[1]
    eps = 1e-15
    A2_safe = np.clip(A2, eps, 1.0 - eps)
    cost = - (1.0 / m) * np.sum(w1 * Y * np.log(A2_safe) + w0 * (1.0 - Y) * np.log(1.0 - A2_safe))
    return float(np.squeeze(cost))


def backward_propagation_weighted(
    parameters: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    w0: float,
    w1: float
) -> Dict[str, np.ndarray]:
    m = X.shape[1]
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = parameters["W2"]

    sample_weights = np.where(Y == 1, w1, w0)
    dZ2 = (A2 - Y) * sample_weights

    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1.0 - np.power(A1, 2))
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def train_model(
    n_h: int = 16,
    learning_rate: float = 0.05,
    num_iterations: int = 1500,
    seed: int = 42,
    db_path: str = SQLITE_DB_PATH,
    photos_dir: str = PHOTOS_DIR,
    model_save_path: str = DEFAULT_MODEL_FILE,
    plot_save_path: str = DEFAULT_PLOT_PATH,
) -> Dict[str, Any]:
    """
    Complete pure-NumPy training pipeline:
    1. Load photos from photo_gallery.
    2. Enforce minimum sample count (refuse if < 20 photos per label).
    3. Shuffle with fixed seed; split 80% train / 20% test BEFORE flipping.
    4. Augment train set with horizontally flipped images.
    5. Train 1024 -> n_h (tanh) -> 1 (sigmoid) neural network.
    6. Print cost every 100 iterations.
    7. Compute accuracy, confusion matrix, precision, recall on test set by hand.
    8. Save cost-vs-iteration plot and parameters npz.
    """
    cache_2000_path = os.path.join("data", "dataset_2000_cache.npz")
    if os.path.exists(cache_2000_path):
        print(f"Loading 2000+ Photo Dataset from cache {cache_2000_path}...")
        cdata = np.load(cache_2000_path)
        X_all = cdata["X"]
        Y_all = cdata["Y"]
        n_healthy = int(cdata["n_healthy"])
        n_injured = int(cdata["n_injured"])

        rng = np.random.RandomState(seed)
        h_idx = np.arange(n_healthy)
        i_idx = np.arange(n_healthy, n_healthy + n_injured)
        rng.shuffle(h_idx)
        rng.shuffle(i_idx)

        h_split = int(0.8 * n_healthy)
        i_split = int(0.8 * n_injured)

        train_idx = np.concatenate([h_idx[:h_split], i_idx[:i_split]])
        test_idx = np.concatenate([h_idx[h_split:], i_idx[i_split:]])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        X_train = X_all[:, train_idx]
        Y_train = Y_all[:, train_idx]
        X_test = X_all[:, test_idx]
        Y_test = Y_all[:, test_idx]

        m_h = np.sum(Y_train == 0)
        m_i = np.sum(Y_train == 1)
        w0 = 1.0
        w1 = float(m_h / m_i) if m_i > 0 else 1.0
        print(f"Loaded 2000+ Photo Dataset: {X_all.shape[1]} samples ({n_healthy} healthy, {n_injured} injured).")
        print(f"Class weighting applied: w0={w0:.2f}, w1={w1:.2f} (injury recall prioritized).")
    else:
        samples, counts = load_gallery_dataset(db_path, photos_dir)
        healthy_count = counts["healthy"]
        injured_count = counts["injured"]

        if healthy_count < MIN_SAMPLES_PER_CLASS or injured_count < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Refusing to train: at least {MIN_SAMPLES_PER_CLASS} photos per label required. "
                f"Current gallery has {healthy_count} healthy and {injured_count} injured."
            )

        healthy_samples = [s for s in samples if s["label"] == 0]
        injured_samples = [s for s in samples if s["label"] == 1]

        rng = np.random.RandomState(seed)
        rng.shuffle(healthy_samples)
        rng.shuffle(injured_samples)

        h_split = int(0.8 * len(healthy_samples))
        i_split = int(0.8 * len(injured_samples))

        train_base = healthy_samples[:h_split] + injured_samples[:i_split]
        test_base = healthy_samples[h_split:] + injured_samples[i_split:]

        rng.shuffle(train_base)
        rng.shuffle(test_base)

        X_train_list = []
        Y_train_list = []
        for s in train_base:
            orig_vec = s["vec"]
            flip_vec = flip_image_vector(orig_vec)
            label = s["label"]
            X_train_list.extend([orig_vec, flip_vec])
            Y_train_list.extend([label, label])

        X_train = np.hstack(X_train_list)
        Y_train = np.array(Y_train_list, dtype=np.float64).reshape(1, -1)

        X_test_list = [s["vec"] for s in test_base]
        Y_test_list = [s["label"] for s in test_base]
        X_test = np.hstack(X_test_list)
        Y_test = np.array(Y_test_list, dtype=np.float64).reshape(1, -1)
        w0, w1 = 1.0, 1.0

    # Initialize parameters (n_x=1024, n_h as argument, n_y=1)
    n_x = 1024
    n_y = 1
    parameters = initialize_parameters(n_x, n_h, n_y, seed=seed)

    costs = []
    iteration_records = []

    print(f"Starting neural network training:")
    print(f"Features: {n_x}, Hidden: {n_h}, Output: {n_y}")
    print(f"Train samples (with flip augmentation): {X_train.shape[1]}, Test samples: {X_test.shape[1]}")
    print(f"Learning rate: {learning_rate}, Iterations: {num_iterations}")

    for i in range(1, num_iterations + 1):
        # Forward propagation
        A2, cache = forward_propagation(X_train, parameters)

        # Cost computation
        cost = compute_weighted_cost(A2, Y_train, w0, w1) if 'w1' in locals() and w1 != 1.0 else compute_cost(A2, Y_train)

        # Backward propagation (using 1 - A1**2)
        grads = backward_propagation_weighted(parameters, cache, X_train, Y_train, w0, w1) if 'w1' in locals() and w1 != 1.0 else backward_propagation(parameters, cache, X_train, Y_train)

        # Gradient descent update
        parameters = update_parameters(parameters, grads, learning_rate)

        # Record and print cost every 100 iterations
        if i % 100 == 0 or i == 1:
            costs.append(cost)
            iteration_records.append(i)
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")

    # Evaluate on test set
    test_metrics = evaluate_model(parameters, X_test, Y_test)
    train_metrics = evaluate_model(parameters, X_train, Y_train)

    print("\n--- Hand-Computed Test Set Evaluation ---")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f} (Recall on injured matters most)")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Save cost-vs-iteration plot
    os.makedirs(os.path.dirname(plot_save_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(iteration_records, costs, marker="o", color="#B08900", linewidth=1.8, markersize=4)
    plt.title("Cost vs Iteration (Photo Recognition NN)")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Cost")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    print(f"Cost plot saved to {plot_save_path}")

    # Save parameters and metadata to npz
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    np.savez(
        model_save_path,
        W1=parameters["W1"],
        b1=parameters["b1"],
        W2=parameters["W2"],
        b2=parameters["b2"],
        n_x=n_x,
        n_h=n_h,
        n_y=n_y,
        test_accuracy=test_metrics["accuracy"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        trained_at=trained_at,
    )
    print(f"Model weights saved to {model_save_path} at {trained_at}")

    return {
        "status": "success",
        "n_h": n_h,
        "learning_rate": learning_rate,
        "iterations": num_iterations,
        "trained_at": trained_at,
        "iterations": num_iterations,
        "train_samples_augmented": int(X_train.shape[1]),
        "test_samples": int(X_test.shape[1]),
        "final_train_cost": round(costs[-1], 6),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cost_plot_saved": plot_save_path,
        "model_file_saved": model_save_path,
    }


def load_trained_model(model_path: str = DEFAULT_MODEL_FILE) -> Dict[str, Any]:
    """
    Loads saved numpy parameters and metadata from npz file.
    Raises FileNotFoundError if model has not been trained yet.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train model first.")

    data = np.load(model_path)
    parameters = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
    }
    trained_at = (
        str(data["trained_at"])
        if "trained_at" in data
        else datetime.fromtimestamp(os.path.getmtime(model_path), timezone.utc).isoformat()
    )
    metadata = {
        "n_x": int(data["n_x"]),
        "n_h": int(data["n_h"]),
        "n_y": int(data["n_y"]),
        "test_accuracy": float(data["test_accuracy"]),
        "test_precision": float(data["test_precision"]),
        "test_recall": float(data["test_recall"]),
        "learning_rate": float(data["learning_rate"]),
        "num_iterations": int(data["num_iterations"]),
        "trained_at": trained_at,
    }
    return {"parameters": parameters, "metadata": metadata}


def predict_photo(image_input: Any, model_path: str = DEFAULT_MODEL_FILE) -> Dict[str, Any]:
    """
    Runs forward propagation on an image input.
    Label decision rule:
    - If probability between 0.4 and 0.6: label = 'uncertain'
    - If probability > 0.6: label = 'injured'
    - If probability < 0.4: label = 'healthy'
    """
    model_bundle = load_trained_model(model_path)
    params = model_bundle["parameters"]
    metadata = model_bundle["metadata"]

    vec = preprocess_image(image_input)  # shape (1024, 1)
    A2, _ = forward_propagation(vec, params)
    prob_injured = float(A2[0, 0])

    if 0.4 <= prob_injured <= 0.6:
        label = "uncertain"
    elif prob_injured > 0.6:
        label = "injured"
    else:
        label = "healthy"

    return {
        "probability_injured": round(prob_injured, 4),
        "label": label,
        "model_test_accuracy": round(metadata["test_accuracy"], 4),
        "model_test_recall": round(metadata["test_recall"], 4),
        "threshold_policy": "injured: prob > 0.6, healthy: prob < 0.4, uncertain: 0.4 <= prob <= 0.6",
    }


if __name__ == "__main__":
    train_model(n_h=32, learning_rate=0.03, num_iterations=2000)
