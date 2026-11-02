"""
Phase 1: ML v1 Baseline — Unsupervised Wildlife Health Anomaly Detection
Kaggle Animal Disease Dataset

This script implements:
1. Data loading and preprocessing for Kaggle animal disease dataset.
2. Exploratory Data Analysis (EDA): class balance, frequent symptoms, missing data.
3. String cleaning and spelling variant standardization with manual mapping.
4. Multi-label one-hot symptom feature matrix (X).
5. Unsupervised modeling with Isolation Forest and Local Outlier Factor (LOF) on X only.
6. Combined anomaly flag ('Unusual' if either model flags).
7. Evaluation against ground truth 'Dangerous' labels (Recall prioritized).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------
# Step 1: Load Raw Dataset
# ---------------------------------------------------------
def load_data(filepath="data/raw/data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath} with shape: {df.shape}")
    return df


# ---------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
def run_eda(df):
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)

    # 1. Missing data check
    print("\n--- Missing Data Check ---")
    missing = df.isnull().sum()
    print(missing)

    # 2. Target class balance
    print("\n--- Class Balance of Target ('Dangerous') ---")
    target_counts = df["Dangerous"].value_counts(dropna=False)
    print(target_counts)
    print(
        f"Dangerous Percentage (Yes): {df['Dangerous'].value_counts(normalize=True).get('Yes', 0) * 100:.2f}%"
    )

    # 3. Animal species distribution
    print("\n--- Animal Species Breakdown (Top 10) ---")
    print(df["AnimalName"].value_counts().head(10))

    # 4. Save EDA bar plot for class balance
    os.makedirs("data", exist_ok=True)
    plt.figure(figsize=(6, 4))
    colors = ["#B08900", "#5A8B67", "#888880"]
    labels = [str(k) for k in target_counts.index]
    plt.bar(labels, target_counts.values, color=colors[: len(labels)], width=0.5)
    plt.title("Class Balance of 'Dangerous' in Kaggle Dataset", fontsize=12)
    plt.xlabel("Dangerous Label")
    plt.ylabel("Number of Records")
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 10, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    eda_plot_path = os.path.join("data", "eda_class_balance.png")
    plt.savefig(eda_plot_path, dpi=150)
    plt.close()
    print(f"Saved class balance plot to {eda_plot_path}")


# ---------------------------------------------------------
# Step 3: Symptom Cleaning & Standardization
# ---------------------------------------------------------
SPELLING_MAPPINGS = {
    # Breathing issues
    "diffculty breathing": "difficulty in breathing",
    "difficulty breathing": "difficulty in breathing",
    "difficulty breating": "difficulty in breathing",
    "difficultty in breathing": "difficulty in breathing",
    "effort to breath": "difficulty in breathing",
    "fast breathing": "rapid breathing",
    "noisy breathing": "loud breathing",
    "lound breathing": "loud breathing",
    # Appetite and feeding
    "loss of  appetite": "loss of appetite",
    "loss of appettite": "loss of appetite",
    "loss od appetite": "loss of appetite",
    "no appp": "poor appetite",
    "decrease appetite": "poor appetite",
    "decreased appetite": "poor appetite",
    "reduced appetite": "poor appetite",
    "inappetence": "loss of appetite",
    "loss of eat": "loss of appetite",
    "unable to eat": "anorexia",
    # Vomiting
    "vommiting": "vomiting",
    "periodic vommiting": "vomiting",
    # Weakness
    "weekness": "weakness",
    "severe weekness and depression": "weakness and depression",
    "week legs": "weakness in legs",
    "weekness in the back legs": "weakness in back legs",
    "week pulse": "weak pulse",
    # Pain
    "pains": "pain",
    "painfull": "pain",
    "pain on leg": "leg pain",
    "pain on face": "facial pain",
    "painfull swalling": "painful swelling",
    "abdminal pain": "abdominal pain",
    "abdonormal pain": "abdominal pain",
    "join pains": "joint pain",
    "muscle pain": "pain",
    "muscle stiffness": "stiffness",
    # Seizures
    "seizuers": "seizures",
    # Pneumonia
    "pnemonia": "pneumonia",
    # Anorexia
    "anoxeria": "anorexia",
    # Diarrhea
    "diarrhoea": "diarrhea",
    "watering diarrhea": "watery diarrhea",
    "diarrhea with muscus": "diarrhea with mucus",
    # Dullness
    "dull ness": "dullness",
    # Anemia
    "anaemia": "anemia",
    "aneamia": "anemia",
    # Lesions & skin
    "lession": "lesion",
    "lessions": "lesions",
    "lession on the skin": "lesions on skin",
    "lession on cat skin": "lesions on skin",
    # Swelling
    "swelling ": "swelling",
    "swelling problem": "swelling",
    "swallon": "swollen",
    # Reproductive & obstetric
    "fetopelvic dispropotion": "fetopelvic disproportion",
    "uteria inertia": "uterine inertia",
    # Scratching & discharge
    "scartching": "scratching",
    "scartch": "scratching",
    "eye disharge": "eye discharge",
    "blooody fluid": "bloody fluid",
    "blood on faces": "blood in feces",
    "blood in faces": "blood in feces",
}


def clean_symptom(text):
    if pd.isnull(text):
        return None
    # Lowercase, strip, collapse multiple spaces
    s = re.sub(r"\s+", " ", str(text).lower().strip())
    # Apply manual mapping dictionary
    return SPELLING_MAPPINGS.get(s, s)


def clean_dataset(df):
    symptom_cols = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
    cleaned_df = df.copy()

    # Apply cleaning to all symptom columns
    for col in symptom_cols:
        cleaned_df[col] = cleaned_df[col].apply(clean_symptom)

    # Collect cleaned list per animal
    cleaned_symptoms_list = []
    for _, row in cleaned_df.iterrows():
        syms = [row[c] for c in symptom_cols if pd.notnull(row[c]) and row[c] != ""]
        cleaned_symptoms_list.append(syms)

    cleaned_df["cleaned_symptoms"] = cleaned_symptoms_list
    return cleaned_df


# ---------------------------------------------------------
# Step 4: Build Multi-Label One-Hot Symptom Feature Matrix
# ---------------------------------------------------------
def build_feature_matrix(cleaned_df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(cleaned_df["cleaned_symptoms"])
    vocab = mlb.classes_
    print(f"\nBuilt one-hot symptom matrix X:")
    print(f"- Samples (records): {X.shape[0]}")
    print(f"- Symptom features (vocabulary size): {X.shape[1]}")
    return X, vocab, mlb


# ---------------------------------------------------------
# Step 5: Fit Isolation Forest & Local Outlier Factor (Unsupervised)
# ---------------------------------------------------------
def train_unsupervised_models(X, cleaned_df, contamination=0.10):
    print("\n" + "=" * 50)
    print("UNSUPERVISED MODEL TRAINING")
    print("=" * 50)
    # IMPORTANT: Never pass 'Dangerous' into model training!
    print("Training models on feature matrix X only (unsupervised)...")

    # 1. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_preds = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    iso_scores = iso.decision_function(X)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_preds = lof.fit_predict(X)  # -1 = anomaly, 1 = normal
    lof_scores = lof.negative_outlier_factor_

    # 3. Combine into single flag
    # A record is flagged as 'Unusual' if either model flags it as -1
    result_df = cleaned_df.copy()
    result_df["iso_pred"] = iso_preds
    result_df["iso_score"] = iso_scores
    result_df["lof_pred"] = lof_preds
    result_df["lof_score"] = lof_scores
    result_df["flag"] = np.where(
        (iso_preds == -1) | (lof_preds == -1), "Unusual", "Normal"
    )

    unusual_count = (result_df["flag"] == "Unusual").sum()
    print(
        f"Flagged {unusual_count} of {len(result_df)} records as 'Unusual' "
        f"({unusual_count / len(result_df) * 100:.1f}%)"
    )
    return result_df


# ---------------------------------------------------------
# Step 6: Evaluation Against Ground Truth Labels
# ---------------------------------------------------------
def evaluate_models(result_df):
    print("\n" + "=" * 50)
    print("MODEL EVALUATION: FLAG VS DANGEROUS")
    print("=" * 50)

    # Note: In veterinary epidemiology and wildlife health surveillance,
    # RECALL MATTERS MOST: missing a dangerous case (False Negative) is far worse
    # than a false alarm (False Positive), because an undetected outbreak can spread unchecked.

    valid_df = result_df.dropna(subset=["Dangerous"]).copy()
    # Map predictions to match target classes: 'Unusual' -> 'Yes', 'Normal' -> 'No'
    y_true = valid_df["Dangerous"]
    y_pred = valid_df["flag"].map({"Unusual": "Yes", "Normal": "No"})

    print("\nConfusion Matrix (Labels: ['Yes', 'No']):")
    cm = confusion_matrix(y_true, y_pred, labels=["Yes", "No"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Dangerous (Yes)", "Actual Safe (No)"],
        columns=["Predicted Unusual (Yes)", "Predicted Normal (No)"],
    )
    print(cm_df)

    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, labels=["Yes", "No"], zero_division=0
    )
    print(report)

    # Print clinical priority summary
    print(
        """
[CLINICAL EVALUATION NOTE]:
- In wildlife health anomaly detection, RECALL on dangerous cases matters most.
- Missing a dangerous case (False Negative) risks undetected epidemic spread or fatal illness.
- A false alarm (False Positive) merely warrants a follow-up veterinary inspection.
- Because 97.7% of this dataset is labeled 'Dangerous=Yes', unsupervised anomaly detection
  (which assumes anomalies are rare ~5-10% outliers) flags the most statistically unusual symptom
  clusters. Contamination and threshold tuning directly govern the trade-off between recall and alert volume.
"""
    )


# ---------------------------------------------------------
# Step 7: Main Pipeline Execution & Export
# ---------------------------------------------------------
def main():
    # 1. Load data
    df = load_data()

    # 2. EDA
    run_eda(df)

    # 3. Clean symptoms
    cleaned_df = clean_dataset(df)

    # 4. Build feature matrix X
    X, vocab, mlb = build_feature_matrix(cleaned_df)

    # 5. Train unsupervised models
    result_df = train_unsupervised_models(X, cleaned_df, contamination=0.10)

    # 6. Evaluate
    evaluate_models(result_df)

    # 7. Export deliverable
    output_path = os.path.join("data", "phase1_flagged_kaggle.csv")
    export_cols = [
        "AnimalName",
        "symptoms1",
        "symptoms2",
        "symptoms3",
        "symptoms4",
        "symptoms5",
        "Dangerous",
        "iso_pred",
        "iso_score",
        "lof_pred",
        "lof_score",
        "flag",
    ]
    result_df[export_cols].to_csv(output_path, index=False)
    print(f"Saved Phase 1 flagged output to {output_path}")


if __name__ == "__main__":
    main()
