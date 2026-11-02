"""
Phase 2: ML v2 — Incorporating WAHIS Global Surveillance & Population Context
Wildlife Health Anomaly Detection

This script:
1. Loads Kaggle animal clinical dataset and WAHIS exports (Animal population, SMR, AR).
2. Derives species-level contextual features from WAHIS (surveillance status, global population, premises, country breadth).
3. Joins WAHIS context onto Kaggle rows strictly across valid keys (species category) without forcing mismatched granularity.
4. Cleans symptoms using the standardized manual mapping dictionary.
5. Builds feature matrices: X_v1 (symptoms only) and X_v2 (symptoms + WAHIS context).
6. Runs unsupervised Isolation Forest & LOF on both feature sets.
7. Evaluates and reports v1 vs v2 metrics side-by-side (Confusion Matrix, Precision, Recall, F1).
8. Exports deliverable to data/phase2_flagged_wahis_kaggle.csv.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# Step 1: Ingestion
# ---------------------------------------------------------
def load_datasets():
    kaggle_path = os.path.join("data", "raw", "data.csv")
    pop_path = os.path.join("data", "raw", "Animal population 2026-10-26.csv")
    smr_path = os.path.join("data", "raw", "SMR_2026-10-26_16-29.csv")
    ar_path = os.path.join("data", "raw", "AR_2026-10-26.csv")

    df_kaggle = pd.read_csv(kaggle_path)
    df_pop = pd.read_csv(pop_path)
    df_smr = pd.read_csv(smr_path, sep=";")
    df_ar = pd.read_csv(ar_path, sep=";")

    print(f"Loaded Kaggle Dataset: {df_kaggle.shape[0]} rows, {df_kaggle.shape[1]} cols")
    print(f"Loaded WAHIS Population: {df_pop.shape[0]} rows, {df_pop.shape[1]} cols")
    print(f"Loaded WAHIS SMR: {df_smr.shape[0]} rows, {df_smr.shape[1]} cols")
    print(f"Loaded WAHIS AR: {df_ar.shape[0]} rows, {df_ar.shape[1]} cols")

    return df_kaggle, df_pop, df_smr, df_ar


# ---------------------------------------------------------
# Step 2: Derive WAHIS Contextual Features
# ---------------------------------------------------------
def process_wahis_context(df_pop, df_smr, df_ar):
    print("\n--- Deriving WAHIS Species Context ---")
    pop = df_pop.copy()

    # Clean numeric fields in population
    pop["Nb live animals"] = pd.to_numeric(
        pop["Nb live animals"].astype(str).str.replace(" ", "").str.replace("-", ""),
        errors="coerce",
    )
    pop["Nb slaughtered animals"] = pd.to_numeric(
        pop["Nb slaughtered animals"].astype(str).str.replace(" ", "").str.replace("-", ""),
        errors="coerce",
    )
    pop["Nb animals premises"] = pd.to_numeric(
        pop["Nb animals premises"].astype(str).str.replace(" ", "").str.replace("-", ""),
        errors="coerce",
    )

    # Aggregate by Animal category across recent reporting period (2023-2025)
    species_ctx = pop.groupby("Animal category").agg(
        wahis_countries=("Country/Territory", "nunique"),
        wahis_records=("Report ID", "count"),
        wahis_live_pop=("Nb live animals", "sum"),
        wahis_slaughtered=("Nb slaughtered animals", "sum"),
        wahis_premises=("Nb animals premises", "sum"),
    ).reset_index()

    # Log transformations for scale invariance
    species_ctx["wahis_log_live_pop"] = np.log1p(species_ctx["wahis_live_pop"])
    species_ctx["wahis_log_premises"] = np.log1p(species_ctx["wahis_premises"])
    species_ctx["wahis_monitored"] = 1  # Formally surveyed in WAHIS

    print(f"Derived context for {len(species_ctx)} WAHIS animal categories:")
    for _, r in species_ctx.iterrows():
        print(f"  - {r['Animal category']:<25} | Countries: {r['wahis_countries']:<3} | Log Pop: {r['wahis_log_live_pop']:.2f}")

    return species_ctx


# ---------------------------------------------------------
# Step 3: Granularity-Safe Join onto Kaggle Rows
# ---------------------------------------------------------
def map_kaggle_to_wahis_category(animal_name):
    """
    Maps Kaggle AnimalName to the standardized WAHIS Animal Category.
    Respects granularity: joins on species category without hallucinating country/date.
    """
    n = str(animal_name).lower().strip()
    if any(b in n for b in ["bird", "chicken", "duck", "fowl"]):
        return "Birds"
    if any(c in n for c in ["cattle", "cow"]):
        return "Cattle"
    if "buffalo" in n:
        return "Buffaloes"
    if "sheep" in n:
        return "Sheep"
    if "goat" in n:
        return "Goats"
    if "pig" in n:
        return "Swine"
    if any(e in n for e in ["horse", "donkey", "mule"]):
        return "Equidae"
    if any(d in n for d in ["dog", "fox", "wolv", "hyaena"]):
        return "Dogs"
    if any(f in n for f in ["cat", "lion", "tiger"]):
        return "Cats"
    if any(c in n for c in ["deer", "elk", "wapiti", "moos", "reindeer"]):
        return "Cervidae"
    if "rabbit" in n:
        return "Rabbits / hares (mixed group)"
    return "Other Wildlife / Exotic"


def join_context(df_kaggle, species_ctx):
    df = df_kaggle.copy()
    df["wahis_category"] = df["AnimalName"].apply(map_kaggle_to_wahis_category)

    merged = df.merge(
        species_ctx,
        left_on="wahis_category",
        right_on="Animal category",
        how="left",
    )

    # Impute missing values for unmonitored / exotic wildlife species
    merged["wahis_monitored"] = merged["wahis_monitored"].fillna(0).astype(int)
    merged["wahis_countries"] = merged["wahis_countries"].fillna(0)
    merged["wahis_records"] = merged["wahis_records"].fillna(0)
    merged["wahis_log_live_pop"] = merged["wahis_log_live_pop"].fillna(0.0)
    merged["wahis_log_premises"] = merged["wahis_log_premises"].fillna(0.0)

    print(f"\nJoined WAHIS context onto all {len(merged)} Kaggle observation rows.")
    print(f"Monitored species rows: {(merged['wahis_monitored'] == 1).sum()} / {len(merged)}")
    print(f"Unmonitored / exotic rows: {(merged['wahis_monitored'] == 0).sum()} / {len(merged)}")

    return merged


# ---------------------------------------------------------
# Step 4: Symptom Cleaning & Feature Matrix Preparation
# ---------------------------------------------------------
SPELLING_MAPPINGS = {
    "diffculty breathing": "difficulty in breathing",
    "difficulty breathing": "difficulty in breathing",
    "difficulty breating": "difficulty in breathing",
    "difficultty in breathing": "difficulty in breathing",
    "effort to breath": "difficulty in breathing",
    "loss of  appetite": "loss of appetite",
    "loss of appettite": "loss of appetite",
    "loss od appetite": "loss of appetite",
    "no appp": "poor appetite",
    "decrease appetite": "poor appetite",
    "decreased appetite": "poor appetite",
    "reduced appetite": "poor appetite",
    "inappetence": "loss of appetite",
    "vommiting": "vomiting",
    "periodic vommiting": "vomiting",
    "weekness": "weakness",
    "severe weekness and depression": "weakness and depression",
    "week legs": "weakness in legs",
    "weekness in the back legs": "weakness in back legs",
    "week pulse": "weak pulse",
    "pains": "pain",
    "painfull": "pain",
    "abdminal pain": "abdominal pain",
    "join pains": "joint pain",
    "seizuers": "seizures",
    "pnemonia": "pneumonia",
    "anoxeria": "anorexia",
    "diarrhoea": "diarrhea",
    "dull ness": "dullness",
    "anaemia": "anemia",
    "aneamia": "anemia",
    "lession": "lesion",
    "lessions": "lesions",
    "lession on the skin": "lesions on skin",
    "swelling ": "swelling",
    "swelling problem": "swelling",
    "swallon": "swollen",
    "fetopelvic dispropotion": "fetopelvic disproportion",
    "uteria inertia": "uterine inertia",
    "scartching": "scratching",
    "eye disharge": "eye discharge",
}

def clean_symptom(text):
    if pd.isnull(text):
        return None
    s = re.sub(r"\s+", " ", str(text).lower().strip())
    return SPELLING_MAPPINGS.get(s, s)


def build_feature_matrices(merged_df):
    symptom_cols = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
    cleaned_symptoms = []
    for _, row in merged_df.iterrows():
        syms = [clean_symptom(row[c]) for c in symptom_cols if pd.notnull(row[c])]
        cleaned_symptoms.append(syms)

    mlb = MultiLabelBinarizer()
    X_sym = mlb.fit_transform(cleaned_symptoms)

    wahis_cols = [
        "wahis_monitored",
        "wahis_countries",
        "wahis_records",
        "wahis_log_live_pop",
        "wahis_log_premises",
    ]
    scaler = StandardScaler()
    X_ctx = scaler.fit_transform(merged_df[wahis_cols])

    X_v1 = X_sym
    X_v2 = np.hstack([X_sym, X_ctx])

    print(f"\nFeature Matrices Created:")
    print(f"- X_v1 (Symptoms Only): {X_v1.shape}")
    print(f"- X_v2 (Symptoms + WAHIS Context): {X_v2.shape}")

    return X_v1, X_v2, mlb


# ---------------------------------------------------------
# Step 5: Unsupervised Modeling (v1 vs v2)
# ---------------------------------------------------------
def run_unsupervised_models(X_v1, X_v2, merged_df, contamination=0.10):
    print("\n" + "=" * 55)
    print("UNSUPERVISED MODEL TRAINING (v1 vs v2)")
    print("=" * 55)
    # Strictly unsupervised: never pass 'Dangerous' into training!

    # --- Pipeline v1: Symptoms Only ---
    iso_v1 = IsolationForest(contamination=contamination, random_state=42)
    lof_v1 = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    iso_p1 = iso_v1.fit_predict(X_v1)
    lof_p1 = lof_v1.fit_predict(X_v1)
    flag_v1 = np.where((iso_p1 == -1) | (lof_p1 == -1), "Unusual", "Normal")

    # --- Pipeline v2: Symptoms + WAHIS Context ---
    iso_v2 = IsolationForest(contamination=contamination, random_state=42)
    lof_v2 = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    iso_p2 = iso_v2.fit_predict(X_v2)
    lof_p2 = lof_v2.fit_predict(X_v2)
    flag_v2 = np.where((iso_p2 == -1) | (lof_p2 == -1), "Unusual", "Normal")

    results = merged_df.copy()
    results["iso_pred_v1"] = iso_p1
    results["iso_score_v1"] = iso_v1.decision_function(X_v1)
    results["lof_pred_v1"] = lof_p1
    results["lof_score_v1"] = lof_v1.negative_outlier_factor_
    results["flag_v1"] = flag_v1

    results["iso_pred_v2"] = iso_p2
    results["iso_score_v2"] = iso_v2.decision_function(X_v2)
    results["lof_pred_v2"] = lof_p2
    results["lof_score_v2"] = lof_v2.negative_outlier_factor_
    results["flag_v2"] = flag_v2

    print(f"v1 Flagged Unusual: {(flag_v1 == 'Unusual').sum()} / {len(results)} ({(flag_v1 == 'Unusual').mean()*100:.1f}%)")
    print(f"v2 Flagged Unusual: {(flag_v2 == 'Unusual').sum()} / {len(results)} ({(flag_v2 == 'Unusual').mean()*100:.1f}%)")
    agreement = (flag_v1 == flag_v2).mean() * 100
    print(f"v1 vs v2 Flag Agreement: {agreement:.2f}%")

    return results


# ---------------------------------------------------------
# Step 6: Side-by-Side Evaluation & Comparison
# ---------------------------------------------------------
def compare_metrics(results_df):
    print("\n" + "=" * 55)
    print("SIDE-BY-SIDE EVALUATION: v1 vs v2")
    print("=" * 55)

    valid = results_df.dropna(subset=["Dangerous"]).copy()
    y_true = valid["Dangerous"]
    y_p1 = valid["flag_v1"].map({"Unusual": "Yes", "Normal": "No"})
    y_p2 = valid["flag_v2"].map({"Unusual": "Yes", "Normal": "No"})

    cm1 = confusion_matrix(y_true, y_p1, labels=["Yes", "No"])
    cm2 = confusion_matrix(y_true, y_p2, labels=["Yes", "No"])

    print("\n--- Confusion Matrix: v1 (Symptoms Only) ---")
    print(pd.DataFrame(cm1, index=["Actual Yes", "Actual No"], columns=["Pred Yes", "Pred No"]))

    print("\n--- Confusion Matrix: v2 (Symptoms + WAHIS Context) ---")
    print(pd.DataFrame(cm2, index=["Actual Yes", "Actual No"], columns=["Pred Yes", "Pred No"]))

    rep1 = classification_report(y_true, y_p1, labels=["Yes", "No"], output_dict=True, zero_division=0)
    rep2 = classification_report(y_true, y_p2, labels=["Yes", "No"], output_dict=True, zero_division=0)

    comparison_df = pd.DataFrame({
        "Metric": [
            "Precision (Dangerous / Yes)",
            "Recall (Dangerous / Yes)",
            "F1-Score (Dangerous / Yes)",
            "Precision (Safe / No)",
            "Recall (Safe / No)",
            "F1-Score (Safe / No)",
            "Overall Accuracy",
            "Macro Avg Recall",
            "Macro Avg F1"
        ],
        "v1 (Symptoms Only)": [
            f"{rep1['Yes']['precision']:.4f}",
            f"{rep1['Yes']['recall']:.4f}",
            f"{rep1['Yes']['f1-score']:.4f}",
            f"{rep1['No']['precision']:.4f}",
            f"{rep1['No']['recall']:.4f}",
            f"{rep1['No']['f1-score']:.4f}",
            f"{rep1['accuracy']:.4f}",
            f"{rep1['macro avg']['recall']:.4f}",
            f"{rep1['macro avg']['f1-score']:.4f}",
        ],
        "v2 (+ WAHIS Context)": [
            f"{rep2['Yes']['precision']:.4f}",
            f"{rep2['Yes']['recall']:.4f}",
            f"{rep2['Yes']['f1-score']:.4f}",
            f"{rep2['No']['precision']:.4f}",
            f"{rep2['No']['recall']:.4f}",
            f"{rep2['No']['f1-score']:.4f}",
            f"{rep2['accuracy']:.4f}",
            f"{rep2['macro avg']['recall']:.4f}",
            f"{rep2['macro avg']['f1-score']:.4f}",
        ]
    })

    print("\n--- Side-by-Side Metrics Table ---")
    print(comparison_df.to_string(index=False))

    return comparison_df


# ---------------------------------------------------------
# Step 7: Main Execution & Export
# ---------------------------------------------------------
def main():
    df_kaggle, df_pop, df_smr, df_ar = load_datasets()
    species_ctx = process_wahis_context(df_pop, df_smr, df_ar)
    merged_df = join_context(df_kaggle, species_ctx)
    X_v1, X_v2, mlb = build_feature_matrices(merged_df)
    results_df = run_unsupervised_models(X_v1, X_v2, merged_df, contamination=0.10)
    compare_metrics(results_df)

    output_path = os.path.join("data", "phase2_flagged_wahis_kaggle.csv")
    export_cols = [
        "AnimalName",
        "wahis_category",
        "wahis_monitored",
        "wahis_countries",
        "wahis_records",
        "wahis_log_live_pop",
        "wahis_log_premises",
        "Dangerous",
        "iso_pred_v1",
        "lof_pred_v1",
        "flag_v1",
        "iso_pred_v2",
        "lof_pred_v2",
        "flag_v2",
    ]
    results_df[export_cols].to_csv(output_path, index=False)
    print(f"\nSaved Phase 2 flagged results to: {output_path}")


if __name__ == "__main__":
    main()
