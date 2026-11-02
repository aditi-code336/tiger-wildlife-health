"""
Train and export the final validated Phase 2 model bundle (v2: Symptoms + WAHIS Context).
Exports model.joblib containing:
- MultiLabelBinarizer (symptom encoding)
- IsolationForest (unsupervised partition isolation)
- LocalOutlierFactor (with novelty=True for fast inference on unseen requests)
- StandardScaler (WAHIS context scaling)
- WAHIS species context lookup dictionary
- Cleaning & spelling mapping rules
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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
    "swelling ": "swelling",
    "swelling problem": "swelling",
    "swallon": "swollen",
    "fetopelvic dispropotion": "fetopelvic disproportion",
    "uteria inertia": "uterine inertia",
    "scartching": "scratching",
    "eye disharge": "eye discharge",
}


def clean_symptom(text):
    if not text:
        return ""
    s = re.sub(r"\s+", " ", str(text).lower().strip())
    return SPELLING_MAPPINGS.get(s, s)


def map_species_to_wahis(name):
    if not name:
        return "Other"
    n = str(name).lower().strip()
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
    return "Other"


def train_and_export():
    print("Loading raw training datasets...")
    df_kaggle = pd.read_csv("data/raw/data.csv")
    df_pop = pd.read_csv("data/raw/Animal population 2026-10-26.csv")

    # Clean WAHIS population
    pop = df_pop.copy()
    for col in ["Nb live animals", "Nb slaughtered animals", "Nb animals premises"]:
        pop[col] = pd.to_numeric(
            pop[col].astype(str).str.replace(" ", "").str.replace("-", ""),
            errors="coerce",
        )

    species_ctx = (
        pop.groupby("Animal category")
        .agg(
            wahis_countries=("Country/Territory", "nunique"),
            wahis_records=("Report ID", "count"),
            wahis_live_pop=("Nb live animals", "sum"),
            wahis_premises=("Nb animals premises", "sum"),
        )
        .reset_index()
    )

    species_ctx["wahis_log_live_pop"] = np.log1p(species_ctx["wahis_live_pop"])
    species_ctx["wahis_log_premises"] = np.log1p(species_ctx["wahis_premises"])
    species_ctx["wahis_monitored"] = 1

    # Build species context dictionary
    wahis_cols = [
        "wahis_monitored",
        "wahis_countries",
        "wahis_records",
        "wahis_log_live_pop",
        "wahis_log_premises",
    ]
    species_dict = {}
    for _, row in species_ctx.iterrows():
        species_dict[row["Animal category"]] = [float(row[c]) for c in wahis_cols]

    # Clean Kaggle symptoms
    symptom_cols = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
    cleaned_symptoms_list = []
    for _, row in df_kaggle.iterrows():
        syms = [clean_symptom(row[c]) for c in symptom_cols if pd.notnull(row[c])]
        cleaned_symptoms_list.append([s for s in syms if s])

    # Fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    X_sym = mlb.fit_transform(cleaned_symptoms_list)

    # Attach WAHIS context for training
    df_kaggle["wahis_category"] = df_kaggle["AnimalName"].apply(map_species_to_wahis)
    merged = df_kaggle.merge(
        species_ctx, left_on="wahis_category", right_on="Animal category", how="left"
    )
    for c in wahis_cols:
        merged[c] = merged[c].fillna(0.0)

    scaler = StandardScaler()
    X_ctx = scaler.fit_transform(merged[wahis_cols])
    X_train = np.hstack([X_sym, X_ctx])

    print(f"Training Feature Matrix Shape: {X_train.shape}")

    # Train Isolation Forest
    contamination = 0.10
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train)

    # Train Local Outlier Factor with novelty=True for runtime serving
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X_train)

    # Package model bundle
    bundle = {
        "version": "v2_wahis_context",
        "mlb": mlb,
        "iso": iso,
        "lof": lof,
        "scaler": scaler,
        "species_dict": species_dict,
        "wahis_cols": wahis_cols,
        "spelling_mappings": SPELLING_MAPPINGS,
        "contamination": contamination,
        "symptom_vocabulary_size": len(mlb.classes_),
    }

    model_path = "model.joblib"
    joblib.dump(bundle, model_path)
    print(f"Successfully trained and saved model bundle to {model_path}!")


if __name__ == "__main__":
    train_and_export()
