import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def main():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    obs_file = os.path.join("data", "observations.csv")
    output_file = os.path.join("data", "flagged_observations.csv")

    # Step 2 — Build or load dataset
    n_normal, n_anomaly = 95, 5
    if os.path.exists(obs_file):
        print(f"Reading observations from {obs_file}...")
        df = pd.read_csv(obs_file)
    else:
        print("Generating synthetic observation dataset...")
        np.random.seed(1)
        normal_df = pd.DataFrame({
            "activity": np.random.normal(70, 8, n_normal),
            "feeding": np.random.normal(65, 7, n_normal),
            "movement": np.random.normal(75, 6, n_normal),
            "injury": np.zeros(n_normal)
        })
        anomaly_df = pd.DataFrame({
            "activity": np.random.normal(25, 5, n_anomaly),
            "feeding": np.random.normal(20, 5, n_anomaly),
            "movement": np.random.normal(30, 5, n_anomaly),
            "injury": np.ones(n_anomaly)
        })
        df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        df["animal_id"] = range(1, len(df) + 1)
        df = df[["animal_id", "activity", "feeding", "movement", "injury"]]
        df.to_csv(obs_file, index=False)
        print(f"Saved observations dataset to {obs_file}")

    # Step 3 — Run Isolation Forest
    X = df[["activity", "feeding", "movement", "injury"]].values

    iso = IsolationForest(contamination=0.05, random_state=1)
    df["iso_pred"] = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
    df["iso_score"] = iso.decision_function(X)    # lower score = more abnormal

    # Step 4 — Run Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05)
    df["lof_pred"] = lof.fit_predict(X)          # -1 = anomaly, 1 = normal
    df["lof_score"] = lof.negative_outlier_factor_ # lower = more abnormal

    # Step 5 — Combine into a single flag
    df["flag"] = np.where(
        (df["iso_pred"] == -1) | (df["lof_pred"] == -1),
        "Unusual", "Normal"
    )

    flagged = df[df["flag"] == "Unusual"]
    print(f"\nFlagged {len(flagged)} of {len(df)} records")
    print(flagged)

    # Step 6 — Validate
    df["actual"] = np.where(df["animal_id"] > n_normal, "Abnormal", "Normal")
    print("\n--- Validation: Actual vs Flagged ---")
    print(pd.crosstab(df["actual"], df["flag"]))

    # Step 7 — Output
    df.to_csv(output_file, index=False)
    print(f"\nSaved final results to {output_file}")

    # Generate historical observations for inline detail view (3 past periods + current)
    records = []
    np.random.seed(42)
    for idx, row in df.iterrows():
        is_abnormal = row["animal_id"] > n_normal
        hist = []
        if is_abnormal:
            # Show progression toward abnormal state
            hist.append({
                "period": "3 days ago",
                "activity": round(float(np.random.normal(68, 4)), 1),
                "feeding": round(float(np.random.normal(63, 4)), 1),
                "movement": round(float(np.random.normal(72, 4)), 1),
                "injury": 0
            })
            hist.append({
                "period": "2 days ago",
                "activity": round(float(np.random.normal(50, 5)), 1),
                "feeding": round(float(np.random.normal(45, 5)), 1),
                "movement": round(float(np.random.normal(55, 5)), 1),
                "injury": 0
            })
            hist.append({
                "period": "Yesterday",
                "activity": round(float(np.random.normal(32, 4)), 1),
                "feeding": round(float(np.random.normal(28, 4)), 1),
                "movement": round(float(np.random.normal(38, 4)), 1),
                "injury": int(row["injury"])
            })
        else:
            # Stable normal trajectory
            for days_ago in ["3 days ago", "2 days ago", "Yesterday"]:
                hist.append({
                    "period": days_ago,
                    "activity": round(float(np.random.normal(row["activity"], 3)), 1),
                    "feeding": round(float(np.random.normal(row["feeding"], 3)), 1),
                    "movement": round(float(np.random.normal(row["movement"], 3)), 1),
                    "injury": 0
                })
        
        # Latest record (current observation)
        hist.append({
            "period": "Latest",
            "activity": round(float(row["activity"]), 1),
            "feeding": round(float(row["feeding"]), 1),
            "movement": round(float(row["movement"]), 1),
            "injury": int(row["injury"])
        })

        records.append({
            "animal_id": int(row["animal_id"]),
            "activity": round(float(row["activity"]), 1),
            "feeding": round(float(row["feeding"]), 1),
            "movement": round(float(row["movement"]), 1),
            "injury": int(row["injury"]),
            "iso_pred": int(row["iso_pred"]),
            "iso_score": round(float(row["iso_score"]), 3),
            "lof_pred": int(row["lof_pred"]),
            "lof_score": round(float(row["lof_score"]), 3),
            "flag": str(row["flag"]),
            "actual": str(row["actual"]),
            "history": hist
        })

    import json
    js_data_path = os.path.join("data", "observations_data.js")
    with open(js_data_path, "w", encoding="utf-8") as f:
        f.write("window.OBSERVATIONS_DATA = " + json.dumps(records, indent=2) + ";\n")
    print(f"Exported frontend dataset to {js_data_path}")

    # Optional visualization
    try:
        plt.figure(figsize=(8, 5))
        normal_pts = df[df["flag"] == "Normal"]
        unusual_pts = df[df["flag"] == "Unusual"]

        plt.scatter(normal_pts["activity"], normal_pts["movement"], color="#2b5c8f", alpha=0.7, label="Normal")
        plt.scatter(unusual_pts["activity"], unusual_pts["movement"], color="#d9534f", marker="x", s=90, linewidths=2, label="Unusual")

        plt.title("Wildlife Health Anomaly Detection (Activity vs Movement)")
        plt.xlabel("Activity Level Score")
        plt.ylabel("Movement Pattern Score")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plot_file = os.path.join("data", "anomaly_plot.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved visualization plot to {plot_file}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
