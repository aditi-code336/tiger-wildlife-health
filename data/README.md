# Data Audit & Catalog (Phase 0 & Phase 2)

This document provides the formal audit and data dictionary for all raw data sources ingested into `data/raw/` for the Wildlife Health Anomaly Detection project.

> [!IMPORTANT]
> **Granularity & Architecture Rule**:
> Do **NOT** attempt to merge these data sources into a single table without respecting granularity boundaries. They represent fundamentally different granularities and modeling objectives (individual clinical cases vs. macro-level geographic outbreak/population aggregates) and are integrated solely across valid shared entities (e.g. species categories).

---

## 1. Kaggle Animal Condition Classification Dataset (`data/raw/data.csv`)

- **Source**: Kaggle (`gracehephzibahm/animal-disease`)
- **File Path**: `data/raw/data.csv` (871 rows, 7 columns)
- **Granularity**: **Per-animal-case granularity** (clinical observation of a single individual animal).
- **Target / Label**: **Yes** — column `Dangerous` (`Yes` [849], `No` [20], 2 missing values).
- **Temporal & Spatial Coverage**: None. Does not contain timestamp, date, or geographic coordinates.

### Schema & Data Types

| Column Name | Data Type | Meaning / Description | Example Values |
|---|---|---|---|
| `AnimalName` | String / Object | Species or animal category | `Dog`, `cat`, `Deer`, `Buffaloes`, `Rabbit`, `cow` |
| `symptoms1` | String / Object | Primary observed symptom | `Fever`, `Ulcers`, `Limping`, `Swelling on leg` |
| `symptoms2` | String / Object | Secondary observed symptom | `Diarrhea`, `Respiratory distress`, `Seizures` |
| `symptoms3` | String / Object | Tertiary observed symptom | `Vomiting`, `Coughing`, `Poor Appetite` |
| `symptoms4` | String / Object | Quaternary observed symptom | `Weight loss`, `Lethargy`, `Lameness` |
| `symptoms5` | String / Object | Quinary observed symptom | `Dehydration`, `Pains`, `Blue Eye`, `Anorexia` |
| `Dangerous` | String / Object | Clinical severity label (Target) | `Yes`, `No` |

---

## 2. WAHIS Exports (World Organisation for Animal Health / WOAH)

Three distinct WAHIS export files were placed in `data/raw/`:

### A. WAHIS Animal Population Export (`data/raw/Animal population 2026-10-26.csv`)
- **Rows**: 9,899 rows, 13 columns (comma-separated `,`).
- **Granularity**: Country-level census records by year (2023–2025) and animal category.
- **Unit of Representation**: Aggregate animal census counts per country, year, and species category.
- **Key Columns**:
  - `Country/Territory`, `Country(ISO3)`: Reporting nation.
  - `Year`: Reporting year (2023, 2024, 2025).
  - `Animal type`: `Terrestrial` or `Aquatic`.
  - `Animal category`: `Birds`, `Cattle`, `Equidae`, `Swine`, `Sheep`, `Goats`, `Buffaloes`, `Dogs`, `Cats`, `Cervidae`, `Rabbits / hares (mixed group)`.
  - `Nb live animals`: Estimated population size in country.
  - `Nb slaughtered animals`: Annual slaughter volume.
  - `Nb animals premises`: Number of livestock farms / holding premises.
- **Use in Phase 2**: Aggregated by species category to derive global population scale (`log_live_pop`), global reporting breadth (`reporting_countries_count`), farm density (`log_premises`), and formal surveillance tracking status (`wahis_monitored`).

### B. WAHIS Six-Monthly Reports (`data/raw/SMR_2026-10-26_16-29.csv`)
- **Rows**: 14,184 rows, 9 columns (semicolon-separated `;`).
- **Granularity**: Validated national semi-annual monitoring submissions (2005–2026).
- **Key Columns**: `Country/Territory`, `Status`, `Report ID`, `Animal type` (Terrestrial, Aquatic), `Report period` (First/Second semester), `Year`, `Validation date`.
- **Unit of Representation**: Official biannual disease monitoring dossier submission per country and animal type.

### C. WAHIS Annual Reports (`data/raw/AR_2026-10-26.csv`)
- **Rows**: 354 rows, 5 columns (semicolon-separated `;`).
- **Granularity**: Validated national annual summaries (2023–2025).
- **Key Columns**: `Country/Territory`, `Report ID`, `Year`, `Report date`, `Validation date`.

---

## 3. Dataful.in Dataset

- **Source**: dataful.in (India Open Data Portal Aggregator)
- **Status**: Ready for file placement in `data/raw/`
- **Granularity & Schema**:
  - Actual schema to be inspected once placed; will not be assumed to match Kaggle or WAHIS.
- **Temporal & Spatial Coverage**: India (National, State, District levels).

---

## Summary of Granularity Differences

| Dataset Source | Granularity | Key Identification Units | Use Case in Project |
|---|---|---|---|
| **Kaggle Animal Disease** | Individual animal case | Single animal + 5 clinical symptoms | Micro-level anomaly detection & severity classification (Phase 1) |
| **WAHIS (WOAH)** | Regional aggregate census & reports | Country / Species Category + Year | Macro-level contextual features joined by species key (Phase 2) |
| **Dataful.in** | State / District aggregates | Indian State/District + Reporting Period | Regional localization & India-specific livestock surveillance (Phase 3) |
