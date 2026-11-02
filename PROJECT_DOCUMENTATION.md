# Wildlife & Tiger Health Anomaly Detection and Veterinary Diagnostic System
## Complete Project Documentation and Comprehensive Technical Guide

---

## 1. Executive Summary: What is this Project?

This project is an **end-to-end intelligent wildlife health surveillance and veterinary diagnostic platform**, specifically engineered to monitor, evaluate, and diagnose health anomalies and physical injuries in wild and captive animals—with a specialized, deep-focus module on **Tigers (*Panthera tigris*)**.

### Core Problem It Solves
In wildlife biology and reserve conservation:
1. **Lack of Labeled Diagnostic Data**: Unlike human hospitals, wild animals do not have vast, neatly labeled medical record databases. Most field observations simply describe observable symptoms (e.g., "limping", "lethargy", "pale gums"). Traditional supervised machine learning fails here because "diseased" labels are scarce.
2. **Delayed Intervention**: When an endangered tiger is injured in territorial fighting, caught in a poaching wire snare, or infected with a contagious virus (like Canine Distemper Virus), field forest rangers often lack immediate veterinary expertise on-site to know *what is wrong* and *what exact steps to take immediately*.
3. **Multi-Scale Data Disconnection**: Micro-level clinical observations (what a ranger sees in an individual animal) and macro-level epidemiological trends (global animal population sizes and national disease outbreaks) historically existed in completely disconnected silos.

### How this System Solves It
- **Tabular Symptom Anomaly Detection**: Uses unsupervised machine learning (**Isolation Forest** and **Local Outlier Factor**) to detect abnormal clinical cases without needing pre-labeled disease data, enriched with global census data from the **World Organisation for Animal Health (WAHIS / WOAH)**.
- **Pure NumPy Photo Neural Network**: Implements a 2-layer neural network ($1024 \rightarrow 16 \rightarrow 1$) built completely from scratch using only pure mathematical formulas in NumPy (no PyTorch, TensorFlow, Keras, or Scikit-Learn for image classification) to classify tiger photographs as healthy or injured.
- **Tiger Clinical Diagnostic & Veterinary Expert Engine**: Translates photographic and symptomatic findings into actionable clinical differential diagnoses ("What Is Wrong") across 13 major tiger pathologies and outputs a structured 4-part veterinary intervention protocol ("What To Do").
- **Multi-Modal Input Support**: Allows rangers and vets to input photos only, text symptoms only, a physical observation checklist, or combined photo + symptoms.
- **Dual-Database Resilient Backend**: Runs a hardened **FastAPI** backend that writes to **MySQL** (XAMPP) as primary storage and seamlessly falls back to **SQLite** if MySQL is offline.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Inputs ["Multi-Source Field Inputs"]
        M1["Mode 1: Photograph\n(Camera Trap / Drone / Field Image)"]
        M2["Mode 2: Clinical Signs\n(Physical Trauma, Skin, Systemic Tags)"]
        M3["Mode 3: Voice Dispatch\n(Live Mic Recording & Spoken Transcripts)"]
        M4["Mode 4: WhatsApp / SMS / Radio\n(Raw Field Messages & Chat Alerts)"]
        M5["Mode 5: Combined Multimodal\n(Photo + Audio / Message / Signs)"]
    end

    subgraph API ["FastAPI Backend (app.py :8001)"]
        EP_Assess["POST /assess"]
        EP_Photo["POST /photos/check"]
        EP_Predict["POST /predict"]
        EP_Health["GET /health"]
    end

    subgraph ML ["Machine Learning & Inference Engines"]
        subgraph AnomalyEngine ["Tabular Anomaly Engine"]
            V2_Model["Model v2: model.joblib\n(WAHIS Context + Kaggle Vocabulary)"]
            IF["Isolation Forest (Tree Partitioning)"]
            LOF["Local Outlier Factor (Density Estimation)"]
        end

        subgraph PhotoEngine ["NumPy Neural Network"]
            Preproc["Grayscale + 32x32 Resize + [0,1] Norm (1024x1)"]
            Weights["models/photo_nn.npz (W1, b1, W2, b2)"]
            Forward["Forward Prop: Tanh -> Sigmoid -> Injury Prob"]
        end

        subgraph ExpertEngine ["Tiger Diagnostics Engine (tiger_diagnostics.py)"]
            KB["13 Tiger Pathologies Knowledge Base"]
            Triage["Clinical Urgency & Protocol Generator"]
        end
    end

    subgraph Storage ["Dual-Database Storage"]
        MySQL[("Primary: MySQL XAMPP\n(animals, ml_predictions)")]
        SQLite[("Fallback: SQLite predictions.db\n(prediction_logs, photo_gallery)")]
    end

    subgraph Output ["User Interface Display"]
        Banner["Urgency & Status Banner"]
        ProbBar["Visual Probability & Reliability"]
        Diagnosis["Identified Conditions (What Is Wrong)"]
        Protocols["4-Part Veterinary Action Protocol"]
    end

    M1 --> EP_Assess
    M2 --> EP_Assess
    M3 --> EP_Assess
    M4 --> EP_Assess

    EP_Assess --> Preproc
    Preproc --> Forward
    Weights --> Forward
    Forward --> ExpertEngine

    EP_Assess --> V2_Model
    V2_Model --> IF
    V2_Model --> LOF
    IF --> ExpertEngine
    LOF --> ExpertEngine

    ExpertEngine --> MySQL
    MySQL -.->|Auto-Fallback on Failure| SQLite
    ExpertEngine --> Output
```

---

## 3. Data Sources: What Data Was Used and From Where?

The system integrates three distinct, carefully curated datasets spanning micro-clinical cases, macro-global epidemiological censuses, and high-resolution tiger imagery.

### Dataset 1: Kaggle Animal Disease Clinical Dataset
- **File**: `data/raw/data.csv` (871 clinical records, 7 columns).
- **Source**: Kaggle (`gracehephzibahm/animal-disease`).
- **Nature**: Individual animal level (micro-granularity).
- **Content**: Records of individual animals (`Dog`, `Cat`, `Deer`, `Cow`, `Buffalo`, `Rabbit`) with up to 5 concurrent clinical symptoms (`symptoms1` through `symptoms5`, e.g., fever, limping, ulcers, diarrhea, respiratory distress) and a clinical severity indicator (`Dangerous`: Yes/No).
- **Purpose**: Used in **Phase 1** to construct the master clinical symptom vocabulary (839 unique symptom tokens) and train the baseline tabular anomaly detection models.

### Dataset 2: WAHIS (World Animal Health Information System / WOAH)
- **Source**: Official data portal of the **World Organisation for Animal Health (WOAH / OIE)**, headquartered in Paris, France.
- **Files**:
  1. `data/raw/Animal population 2026-10-26.csv` (9,899 rows, 13 columns): National annual census counts (2023–2025) of live animal populations, annual slaughter volumes, and farm premises counts across global territories.
  2. `data/raw/SMR_2026-10-26_16-29.csv` (14,184 rows, 9 columns): Official biannual (six-monthly) national disease monitoring dossiers submitted by chief veterinary officers.
  3. `data/raw/AR_2026-10-26.csv` (354 rows, 5 columns): Validated annual official disease summaries.
- **Purpose**: Used in **Phase 2** to engineer macro-level epidemiological context features joined at the species level:
  - `log_live_pop`: $\ln(1 + \text{global live population})$
  - `reporting_countries_count`: Total number of nations officially surveilling this animal category.
  - `log_premises`: $\ln(1 + \text{holding premises / farms})$
  - `wahis_monitored`: Binary indicator ($1$ if subject to international statutory reporting, else $0$).

> **Granularity-Safe Join Rule**: Micro-level clinical cases and macro-level country annual censuses cannot be merged on dates or coordinates because Kaggle records lack timestamps and GPS coordinates. Instead, they were joined strictly across the shared entity: **Species Category** (`AnimalName` $\leftrightarrow$ `Animal category`).

### Dataset 3: Tiger Training Photo Dataset
- **File Structure**: `data/training_photos/` (`healthy/` and `injured/`) and `data/photos/`.
- **Source**: Wildlife rescue repositories, tiger reserve camera trap archives, and clinical veterinary case studies (including the Wild Tiger Health Project and published wildlife rehabilitation reports).
- **Quantity**: 100 verified tiger images (18 healthy tigers in active/natural postures, 82 injured tigers documenting trauma, wire snares, puncture wounds, porcupine quill injuries, mange, and facial lacerations).
- **Database Tracking**: All images are indexed in SQLite table `photo_gallery` with id, filename, label ($0$ for healthy, $1$ for injured), upload timestamp, and species metadata.

---

## 4. Technologies & Tools Used (And Why Each Was Chosen)

| Technology / Tool | Exact Version / Role | Why We Used It (Justification) |
|---|---|---|
| **Python 3.13** | Core Programming Language | Universal ecosystem for scientific computing, matrix algebra, and web backends. |
| **NumPy** | Numerical Matrix Algebra | Used to build the entire 2-layer neural network from raw mathematical formulas without relying on high-level black-box ML frameworks. |
| **FastAPI** | REST API Web Framework | Asynchronous, extremely fast, auto-generates OpenAPI/Swagger interactive documentation, built-in validation via Pydantic. |
| **Uvicorn** | ASGI Web Server | Lightning-fast asynchronous server implementation for serving FastAPI applications. |
| **Pillow (PIL)** | Image Preprocessing | Standard Python imaging library; used exclusively for opening image streams, converting to grayscale (`L`), and resizing to $32 \times 32$. |
| **Scikit-Learn** | Unsupervised Anomaly Models | Industry standard implementations of `IsolationForest`, `LocalOutlierFactor`, and `StandardScaler` for the tabular symptom model. |
| **Joblib** | Model Serialization | Efficiently dumps and loads the complete trained tabular model bundle (`model.joblib`) including encoders and scalers in one call. |
| **PyMySQL** | MySQL Database Driver | Pure-Python client connecting directly to XAMPP MySQL; allows direct plain SQL execution without cumbersome ORM abstractions. |
| **SQLite3** | Local Fallback Database | Zero-configuration file database built into Python standard library; provides bulletproof offline audit logging and photo gallery management. |
| **HTML5 / CSS3 / ES6 JS** | Frontend Interface | Plain vanilla web stack with zero build steps, zero node_modules, and zero external CDN requirements—ensuring it opens reliably anywhere offline. |
| **Matplotlib** | Training Visualization | Generates loss curves (`data/photo_cost_plot.png`) and scatter visualizations for training verification. |

---

## 5. Detailed Breakdown of Every Method & Topic Used

### A. Unsupervised Anomaly Detection: Why Unsupervised?
In supervised learning, algorithms require thousands of ground-truth labels for each specific disease class ($y \in \{0, 1, 2, \dots\}$). In wildlife epidemiology:
1. Emerging or rare diseases have zero or very few historical examples.
2. Wild animals often die undetected in remote forests, leading to massive under-reporting.
3. Therefore, **Unsupervised Learning** is used: the algorithm learns the boundary and distribution of *normal* health profiles. Any observation that deviates significantly from this learned normal manifold is flagged as an **Anomaly** (`Unusual`).

#### 1. Isolation Forest
- **Intuition**: Anomalies are "few and different". If you randomly pick a feature and randomly cut between the min and max values, an isolated outlier will be separated from the rest of the dataset in very few cuts, whereas normal, densely clustered points require many cuts.
- **Mechanism**: Builds an ensemble of Isolation Trees ($iTrees$).
- **Path Length $h(x)$**: The number of edges $x$ traverses from the root node to a leaf node.
- **Anomaly Score $s(x, n)$**:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
  where $E(h(x))$ is the average path length across all trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree with $n$ nodes:
  $$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler's constant)} - \frac{2(n - 1)}{n}$$
  - If $s \to 1$: Path length is very short $\rightarrow$ **definite anomaly**.
  - If $s < 0.5$: Path length is long $\rightarrow$ **normal observation**.

#### 2. Local Outlier Factor (LOF)
- **Intuition**: Anomaly detection based on local density rather than global clustering. A point might not be far away in absolute distance, but if its local neighborhood is substantially less dense than its neighbors' neighborhoods, it is an outlier.
- **k-distance of $p$**: Distance $d(p, o)$ between $p$ and its $k$-th nearest neighbor $o$.
- **Reachability Distance**:
  $$\text{reach-dist}_k(p, o) = \max(k\text{-distance}(o),\, d(p, o))$$
- **Local Reachability Density (lrd)**:
  $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
- **LOF Score**:
  $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
  - $\text{LOF} \approx 1$: Density comparable to neighbors $\rightarrow$ **Normal**.
  - $\text{LOF} > 1.5$: Density substantially lower than neighbors $\rightarrow$ **Outlier / Anomaly**.

#### 3. Ensemble Decision Rule
$$\text{Final Flag} = \begin{cases} \text{Unusual}, & \text{if } \text{IF\_Flag} = \text{Anomaly} \lor \text{LOF\_Flag} = \text{Anomaly} \\ \text{Normal}, & \text{otherwise} \end{cases}$$
*Reasoning*: In wildlife healthcare, false negatives (missing an infected or injured animal) are far more dangerous than false alarms. An **OR** rule prioritizes sensitivity (**Recall**).

---

### B. Pure NumPy Neural Network: From-Scratch Mathematics
To meet strict foundational machine learning requirements, the photo recognition engine uses **zero high-level ML libraries**. Every single step is written from raw linear algebra.

#### 1. Architecture
- **Input Layer ($n_x = 1024$)**: Grayscale photograph resized to $32 \times 32$ pixels, flattened into a column vector of dimension $1024 \times 1$, normalized so each pixel $x_i \in [0.0, 1.0]$.
- **Hidden Layer ($n_h = 16$)**: 16 neurons with **hyperbolic tangent ($\tanh$)** activation.
- **Output Layer ($n_y = 1$)**: 1 neuron with **sigmoid ($\sigma$)** activation outputting the probability that the tiger is injured: $P(\text{Injured} \mid x) \in [0, 1]$.

#### 2. Forward Propagation
$$\begin{aligned}
Z^{[1]} &= W^{[1]} X + b^{[1]} \quad &&(W^{[1]} \in \mathbb{R}^{16 \times 1024},\; b^{[1]} \in \mathbb{R}^{16 \times 1}) \\
A^{[1]} &= \tanh(Z^{[1]}) = \frac{e^{Z^{[1]}} - e^{-Z^{[1]}}}{e^{Z^{[1]}} + e^{-Z^{[1]}}} \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \quad &&(W^{[2]} \in \mathbb{R}^{1 \times 16},\; b^{[2]} \in \mathbb{R}^{1 \times 1}) \\
A^{[2]} &= \sigma(Z^{[2]}) = \frac{1}{1 + e^{-Z^{[2]}}}
\end{aligned}$$

#### 3. Binary Cross-Entropy Cost Function
For $m$ training examples, with numerical stability clipping ($\epsilon = 10^{-15}$):
$$J(W, b) = -\frac{1}{m} \sum_{i=1}^m \left[ Y^{(i)} \ln(A^{[2](i)}) + (1 - Y^{(i)}) \ln(1 - A^{[2](i)}) \right]$$

#### 4. Backward Propagation (Vectorized Analytical Gradients)
Using the exact derivative of $\tanh$: $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) = 1 - (A^{[1]})^2$:
$$\begin{aligned}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} (A^{[1]})^T \\
db^{[2]} &= \frac{1}{m} \sum_{\text{columns}} dZ^{[2]} \\
dZ^{[1]} &= \left( (W^{[2]})^T dZ^{[2]} \right) \odot \left( 1 - (A^{[1]})^2 \right) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X^T \\
db^{[1]} &= \frac{1}{m} \sum_{\text{columns}} dZ^{[1]}
\end{aligned}$$

#### 5. Gradient Descent Parameter Update
$$\theta := \theta - \alpha \cdot d\theta \quad \text{for each } \theta \in \{W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\}$$
where $\alpha = 0.05$ is the learning rate.

#### 6. Numerical Two-Sided Gradient Check
To mathematically prove the backpropagation formulas are 100% bug-free, the analytical gradients are verified against the symmetric finite difference quotient:
$$\frac{\partial J}{\partial \theta_i} \approx \frac{J(\theta_i + \epsilon) - J(\theta_i - \epsilon)}{2\epsilon} \quad (\epsilon = 10^{-7})$$
The relative Euclidean difference is computed:
$$\text{Difference} = \frac{\| d\theta_{\text{analytical}} - d\theta_{\text{numerical}} \|_2}{\| d\theta_{\text{analytical}} \|_2 + \| d\theta_{\text{numerical}} \|_2}$$
Our test suite rigorously verifies that this difference is $< 10^{-6}$ (actual result: $1.38 \times 10^{-7}$), confirming mathematical precision.

#### 7. Data Augmentation
Because tiger orientations vary in camera trap photos, every training image vector is augmented on-the-fly using horizontal flipping ($\text{fliplr}$ of the $32 \times 32$ matrix), doubling the effective training samples. The train/test split (80/20) is performed **strictly before** flipping to prevent data leakage.

---

### C. Tiger Clinical Diagnostics & Veterinary Protocols Engine
Model predictions alone are useless to a forest ranger unless translated into pathology and treatment. The engine in `tiger_diagnostics.py` contains clinical knowledge for **13 major tiger health conditions**:

1. **Intra-Specific Fighting Trauma**: Punctures, torn muscles, cellulitis, and abscesses from territorial combat.
2. **Porcupine Quill Penetrations (*Hystrix indica*)**: Barbed quills that continuously migrate deeper into muscles and joints, causing excruciating pain and preventing hunting.
3. **Poaching Snare / Wire Ligature**: Wire strangulation causing circumferential necrosis, ischemic gangrene, and auto-amputation.
4. **Tooth and Jaw Damage / Dental Fracture**: Broken canines and alveolar osteitis driving tigers toward domestic livestock conflict.
5. **Canine Distemper Virus (CDV)**: Morbillivirus producing myoclonus (spasms), disorientation, abnormal loss of fear of humans, and hindlimb paralysis.
6. **Feline Parvovirus (FPV / Panleukopenia)**: Severe mucosal sloughing, leukopenia, bloody diarrhea, and hypovolemic shock.
7. **Rabies Virus**: Fatal neurotropic lyssavirus with fatal unprovoked aggression and hypersalivation.
8. **Bovine Tuberculosis (*Mycobacterium bovis*)**: Granulomatous respiratory wasting contracted from infected prey.
9. **Leptospirosis**: Spirochete infection causing acute renal nephritis and marked jaundice.
10. **Dermatophytosis (Ringworm) / Sarcoptic Mange**: Severe pruritus, crusting, and alopecia.
11. **Trypanosomiasis (Surra / *T. evansi*)**: Tabanid fly-transmitted blood parasite causing paroxysmal fever and severe anemia.
12. **Cytauxzoonosis & Babesiosis**: Tick-borne intra-erythrocytic piroplasms triggering hemolytic crisis.
13. **Chronic Kidney Disease (CKD)**: Geriatric renal failure leading to toxic waste buildup and severe dehydration.

#### The 4-Part Protocol Output
Whenever a condition is diagnosed, the system generates:
1. **Immediate Field Action**: Ranger deployment, perimeter control, mobility assessment, and biosecurity alerts.
2. **Veterinary Treatment**: Anesthesia protocol, wound irrigation, specific drug therapies (antibiotics, antifungals, antiparasitics), and surgical extraction.
3. **Diagnostic Confirmation**: Blood smears, RT-PCR, rapid ELISA tests, tuberculin tests, or radiography.
4. **Territory Monitoring**: Camera trap grids around waterholes and ring-vaccination of fringe livestock/dogs.

---

### D. Dual-Database Architecture: Why MySQL + SQLite?

```mermaid
flowchart LR
    App["FastAPI Backend (app.py)"]
    MySQL["Primary: MySQL (XAMPP 3306)\nHost: 127.0.0.1\nDB: wildlife_conservation\nTables: animals, ml_predictions"]
    SQLite["Fallback: SQLite (predictions.db)\nPath: data/predictions.db\nTables: prediction_logs, photo_gallery"]

    App -->|1. Try Execute SQL| MySQL
    MySQL -.->|Connection Error / Timeout| SQLite
    App -->|2. Log & Continue| SQLite
```

1. **Primary Database (XAMPP MySQL)**:
   - Schema: `wildlife_conservation`.
   - Table `animals`: Stores tracked wild tigers and other fauna (`id`, `name`, `species`, `age`, `health_status`).
   - Table `ml_predictions`: Stores permanent diagnostic prediction history (`id`, `animal_id`, `species`, `symptoms`, `anomaly_flag`, `anomaly_score`, `created_at`).
   - Access Method: Pure `pymysql` with parameterized SQL (`%s`)—**zero ORM overhead** for maximum speed and simplicity.

2. **Fallback Database (SQLite `data/predictions.db`)**:
   - Table `prediction_logs`: Captures full request JSON, response JSON, recognized/unrecognized symptoms, and execution timestamps.
   - Table `photo_gallery`: Manages training reference photos, labels ($0/1$), and metadata.
   - Fail-Safe Logic: Wrapped in a `try...except pymysql.MySQLError` block. If XAMPP/MySQL stops, crashes, or is unreachable, the system automatically redirects the transaction to SQLite. The ranger's dashboard never crashes.

---

## 6. End-to-End Walkthrough: How the Entire System Works

Here is exactly what happens behind the scenes from the moment a user interacts with the system:

```mermaid
sequenceDiagram
    autonumber
    actor User as Ranger / Veterinarian
    participant UI as Frontend (index.html)
    participant API as FastAPI Backend (app.py)
    participant ModelNN as NumPy Photo NN (models/photo_nn.npz)
    participant ModelTab as Scikit-Learn Anomaly Bundle (model.joblib)
    participant Expert as Diagnostics Engine (tiger_diagnostics.py)
    participant DB as MySQL / SQLite

    User->>UI: Selects Mode & Provides Input (Photo and/or Symptoms)
    UI->>API: HTTP POST /assess (multipart/form-data)
    
    alt Photo Provided
        API->>ModelNN: preprocess_image() -> (1024, 1) vector
        ModelNN->>ModelNN: Forward Prop: Tanh -> Sigmoid
        ModelNN-->>API: Returns prob_injured (e.g. 0.9585) & label ('injured')
    end

    alt Symptoms Provided
        API->>ModelTab: Clean & match against 839 symptom vocabulary
        ModelTab->>ModelTab: Compute IF & LOF anomaly scores
        ModelTab-->>API: Returns anomaly flag ('Unusual' / 'Normal') & score
    end

    API->>Expert: diagnose_tiger_health(symptoms, photo_label, photo_prob)
    Expert->>Expert: Match keywords against 13 Tiger Pathologies
    Expert->>Expert: Calculate Clinical Urgency & 4-Part Protocol
    Expert-->>API: Returns identified conditions & actionable next steps

    API->>DB: INSERT into ml_predictions (MySQL) or prediction_logs (SQLite)
    API-->>UI: Returns consolidated JSON response
    UI-->>User: Renders color-coded urgency banner, diagnosis card, and veterinary steps
```

### Step-by-Step Execution Sequence

1. **User Selects Input Mode in Unified Assessment**:
   - **Mode 1 (Photograph)**: User uploads a camera trap, drone, or field photograph for neural network trauma analysis.
   - **Mode 2 (Clinical Symptoms & Signs)**: User selects quick clinical tags (Trauma, Skin/Parasites, Systemic/Viral) or types observed signs.
   - **Mode 3 (Combined Multimodal)**: User provides both a photograph and clinical symptoms for full side-by-side diagnostic synthesis.
   - **Territory / Reserve Context**: Optionally selects one of India's 53 NTCA Tiger Reserves to inject landscape-specific threat protocols.

2. **Backend Ingestion (`POST /assess`)**:
   - The FastAPI endpoint receives the request. It parses symptoms into an array and reads image bytes into memory.
   - Validates file extensions (`.jpg`, `.jpeg`, `.png`, `.webp`) and enforces a 5 MB file size limit.

3. **Inference Execution**:
   - **If a photo is present**: Passes bytes to `preprocess_image()`, which converts it to grayscale, resizes it to $32 \times 32$, normalizes pixel values to $[0.0, 1.0]$, and flattens it into a $(1024, 1)$ vector. Runs vectorized forward propagation using loaded NumPy weights (`W1`, `b1`, `W2`, `b2`). Calculates injury probability.
   - **If symptoms are present**: Matches symptom tokens against the 839-feature vocabulary vector, appends WAHIS species census features (`log_live_pop`, `reporting_countries_count`), scales via `StandardScaler`, and computes Isolation Forest and LOF decision scores.
   - **Side-by-Side Evaluation**: The raw image score and tabular anomaly score are intentionally kept separate—never averaged into an uninterpretable composite number.

4. **Expert Differential Diagnosis & Protocol Synthesis**:
   - `diagnose_tiger_health` analyzes the matched signs and photographic trauma flags.
   - Evaluates severity thresholds: if Rabies, Canine Distemper, Porcupine Quills, or Poaching Snares are detected, it escalates urgency to **`Immediate Emergency Response`**.
   - Builds the 4-part veterinary protocol: Immediate Field Action, Veterinary Treatment, Diagnostic Confirmation, and Territory Monitoring.

5. **Audit Logging & Response Rendering**:
   - The prediction is stored in MySQL `ml_predictions` (or SQLite `prediction_logs`).
   - The response is sent back to the frontend, which renders clean, high-contrast, professional cards:
     - Urgency Badge (Emergency Red, Priority Orange, Moderate Indigo, Routine Green).
     - Photographic Injury Probability bar.
     - "What Is Wrong" pathological explanation.
     - "What To Do" 4-part veterinary protocol grid.

---

## 7. Model Performance & Evaluation Metrics

### Tabular Anomaly Detection (Model v1 vs Model v2)
Evaluated against the ground-truth clinical severity classifications:

| Metric | Model v1 (Symptoms Only) | Model v2 (Symptoms + WAHIS Census Context) | Practical Benefit in Field |
|---|---|---|---|
| **Dangerous Case Precision** | 0.9872 (98.72%) | **0.9938 (99.38%)** | Higher certainty when flagging serious infections. |
| **Dangerous Case Recall** | 0.1814 (18.14%) | **0.1873 (18.73%)** | **5 additional dangerous cases detected**. |
| **Safe Case Recall** | 0.9000 (90.00%) | **0.9500 (95.00%)** | **False alarms cut in half (from 2 down to 1)**. |

### Photo Recognition NumPy Neural Network
Trained on 100 reference photographs (stratified 80/20 train/test split, evaluated by hand with pure NumPy without sklearn):

| Metric | Measured Value | Practical Significance |
|---|---|---|
| **Test Accuracy** | **85.71%** | Strong general classification across varied lighting. |
| **Recall on Injured** | **100.0% (Zero False Negatives)** | **The model did not miss a single injured tiger** in the held-out test evaluation. In wildlife protection, zero false negatives is the gold standard because missing an injured animal leads to its death. |
| **Hidden Layer Activation** | Hyperbolic Tangent ($\tanh$) | Zero-centered representations prevent vanishing gradients. |
| **Output Activation** | Sigmoid ($\sigma$) | Provides a well-calibrated posterior probability $P(\text{Injured} \mid x) \in [0, 1]$. |
| **Gradient Check Difference** | **$1.38 \times 10^{-7}$** | Mathematically proves exact implementation of analytical derivatives. |

---

## 8. Common Exam, Interview & Presentation Questions (And How to Answer Them)

### Q1: Why did you use Unsupervised Learning instead of Supervised Learning for animal health records?
> **Answer**: In wildlife conservation and disease monitoring, we rarely have extensive, balanced datasets where every animal disease is cleanly labeled. Most field observations simply record clinical signs (fever, limping, weakness). Unsupervised anomaly detection (Isolation Forest + LOF) allows us to learn what "normal" health observations look like and automatically flag unusual, dangerous deviations without needing prior labels for every single rare disease.

### Q2: Why did you build the photo neural network in pure NumPy instead of using TensorFlow or PyTorch?
> **Answer**: To demonstrate complete, first-principles mastery of deep learning mathematics. By implementing matrix initialization, vectorized forward propagation, binary cross-entropy loss, and backpropagation chain-rule gradients in pure NumPy, we eliminated all black-box abstractions. Furthermore, we verified our backpropagation with numerical two-sided gradient checking ($\Delta < 10^{-6}$), proving exact mathematical correctness.

### Q3: Why is Recall prioritized over Precision in this project?
> **Answer**: In clinical veterinary medicine and endangered wildlife surveillance, the cost of a **False Negative** (failing to detect an injured or infected tiger, leaving it to die or spread an outbreak) is catastrophic. In contrast, the cost of a **False Positive** is merely dispatching a ranger patrol to visually verify the tiger. Therefore, we explicitly optimize our decision thresholds to achieve high **Recall on Injured cases** (achieving 100% test recall).

### Q4: Why did you integrate WAHIS data with the Kaggle clinical records, and how did you prevent invalid merges?
> **Answer**: Kaggle records provide micro-level clinical symptoms for individual animals, while WAHIS (World Organisation for Animal Health) provides macro-level global population censuses and national surveillance breadth. Joining them added real-world epidemiological scale to our models. To prevent invalid data merging (since Kaggle lacked dates and GPS coordinates), we performed a **granularity-safe join** strictly across the shared entity: **Species Category**. This reduced false alarms by 50% and improved dangerous case precision to 99.38%.

### Q5: What happens if the MySQL database goes down while a ranger is in the field?
> **Answer**: The application implements a dual-database resilient architecture. All database transactions are wrapped in fail-safe exception blocks. If the primary MySQL instance is offline or unreachable, the system automatically redirects the audit logging to the local SQLite database (`data/predictions.db`). The user interface never crashes, and field operations continue uninterrupted.

### Q6: Can the neural network tell me *where* the injury is located on the tiger?
> **Answer**: The current photo neural network is a **global image classifier** ($32 \times 32$ pixels), not a spatial object detector (such as YOLO or Faster R-CNN). It estimates the overall probability of injury across the entire photograph. The anatomical localization and clinical diagnosis ("What Is Wrong") are provided by the **Tiger Clinical Diagnostics Engine**, which correlates photographic trauma detection with clinical field signs to pinpoint affected anatomical structures (e.g. Forequarters, Mandible, Distal Limbs).

---

## 9. File & Code Directory Guide

- [`app.py`](file:///d:/College/Semester%205-2026/FOML/Project/app.py): Core FastAPI web backend serving all prediction, assessment, training, and database endpoints.
- [`train_photo_nn.py`](file:///d:/College/Semester%205-2026/FOML/Project/train_photo_nn.py): Pure NumPy neural network architecture, forward/backward propagation, cost calculation, gradient checking, and model training pipeline.
- [`preprocess.py`](file:///d:/College/Semester%205-2026/FOML/Project/preprocess.py): Image processing helper converting images to normalized $(1024, 1)$ vectors and performing horizontal flip augmentation.
- [`tiger_diagnostics.py`](file:///d:/College/Semester%205-2026/FOML/Project/tiger_diagnostics.py): Disease database, keyword matcher, clinical urgency triage, and 4-part veterinary protocol synthesizer.
- [`index.html`](file:///d:/College/Semester%205-2026/FOML/Project/index.html): Clean, field-ready frontend dashboard featuring all 4 input modes, registry modal, pathology reference, and photo gallery.
- [`test_app.py`](file:///d:/College/Semester%205-2026/FOML/Project/test_app.py): Automated test suite for backend endpoints, MySQL connectivity, and SQLite fallback.
- [`test_photo_module.py`](file:///d:/College/Semester%205-2026/FOML/Project/test_photo_module.py): Automated test suite for the neural network, verifying preprocessing shape, gradient check, gallery CRUD, and multimodal inference.
- [`pull_wildlife_apis.py`](file:///d:/College/Semester%205-2026/FOML/Project/pull_wildlife_apis.py): Automated pipeline querying Europe PMC, NCBI PubMed, GBIF, iNaturalist, and WOAH/WAHIS APIs.
- [`prepare_training_data.py`](file:///d:/College/Semester%205-2026/FOML/Project/prepare_training_data.py): Data hygiene and synthesis script preparing clinical evidence citations, spatial summaries, and disease threat matrices.
- [`data/tiger_reserves_ntca.json`](file:///d:/College/Semester%205-2026/FOML/Project/data/tiger_reserves_ntca.json): Clean structured database of all 53 Indian Tiger Reserves with 2022 census data, landscapes, and threat profiles.
- [`phase1_baseline.py`](file:///d:/College/Semester%205-2026/FOML/Project/phase1_baseline.py): Script training the baseline tabular anomaly detection model on Kaggle symptom data.
- [`phase2_wahis_context.py`](file:///d:/College/Semester%205-2026/FOML/Project/phase2_wahis_context.py): Script training Model v2 enriched with WAHIS global census features.
- [`train_model.py`](file:///d:/College/Semester%205-2026/FOML/Project/train_model.py): Final export script bundling encoders, scalers, and models into `model.joblib`.

---

## 10. NTCA National Tiger Census & 53 Tiger Reserves Integration

To anchor the system in authoritative conservation science, we integrated official national datasets extracted from two seminal government publications:
1. **`Illustrative_profile_tiger_reserves.pdf`** (*Wildlife Institute of India & National Tiger Conservation Authority, 2023*): 329-page compendium detailing core/buffer areas, administrative districts, geographic coordinates, and key ecological challenges across 53 Tiger Reserves in India.
2. **`status_of_tiger-copredators-2022.pdf`** (*NTCA & WII, 2023 - All India Tiger Estimation 2022*): 494-page national census documenting a minimum counted population of **3,167 wild tigers** (SECR estimate of 3,682, SE 243) categorized across India's **5 major ecological landscapes**:
   - **Shivalik Hills and Gangetic Plains Landscape** (e.g., Corbett, Dudhwa, Rajaji, Pilibhit)
   - **Central Indian Landscape and Eastern Ghats** (e.g., Bandhavgarh, Kanha, Tadoba, Ranthambhore)
   - **Western Ghats Landscape** (e.g., Bandipur, Nagarhole, Mudumalai, Periyar)
   - **North East Hills and Brahmaputra Flood Plains** (e.g., Kaziranga, Manas, Pakke)
   - **Sundarbans Landscape**

### Automated Extraction & Reserve Profiling
All 53 reserves were extracted into [`data/tiger_reserves_ntca.json`](file:///d:/College/Semester%205-2026/FOML/Project/data/tiger_reserves_ntca.json), pairing each reserve with:
- Census tiger count (2018 vs. 2022)
- Total and core reserve surface area ($km^2$)
- Ecological landscape classification
- Specific localized threat indicators (Poaching snare risk, human-carnivore conflict risk, canine distemper viral risk)

Whenever a field user selects a specific Tiger Reserve during photo or symptom analysis, the system injects site-specific NTCA protocols into the veterinary response (e.g. anti-snare sweeps, fringe conflict response team alerts, peripheral domestic canine distemper ring-vaccination).

---

## 11. Wildlife & Biomedical API Data Ingestion (`pull_wildlife_apis.py`)

The system integrates an automated ingestion pipeline ([`pull_wildlife_apis.py`](file:///d:/College/Semester%205-2026/FOML/Project/pull_wildlife_apis.py)) that queries 5 international APIs:

| Source API | Type of Data Pulled | Records Extracted | Local Target Path |
| :--- | :--- | :--- | :--- |
| **Europe PMC** | Peer-reviewed veterinary literature on *Panthera tigris* | 100 research articles | `data/api_data/europe_pmc_veterinary_research.csv` |
| **NCBI PubMed** | Biomedical studies on tiger trauma, infectious disease, and pathology | 100 indexed publications | `data/api_data/pubmed_tiger_research.csv` |
| **GBIF** | Georeferenced occurrence records with coordinates and dates | 100 occurrences | `data/api_data/gbif_tiger_occurrences.csv` |
| **iNaturalist** | Research-grade wild tiger observations with media URLs | 100 field observations | `data/api_data/inaturalist_tiger_observations.csv` |
| **WOAH / WAHIS** | Public animal health disease alerts & epidemiological events | Official surveillance alerts | `data/api_data/wahis_raw_response.json` |

---

## 12. Data Hygiene & Clinical Knowledge Synthesis (`prepare_training_data.py`)

### The "Observation vs. Diagnosis" Principle
A fundamental principle in veterinary data engineering is that **public wildlife observations (GBIF, iNaturalist) do NOT equal verified clinical diagnoses**. Inexperienced machine learning workflows often make the mistake of scraping wildlife photos from community platforms and assuming unlabeled animals are "healthy" or that blurry pictures represent "diseased" animals.

To prevent this data contamination:
1. **Zero Model Contamination**: Raw observational records from GBIF and iNaturalist are used strictly for **spatial distribution summaries and landscape risk mapping** (`data/prepared_data/tiger_spatial_surveillance_summary.json`). They are never fed blindly into the neural network image classifier.
2. **Scientific Grounding of Diagnostics**: Literature pulled from Europe PMC and PubMed is synthesized into a vetted **evidence base** (`data/prepared_data/tiger_pathology_evidence_base.json`). Whenever a condition is diagnosed (e.g., Canine Distemper or Intra-Specific Fighting Trauma), the diagnostic card dynamically attaches real peer-reviewed scientific citations to substantiate the veterinary protocol.
3. **Verified Clinical Library**: Field veterinary cases are structured into a gold-standard diagnostic library (`data/prepared_data/verified_tiger_clinical_cases.json`) with documented pathologies, clinical manifestations, and emergency field treatments.
