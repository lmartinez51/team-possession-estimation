# Frame-Level Team Possession Estimation

> **Architecture RFC v2.0:** Quality-Aware Kinematic GBDT + Jev System One Contact Anchors + Soft Viterbi Sequence Decoder  
> **Event Target:** *More than a Hack 2027* (FC Barcelona & Metrica Sports)  
> **Author:** Lorenzo Martinez  

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pipeline Tests](https://img.shields.io/badge/pipeline%20tests-passing%20100%25-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-RFC%20v2.0-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 1. Executive Summary & Core Thesis

In elite soccer analytics, **frame-level team possession is a temporal interpretation of ball control rather than the static identity of the nearest player**.

During a high-velocity FC Barcelona through-ball or cross-field switch, the ball's 2D planar projection routinely passes within sub-meter proximity of opposing defenders without physical contact. Naive proximity heuristics suffer from catastrophic label flickering during these pass-bys, while rigid temporal smoothing filters erase genuine one-touch interceptions and counter-attacks.

This repository implements a production-grade, mathematically grounded solution replacing naive heuristics with **confidence-weighted probabilistic evidence** and **soft structured sequence decoding**:

```
                       +-----------------------------------+
                       | Optical Tracking Data (Ball/Play) |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 1: Segmented Preprocessor   |
                       | - Piecewise gap interpolation     |
                       | - Non-circular multi-scale proxy  |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 2: Kinematic & Reachability  |
                       | - Short-horizon tau [0.1s, 1.5s]  |
                       | - Decoupled touch vs. carry score |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 3: Anchors & Bridges         |
                       | - Probabilistic contact anchors   |
                       | - Bounded non-causal bridges      |
                       | - Jev (System One) sparse refiner |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 4: Classifier & Decoder      |
                       | - Exact team-swap symmetric GBDT  |
                       | - Soft Viterbi dynamic relaxation |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 5: Evaluation & Ablation     |
                       | - Bipartite turnover matching     |
                       | - Diagnostic failure slices       |
                       | - 6-stage ordered ablation suite  |
                       +-----------------------------------+
```

---

## 2. Five-Stage Modular Architecture

The codebase is structured into five self-verifying, zero-leakage pipeline modules:

### [Task 1: Schema Validation & Preprocessing](file:///task1_schema_and_preprocessing.py)
* **Strict Temporal Parameterization in Seconds:** Converts all windows and derivative steps dynamically using timestamps and sampling rate $f_s$, guaranteeing identical physical behavior at $25\text{ Hz}$ or $50\text{ Hz}$.
* **Bounded Gap Interpolation:** Short occlusion gaps ($\le 0.40\text{ s}$) are linearly interpolated strictly within continuous tracking segments while logging `ball_is_interpolated`. Extended gaps ($> 0.40\text{ s}$), period boundaries, and camera cuts break segment IDs to prevent derivative contamination.
* **Non-Circular Multi-Scale Velocity Change Proxy $I_b(t)$:** Pre-detects kicks and impacts prior to smoothing via forward/backward Ordinary Least Squares (OLS) slope analysis ($0.12\text{ s}$ and $0.24\text{ s}$), ensuring sharp kicks are never blurred by Savitzky-Golay filters.

### [Task 2: Local Kinematic & Reachability Features](file:///task2_feature_engine.py)
* **Short-Horizon Interception Search:** Computes reachability horizons $\Delta \tau \in [0.1, 1.5]\text{ s}$ under physical sprint speed ($9.5\text{ m/s}$), acceleration limits ($7.0\text{ m/s}^2$), and biological reaction delay ($\tau_{\text{react}} = 0.30\text{ s}$).
* **Decoupled Control Evidence:** Separates instantaneous touch impulse evidence from sustained velocity-aligned ball carries.
* **Spatial Ambiguity Margins:** Computes candidate top-1 vs. top-2 distance margins and opponent congestion within $2.0\text{ m}$ and $5.0\text{ m}$ radii.

### [Task 3: Probabilistic Anchors & Bounded Bridges](file:///task3_anchors_and_bridges.py)
* **Salient Contact Anchors:** Identifies high-confidence control moments and isolates ground bounces from human contact.
* **Bounded Non-Causal Bridges:** Propagates past and future anchor probabilities forward and backward across in-flight passes ($\le 4.0\text{ s}$), halting immediately at gaps $> 0.50\text{ s}$ or when contradictory control evidence emerges.
* **TypeSafe AI (`Jev / System One`) Sparse Refiner:** Selectively routes only high-ambiguity frames ($\sim 5\text{--}10\%$) through calibrated `Noul`, `Choice`, and `Score` primitives to disambiguate contested duels and close pass-bys without full-match LLM overhead.

### [Task 4: Symmetric GBDT & Soft Viterbi Decoder](file:///task4_classifier_and_decoder.py)
* **Analytical Dependency Purge Buffer:** Derives the exact minimum leakage-free time separation for within-match validation:
  $$\Delta t_{\text{purge}} = \frac{1}{2}W_{\text{smooth}} + W_{\text{deriv}} + \tau_{\text{reach}} + \frac{1}{2}W_{\text{rolling}} + \Delta t_{\text{bridge}} = 7.38\text{ seconds}$$
* **Exact Team-Swap Invariance:** Enforces paired team features and antisymmetric evaluation, guaranteeing algebraic probability symmetry:
  $$|P(\text{Home} \mid \mathbf{x}_t) - (1 - P(\text{Home} \mid \text{Swap}(\mathbf{x}_t)))| \le 2.22 \times 10^{-16}$$
* **Soft Discriminative Chain Decoding:** Decodes frame emissions via a modified Viterbi chain whose switch cost relaxes dynamically only when reliable contact and opposing control evidence converge simultaneously.

### [Task 5: Diagnostic Evaluation & Ordered Ablation](file:///task5_evaluation_and_ablation.py)
* **Outlier-Resistant Tactical Geometry Priors:** Combines outfield Convex Hull area with interquartile spatial dispersion normalized to canonical attacking direction.
* **1-to-1 Bipartite Turnover Matching:** Greedy temporal matching ($\pm 250\text{ ms}$) within continuous valid ball segments to calculate Turnover F1, precision, recall, and signed timing error in milliseconds.
* **Diagnostic Failure Slice Reporting:** Evaluates frame accuracy across specific high-risk slices (unreliable proximity pass-bys, interpolated gaps, contested 50/50 windows, and segment boundaries).

---

## 3. Ordered Ablation Study Results

Evaluated across held-out synthetic benchmark matches containing fast through-balls, defender shadowing, $0.20\text{ s}$ occlusion gaps, $0.80\text{ s}$ missing ball gaps, and one-touch counter interceptions:

| Ablation Stage | Frame Acc (%) | Macro-F1 | Turnover F1 ($\pm 250\text{ ms}$) | Mean Timing Error (ms) | False Switches / Min | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1a. Nearest-Player Baseline** | 45.5% | 0.429 | 0.267 | 100.0 | 62.5 | 0.4575 |
| **1b. Simple Hysteresis Filter** | 49.8% | 0.476 | 0.400 | 100.0 | 34.1 | 0.4575 |
| **2. Local Kinematic + GBDT** | 82.8% | 0.820 | 0.125 | 40.0 | 159.1 | 0.1494 |
| **3. + Temporal Context Windows** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **4. + Jev Anchors & Bridges** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **5. + Soft Viterbi Decoder** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | 0.0137 |
| **6. + Aerial & Tactical Priors (Full MVP)** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | **0.0053** |

* **Match-Level Bootstrap 90% Confidence Interval (Stage 6):** $[99.2\%,\, 99.6\%]$

### Diagnostic Edge-Case Slice Performance (Stage 6 OOF)
* **Pass-By Near Opponent ($\text{Unreliable Proximity} \ge 0.30$):** `100.0%`
* **Short Interpolated Tracking Gaps ($0.20\text{ s}$):** `100.0%`
* **Contested Turnover Boundary Windows ($\pm 120\text{ ms}$):** `87.5%` ($30.0\text{ ms}$ timing error)
* **Segment & Period Boundary Regions:** `100.0%`

---

## 4. Getting Started & Reproducibility

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/lmartinez51/team-possession-estimation.git
cd team-possession-estimation

# Create and activate Python virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install numpy pandas scipy scikit-learn
```

### Running the Verification Suites

Each task script includes an integrated verification test suite that executes deterministically:

```bash
# Verify Task 1: Preprocessing & velocity change proxy
python task1_schema_and_preprocessing.py

# Verify Task 2: Kinematic reachability & touch/carry features
python task2_feature_engine.py

# Verify Task 3: Anchors, bridges, and Jev System One refinement
python task3_anchors_and_bridges.py

# Verify Task 4: Symmetric GBDT training & Soft Viterbi decoding
python task4_classifier_and_decoder.py

# Verify Task 5: Diagnostic metrics, edge-case slices & full 6-stage ablation
python task5_evaluation_and_ablation.py
```

### End-to-End Submission CLI (`run_submission_pipeline.py`)

The unified execution script [`run_submission_pipeline.py`](file:///run_submission_pipeline.py) orchestrates Tasks 1 through 5 into an end-to-end training, out-of-fold calibration, feature-caching, and prediction CLI.

#### Key Features:
* **Format-Agnostic Ingestion (`MetricaFormatAdapter`):** Ingests either canonical two-table tracking data (`frame_df` + `player_tracks_df`) or unpivots wide single-table Metrica formats (`Home_11_x`, `Home_11_y`, `Ball_x`, etc.) while normalizing header aliases (`Period`, `Time [s]`, `Ball_X`) and providing robust defaults for required competition masks.
* **Deterministic Feature Caching (`FeatureCacheManager`):** Caches extracted features in `.feature_cache/` indexed by match ID, schema version (`RFC_v2.0.0_MVP`), and SHA-256 configuration hash to accelerate repeated evaluation.
* **Explicit Latent State Preservation:** Generates competition-compliant binary predictions (`Home` vs. `Away`) while preserving distinct latent states (`Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`) in the output submission CSV.
* **TypeSafe AI (`Jev / System One`) Integration:** Supports sparse ambiguity refinement using live API calls (`--live-jev`) or heuristic fallbacks.

#### CLI Usage Examples:

```powershell
# 1. Prediction on custom match data with live TypeSafe AI (Jev / System One) refinement:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_frames.csv --players ruta_players.csv --output artifacts\mi_submission.csv --live-jev

# 2. Self-contained end-to-end verification gate (trains, OOF-tunes, serializes bundle, and tests held-out match):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode verify

# 3. Train and export model bundle on benchmark or custom training matches:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode train --model-bundle artifacts\barca_possession_mvp_v2.pkl

# 4. Predict on a single wide-format Metrica tracking table:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_Sample_Game_1_RawTrackingData.csv --output artifacts\mi_submission.csv
```

#### CLI Flag Reference:
* `--mode`: Execution mode (`verify`, `train`, `predict`). Default is `verify`.
* `--frames`: Path to input `frame_df` CSV/Parquet or single wide-format Metrica tracking table.
* `--players`: Optional path to input `player_tracks_df` CSV/Parquet (required if `--frames` is not a wide-format table).
* `--meta`: Optional path to match metadata JSON (`match_id`, `sampling_rate_hz`, pitch dimensions).
* `--model-bundle`: Path to serialized model `.pkl` bundle (default: `artifacts/barca_possession_mvp_v2.pkl`).
* `--output`: Path to write the output submission CSV (default: `artifacts/submission_output.csv`).
* `--live-jev`: Enables live TypeSafe AI (`Jev / System One`) API queries for ambiguous candidate frames using `TYPESAFE_API_KEY`.


---

## 5. Repository Structure

```
.
├── docs/
│   └── technical_submission_report.md     # Detailed RFC v2.0 technical report
├── run_submission_pipeline.py             # Unified CLI submission & prediction pipeline
├── task1_schema_and_preprocessing.py      # Quality-aware segmented preprocessing
├── task2_feature_engine.py                # Kinematics, reachability & control evidence
├── task3_anchors_and_bridges.py           # Anchors, bounded bridges & Jev refiner
├── task4_classifier_and_decoder.py        # Team-swap GBDT & Soft Viterbi decoder
├── task5_evaluation_and_ablation.py       # Metrics, turnover matching & ablation runner
├── .gitignore                             # Ignored caches and virtual environments
└── README.md                              # Project documentation & benchmark overview
```

---

## 6. License & Attribution

This project is released under the **MIT License**.  
Developed for the *More than a Hack 2027* soccer tracking data competition (FC Barcelona & Metrica Sports).
