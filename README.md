# Frame-Level Team Possession Estimation

🌐 **Language / Idioma / Llengua:** [English](README.md) | [Español](README.es.md) | [Català](README.ca.md)

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

## 2. Six-Stage Modular Architecture

The codebase is structured into six self-verifying, zero-leakage pipeline modules:

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

### [Task 6: Match Visualizer & Diagnostic Plotter](file:///task6_match_visualizer.py)
* **Multi-Panel Tactical Dashboard:** Renders a 3-panel professional match visualizer combining a 2D pitch view (with FC Barcelona Blaugrana color styling, metric coordinates, player IDs, and ball tracking), sequence-level posterior probability curves with ground truth and shaded possession dominance, and kinematic discontinuities ($I_b$, aerial/pass-by likelihood, bridge agreement, and Jev refiner activations).
* **Defensive Schema Ingestion:** Automatically inspects frame columns with graceful fallbacks for raw predictions, competition test sets (missing ground truth), or binary submission tables.
* **Headless & Scalable Execution:** Native `matplotlib.use("Agg")` rendering, robust pickle unpickling, and optional temporal window zooming (`time_window_s=(t_start, t_end)`).

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
pip install numpy pandas scipy scikit-learn matplotlib
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

# Verify Task 6: Match visualizer & 3-panel tactical diagnostic dashboard
python task6_match_visualizer.py
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

# 5. Predict on official Metrica split Home & Away tracking tables (auto-detected multi-row format):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames data_metrica\Sample_Game_1_RawTrackingData_Home_Team.csv --players data_metrica\Sample_Game_1_RawTrackingData_Away_Team.csv --output artifacts\reintento_metrica.csv --live-jev --max-frames 2500
```

#### CLI Flag Reference:
* `--mode`: Execution mode (`verify`, `train`, `predict`). Default is `verify`.
* `--frames`: Path to input `frame_df` CSV/Parquet, single wide-format tracking table, or Metrica Home tracking table.
* `--players`: Path to input `player_tracks_df` CSV/Parquet, or Metrica Away tracking table (auto-detected).
* `--meta`: Optional path to match metadata JSON (`match_id`, `sampling_rate_hz`, pitch dimensions).
* `--model-bundle`: Path to serialized model `.pkl` bundle (default: `artifacts/barca_possession_mvp_v2.pkl`).
* `--output`: Path to write the output submission CSV (default: `artifacts/submission_output.csv`).
* `--live-jev`: Enables live TypeSafe AI (`Jev / System One`) API queries for ambiguous candidate frames using `TYPESAFE_API_KEY`.
* `--max-frames`: Optional maximum number of frames to load for rapid testing.

### Automated Batch Ingestion for Real Competition Data (`auto_process_competition.py`)

When the final competition tracking data is released by FC Barcelona and Metrica Sports, [`auto_process_competition.py`](file:///auto_process_competition.py) serves as the **zero-configuration batch orchestrator**. It recursively discovers all tracking files in a target folder, detects their internal structure (split Metrica tables, single wide tables, or canonical pairs), pairs corresponding matches, executes full inference through the 5-task pipeline, and exports competition-compliant submission CSVs and diagnostic visualizations.

#### 1. Supported Data Formats & Auto-Detection:
* **Metrica Sports Split Tables (`metrica_split`):**
  - Matches files like `*Home_Team.csv` and `*Away_Team.csv` for the same game.
  - Automatically parses Metrica's 3-row headers, aligns timestamps, extracts ball coordinates, and transforms normalized `[0, 1]` coordinates into pitch meters centered at `(0, 0)` (`[-52.5, 52.5]` m x `[-34.0, 34.0]` m).
* **Canonical Relational Tables (`canonical_pair`):**
  - Matches frame metadata tables (`*frame*.csv/.parquet`) with long-format player trajectory tables (`*player*.csv/.parquet`).
* **Single Wide Tracking Tables (`wide_single`):**
  - Ingests single-file tracking dumps containing players and ball across columns (`Home_11_x, Home_11_y, Ball_x...`).
* **Smart Filter & Noise Rejection:**
  - Automatically ignores event logs (`*Events*.csv`), referee tables, lineups, and ancillary metadata so you can drop raw match archives directly into `data/` without manual cleanup.

#### 2. Directory Layout:
Simply place your raw downloaded tracking files into the `data/` directory (or any subfolder inside it):

```text
Possession-Estimation/
├── data/                                         # Place downloaded match files here
│   ├── Match_1_RawTrackingData_Home_Team.csv     # Metrica Home tracking table
│   ├── Match_1_RawTrackingData_Away_Team.csv     # Metrica Away tracking table
│   ├── Match_1_RawEventsData.csv                 # (Auto-ignored event stream)
│   ├── Match_2_RawTrackingData_Home_Team.csv
│   └── Match_2_RawTrackingData_Away_Team.csv
├── artifacts/                                    # Target directory for output submissions
│   ├── submission_Match_1.csv                    # Clean, validated submission file
│   ├── submission_Match_2.csv
│   └── visualizations/                           # Tactical pitch & posterior dashboards
│       ├── Match_1_diagnostic.png
│       └── Match_2_diagnostic.png
└── auto_process_competition.py
```

#### 3. Environment & Prerequisites:
If using live TypeSafe AI (`Jev / System One`) reasoning for boundary edge-cases, set your API key in your terminal session:

```powershell
# Windows PowerShell:
$env:TYPESAFE_API_KEY = "your_typesafe_api_key_here"

# Linux / macOS Bash:
export TYPESAFE_API_KEY="your_typesafe_api_key_here"
```
*(Note: If no API key is provided, the pipeline seamlessly falls back to deterministic kinematic heuristics without failing).*

#### 4. Execution Commands:

##### A. Full Competition Batch Run (Production):
Processes every match detected in `data/`, applies out-of-fold calibrated models and Jev boundary refinement, validates submission contracts, and exports tactical diagnostic plots:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --visualize
```

##### B. Fast Sanity Check (Recommended First Step):
Before processing full 90-minute matches (~145,000 frames each), run a quick verification on the first 2,500 frames (~100 seconds) using `--max-frames`:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --max-frames 2500 --visualize
```

##### C. Offline / Heuristic Run (No API Key Required):
Run inference using purely deterministic kinematic classification and Soft Viterbi temporal relaxation:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --visualize
```

#### 5. Output Verification & Contract:
The pipeline produces submission files with zero missing values, guaranteed monotonicity, and strict schema compliance:
* **Output Path:** `artifacts/submission_<match_id>.csv`
* **Output Columns:**
  - `frame`: Frame index (integer, strictly ascending).
  - `possession_team`: Final decoded team label (`Home` or `Away`).
  - `confidence`: Calibrated posterior probability `[0.50, 1.00]`.
  - `possession_state`: Fine-grained latent state (`Home`, `Away`, `Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`).
* **Visual Dashboard:** Multi-panel tactical 2D pitch map with player velocity vectors, ball trajectory, and rolling team possession probability saved to `artifacts/visualizations/<match_id>_diagnostic.png`.

#### 6. Batch CLI Flag Reference:
| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Directory where competition tracking files are located (searched recursively). |
| `--output-dir` | `artifacts` | Destination directory for generated submission CSVs and diagnostic plots. |
| `--live-jev` | `False` | Enables live TypeSafe AI (`Jev / System One`) LLM queries for sparse boundary ambiguity frames. |
| `--max-frames` | `None` | Caps frames processed per match (e.g. `2500` for ~100s smoke tests). |
| `--visualize` | `False` | Renders and saves high-resolution 3-panel tactical dashboard PNGs. |
| `--model-bundle` | `artifacts/barca_possession_mvp_v2.pkl` | Path to the trained LightGBM model bundle. |

### Interactive 3D WebGL Broadcast Replay (`generate_3d_replay_html.py`)

For technical demonstrations, fan experiences, and tactical debriefs, [`generate_3d_replay_html.py`](file:///generate_3d_replay_html.py) compiles tracking predictions into a self-contained **3D WebGL broadcast experience** powered by Three.js. It translates abstract mathematical concepts (bounded bridges, Viterbi states, and Jev contact resolutions) into intuitive visual cues.

```powershell
# 1. Generate 3D interactive broadcast replay (default: 60s at 25Hz):
.\.venv\Scripts\python.exe generate_3d_replay_html.py --max-frames 1500 --output artifacts\visualizations\barca_3d_interactive_replay.html

# 2. Generate from specific match files or custom data directory:
.\.venv\Scripts\python.exe generate_3d_replay_html.py --data-dir data --max-frames 2500
```

#### Key Visual & Tactical Features:
* **Interactive AI Comparison Switch:** Toggle in real-time between **🧠 Nuestra IA** (Viterbi + Jev) and **🤖 IA Común** (Nearest-Player heuristic) to see naive proximity errors and false turnovers highlighted on the fly.
* **Escudo 3D Anti-Robo (Pass Shield):** An animated golden icosahedron wireframe that envelops the ball during rapid through-balls or cross-field switches, visually demonstrating why passing near a defender does not forfeit team control.
* **Hilo Dorado de Control (3D Tether Arc):** A dynamic Bezier curve joining the legitimate ball controller to the ball trajectory.
* **Expanding Shockwave Rings:** Radial ground waves marking physical contact anchors and defensive turnovers.
* **4 Camera Perspectives:**
  - 🎥 **TV Broadcast:** Smooth tracking following match action.
  - 🚁 **Dron Cercano:** Close-up cinematic follow camera.
  - 🏟️ **Pizarra Aérea:** Orthogonal 2D tactical board view.
  - 🕹️ **Vista Libre (Órbita Manual):** Full 360° mouse drag orbit and scroll-wheel zoom (12m to 140m).
* **Keyboard Hotkeys:** `Space` (Play/Pause), `← / →` (-1s / +1s), `1-4` (Camera Modes), `M` (AI Mode Switch).


---

## 5. Repository Structure

```
.
├── docs/
│   └── technical_submission_report.md     # Detailed RFC v2.0 technical report
├── auto_process_competition.py            # Automated batch scanning & inference for competition data
├── generate_3d_replay_html.py             # 3D interactive WebGL/Three.js broadcast replay generator
├── ingest_and_test_metrica_sample.py      # Official Metrica Sample Game 1 ingestion & test suite
├── run_submission_pipeline.py             # Unified CLI submission & prediction pipeline
├── task1_schema_and_preprocessing.py      # Quality-aware segmented preprocessing
├── task2_feature_engine.py                # Kinematics, reachability & control evidence
├── task3_anchors_and_bridges.py           # Anchors, bounded bridges & Jev refiner
├── task4_classifier_and_decoder.py        # Team-swap GBDT & Soft Viterbi decoder
├── task5_evaluation_and_ablation.py       # Metrics, turnover matching & ablation runner
├── task6_match_visualizer.py              # 2D pitch & multi-panel diagnostic dashboard
├── .gitignore                             # Ignored caches, datasets and virtual environments
├── README.md                              # Main documentation in American English (Default)
├── README.es.md                           # Complete documentation in Spanish
└── README.ca.md                           # Complete documentation in Catalan
```

---

## 6. License & Attribution

This project is released under the **MIT License**.  
Developed for the *More than a Hack 2027* soccer tracking data competition (FC Barcelona & Metrica Sports).
