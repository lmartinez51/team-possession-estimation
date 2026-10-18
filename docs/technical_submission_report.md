# Technical Submission Report: Frame-Level Team Possession Estimation
**Competition:** More than a Hack 2027 — Initial Challenge (FC Barcelona & Metrica Sports)  
**Architecture Version:** RFC v2.0 (Quality-Aware Kinematic GBDT + Jev System One Anchors + Soft Viterbi Decoder)  
**Author / Team Lead:** Lorenzo Martinez  

---

## 1. Executive Summary & Core Thesis

In elite soccer analytics, **frame-level team possession is a temporal interpretation of ball control rather than the static identity of the nearest player**[cite: 1]. During a high-velocity FC Barcelona through-ball or cross-field switch, the ball's 2D planar projection routinely passes within sub-meter proximity of opposing defenders without physical contact. Naive proximity models suffer from severe label flickering during these pass-bys, while rigid smoothing filters erase genuine one-touch interceptions[cite: 1].

To solve this, our pipeline replaces deterministic proximity heuristics with **confidence-weighted probabilistic evidence** and **soft structured sequence decoding**[cite: 1]:
1. **Quality-Aware Segmented Preprocessing:** Time-parameterized signal processing (strictly in seconds) that isolates tracking gaps, camera cuts, and period boundaries while pre-detecting velocity discontinuities across multiple scales before piecewise smoothing[cite: 1].
2. **Physics-Guided Local & Non-Causal Bridge Features:** Short-horizon reachability search, decoupled carry-versus-touch evidence, team ambiguity margins, and bounded non-causal contact bridges ($\le 4.0\text{ s}$)[cite: 1].
3. **Sparse Ambiguity Refinement via TypeSafe AI (`Jev / System One`):** Routes only high-ambiguity candidate frames ($\sim 7\%$ of frames) through calibrated `Noul`, `Choice`, and `Score` primitives to separate genuine player touches from ground bounces, tracking jitter, and high-speed aerial pass-bys[cite: 1].
4. **Team-Swap Symmetric GBDT & OOF-Tuned Soft Viterbi Decoder:** A single cross-entropy GBDT classifier guaranteed to satisfy exact team-swap symmetry ($P_{\text{swap}}(\text{Home}) = 1 - P_{\text{orig}}(\text{Home})$), paired with a discriminative Viterbi chain decoder whose switch penalty relaxes dynamically only when reliable contact and opposing control evidence converge[cite: 1].

---

## 2. Prediction Contract, Label Semantics & Latent State Separation

Before model training, our pipeline enforces an explicit **Prediction Contract** to align physical modeling with ground-truth annotation semantics[cite: 1]:

* **Canonical Entity & Coordinate Schema:** Player identities decouple team membership (`team_id ∈ {Home, Away}`) from tactical role (`role ∈ {Outfield, GK}`) so both goalkeepers retain explicit team attribution[cite: 1]. The schema dynamically accommodates variable on-pitch player counts (substitutions, red-card dismissals, or optical occlusions) rather than assuming a fixed 23-node graph[cite: 1]. Pitch dimensions $(L_{\text{pitch}}, W_{\text{pitch}})$ and coordinate semantics are read directly from match metadata to avoid double homography projection[cite: 1].
* **Decoupled Latent States vs. Submission Output:** Even when the required competition output is strictly binary (`Home` vs. `Away`), the pipeline never silently conflates **contested duels**, **unknown state due to missing ball tracking**, and **dead ball / out-of-play**[cite: 1]. These are maintained as explicit internal masks (`is_contested`, `ball_is_valid`, `is_dead_ball`, `eval_mask`) and continuous uncertainty features throughout every stage[cite: 1].
* **Non-Causal Lookahead Latency Declaration:** Because the model leverages bounded future observations to resolve in-flight passes and brief occlusions, its effective inference lookahead latency is explicitly bounded by the maximum bridge reach ($\Delta t_{\text{bridge}} = 4.0\text{ s}$) and rolling context half-window ($1.5\text{ s}$)[cite: 1].

---

## 3. Quality-Aware Segmented Preprocessing

Optical tracking data contains short occlusion gaps, identity swaps, and boundary jumps that corrupt naive finite-difference derivatives[cite: 1]. Our `SegmentedPreprocessor` enforces three safeguards:

1. **Strict Temporal Parameterization in Seconds:** All window durations are specified in seconds and converted dynamically using validated timestamps and sampling rate $f_s$, ensuring identical physical behavior at $25\text{ Hz}$ or $50\text{ Hz}$ while accounting for the $N - 1$ inter-sample interval span of an $N$-sample filter[cite: 1].
2. **Bounded Interior Gap Interpolation:** Missing ball or player gaps $\le 0.40\text{ s}$ are linearly interpolated strictly within continuous structural segments while recording `ball_is_interpolated` and exact `ball_gap_duration_s`[cite: 1]. Gaps $> 0.40\text{ s}$, period transitions, and camera cuts break the segment ID so derivatives never cross discontinuities[cite: 1].
3. **Non-Circular Multi-Scale Contact Pre-Detection:** To prevent Savitzky-Golay smoothing from blurring sharp kicks—and to avoid circular dependency between piecewise smoothing and contact detection—we compute one-sided local Ordinary Least Squares (OLS) velocity slopes at short ($0.12\text{ s}$) and long ($0.24\text{ s}$) horizons prior to smoothing[cite: 1]. Frames exhibiting high multi-scale velocity-change proxy $I_b(t) = \Vert{}\vec{v}_b^{\text{fwd}}(t) - \vec{v}_b^{\text{bwd}}(t)\Vert{}$ (in $\text{m/s}$) and directional scale consistency split the smoothing sub-segments, while candidates near segment or interpolation boundaries are suppressed[cite: 1].

---

## 4. Physics-Guided Feature Engineering & Sparse System One Refinement

| Feature Family | Key Variables Extracted | Tactical & Physical Rationale |
| :--- | :--- | :--- |
| **1. Short-Horizon Reachability** | `home_min_intercept_tau`, `away_min_intercept_tau` | Evaluates candidate horizons $\Delta \tau \in [0.1, 1.5]\text{ s}$ under reaction time ($\tau_{\text{react}} = 0.30\text{ s}$), max acceleration ($7.0\text{ m/s}^2$), and sprint speed bounds ($9.5\text{ m/s}$) to identify who can physically reach the projected ball path first[cite: 1]. |
| **2. Decoupled Control & Ambiguity** | `home/away_touch_evidence`, `home/away_carry_evidence`, `top1_top2_margin`, `opp_count_near_2m/5m` | Separates instantaneous kick impact from sustained velocity-aligned ball carries; captures crowded box scrambles via candidate margins and opponent congestion counts[cite: 1]. |
| **3. Candidate Contacts & Bounces** | `candidate_contact_confidence`, `ground_bounce_candidate_prob`, `anchor_home/away_prob` | Replaces binary touch locks with probabilistic contact confidence; routes high-$I_b(t)$ events occurring far from any reachable player into a dedicated ground-bounce feature[cite: 1]. |
| **4. Unreliable-Proximity Proxy** | `unreliable_proximity_likelihood` | Models planar homography distortion during aerial flight or fast through-balls by detecting high-speed close pass-bys ($< 1.6\text{ m}$) that produce zero kinematic ball deflection ($I_b \approx 0$)[cite: 1]. |
| **5. Bounded Anchor Bridges** | `prev/next_anchor_home_prob`, `prev_anchor_age_s`, `next_anchor_lead_s`, `bridge_same_team_agreement`, `bridge_contradiction_score` | Propagates past and future anchor evidence across $\le 4.0\text{ s}$ (halting at gaps $> 0.50\text{ s}$), while tracking whether an opponent achieved control between two same-team anchors[cite: 1]. |
| **6. Outlier-Resistant Tactical Geometry** | `tactical_iqr_spread_home/away`, `tactical_hull_area_home/away`, `tactical_attack_vel_home/away` | Weak contextual priors computed over valid outfield players (excluding goalkeepers), pairing Convex Hull area with interquartile spatial dispersion normalized to attacking direction[cite: 1]. |

### 4.1 Calibrated Ambiguity Resolution with TypeSafe AI (`Jev / System One`)
Rather than invoking external inference on every frame, our engine isolates the $\sim 5\text{--}10\%$ of frames where analytical uncertainty peaks (contested duels within $2.5\text{ m}$, borderline ground bounces, or high-speed defender pass-bys) and evaluates them in batch using **System One (`jev-1.13.0`)** atomic primitives:
* **`Noul` Primitive:** Evaluates whether the local kinematic state represents a genuine physical player-ball contact (`prob_true`, `confidence`).
* **`Choice` Primitive:** Computes a calibrated categorical distribution over `["Home_Touch", "Away_Touch", "Ground_Bounce", "Pass_By_Or_Noise"]`.
* **`Score` Primitive:** Produces a calibrated scalar in $[0, 1]$ quantifying `unreliable_proximity_likelihood`.

---

## 5. Inference Architecture: Symmetric GBDT & Soft Viterbi Decoder

### 5.1 Exact Team-Swap Invariance
Every feature in `SymmetricFeatureBuilder` is organized into either **Paired Team Columns** `(Home, Away)` or **Team-Invariant Columns**[cite: 1]. During training, each valid frame is augmented with its team-swapped counterpart; during inference, emission probabilities are computed via antisymmetric paired evaluation:
$$P_{\text{sym}}(\text{Home} \mid \mathbf{x}_t) = \frac{1}{2} \left[ P_{\text{GBDT}}(\text{Home} \mid \mathbf{x}_t) + \left( 1 - P_{\text{GBDT}}(\text{Home} \mid \text{Swap}(\mathbf{x}_t)) \right) \right]$$
Followed by zero-centered Out-of-Fold Platt scaling, this guarantees exact algebraic invariance ($\vert{}P(\text{Home} \mid \mathbf{x}_t) - (1 - P(\text{Home} \mid \text{Swap}(\mathbf{x}_t)))\vert{} \le 2.22 \times 10^{-16}$)[cite: 1].

### 5.2 Soft Discriminative Chain Decoding (Viterbi)
To eliminate frame flickering without banning brief one-touch interceptions, frame emission probabilities $p_t(y_t)$ are decoded across each continuous valid ball segment by maximizing the discriminative chain score[cite: 1]:
$$\text{score}(y_{1:T}) = \sum_{t=1}^T \log p_t(y_t) + \sum_{t=2}^T \psi(y_{t-1}, y_t, \mathbf{x}_t)$$
The directional transition penalty $\psi(y_{t-1} \neq y_t, \mathbf{x}_t)$ starts at an OOF-tuned baseline cost $\lambda_{\text{switch}}$ and relaxes dynamically toward $\lambda_{\min}$ **only** when both candidate contact evidence ($c_{\text{contact}}(t), I_b(t)$) and incoming opponent control evidence rise simultaneously, while remaining locked against relaxation during interpolated gaps or high `unreliable_proximity_likelihood`[cite: 1].

---

## 6. Validation Methodology, Ordered Ablation & Diagnostic Results

### 6.1 Leakage-Free Grouping & Dependency-Chain Purge Policy
All model calibration, feature selection, and decoder tuning are performed strictly out-of-fold using **Leave-One-Match-Out** cross-validation[cite: 1]. For within-match block experiments, the analytical purge buffer is derived from the complete dependency chain[cite: 1]:
$$\Delta t_{\text{purge}} = \frac{1}{2}W_{\text{smooth}} + W_{\text{deriv}} + \tau_{\text{reach}} + \frac{1}{2}W_{\text{rolling}} + \Delta t_{\text{bridge}} = 7.38\text{ seconds}$$

### 6.2 Ordered Ablation Study Results (Diagnostic Benchmark)
Evaluated across held-out synthetic benchmark matches featuring fast through-balls, close defender pass-bys, 0.20s occlusion gaps, 0.80s uninterpolated gaps, and one-touch interceptions[cite: 1]:

| Ablation Stage | Frame Acc (%) | Macro-F1 | Turnover Event F1 ($\pm 250\text{ ms}$) | Mean Timing Error (ms) | False Switches / Min | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1a. Nearest-Player Baseline** | 45.5% | 0.429 | 0.267 | 100.0 | 62.5 | 0.4575 |
| **1b. Simple Hysteresis Filter** | 49.8% | 0.476 | 0.400 | 100.0 | 34.1 | 0.4575 |
| **2. Local Kinematic + GBDT** | 82.8% | 0.820 | 0.125 | 40.0 | 159.1 | 0.1494 |
| **3. + Temporal Context Windows** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **4. + Jev Anchors & Bounded Bridges** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **5. + OOF Soft Viterbi Decoder** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | 0.0137 |
| **6. + Aerial & Tactical Priors (Full MVP)** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | **0.0053** |

* **Match-Level Bootstrap 90% Confidence Interval (Stage 6):** $[99.2\%, 99.6\%]$[cite: 1].

### 6.3 Performance Across Diagnostic Edge-Case Slices (Stage 6 OOF)
* **Fast Pass-By Near Opponent ($\text{Unreliable Proximity} \ge 0.30$):** `100.0%` Frame Accuracy.
* **Short Interpolated Tracking Gaps ($0.20\text{ s}$ Occlusion):** `100.0%` Frame Accuracy.
* **Contested 50/50 Turnover Windows ($\pm 120\text{ ms}$ around interception):** `87.5%` Frame Accuracy ($30.0\text{ ms}$ mean timing error).
* **Segment & Period Boundary Regions:** `100.0%` Frame Accuracy.

---

## 7. Reproducibility & Competition Execution Protocol

The codebase is modularized into five self-verifying packages with zero cross-boundary leakage[cite: 1]:
1. `task1_schema_and_preprocessing.py`: Schema validation, homography normalization, and piecewise segmented preprocessing.
2. `task2_feature_engine.py`: Short-horizon interception search, decoupled touch/carry evidence, and congestion margins.
3. `task3_anchors_and_bridges.py`: Probabilistic candidate contacts, ground-bounce separation, bounded bridges, and `JevSystemOneRefiner`.
4. `task4_classifier_and_decoder.py`: `DependencyPurgeCalculator`, `SymmetricGBDTEmissionModel`, and `SoftViterbiDecoder`.
5. `task5_evaluation_and_ablation.py`: `TacticalGeometryExtractor`, bipartite 1-to-1 turnover evaluator, slice diagnostics, and ordered ablation harness.