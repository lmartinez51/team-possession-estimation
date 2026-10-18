# Task 4: Symmetric GBDT Emission Classifier, Dependency-Purged OOF Validation,
# and Soft Discriminative Viterbi Sequence Decoder (RFC v2.0 Section 4.6, 4.7, 5 & 6.1).
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import (
    MatchMetadata,
    PreprocessConfig,
    SegmentedPreprocessor,
    TeamID,
    ValidatedTrackingDataset,
    generate_synthetic_diagnostic_match,
)
from task2_feature_engine import FeatureConfig, LocalKinematicFeatureExtractor
from task3_anchors_and_bridges import (
    AnchorBridgeConfig,
    JevSystemOneRefiner,
    KinematicContactAndBridgeEngine,
)

# Optional high-performance GBDT backends with seamless pure-NumPy fallback
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# 1. CONFIGURATION & DEPENDENCY-CHAIN PURGE CALCULATOR (RFC v2.0 Section 6.1)
# ============================================================================

@dataclass
class TemporalWindowConfig:
    # Multi-scale temporal context windows strictly in seconds (RFC v2.0 Section 4.2 & 6.3)
    short_context_s: float = 1.00
    long_context_s: float = 3.00


@dataclass
class DecoderConfig:
    # Soft Viterbi transition penalty parameters tuned on Out-of-Fold predictions
    base_switch_penalty: float = 2.80
    min_switch_penalty: float = 0.15
    turnover_relax_gain: float = 0.90
    contact_vs_dv_weight: float = 0.65
    prob_clip_eps: float = 1e-4


@dataclass
class DependencyPurgeCalculator:
    # Analytically computes the required within-match validation purge buffer (in seconds)
    # from the complete feature & preprocessing dependency chain (RFC v2.0 Section 6.1).
    prep_cfg: PreprocessConfig = field(default_factory=PreprocessConfig)
    feat_cfg: FeatureConfig = field(default_factory=FeatureConfig)
    bridge_cfg: AnchorBridgeConfig = field(default_factory=AnchorBridgeConfig)
    win_cfg: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)

    def compute_purge_buffer_seconds(self) -> float:
        smooth_half_s = 0.5 * max(
            self.prep_cfg.player_smooth_win_s, self.prep_cfg.ball_smooth_win_s
        )
        deriv_horizon_s = max(
            self.prep_cfg.multiscale_long_win_s, self.prep_cfg.boundary_suppress_s
        )
        reachability_s = max(self.feat_cfg.reachability_horizons_s)
        rolling_half_s = 0.5 * self.win_cfg.long_context_s
        bridge_span_s = self.bridge_cfg.max_bridge_span_s

        total_purge_s = (
            smooth_half_s
            + deriv_horizon_s
            + reachability_s
            + rolling_half_s
            + bridge_span_s
        )
        return float(np.round(total_purge_s, 4))

    def generate_purged_time_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        purge_buffer_s: Optional[float] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates leak-free train/val boolean masks within a single match by purging
        a safety buffer of `purge_buffer_s` seconds before and after each validation block
        (RFC v2.0 Section 6.1).
        """
        if purge_buffer_s is None:
            purge_buffer_s = self.compute_purge_buffer_seconds()

        n = len(df)
        if n == 0:
            return [(np.ones(0, dtype=bool), np.zeros(0, dtype=bool))]

        t = df["timestamp_s"].to_numpy(dtype=float) if "timestamp_s" in df.columns else np.arange(n, dtype=float)
        t_min, t_max = float(np.min(t)), float(np.max(t))
        t_total = t_max - t_min

        if t_total <= 0.0:
            return [(np.ones(n, dtype=bool), np.zeros(n, dtype=bool))]

        block_size = t_total / n_splits
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        for k in range(n_splits):
            val_start = t_min + k * block_size
            val_end = t_min + (k + 1) * block_size

            val_mask = (t >= val_start) & (t < val_end if k < n_splits - 1 else t <= val_end)

            purge_start = val_start - purge_buffer_s
            purge_end = val_end + purge_buffer_s

            train_mask = (t < purge_start) | (t > purge_end)
            splits.append((train_mask, val_mask))

        return splits



# ============================================================================
# 2. MULTI-SCALE TEMPORAL CONTEXT & SYMMETRIC FEATURE BUILDER
# ============================================================================

class SymmetricFeatureBuilder:
    # Organizes all features into Paired (Home, Away) columns and Team-Invariant columns
    # so that swapping Home and Away is an exact algebraic permutation of paired columns.

    PAIRED_COLUMNS: List[Tuple[str, str]] = [
        ("min_dist_home", "min_dist_away"),
        ("home_min_intercept_tau", "away_min_intercept_tau"),
        ("home_touch_evidence", "away_touch_evidence"),
        ("home_carry_evidence", "away_carry_evidence"),
        ("home_top1_top2_margin", "away_top1_top2_margin"),
        ("opp_count_near_home_2m", "opp_count_near_away_2m"),
        ("opp_count_near_home_5m", "opp_count_near_away_5m"),
        ("visible_home_players", "visible_away_players"),
        ("anchor_home_prob", "anchor_away_prob"),
        ("prev_anchor_home_prob", "prev_anchor_away_prob"),
        ("next_anchor_home_prob", "next_anchor_away_prob"),
        ("roll_short_home_ctrl", "roll_short_away_ctrl"),
        ("roll_long_home_ctrl", "roll_long_away_ctrl"),
        ("roll_short_min_dist_home", "roll_short_min_dist_away"),
        ("roll_long_min_dist_home", "roll_long_min_dist_away"),
    ]

    INVARIANT_COLUMNS: List[str] = [
        "ball_speed",
        "ball_accel",
        "ball_vel_change_proxy_mps",
        "ball_multiscale_consistency",
        "ball_is_interpolated",
        "ball_gap_duration_s",
        "candidate_contact_confidence",
        "ground_bounce_candidate_prob",
        "unreliable_proximity_likelihood",
        "jev_decision_confidence",
        "prev_anchor_age_s",
        "prev_anchor_confidence",
        "next_anchor_lead_s",
        "next_anchor_confidence",
        "bridge_same_team_agreement",
        "bridge_contradiction_score",
    ]

    def __init__(self, win_config: Optional[TemporalWindowConfig] = None):
        self.win_config = win_config or TemporalWindowConfig()

    def add_segmented_temporal_windows(
        self, df: pd.DataFrame, metadata: MatchMetadata
    ) -> pd.DataFrame:
        # Computes centered rolling temporal context features strictly isolated within
        # structural_seg_id and valid ball segments so windows never cross boundaries.
        roll_cols = [
            "roll_short_home_ctrl", "roll_short_away_ctrl",
            "roll_long_home_ctrl", "roll_long_away_ctrl",
            "roll_short_min_dist_home", "roll_short_min_dist_away",
            "roll_long_min_dist_home", "roll_long_min_dist_away",
        ]
        out = df.sort_values("frame_id").reset_index(drop=True).copy()
        n = len(out)
        if n == 0:
            for c in roll_cols:
                out[c] = pd.Series(dtype=float)
            return out

        w_short = max(3, metadata.seconds_to_samples(self.win_config.short_context_s, odd_window=True))
        w_long = max(3, metadata.seconds_to_samples(self.win_config.long_context_s, odd_window=True))

        # Safe extraction of local control proxies
        h_touch = out["home_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_touch_evidence" in out.columns else np.zeros(n, dtype=float)
        h_carry = out["home_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_carry_evidence" in out.columns else np.zeros(n, dtype=float)
        a_touch = out["away_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_touch_evidence" in out.columns else np.zeros(n, dtype=float)
        a_carry = out["away_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_carry_evidence" in out.columns else np.zeros(n, dtype=float)

        home_inst_ctrl = np.maximum(h_touch, h_carry)
        away_inst_ctrl = np.maximum(a_touch, a_carry)
        d_home = out["min_dist_home"].fillna(999.0).to_numpy(dtype=float) if "min_dist_home" in out.columns else np.full(n, 999.0, dtype=float)
        d_away = out["min_dist_away"].fillna(999.0).to_numpy(dtype=float) if "min_dist_away" in out.columns else np.full(n, 999.0, dtype=float)

        rs_h_ctrl = np.zeros(n, dtype=float)
        rs_a_ctrl = np.zeros(n, dtype=float)
        rl_h_ctrl = np.zeros(n, dtype=float)
        rl_a_ctrl = np.zeros(n, dtype=float)

        rs_d_home = np.full(n, 999.0, dtype=float)
        rs_d_away = np.full(n, 999.0, dtype=float)
        rl_d_home = np.full(n, 999.0, dtype=float)
        rl_d_away = np.full(n, 999.0, dtype=float)

        # Derive ball_valid_seg_id if missing
        if "ball_valid_seg_id" not in out.columns:
            if "ball_is_valid" in out.columns:
                valid = out["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
                seg_breaks = np.concatenate([[True], valid[1:] != valid[:-1]])
                seg_ids = np.cumsum(seg_breaks)
                out["ball_valid_seg_id"] = np.where(valid, seg_ids, -1)
            else:
                out["ball_valid_seg_id"] = 0

        for seg_id, group in out.groupby("ball_valid_seg_id", sort=False):
            if seg_id < 0:
                continue
            pos_idx = group.index.to_numpy(dtype=int)

            s_hc = pd.Series(home_inst_ctrl[pos_idx])
            s_ac = pd.Series(away_inst_ctrl[pos_idx])
            s_dh = pd.Series(d_home[pos_idx])
            s_da = pd.Series(d_away[pos_idx])

            rs_h_ctrl[pos_idx] = s_hc.rolling(w_short, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rs_a_ctrl[pos_idx] = s_ac.rolling(w_short, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rl_h_ctrl[pos_idx] = s_hc.rolling(w_long, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rl_a_ctrl[pos_idx] = s_ac.rolling(w_long, min_periods=1, center=True).mean().to_numpy(dtype=float)

            rs_d_home[pos_idx] = s_dh.rolling(w_short, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rs_d_away[pos_idx] = s_da.rolling(w_short, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rl_d_home[pos_idx] = s_dh.rolling(w_long, min_periods=1, center=True).mean().to_numpy(dtype=float)
            rl_d_away[pos_idx] = s_da.rolling(w_long, min_periods=1, center=True).mean().to_numpy(dtype=float)

        out["roll_short_home_ctrl"] = rs_h_ctrl
        out["roll_short_away_ctrl"] = rs_a_ctrl
        out["roll_long_home_ctrl"] = rl_h_ctrl
        out["roll_long_away_ctrl"] = rl_a_ctrl

        out["roll_short_min_dist_home"] = rs_d_home
        out["roll_short_min_dist_away"] = rs_d_away
        out["roll_long_min_dist_home"] = rl_d_home
        out["roll_long_min_dist_away"] = rl_d_away
        return out

    @classmethod
    def get_feature_names(cls) -> List[str]:
        names: List[str] = []
        for h_col, a_col in cls.PAIRED_COLUMNS:
            names.append(h_col)
            names.append(a_col)
        for inv_col in cls.INVARIANT_COLUMNS:
            names.append(inv_col)
        return names

    @classmethod
    def extract_matrix(cls, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols: List[np.ndarray] = []
        for col_name in cls.get_feature_names():
            if col_name in df.columns:
                arr = df[col_name].to_numpy(dtype=float)
                arr = np.nan_to_num(arr, nan=0.0, posinf=999.0, neginf=-999.0)
            else:
                arr = np.zeros(n, dtype=float)
            cols.append(arr)
        return np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)

    @classmethod
    def swap_feature_matrix(cls, X: np.ndarray) -> np.ndarray:
        # Swaps every (Home, Away) paired column while preserving invariant columns.
        X_swap = X.copy()
        n_pairs = len(cls.PAIRED_COLUMNS)
        for p_idx in range(n_pairs):
            c_home = 2 * p_idx
            c_away = 2 * p_idx + 1
            X_swap[:, c_home] = X[:, c_away]
            X_swap[:, c_away] = X[:, c_home]
        return X_swap

    @classmethod
    def swap_dataframe_teams(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Swaps Home and Away columns directly in a feature DataFrame for testing
        swapped = df.copy()
        for h_col, a_col in cls.PAIRED_COLUMNS:
            if h_col in df.columns and a_col in df.columns:
                swapped[h_col] = df[a_col].copy()
                swapped[a_col] = df[h_col].copy()
        if "home_away_min_dist_diff" in df.columns:
            swapped["home_away_min_dist_diff"] = -df["home_away_min_dist_diff"].copy()
        if "gt_possession_team" in df.columns:
            swapped["gt_possession_team"] = df["gt_possession_team"].map(
                lambda v: TeamID.AWAY.value if v == TeamID.HOME.value else (
                    TeamID.HOME.value if v == TeamID.AWAY.value else v
                )
            )
        return swapped


# ============================================================================
# 3. PURE-NUMPY GBDT FALLBACK & SYMMETRIC GBDT EMISSION MODEL (RFC Sec 5.2)
# ============================================================================

class PureNumpyGBDT:
    # Lightweight, high-performance gradient boosted decision stumps in pure NumPy (log-loss)
    # used automatically if neither LightGBM nor Scikit-learn is installed.
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.15):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees: List[Tuple[int, float, float, float]] = []
        self.base_logit: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        if n_samples == 0:
            return
        pos_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
        self.base_logit = float(np.log(pos_rate / (1.0 - pos_rate)))
        f_pred = np.full(n_samples, self.base_logit, dtype=float)
        self.trees = []

        # Precompute candidate split masks across all features to avoid repetitive percentiles
        candidate_splits: List[Tuple[int, float, np.ndarray]] = []
        for j in range(n_features):
            col = X[:, j]
            quantiles = np.unique(np.percentile(col, [15, 30, 50, 70, 85]))
            for thr in quantiles:
                mask = col <= thr
                cnt = int(np.count_nonzero(mask))
                if 0 < cnt < n_samples:
                    candidate_splits.append((j, float(thr), mask))

        for _ in range(self.n_estimators):
            prob = self._sigmoid(f_pred)
            residual = y - prob
            hessian = np.maximum(prob * (1.0 - prob), 1e-4)

            g_total = float(np.sum(residual))
            h_total = float(np.sum(hessian))

            best_gain = -1.0
            best_tree = (0, 0.0, 0.0, 0.0)
            best_mask: Optional[np.ndarray] = None

            for j, thr, left_m in candidate_splits:
                g_left = float(np.sum(residual[left_m]))
                h_left = float(np.sum(hessian[left_m])) + 1.0
                g_right = g_total - g_left
                h_right = (h_total - (h_left - 1.0)) + 1.0

                gain = 0.5 * ((g_left ** 2) / h_left + (g_right ** 2) / h_right)
                if gain > best_gain:
                    best_gain = gain
                    v_left = g_left / h_left
                    v_right = g_right / h_right
                    best_tree = (j, thr, float(v_left), float(v_right))
                    best_mask = left_m

            if best_gain <= 0.0 or best_mask is None:
                break

            feat_idx, thr, v_left, v_right = best_tree
            step = np.where(best_mask, v_left, v_right)
            f_pred += self.learning_rate * step
            self.trees.append(best_tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        f_pred = np.full(X.shape[0], self.base_logit, dtype=float)
        for feat_idx, thr, v_left, v_right in self.trees:
            f_pred += self.learning_rate * np.where(X[:, feat_idx] <= thr, v_left, v_right)
        p_home = self._sigmoid(f_pred)
        return np.column_stack([1.0 - p_home, p_home])


class SymmetricGBDTEmissionModel:
    # Single GBDT classifier trained with Cross-Entropy Loss on team-swap augmented pairs
    # and evaluated via antisymmetric pairing to guarantee exact Team-Swap Consistency:
    # P_swapped(Home) == 1.0 - P_original(Home) (RFC v2.0 Section 5.2).

    def __init__(
        self,
        n_estimators: int = 60,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Any = None
        self.backend_name: str = "uninitialized"
        self.platt_scale_a: float = 1.0
        self.platt_scale_b: float = 0.0

    def _build_base_estimator(self) -> Any:
        if HAS_LIGHTGBM:
            self.backend_name = "LightGBM"
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=min(15, 2 ** self.max_depth - 1),
                min_child_samples=5,
                objective="binary",
                random_state=self.random_state,
                verbose=-1,
            )
        if HAS_SKLEARN:
            self.backend_name = "Sklearn_HistGBDT"
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=self.random_state,
            )
        self.backend_name = "PureNumpy_GBDT"
        return PureNumpyGBDT(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
        )

    def fit(self, df: pd.DataFrame) -> None:
        if "gt_possession_team" not in df.columns:
            raise ValueError("DataFrame passed to SymmetricGBDTEmissionModel.fit must contain 'gt_possession_team'.")

        eval_m = df["eval_mask"].fillna(True).to_numpy(dtype=bool) if "eval_mask" in df.columns else np.ones(len(df), dtype=bool)
        ball_m = df["ball_is_valid"].fillna(False).to_numpy(dtype=bool) if "ball_is_valid" in df.columns else np.ones(len(df), dtype=bool)
        dead_m = df["is_dead_ball"].fillna(False).to_numpy(dtype=bool) if "is_dead_ball" in df.columns else np.zeros(len(df), dtype=bool)

        gt_raw = df["gt_possession_team"].to_numpy(dtype=str)
        is_labeled_team = np.isin(gt_raw, [TeamID.HOME.value, TeamID.AWAY.value])

        valid_mask = eval_m & ball_m & (~dead_m) & is_labeled_team
        train_df = df.loc[valid_mask].reset_index(drop=True)
        if len(train_df) == 0:
            raise ValueError("No valid training frames available after applying masks.")

        X_orig = SymmetricFeatureBuilder.extract_matrix(train_df)
        y_orig = (train_df["gt_possession_team"].to_numpy(dtype=str) == TeamID.HOME.value).astype(int)

        # Augment with team-swapped counterparts so the GBDT learns balanced splits
        X_swap = SymmetricFeatureBuilder.swap_feature_matrix(X_orig)
        y_swap = 1 - y_orig

        X_aug = np.vstack([X_orig, X_swap])
        y_aug = np.concatenate([y_orig, y_swap])

        self.model = self._build_base_estimator()
        self.model.fit(X_aug, y_aug)

    def predict_raw_home_prob(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SymmetricGBDTEmissionModel must be fit before calling predict.")

        n = len(df)
        if n == 0:
            return np.zeros(0, dtype=float)

        X_orig = SymmetricFeatureBuilder.extract_matrix(df)
        X_swap = SymmetricFeatureBuilder.swap_feature_matrix(X_orig)

        p_orig = self.model.predict_proba(X_orig)[:, 1]
        p_swap = self.model.predict_proba(X_swap)[:, 1]

        # Antisymmetric paired averaging guarantees exact Team-Swap Consistency
        p_sym = 0.5 * (p_orig + (1.0 - p_swap))

        # Apply zero-centered symmetric Platt calibration
        logit = np.log(np.clip(p_sym, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_sym, 1e-6, 1.0 - 1e-6))
        cal_logit = self.platt_scale_a * logit
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(cal_logit, -20.0, 20.0)))

        # Return neutral 0.5 probability where ball tracking is missing
        valid_ball = df["ball_is_valid"].fillna(False).to_numpy(dtype=bool) if "ball_is_valid" in df.columns else np.ones(n, dtype=bool)
        return np.where(valid_ball, p_cal, 0.5)

    def fit_oof_calibration(self, oof_probs: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray) -> float:
        # Tunes symmetric temperature/Platt scaling parameter `platt_scale_a` on Out-of-Fold
        # predictions to minimize cross-entropy (Log Loss) without breaking team-swap symmetry.
        p = oof_probs[valid_mask]
        y = y_true[valid_mask]
        if len(p) == 0:
            return 1.0

        logits = np.log(np.clip(p, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p, 1e-6, 1.0 - 1e-6))

        try:
            from scipy.optimize import minimize_scalar
            def loss_fn(a: float) -> float:
                cal_p = 1.0 / (1.0 + np.exp(-np.clip(a * logits, -20.0, 20.0)))
                cal_p = np.clip(cal_p, 1e-6, 1.0 - 1e-6)
                log_loss = -float(np.mean(y * np.log(cal_p) + (1.0 - y) * np.log(1.0 - cal_p)))
                # Subtle regularization towards 1.0 to avoid unbounded temperature on separable sets
                return log_loss + 1e-4 * (a - 1.0) ** 2

            res = minimize_scalar(loss_fn, bounds=(0.2, 4.0), method="bounded")
            best_a = float(res.x) if res.success else 1.0
        except Exception:
            best_a = 1.0
            best_loss = 1e9
            for candidate_a in np.linspace(0.2, 4.0, 77):
                cal_p = 1.0 / (1.0 + np.exp(-np.clip(candidate_a * logits, -20.0, 20.0)))
                cal_p = np.clip(cal_p, 1e-6, 1.0 - 1e-6)
                loss = -float(np.mean(y * np.log(cal_p) + (1.0 - y) * np.log(1.0 - cal_p)))
                if loss < best_loss:
                    best_loss = loss
                    best_a = float(candidate_a)

        self.platt_scale_a = best_a
        return best_a


# ============================================================================
# 4. SOFT DISCRIMINATIVE VITERBI DECODER (RFC v2.0 Section 4.6 & 5.3)
# ============================================================================

class SoftViterbiDecoder:
    # Decodes the optimal possession state sequence within each continuous valid segment
    # by maximizing:
    #   score(y) = sum_t log p_t(y_t) + sum_t transition_score(y_{t-1}, y_t, x_t)
    # where the switch cost decreases ONLY when both contact/velocity-change evidence
    # AND incoming opponent control evidence rise, and NEVER relaxes during tracking artifacts.

    def __init__(self, config: Optional[DecoderConfig] = None):
        self.config = config or DecoderConfig()

    def compute_directional_switch_penalties(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Returns (penalty_away_to_home, penalty_home_to_away) as positive cost arrays >= min_switch_penalty
        n = len(df)
        if n == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        c_conf = df["candidate_contact_confidence"].fillna(0.0).to_numpy(dtype=float) if "candidate_contact_confidence" in df.columns else np.zeros(n, dtype=float)
        dv_arr = df["ball_vel_change_proxy_mps"].fillna(0.0).to_numpy(dtype=float) if "ball_vel_change_proxy_mps" in df.columns else np.zeros(n, dtype=float)
        dv_norm = np.clip(dv_arr / 8.0, 0.0, 1.0)

        h_touch = df["home_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_touch_evidence" in df.columns else np.zeros(n, dtype=float)
        h_carry = df["home_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_carry_evidence" in df.columns else np.zeros(n, dtype=float)
        anc_h = df["anchor_home_prob"].fillna(0.5).to_numpy(dtype=float) if "anchor_home_prob" in df.columns else np.full(n, 0.5, dtype=float)

        a_touch = df["away_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_touch_evidence" in df.columns else np.zeros(n, dtype=float)
        a_carry = df["away_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_carry_evidence" in df.columns else np.zeros(n, dtype=float)
        anc_a = df["anchor_away_prob"].fillna(0.5).to_numpy(dtype=float) if "anchor_away_prob" in df.columns else np.full(n, 0.5, dtype=float)

        # Incoming control evidence for Home (when switching Away -> Home)
        in_home_ctrl = np.maximum.reduce([h_touch, h_carry, anc_h * c_conf])

        # Incoming control evidence for Away (when switching Home -> Away)
        in_away_ctrl = np.maximum.reduce([a_touch, a_carry, anc_a * c_conf])

        # Tracking artifact protection factor: never relax transition costs during
        # interpolated gaps, unreliable segment boundaries, or high aerial pass-by likelihood.
        is_interp = df["ball_is_interpolated"].fillna(False).to_numpy(dtype=bool) if "ball_is_interpolated" in df.columns else np.zeros(n, dtype=bool)
        unrel_bound = df["ball_boundary_unreliable"].fillna(True).to_numpy(dtype=bool) if "ball_boundary_unreliable" in df.columns else np.zeros(n, dtype=bool)
        unrel_prox = df["unreliable_proximity_likelihood"].fillna(0.0).to_numpy(dtype=float) if "unreliable_proximity_likelihood" in df.columns else np.zeros(n, dtype=float)

        artifact_guard = (
            (~is_interp).astype(float)
            * (~unrel_bound).astype(float)
            * (1.0 - 0.75 * np.clip(unrel_prox, 0.0, 1.0))
        )

        w_c = self.config.contact_vs_dv_weight
        kin_trigger = w_c * c_conf + (1.0 - w_c) * dv_norm

        turnover_ev_to_home = np.clip(artifact_guard * kin_trigger * in_home_ctrl, 0.0, 1.0)
        turnover_ev_to_away = np.clip(artifact_guard * kin_trigger * in_away_ctrl, 0.0, 1.0)

        base_p = self.config.base_switch_penalty
        min_p = self.config.min_switch_penalty
        gain = self.config.turnover_relax_gain

        pen_a_to_h = np.maximum(min_p, base_p * (1.0 - gain * turnover_ev_to_home))
        pen_h_to_a = np.maximum(min_p, base_p * (1.0 - gain * turnover_ev_to_away))
        return pen_a_to_h, pen_h_to_a

    # Maintain alias for backward compatibility with PEP 8 callers
    compute_Directional_switch_penalties = compute_directional_switch_penalties

    def decode(self, df: pd.DataFrame, emission_home_prob: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        if n == 0:
            out["emission_home_prob"] = pd.Series(dtype=float)
            out["decoded_home_prob"] = pd.Series(dtype=float)
            out["predicted_possession_team"] = pd.Series(dtype=str)
            out["diagnostic_confidence"] = pd.Series(dtype=float)
            return out

        eps = self.config.prob_clip_eps
        p_home = np.clip(emission_home_prob, eps, 1.0 - eps)
        p_away = 1.0 - p_home

        pen_a_to_h, pen_h_to_a = self.compute_directional_switch_penalties(out)

        decoded_home_int = np.zeros(n, dtype=int)
        posterior_home_prob = p_home.copy()

        # Derive ball_valid_seg_id if not present
        if "ball_valid_seg_id" not in out.columns:
            if "ball_is_valid" in out.columns:
                valid = out["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
                seg_breaks = np.concatenate([[True], valid[1:] != valid[:-1]])
                out["ball_valid_seg_id"] = np.where(valid, np.cumsum(seg_breaks), -1)
            else:
                out["ball_valid_seg_id"] = 0

        # Decode independently within each continuous valid ball segment
        for seg_id, group in out.groupby("ball_valid_seg_id", sort=False):
            pos_idx = group.index.to_numpy(dtype=int)
            if seg_id < 0 or len(pos_idx) == 0:
                # Missing ball tracking: retain neutral 0.5 probability
                posterior_home_prob[pos_idx] = 0.5
                decoded_home_int[pos_idx] = 0
                continue

            m = len(pos_idx)
            log_emit = np.column_stack([
                np.log(p_away[pos_idx]),
                np.log(p_home[pos_idx]),
            ])
            c_ah = pen_a_to_h[pos_idx]
            c_ha = pen_h_to_a[pos_idx]

            # Forward Viterbi dynamic programming table: state 0 = Away, state 1 = Home
            dp = np.zeros((m, 2), dtype=float)
            backptr = np.zeros((m, 2), dtype=int)
            dp[0, 0] = log_emit[0, 0]
            dp[0, 1] = log_emit[0, 1]

            for t in range(1, m):
                # Target state 0 (Away): stay 0->0 (cost 0) vs switch 1->0 (cost c_ha[t])
                score_00 = dp[t - 1, 0]
                score_10 = dp[t - 1, 1] - c_ha[t]
                if score_00 >= score_10:
                    dp[t, 0] = score_00 + log_emit[t, 0]
                    backptr[t, 0] = 0
                else:
                    dp[t, 0] = score_10 + log_emit[t, 0]
                    backptr[t, 0] = 1

                # Target state 1 (Home): stay 1->1 (cost 0) vs switch 0->1 (cost c_ah[t])
                score_11 = dp[t - 1, 1]
                score_01 = dp[t - 1, 0] - c_ah[t]
                if score_11 >= score_01:
                    dp[t, 1] = score_11 + log_emit[t, 1]
                    backptr[t, 1] = 1
                else:
                    dp[t, 1] = score_01 + log_emit[t, 1]
                    backptr[t, 1] = 0

            # Backtrack optimal state sequence
            path = np.zeros(m, dtype=int)
            path[m - 1] = 1 if dp[m - 1, 1] >= dp[m - 1, 0] else 0
            for t in range(m - 1, 0, -1):
                path[t - 1] = backptr[t, path[t]]

            decoded_home_int[pos_idx] = path

            # Exact Backward Max-Marginal pass to eliminate forward-only temporal lag
            # and guarantee that posterior_home_prob >= 0.5 perfectly coincides with the Viterbi choice.
            dp_bwd = np.zeros((m, 2), dtype=float)
            for t in range(m - 2, -1, -1):
                # From state 0 at t: stay 0->0 (cost 0) vs switch 0->1 (cost c_ah[t+1])
                to0 = dp_bwd[t + 1, 0] + log_emit[t + 1, 0]
                to1 = dp_bwd[t + 1, 1] + log_emit[t + 1, 1] - c_ah[t + 1]
                dp_bwd[t, 0] = max(to0, to1)

                # From state 1 at t: stay 1->1 (cost 0) vs switch 1->0 (cost c_ha[t+1])
                to1_from1 = dp_bwd[t + 1, 1] + log_emit[t + 1, 1]
                to0_from1 = dp_bwd[t + 1, 0] + log_emit[t + 1, 0] - c_ha[t + 1]
                dp_bwd[t, 1] = max(to1_from1, to0_from1)

            # Bidirectional max-marginal logit difference
            margin_logit = (dp[:, 1] + dp_bwd[:, 1]) - (dp[:, 0] + dp_bwd[:, 0])
            posterior_home_prob[pos_idx] = 1.0 / (1.0 + np.exp(-np.clip(margin_logit, -15.0, 15.0)))

        valid_ball = out["ball_is_valid"].fillna(False).to_numpy(dtype=bool) if "ball_is_valid" in out.columns else np.ones(n, dtype=bool)
        out["emission_home_prob"] = p_home
        out["decoded_home_prob"] = np.where(valid_ball, posterior_home_prob, 0.5)
        out["predicted_possession_team"] = np.where(
            ~valid_ball,
            "Unknown_Missing_Ball",
            np.where(decoded_home_int == 1, TeamID.HOME.value, TeamID.AWAY.value),
        )
        out["diagnostic_confidence"] = np.where(
            ~valid_ball,
            0.0,
            np.abs(2.0 * posterior_home_prob - 1.0),
        )
        return out

    def tune_on_oof(
        self,
        oof_dfs: List[pd.DataFrame],
        oof_probs: List[np.ndarray],
    ) -> float:
        # Tunes base_switch_penalty strictly on Out-of-Fold validation predictions
        best_penalty = self.config.base_switch_penalty
        best_score = -1e9

        for candidate_pen in [1.2, 2.0, 2.8, 3.6, 4.5]:
            self.config.base_switch_penalty = candidate_pen
            total_correct = 0
            total_valid = 0
            false_switches = 0

            for df_fold, p_fold in zip(oof_dfs, oof_probs):
                dec_df = self.decode(df_fold, p_fold)
                eval_m = dec_df["eval_mask"].to_numpy(dtype=bool) if "eval_mask" in dec_df.columns else np.ones(len(dec_df), dtype=bool)
                ball_m = dec_df["ball_is_valid"].to_numpy(dtype=bool) if "ball_is_valid" in dec_df.columns else np.ones(len(dec_df), dtype=bool)
                mask = eval_m & ball_m
                if "period_id" in dec_df.columns:
                    mask = mask & (dec_df["period_id"].to_numpy(dtype=int) > 0)

                if not mask.any() or "gt_possession_team" not in dec_df.columns:
                    continue

                sub_df = dec_df.loc[mask]
                pred = sub_df["predicted_possession_team"].to_numpy(dtype=str)
                gt = sub_df["gt_possession_team"].to_numpy(dtype=str)
                total_correct += int(np.sum(pred == gt))
                total_valid += len(gt)

                # Count switches strictly within continuous segments to avoid false switch penalties at segment breaks
                seg_cols = [c for c in ["period_id", "ball_valid_seg_id"] if c in sub_df.columns]
                if seg_cols:
                    for _, seg_grp in sub_df.groupby(seg_cols, sort=False):
                        p_seq = seg_grp["predicted_possession_team"].to_numpy(dtype=str)
                        g_seq = seg_grp["gt_possession_team"].to_numpy(dtype=str)
                        if len(p_seq) > 1:
                            p_sw = int(np.sum(p_seq[1:] != p_seq[:-1]))
                            g_sw = int(np.sum(g_seq[1:] != g_seq[:-1]))
                            false_switches += max(0, p_sw - g_sw)
                else:
                    if len(pred) > 1:
                        p_sw = int(np.sum(pred[1:] != pred[:-1]))
                        g_sw = int(np.sum(gt[1:] != gt[:-1]))
                        false_switches += max(0, p_sw - g_sw)

            acc = total_correct / max(1, total_valid)
            # Composite OOF objective rewarding frame accuracy while penalizing false flicker switches
            score = acc - 0.03 * false_switches
            if score > best_score:
                best_score = score
                best_penalty = candidate_pen

        self.config.base_switch_penalty = best_penalty
        return best_penalty


# ============================================================================
# 5. END-TO-END MATCH FEATURE BUILDER & GROUPED OOF VALIDATION HARNESS
# ============================================================================

def build_full_match_features(
    raw_ds: ValidatedTrackingDataset,
    use_jev: bool = True,
    use_live_api: bool = False,
) -> pd.DataFrame:
    prep = SegmentedPreprocessor()
    proc_ds = prep.process(raw_ds)

    t2_ext = LocalKinematicFeatureExtractor()
    t2_df = t2_ext.extract_features(proc_ds)

    ab_cfg = AnchorBridgeConfig(enable_jev_refiner=use_jev)
    t3_eng = KinematicContactAndBridgeEngine(
        config=ab_cfg, jev_refiner=JevSystemOneRefiner(use_live_api=use_live_api)
    )
    t3_df = t3_eng.extract_anchor_and_bridge_features(t2_df, proc_ds.metadata)

    sym_builder = SymmetricFeatureBuilder()
    full_df = sym_builder.add_segmented_temporal_windows(t3_df, proc_ds.metadata)
    full_df["match_id"] = proc_ds.metadata.match_id
    return full_df


def run_leave_one_match_out_oof(
    match_dfs: List[pd.DataFrame],
) -> Tuple[SymmetricGBDTEmissionModel, SoftViterbiDecoder, List[pd.DataFrame]]:
    # Runs leakage-free Leave-One-Match-Out cross-validation to generate Out-of-Fold
    # emission predictions, calibrates probabilities, and tunes Soft Viterbi on OOF.
    if len(match_dfs) == 0:
        raise ValueError("match_dfs list cannot be empty.")
    if len(match_dfs) == 1:
        raise ValueError(
            "Leave-One-Match-Out cross-validation requires at least 2 matches (len(match_dfs) >= 2). "
            "For single-match validation, use DependencyPurgeCalculator.generate_purged_time_splits()."
        )

    oof_probs_list: List[np.ndarray] = []
    all_oof_probs: List[np.ndarray] = []
    all_oof_y: List[np.ndarray] = []
    all_oof_mask: List[np.ndarray] = []

    for i in range(len(match_dfs)):
        train_parts = [match_dfs[j] for j in range(len(match_dfs)) if j != i]
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = match_dfs[i]

        fold_model = SymmetricGBDTEmissionModel()
        fold_model.fit(train_df)
        p_val = fold_model.predict_raw_home_prob(val_df)

        oof_probs_list.append(p_val)
        all_oof_probs.append(p_val)
        all_oof_y.append(
            (val_df["gt_possession_team"].to_numpy(dtype=str) == TeamID.HOME.value).astype(int)
        )
        all_oof_mask.append(
            (val_df["eval_mask"].to_numpy(dtype=bool) if "eval_mask" in val_df.columns else np.ones(len(val_df), dtype=bool))
            & (val_df["ball_is_valid"].to_numpy(dtype=bool) if "ball_is_valid" in val_df.columns else np.ones(len(val_df), dtype=bool))
        )

    # Fit final emission model on all training matches and apply OOF calibration scale
    final_model = SymmetricGBDTEmissionModel()
    final_model.fit(pd.concat(match_dfs, ignore_index=True))
    final_model.fit_oof_calibration(
        np.concatenate(all_oof_probs),
        np.concatenate(all_oof_y),
        np.concatenate(all_oof_mask),
    )

    # Tune Soft Viterbi transition penalty strictly on Out-of-Fold predictions
    decoder = SoftViterbiDecoder()
    decoder.tune_on_oof(match_dfs, oof_probs_list)

    decoded_oof_dfs = [
        decoder.decode(df_m, p_m) for df_m, p_m in zip(match_dfs, oof_probs_list)
    ]
    return final_model, decoder, decoded_oof_dfs


# ============================================================================
# 6. AUTOMATED VERIFICATION & AUDIT COMPLIANCE SUITE FOR TASK 4
# ============================================================================

def run_task4_verification() -> None:
    print("=" * 72)
    print("RUNNING TASK 4 GBDT, TEAM-SWAP SYMMETRY & SOFT VITERBI VERIFICATION")
    print("=" * 72)

    # 1. Verify Analytical Dependency-Chain Purge Buffer calculation & within-match splits (RFC Section 6.1)
    purge_calc = DependencyPurgeCalculator()
    purge_s = purge_calc.compute_purge_buffer_seconds()
    assert purge_s >= 7.0, f"Purge buffer must account for full dependency chain (>=7.0s), got {purge_s}s"

    dummy_df = pd.DataFrame({"timestamp_s": np.linspace(0.0, 100.0, 1000)})
    splits = purge_calc.generate_purged_time_splits(dummy_df, n_splits=5)
    assert len(splits) == 5, f"Expected 5 purged splits, got {len(splits)}"
    for tr_m, val_m in splits:
        assert not np.any(tr_m & val_m), "Train and validation sets must be mutually exclusive."
        v_times = dummy_df.loc[val_m, "timestamp_s"].to_numpy()
        tr_times = dummy_df.loc[tr_m, "timestamp_s"].to_numpy()
        if len(v_times) > 0 and len(tr_times) > 0:
            v_start, v_end = v_times.min(), v_times.max()
            assert np.all((tr_times < v_start - purge_s + 1e-4) | (tr_times > v_end + purge_s - 1e-4)), (
                "Train times must respect purge buffer around validation block."
            )
    print(f"[PASS] 1. Analytical dependency-chain purge buffer verified (purge_buffer = {purge_s:.2f}s).")

    # 2. Build 3 independent synthetic matches with different random seeds for Leave-One-Match-Out OOF
    match_dfs: List[pd.DataFrame] = []
    for m_idx, seed_val in enumerate([42, 101, 202], start=1):
        raw_m = generate_synthetic_diagnostic_match(sampling_rate_hz=25.0, seed=seed_val)
        raw_m = ValidatedTrackingDataset(
            metadata=MatchMetadata(
                match_id=f"SYNTH_MATCH_{m_idx:02d}",
                sampling_rate_hz=raw_m.metadata.sampling_rate_hz,
                pitch_length_m=raw_m.metadata.pitch_length_m,
                pitch_width_m=raw_m.metadata.pitch_width_m,
                coord_semantics=raw_m.metadata.coord_semantics,
            ),
            players=raw_m.players,
            frame_df=raw_m.frame_df,
            player_tracks_df=raw_m.player_tracks_df,
        )
        match_dfs.append(build_full_match_features(raw_m, use_jev=True))

    model, decoder, decoded_oof = run_leave_one_match_out_oof(match_dfs)
    print(
        f"[PASS] 2. Leave-One-Match-Out OOF training & calibration completed "
        f"(Backend={model.backend_name}, Platt_a={model.platt_scale_a:.2f}, "
        f"OOF_switch_penalty={decoder.config.base_switch_penalty:.2f})."
    )

    # 3. Verify Strict Team-Swap Consistency on both GBDT Emissions and Viterbi Decoder (RFC Section 5.2)
    test_df = match_dfs[0]
    swapped_df = SymmetricFeatureBuilder.swap_dataframe_teams(test_df)

    p_orig = model.predict_raw_home_prob(test_df)
    p_swap = model.predict_raw_home_prob(swapped_df)
    max_sym_diff = float(np.max(np.abs(p_orig - (1.0 - p_swap))))
    assert max_sym_diff < 1e-9, f"Team-swap symmetry violated! Max diff = {max_sym_diff}"

    dec_orig = decoder.decode(test_df, p_orig)
    dec_swap = decoder.decode(swapped_df, p_swap)
    valid_m = test_df["ball_is_valid"].to_numpy(dtype=bool)
    orig_home_mask = (dec_orig.loc[valid_m, "predicted_possession_team"] == TeamID.HOME.value).to_numpy()
    swap_away_mask = (dec_swap.loc[valid_m, "predicted_possession_team"] == TeamID.AWAY.value).to_numpy()
    assert np.array_equal(orig_home_mask, swap_away_mask), "Decoded Viterbi path violated team-swap symmetry!"
    print(f"[PASS] 3. Exact Team-Swap Consistency verified (Max probability asymmetry = {max_sym_diff:.2e}).")

    # 4. Verify Pass-By Flicker Elimination vs. Naive Nearest-Player Proximity
    # During t in [1.56s, 2.00s], the ball passes close to Away defenders before the t=2.8s interception.
    m0_oof = decoded_oof[0]
    p1_valid = m0_oof[(m0_oof["period_id"] == 1) & m0_oof["ball_is_valid"]].copy()

    # Naive nearest-player baseline switches falsely during defender pass-by
    naive_pred = np.where(
        p1_valid["min_dist_home"].to_numpy() <= p1_valid["min_dist_away"].to_numpy(),
        TeamID.HOME.value,
        TeamID.AWAY.value,
    )
    viterbi_pred = p1_valid["predicted_possession_team"].to_numpy(dtype=str)
    gt_labels = p1_valid["gt_possession_team"].to_numpy(dtype=str)

    naive_acc = float(np.mean(naive_pred == gt_labels))
    viterbi_acc = float(np.mean(viterbi_pred == gt_labels))

    naive_switches = int(np.sum(naive_pred[1:] != naive_pred[:-1]))
    viterbi_switches = int(np.sum(viterbi_pred[1:] != viterbi_pred[:-1]))

    assert viterbi_acc > naive_acc, (
        f"Expected OOF GBDT+Viterbi accuracy ({viterbi_acc:.3f}) > Naive proximity ({naive_acc:.3f})."
    )
    assert viterbi_switches <= naive_switches, (
        f"Expected Soft Viterbi switches ({viterbi_switches}) <= Naive switches ({naive_switches})."
    )
    print(
        f"[PASS] 4. Held-out OOF evaluation verified: "
        f"Naive Proximity Acc = {naive_acc * 100:.1f}% ({naive_switches} switches) -> "
        f"GBDT + Soft Viterbi Acc = {viterbi_acc * 100:.1f}% ({viterbi_switches} switch)."
    )
    print("=" * 72)
    print("ALL TASK 4 CHECKS PASSED SUCCESSFULLY.")
    print("=" * 72)


if __name__ == "__main__":
    run_task4_verification()