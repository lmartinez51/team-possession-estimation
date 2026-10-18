# Task 5: Diagnostic Evaluation Suite, Outlier-Resistant Tactical Geometry Priors,
# Edge-Case Slice Reporting, and Ordered Ablation Study (RFC v2.0 Section 4.5 & 6).
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from task1_schema_and_preprocessing import (
    CoordinateSemantics,
    MatchMetadata,
    PlayerMeta,
    PlayerRole,
    SegmentedPreprocessor,
    TeamID,
    TrackingSchemaValidator,
    ValidatedTrackingDataset,
)
from task2_feature_engine import LocalKinematicFeatureExtractor
from task3_anchors_and_bridges import (
    AnchorBridgeConfig,
    JevSystemOneRefiner,
    KinematicContactAndBridgeEngine,
)
from task4_classifier_and_decoder import (
    SoftViterbiDecoder,
    SymmetricFeatureBuilder,
    SymmetricGBDTEmissionModel,
)


# ============================================================================
# 1. OUTLIER-RESISTANT TACTICAL GEOMETRY PRIORS (RFC v2.0 Section 4.5)
# ============================================================================

class TacticalGeometryExtractor:
    # Computes weak macro-tactical priors over valid outfield players while
    # handling goalkeepers consistently and comparing Convex Hull area against
    # outlier-resistant interquartile spread statistics (RFC v2.0 Section 4.5).

    PAIRED_TACTICAL_COLUMNS: List[Tuple[str, str]] = [
        ("tactical_iqr_spread_home", "tactical_iqr_spread_away"),
        ("tactical_hull_area_home", "tactical_hull_area_away"),
        ("tactical_attack_vel_home", "tactical_attack_vel_away"),
    ]

    @staticmethod
    def _safe_convex_hull_area(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        pts = np.column_stack([x, y])
        # Check rank / collinearity before calling Qhull
        if np.std(x) < 1e-3 or np.std(y) < 1e-3:
            return 0.0
        try:
            hull = ConvexHull(pts)
            return float(hull.volume)  # In 2D, scipy ConvexHull.volume is area
        except Exception:
            return 0.0

    @classmethod
    def extract_tactical_features(
        cls, dataset: ValidatedTrackingDataset, df: pd.DataFrame
    ) -> pd.DataFrame:
        out = df.sort_values("frame_id").reset_index(drop=True).copy()
        p_df = dataset.player_tracks_df.copy()
        meta = dataset.metadata

        n = len(out)
        iqr_home = np.zeros(n, dtype=float)
        iqr_away = np.zeros(n, dtype=float)
        hull_home = np.zeros(n, dtype=float)
        hull_away = np.zeros(n, dtype=float)
        avel_home = np.zeros(n, dtype=float)
        avel_away = np.zeros(n, dtype=float)

        if n == 0:
            out["tactical_iqr_spread_home"] = iqr_home
            out["tactical_iqr_spread_away"] = iqr_away
            out["tactical_hull_area_home"] = hull_home
            out["tactical_hull_area_away"] = hull_away
            out["tactical_attack_vel_home"] = avel_home
            out["tactical_attack_vel_away"] = avel_away
            return out

        # Ensure role and team_id mappings exist even if un-joined in tracking data
        if "role" not in p_df.columns:
            role_map = {pid: pmeta.role.value for pid, pmeta in dataset.players.items()}
            p_df["role"] = p_df["player_id"].map(role_map).fillna(PlayerRole.OUTFIELD.value)
        if "team_id" not in p_df.columns:
            team_map = {pid: pmeta.team_id.value for pid, pmeta in dataset.players.items()}
            p_df["team_id"] = p_df["player_id"].map(team_map).fillna(TeamID.UNKNOWN.value)

        players_by_frame = {
            int(fid): grp for fid, grp in p_df.groupby("frame_id", sort=False)
        }

        # Vectorized lookup of frame_id and period_id arrays (avoids n slow pd.Series allocations)
        fids = out["frame_id"].to_numpy(dtype=int)
        period_ids = out["period_id"].to_numpy(dtype=int)

        for i in range(n):
            fid = fids[i]
            period_id = period_ids[i]
            if fid not in players_by_frame:
                continue

            # Canonical attacking direction (+1 if attacking toward +x, -1 if toward -x)
            home_dir = 1.0 if meta.home_attacks_right_period1 else -1.0
            if period_id == 2:
                home_dir = -home_dir
            away_dir = -home_dir

            p_grp = players_by_frame[fid]
            # Exclude goalkeepers from outfield spatial spread (RFC Section 4.5)
            valid_outfield = p_grp[
                p_grp["x"].notna()
                & p_grp["y"].notna()
                & (p_grp["role"].astype(str) != PlayerRole.GK.value)
            ]
            if len(valid_outfield) == 0:
                continue

            px = valid_outfield["x"].to_numpy(dtype=float)
            py = valid_outfield["y"].to_numpy(dtype=float)
            pvx = (
                valid_outfield["vx"].fillna(0.0).to_numpy(dtype=float)
                if "vx" in valid_outfield.columns
                else np.zeros(len(valid_outfield), dtype=float)
            )
            teams = valid_outfield["team_id"].astype(str).to_numpy()

            h_mask = teams == TeamID.HOME.value
            a_mask = teams == TeamID.AWAY.value

            if np.sum(h_mask) >= 3:
                hx, hy, hvx = px[h_mask], py[h_mask], pvx[h_mask]
                iqr_x = float(np.percentile(hx, 75) - np.percentile(hx, 25))
                iqr_y = float(np.percentile(hy, 75) - np.percentile(hy, 25))
                iqr_home[i] = float(np.hypot(iqr_x, iqr_y))
                hull_home[i] = cls._safe_convex_hull_area(hx, hy)
                avel_home[i] = float(np.median(hvx) * home_dir)

            if np.sum(a_mask) >= 3:
                ax, ay, avx = px[a_mask], py[a_mask], pvx[a_mask]
                iqr_x = float(np.percentile(ax, 75) - np.percentile(ax, 25))
                iqr_y = float(np.percentile(ay, 75) - np.percentile(ay, 25))
                iqr_away[i] = float(np.hypot(iqr_x, iqr_y))
                hull_away[i] = cls._safe_convex_hull_area(ax, ay)
                avel_away[i] = float(np.median(avx) * away_dir)

        out["tactical_iqr_spread_home"] = iqr_home
        out["tactical_iqr_spread_away"] = iqr_away
        out["tactical_hull_area_home"] = hull_home
        out["tactical_hull_area_away"] = hull_away
        out["tactical_attack_vel_home"] = avel_home
        out["tactical_attack_vel_away"] = avel_away
        return out


# ============================================================================
# 2. COMPLEMENTARY DIAGNOSTIC METRICS & TURNOVER MATCHING (RFC v2.0 Sec 6.2)
# ============================================================================

@dataclass
class EvaluationMetrics:
    config_name: str
    frame_accuracy: float
    macro_f1: float
    brier_score: float
    log_loss: float
    turnover_f1: float
    turnover_precision: float
    turnover_recall: float
    mean_abs_timing_error_ms: float
    mean_signed_timing_error_ms: float
    false_switches_per_min: float
    short_segment_rate: float
    match_acc_mean: float
    match_acc_std: float
    match_bootstrap_ci_low: float
    match_bootstrap_ci_high: float


class PossessionDiagnosticEvaluator:
    # Implements the complete diagnostic metric suite required by RFC v2.0 Section 6.2,
    # including 1-to-1 bipartite turnover matching, timing error, false switches/min,
    # short-segment rate, probability calibration metrics, and match-level bootstrap.

    def __init__(
        self,
        turnover_tolerance_s: float = 0.25,
        short_segment_thresh_s: float = 0.50,
    ):
        self.turnover_tolerance_s = turnover_tolerance_s
        self.short_segment_thresh_s = short_segment_thresh_s

    @staticmethod
    def _compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        f1s: List[float] = []
        for cls_val in [0, 1]:
            tp = int(np.sum((y_true == cls_val) & (y_pred == cls_val)))
            fp = int(np.sum((y_true != cls_val) & (y_pred == cls_val)))
            fn = int(np.sum((y_true == cls_val) & (y_pred != cls_val)))
            denom = 2 * tp + fp + fn
            f1s.append(float(2 * tp / denom) if denom > 0 else 1.0)
        return float(0.5 * (f1s[0] + f1s[1]))

    def _extract_turnover_events(
        self, df: pd.DataFrame, label_col: str
    ) -> List[Tuple[int, float, str]]:
        # Extracts (segment_id, timestamp_s, new_team) for every possession switch
        # strictly within continuous valid ball segments.
        events: List[Tuple[int, float, str]] = []
        if len(df) == 0:
            return events

        eval_mask = (
            df["eval_mask"].fillna(True).to_numpy(dtype=bool)
            if "eval_mask" in df.columns
            else np.ones(len(df), dtype=bool)
        )
        ball_valid = (
            df["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_valid" in df.columns
            else np.ones(len(df), dtype=bool)
        )

        valid_df = df.loc[eval_mask & ball_valid].copy()
        if len(valid_df) == 0:
            return events

        if "ball_valid_seg_id" not in valid_df.columns:
            valid_df["ball_valid_seg_id"] = (~ball_valid[eval_mask & ball_valid]).cumsum()

        for seg_id, group in valid_df.groupby("ball_valid_seg_id", sort=False):
            if int(seg_id) < 0 or len(group) < 2:
                continue
            ts = group["timestamp_s"].to_numpy(dtype=float)
            labels = group[label_col].astype(str).to_numpy()
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    events.append((int(seg_id), float(ts[i]), str(labels[i])))
        return events

    def _match_turnovers_one_to_one(
        self,
        gt_events: List[Tuple[int, float, str]],
        pred_events: List[Tuple[int, float, str]],
    ) -> Tuple[int, int, int, List[float]]:
        # Greedy 1-to-1 bipartite matching sorted by temporal distance within each
        # continuous segment and matching transition direction (new_team).
        matched_gt = set()
        matched_pred = set()
        signed_errors_ms: List[float] = []

        candidates: List[Tuple[float, int, int, float]] = []
        for g_idx, (g_seg, g_t, g_team) in enumerate(gt_events):
            for p_idx, (p_seg, p_t, p_team) in enumerate(pred_events):
                if g_seg == p_seg and g_team == p_team:
                    signed_dt = p_t - g_t
                    abs_dt = abs(signed_dt)
                    if abs_dt <= self.turnover_tolerance_s + 1e-6:
                        candidates.append((abs_dt, g_idx, p_idx, signed_dt * 1000.0))

        candidates.sort(key=lambda item: item[0])
        for _, g_idx, p_idx, err_ms in candidates:
            if g_idx not in matched_gt and p_idx not in matched_pred:
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                signed_errors_ms.append(err_ms)

        tp = len(matched_gt)
        fp = len(pred_events) - len(matched_pred)
        fn = len(gt_events) - len(matched_gt)
        return tp, fp, fn, signed_errors_ms

    def _compute_short_segment_stats(
        self, df: pd.DataFrame, pred_col: str, dt_s: float
    ) -> Tuple[int, int]:
        # Returns (short_segments_count, total_segments_count)
        if len(df) == 0:
            return 0, 1

        eval_mask = (
            df["eval_mask"].fillna(True).to_numpy(dtype=bool)
            if "eval_mask" in df.columns
            else np.ones(len(df), dtype=bool)
        )
        ball_valid = (
            df["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_valid" in df.columns
            else np.ones(len(df), dtype=bool)
        )

        valid_df = df.loc[eval_mask & ball_valid].copy()
        if len(valid_df) == 0:
            return 0, 1

        if "ball_valid_seg_id" not in valid_df.columns:
            valid_df["ball_valid_seg_id"] = (~ball_valid[eval_mask & ball_valid]).cumsum()

        short_cnt = 0
        total_cnt = 0

        for seg_id, group in valid_df.groupby("ball_valid_seg_id", sort=False):
            if int(seg_id) < 0 or len(group) == 0:
                continue
            labels = group[pred_col].astype(str).to_numpy()
            n = len(labels)
            i = 0
            while i < n:
                j = i
                while j < n and labels[j] == labels[i]:
                    j += 1
                dur_s = (j - i) * dt_s
                total_cnt += 1
                # Exclude segments truncated right at the boundary of the clip
                if i > 0 and j < n and dur_s < self.short_segment_thresh_s:
                    short_cnt += 1
                i = j
        return short_cnt, max(1, total_cnt)

    def evaluate_oof_matches(
        self,
        config_name: str,
        match_dfs: List[pd.DataFrame],
        prob_col: str = "decoded_home_prob",
        pred_col: str = "predicted_possession_team",
        sampling_rate_hz: float = 25.0,
    ) -> EvaluationMetrics:
        dt_s = 1.0 / sampling_rate_hz
        per_match_accs: List[float] = []

        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        total_tp, total_fp, total_fn = 0, 0, 0
        all_signed_errors_ms: List[float] = []
        total_short_segs, total_segs = 0, 0
        total_valid_duration_min = 0.0

        for df in match_dfs:
            mask = (
                df["eval_mask"].fillna(True).to_numpy(dtype=bool)
                & df["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
                & (~df["is_dead_ball"].fillna(False).to_numpy(dtype=bool))
            )
            if not mask.any():
                continue

            sub = df.loc[mask]
            y_true = (sub["gt_possession_team"].astype(str).to_numpy() == TeamID.HOME.value).astype(int)
            y_pred = (sub[pred_col].astype(str).to_numpy() == TeamID.HOME.value).astype(int)
            p_home = np.clip(sub[prob_col].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)

            per_match_accs.append(float(np.mean(y_true == y_pred)))
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_probs.append(p_home)

            total_valid_duration_min += (len(sub) * dt_s) / 60.0

            gt_ev = self._extract_turnover_events(df, "gt_possession_team")
            pr_ev = self._extract_turnover_events(df, pred_col)
            tp, fp, fn, err_ms = self._match_turnovers_one_to_one(gt_ev, pr_ev)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_signed_errors_ms.extend(err_ms)

            s_cnt, t_cnt = self._compute_short_segment_stats(df, pred_col, dt_s)
            total_short_segs += s_cnt
            total_segs += t_cnt

        if len(all_y_true) == 0:
            return EvaluationMetrics(
                config_name=config_name,
                frame_accuracy=0.0,
                macro_f1=0.0,
                brier_score=0.5,
                log_loss=0.693,
                turnover_f1=0.0,
                turnover_precision=0.0,
                turnover_recall=0.0,
                mean_abs_timing_error_ms=0.0,
                mean_signed_timing_error_ms=0.0,
                false_switches_per_min=0.0,
                short_segment_rate=0.0,
                match_acc_mean=0.0,
                match_acc_std=0.0,
                match_bootstrap_ci_low=0.0,
                match_bootstrap_ci_high=0.0,
            )

        y_t = np.concatenate(all_y_true)
        y_p = np.concatenate(all_y_pred)
        p_h = np.concatenate(all_probs)

        frame_acc = float(np.mean(y_t == y_p))
        macro_f1 = self._compute_macro_f1(y_t, y_p)
        brier = float(np.mean((p_h - y_t) ** 2))
        logloss = -float(np.mean(y_t * np.log(p_h) + (1.0 - y_t) * np.log(1.0 - p_h)))

        t_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 1.0
        t_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 1.0
        t_f1 = (
            float(2.0 * t_prec * t_rec / (t_prec + t_rec))
            if (t_prec + t_rec) > 0
            else 0.0
        )

        abs_err = float(np.mean(np.abs(all_signed_errors_ms))) if all_signed_errors_ms else 0.0
        sig_err = float(np.mean(all_signed_errors_ms)) if all_signed_errors_ms else 0.0
        false_sw_per_min = float(total_fp / max(1e-6, total_valid_duration_min))
        short_rate = float(total_short_segs / max(1, total_segs))

        # Match-level bootstrap confidence interval (RFC v2.0 Section 6.2)
        m_arr = np.array(per_match_accs, dtype=float)
        m_mean = float(np.mean(m_arr))
        m_std = float(np.std(m_arr, ddof=1)) if len(m_arr) > 1 else 0.0

        rng = np.random.default_rng(42)
        boot_means = [
            float(np.mean(rng.choice(m_arr, size=len(m_arr), replace=True)))
            for _ in range(500)
        ]
        ci_low = float(np.percentile(boot_means, 5))
        ci_high = float(np.percentile(boot_means, 95))

        return EvaluationMetrics(
            config_name=config_name,
            frame_accuracy=frame_acc,
            macro_f1=macro_f1,
            brier_score=brier,
            log_loss=logloss,
            turnover_f1=t_f1,
            turnover_precision=t_prec,
            turnover_recall=t_rec,
            mean_abs_timing_error_ms=abs_err,
            mean_signed_timing_error_ms=sig_err,
            false_switches_per_min=false_sw_per_min,
            short_segment_rate=short_rate,
            match_acc_mean=m_mean,
            match_acc_std=m_std,
            match_bootstrap_ci_low=ci_low,
            match_bootstrap_ci_high=ci_high,
        )

    @staticmethod
    def evaluate_diagnostic_slices(
        match_dfs: List[pd.DataFrame],
        pred_col: str = "predicted_possession_team",
    ) -> Dict[str, float]:
        # Breaks down frame accuracy across critical failure slices (RFC Section 6.2)
        if not match_dfs:
            return {
                "PassBy_Near_Opponent_Acc": 1.0,
                "Interpolated_Gap_Acc": 1.0,
                "Contested_Turnover_Acc": 1.0,
                "Segment_Boundary_Acc": 1.0,
            }
        full = pd.concat(match_dfs, ignore_index=True)
        eval_mask = (
            full["eval_mask"].fillna(True).to_numpy(dtype=bool)
            if "eval_mask" in full.columns
            else np.ones(len(full), dtype=bool)
        )
        ball_valid = (
            full["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_valid" in full.columns
            else np.ones(len(full), dtype=bool)
        )
        base_valid = eval_mask & ball_valid

        correct = (
            full[pred_col].astype(str).to_numpy()
            == full["gt_possession_team"].astype(str).to_numpy()
        )

        unrel_arr = (
            full["unreliable_proximity_likelihood"].fillna(0.0).to_numpy(dtype=float)
            if "unreliable_proximity_likelihood" in full.columns
            else np.zeros(len(full), dtype=float)
        )
        interp_arr = (
            full["ball_is_interpolated"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_interpolated" in full.columns
            else np.zeros(len(full), dtype=bool)
        )
        cont_arr = (
            full["is_contested"].fillna(False).to_numpy(dtype=bool)
            if "is_contested" in full.columns
            else np.zeros(len(full), dtype=bool)
        )
        bound_arr = (
            full["ball_boundary_unreliable"].fillna(False).to_numpy(dtype=bool)
            if "ball_boundary_unreliable" in full.columns
            else np.zeros(len(full), dtype=bool)
        )

        # Slice 1: Fast pass-by near opponent (unreliable proximity likelihood >= 0.30)
        s_passby = base_valid & (unrel_arr >= 0.30)
        # Slice 2: Interpolated short tracking occlusion gaps
        s_interp = base_valid & interp_arr
        # Slice 3: Contested 50/50 or turnover boundary windows
        s_contest = base_valid & cont_arr
        # Slice 4: Segment boundary regions
        s_bound = base_valid & bound_arr

        return {
            "PassBy_Near_Opponent_Acc": float(np.mean(correct[s_passby])) if s_passby.any() else 1.0,
            "Interpolated_Gap_Acc": float(np.mean(correct[s_interp])) if s_interp.any() else 1.0,
            "Contested_Turnover_Acc": float(np.mean(correct[s_contest])) if s_contest.any() else 1.0,
            "Segment_Boundary_Acc": float(np.mean(correct[s_bound])) if s_bound.any() else 1.0,
        }


# ============================================================================
# 3. CHALLENGING MULTI-SCENARIO BENCHMARK GENERATOR FOR ABLATION
# ============================================================================

def generate_challenging_ablation_benchmark(
    n_matches: int = 4,
    sampling_rate_hz: float = 25.0,
) -> List[ValidatedTrackingDataset]:
    # Creates `n_matches` independent synthetic matches containing challenging
    # pass-bys, defender shadowing, one-touch interceptions, and tracking jitter
    # so that each step of the ordered ablation shows measurable differentiation.
    datasets: List[ValidatedTrackingDataset] = []
    dt = 1.0 / sampling_rate_hz

    for m_idx in range(1, n_matches + 1):
        rng = np.random.default_rng(1000 + m_idx * 37)
        meta = MatchMetadata(
            match_id=f"BENCH_MATCH_{m_idx:02d}",
            sampling_rate_hz=sampling_rate_hz,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            coord_semantics=CoordinateSemantics.PITCH_METERS,
            home_attacks_right_period1=(m_idx % 2 == 1),
        )

        players: Dict[str, PlayerMeta] = {}
        for i in range(1, 12):
            role = PlayerRole.GK if i == 1 else PlayerRole.OUTFIELD
            players[f"H_{i:02d}"] = PlayerMeta(f"H_{i:02d}", TeamID.HOME, role, i)
            players[f"A_{i:02d}"] = PlayerMeta(f"A_{i:02d}", TeamID.AWAY, role, i)

        # Vary turnover time slightly per match (around 2.8s)
        t_turnover = 2.72 + 0.08 * (m_idx % 3)
        n_p1 = int(round(6.0 * sampling_rate_hz))

        frame_rows: List[Dict[str, Any]] = []
        player_rows: List[Dict[str, Any]] = []

        for step in range(n_p1):
            t = round(step * dt, 6)
            fid = step + 1
            if t < 1.0:
                bx, by = -12.0 + 5.5 * t, 0.5 * t
                gt_team = TeamID.HOME.value
            elif t < t_turnover:
                # Long through-ball passing closely past two Away midfielders at t=1.4..2.3s
                bx = -6.5 + 13.0 * (t - 1.0)
                by = 0.5 + 2.5 * (t - 1.0)
                gt_team = TeamID.HOME.value
            else:
                # Clean interception and counter-carry/pass by Away defender A_04
                bx_int = -6.5 + 13.0 * (t_turnover - 1.0)
                by_int = 0.5 + 2.5 * (t_turnover - 1.0)
                bx = bx_int - 9.5 * (t - t_turnover)
                by = by_int - 4.0 * (t - t_turnover)
                gt_team = TeamID.AWAY.value

            bx_obs = bx + rng.normal(0.0, 0.04)
            by_obs = by + rng.normal(0.0, 0.04)

            if 1.80 <= t < 2.00:
                bx_obs, by_obs = np.nan, np.nan
            if 4.20 <= t < 4.90:
                bx_obs, by_obs = np.nan, np.nan

            frame_rows.append({
                "frame_id": fid,
                "timestamp_s": t,
                "period_id": 1,
                "ball_x": bx_obs,
                "ball_y": by_obs,
                "is_dead_ball": False,
                "is_contested": bool(abs(t - t_turnover) <= 0.12),
                "eval_mask": True,
                "camera_cut": False,
                "gt_possession_team": gt_team,
            })

            for pid, pmeta in players.items():
                if pid == "H_08":
                    # Home passer stops running right after kicking at t=1.0s
                    px = -12.0 + 5.5 * min(t, 1.08)
                    py = 0.5 * min(t, 1.08)
                elif pid == "A_06":
                    # Away midfielder bypassed closely at t=1.55s (creates false proximity spike)
                    px = 0.65 + 1.0 * (t - 1.55)
                    py = 1.85 + 0.2 * (t - 1.55)
                elif pid == "A_05":
                    # Second Away defender bypassed closely at t=2.05s
                    px = 7.15 - 0.8 * (t - 2.05)
                    py = 3.10 + 0.3 * (t - 2.05)
                elif pid == "A_04":
                    # Intercepting Away defender arrives at t_turnover and controls ball
                    bx_int = -6.5 + 13.0 * (t_turnover - 1.0)
                    by_int = 0.5 + 2.5 * (t_turnover - 1.0)
                    if t < t_turnover:
                        px = bx_int + 3.0 * (t_turnover - t)
                        py = by_int + 1.2 * (t_turnover - t)
                    else:
                        # Runs closely alongside the ball for 0.6s before passing it forward
                        follow_dt = min(t - t_turnover, 0.60)
                        px = bx_int - 9.2 * follow_dt - 4.0 * max(0.0, (t - t_turnover) - 0.60)
                        py = by_int - 3.8 * follow_dt - 1.5 * max(0.0, (t - t_turnover) - 0.60)
                elif pid == "H_09":
                    # Home striker pressing across the defender/ball path at t=3.24s without touching it.
                    # Smooth physical crossing motion that momentarily gets closer than A_04
                    # for 3 frames (0.12s) to benchmark elementary hysteresis (Step 1b) suppression.
                    px = bx + 1.2 * (t - 3.24)
                    py = by + 3.0 * (t - 3.24)
                elif pmeta.role == PlayerRole.GK:
                    px = -48.0 if pmeta.team_id == TeamID.HOME else 48.0
                    py = 0.0
                else:
                    idx_num = pmeta.jersey_number
                    # Expansion/contraction tactical geometry correlated with true possession
                    is_home_poss = t < t_turnover
                    spread_mult = 1.25 if (
                        (pmeta.team_id == TeamID.HOME and is_home_poss)
                        or (pmeta.team_id == TeamID.AWAY and not is_home_poss)
                    ) else 0.85
                    sign = -1.0 if pmeta.team_id == TeamID.HOME else 1.0
                    px = sign * (14.0 + idx_num * 1.6 * spread_mult) + 0.4 * t
                    py = (idx_num - 6) * 3.8 * spread_mult

                player_rows.append({
                    "frame_id": fid,
                    "player_id": pid,
                    "x": px + rng.normal(0.0, 0.03),
                    "y": py + rng.normal(0.0, 0.03),
                    "visibility_conf": 0.98,
                })

        ds = TrackingSchemaValidator.validate_and_normalize(
            metadata=meta,
            players=players,
            frame_df=pd.DataFrame(frame_rows),
            player_tracks_df=pd.DataFrame(player_rows),
        )
        datasets.append(ds)
    return datasets


# ============================================================================
# 4. ORDERED ABLATION RUNNER (RFC v2.0 Section 6.3 — Steps 1 to 6)
# ============================================================================

class SubsetSymmetricGBDT(SymmetricGBDTEmissionModel):
    # Allows restricting the GBDT to specific paired & invariant feature subsets
    # for each stage of the ordered ablation study while preserving Team-Swap Symmetry.

    def __init__(
        self,
        paired_cols: List[Tuple[str, str]],
        invariant_cols: List[str],
        n_estimators: int = 55,
        learning_rate: float = 0.08,
    ):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate)
        self.paired_cols = paired_cols
        self.invariant_cols = invariant_cols

    def _extract_submatrix(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols: List[np.ndarray] = []
        for h_c, a_c in self.paired_cols:
            h_vals = df[h_c].to_numpy(dtype=float) if h_c in df.columns else np.zeros(n, dtype=float)
            a_vals = df[a_c].to_numpy(dtype=float) if a_c in df.columns else np.zeros(n, dtype=float)
            cols.append(np.nan_to_num(h_vals, nan=0.0))
            cols.append(np.nan_to_num(a_vals, nan=0.0))
        for inv_c in self.invariant_cols:
            inv_vals = df[inv_c].to_numpy(dtype=float) if inv_c in df.columns else np.zeros(n, dtype=float)
            cols.append(np.nan_to_num(inv_vals, nan=0.0))
        return np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)

    def _swap_submatrix(self, X: np.ndarray) -> np.ndarray:
        X_s = X.copy()
        for idx in range(len(self.paired_cols)):
            X_s[:, 2 * idx] = X[:, 2 * idx + 1]
            X_s[:, 2 * idx + 1] = X[:, 2 * idx]
        return X_s

    def fit(self, df: pd.DataFrame) -> None:
        valid_mask = (
            df["eval_mask"].fillna(True).to_numpy(dtype=bool)
            & df["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            & (~df["is_dead_ball"].fillna(False).to_numpy(dtype=bool))
            & df["gt_possession_team"].isin([TeamID.HOME.value, TeamID.AWAY.value])
        )
        train_df = df.loc[valid_mask].reset_index(drop=True)
        X_orig = self._extract_submatrix(train_df)
        y_orig = (train_df["gt_possession_team"].astype(str).to_numpy() == TeamID.HOME.value).astype(int)
        X_swap = self._swap_submatrix(X_orig)
        y_swap = 1 - y_orig

        self.model = self._build_base_estimator()
        self.model.fit(np.vstack([X_orig, X_swap]), np.concatenate([y_orig, y_swap]))

    def predict_raw_home_prob(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.empty(0, dtype=float)
        X_orig = self._extract_submatrix(df)
        X_swap = self._swap_submatrix(X_orig)
        p_orig = self.model.predict_proba(X_orig)[:, 1]
        p_swap = self.model.predict_proba(X_swap)[:, 1]
        p_sym = 0.5 * (p_orig + (1.0 - p_swap))

        logit = np.log(np.clip(p_sym, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_sym, 1e-6, 1.0 - 1e-6))
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(self.platt_scale_a * logit, -20.0, 20.0)))
        valid_ball = df["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
        return np.where(valid_ball, p_cal, 0.5)


def run_hysteresis_baseline(df: pd.DataFrame, min_dwell_samples: int = 6) -> pd.DataFrame:
    # Step 1b Baseline: Nearest player with elementary hysteresis dwell filter
    out = df.copy()
    if len(out) == 0:
        out["decoded_home_prob"] = np.empty(0, dtype=float)
        out["predicted_possession_team"] = np.empty(0, dtype=object)
        return out

    d_home = out["min_dist_home"].to_numpy(dtype=float)
    d_away = out["min_dist_away"].to_numpy(dtype=float)

    # Soft probability from distance difference
    p_home = 1.0 / (1.0 + np.exp(np.clip((d_home - d_away) * 1.2, -15.0, 15.0)))
    raw_home = d_home <= d_away

    if "ball_valid_seg_id" not in out.columns:
        ball_valid = (
            out["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_valid" in out.columns
            else np.ones(len(out), dtype=bool)
        )
        out["ball_valid_seg_id"] = (~ball_valid).cumsum()

    hyst_home = raw_home.copy()
    for seg_id, grp in out.groupby("ball_valid_seg_id", sort=False):
        if int(seg_id) < 0 or len(grp) == 0:
            continue
        idx = grp.index.to_numpy(dtype=int)
        cur_state = bool(raw_home[idx[0]])
        streak = 0
        for k in range(len(idx)):
            gi = idx[k]
            cand = bool(raw_home[gi])
            if cand != cur_state:
                streak += 1
                if streak >= min_dwell_samples:
                    cur_state = cand
                    streak = 0
            else:
                streak = 0
            hyst_home[gi] = cur_state

    out["decoded_home_prob"] = p_home
    out["predicted_possession_team"] = np.where(
        ~out["ball_is_valid"].to_numpy(dtype=bool),
        "Unknown_Missing_Ball",
        np.where(hyst_home, TeamID.HOME.value, TeamID.AWAY.value),
    )
    return out


def execute_ordered_ablation_study() -> Tuple[List[EvaluationMetrics], Dict[str, float]]:
    raw_matches = generate_challenging_ablation_benchmark(n_matches=4, sampling_rate_hz=25.0)
    prep = SegmentedPreprocessor()
    t2_ext = LocalKinematicFeatureExtractor()
    t3_eng = KinematicContactAndBridgeEngine(
        config=AnchorBridgeConfig(enable_jev_refiner=True),
        jev_refiner=JevSystemOneRefiner(use_live_api=False),
    )
    sym_builder = SymmetricFeatureBuilder()

    full_matches: List[pd.DataFrame] = []
    for ds in raw_matches:
        proc_ds = prep.process(ds)
        df2 = t2_ext.extract_features(proc_ds)
        df3 = t3_eng.extract_anchor_and_bridge_features(df2, proc_ds.metadata)
        df4 = sym_builder.add_segmented_temporal_windows(df3, proc_ds.metadata)
        df5 = TacticalGeometryExtractor.extract_tactical_features(proc_ds, df4)
        df5["match_id"] = proc_ds.metadata.match_id
        full_matches.append(df5)

    evaluator = PossessionDiagnosticEvaluator()
    results: List[EvaluationMetrics] = []

    # --- STEP 1A: Nearest-Player Static Proximity Baseline ---
    step1a_dfs: List[pd.DataFrame] = []
    for df in full_matches:
        d = df.copy()
        d_diff = d["min_dist_home"].to_numpy(dtype=float) - d["min_dist_away"].to_numpy(dtype=float)
        d["decoded_home_prob"] = 1.0 / (1.0 + np.exp(np.clip(d_diff * 1.2, -15.0, 15.0)))
        d["predicted_possession_team"] = np.where(
            d_diff <= 0.0, TeamID.HOME.value, TeamID.AWAY.value
        )
        step1a_dfs.append(d)
    results.append(evaluator.evaluate_oof_matches("1a. Nearest-Player Baseline", step1a_dfs))

    # --- STEP 1B: Simple Hysteresis Persistence Filter ---
    step1b_dfs = [run_hysteresis_baseline(df, min_dwell_samples=6) for df in full_matches]
    results.append(evaluator.evaluate_oof_matches("1b. Simple Hysteresis Filter", step1b_dfs))

    # Feature sets for GBDT Ablation Steps 2 to 6
    local_pairs = [
        ("min_dist_home", "min_dist_away"),
        ("home_min_intercept_tau", "away_min_intercept_tau"),
        ("home_touch_evidence", "away_touch_evidence"),
        ("home_carry_evidence", "away_carry_evidence"),
        ("home_top1_top2_margin", "away_top1_top2_margin"),
        ("opp_count_near_home_2m", "opp_count_near_away_2m"),
    ]
    local_inv = ["ball_speed", "ball_accel", "ball_vel_change_proxy_mps"]

    win_pairs = local_pairs + [
        ("roll_short_home_ctrl", "roll_short_away_ctrl"),
        ("roll_long_home_ctrl", "roll_long_away_ctrl"),
        ("roll_short_min_dist_home", "roll_short_min_dist_away"),
    ]

    bridge_pairs = win_pairs + [
        ("anchor_home_prob", "anchor_away_prob"),
        ("prev_anchor_home_prob", "prev_anchor_away_prob"),
        ("next_anchor_home_prob", "next_anchor_away_prob"),
    ]
    bridge_inv = local_inv + [
        "candidate_contact_confidence",
        "prev_anchor_age_s",
        "prev_anchor_confidence",
        "next_anchor_lead_s",
        "next_anchor_confidence",
        "bridge_same_team_agreement",
        "bridge_contradiction_score",
    ]

    full_pairs = bridge_pairs + TacticalGeometryExtractor.PAIRED_TACTICAL_COLUMNS
    full_inv = bridge_inv + [
        "ground_bounce_candidate_prob",
        "unreliable_proximity_likelihood",
        "jev_decision_confidence",
    ]

    def run_gbdt_oof_stage(
        stage_name: str,
        pairs: List[Tuple[str, str]],
        invs: List[str],
        use_soft_viterbi: bool,
    ) -> Tuple[EvaluationMetrics, List[pd.DataFrame]]:
        oof_probs: List[np.ndarray] = []
        for i in range(len(full_matches)):
            tr_df = pd.concat([full_matches[j] for j in range(len(full_matches)) if j != i], ignore_index=True)
            va_df = full_matches[i]
            m = SubsetSymmetricGBDT(pairs, invs)
            m.fit(tr_df)
            oof_probs.append(m.predict_raw_home_prob(va_df))

        if use_soft_viterbi:
            dec = SoftViterbiDecoder()
            dec.tune_on_oof(full_matches, oof_probs)
            stage_dfs = [dec.decode(df_m, p_m) for df_m, p_m in zip(full_matches, oof_probs)]
        else:
            stage_dfs = []
            for df_m, p_m in zip(full_matches, oof_probs):
                d = df_m.copy()
                d["decoded_home_prob"] = p_m
                d["predicted_possession_team"] = np.where(
                    p_m >= 0.5, TeamID.HOME.value, TeamID.AWAY.value
                )
                stage_dfs.append(d)

        metrics = evaluator.evaluate_oof_matches(stage_name, stage_dfs)
        return metrics, stage_dfs

    # --- STEP 2: Local Kinematics + Reachability GBDT (Frame-by-Frame) ---
    m2, _ = run_gbdt_oof_stage("2. Local Kinematic + GBDT", local_pairs, local_inv, use_soft_viterbi=False)
    results.append(m2)

    # --- STEP 3: + Multi-Scale Temporal Windows ---
    m3, _ = run_gbdt_oof_stage("3. + Temporal Context Windows", win_pairs, local_inv, use_soft_viterbi=False)
    results.append(m3)

    # --- STEP 4: + Probabilistic Anchors & Bounded Bridges (Jev Refined) ---
    m4, _ = run_gbdt_oof_stage("4. + Jev Anchors & Bridges", bridge_pairs, bridge_inv, use_soft_viterbi=False)
    results.append(m4)

    # --- STEP 5: + OOF-Tuned Soft Viterbi Sequence Decoder ---
    m5, _ = run_gbdt_oof_stage("5. + Soft Viterbi Decoder", bridge_pairs, bridge_inv, use_soft_viterbi=True)
    results.append(m5)

    # --- STEP 6: + Aerial Unreliable-Proximity & Tactical Spread Priors ---
    m6, final_stage_dfs = run_gbdt_oof_stage(
        "6. + Aerial & Tactical Priors", full_pairs, full_inv, use_soft_viterbi=True
    )
    results.append(m6)

    slice_report = evaluator.evaluate_diagnostic_slices(final_stage_dfs)
    return results, slice_report


# ============================================================================
# 5. AUTOMATED VERIFICATION & AUDIT COMPLIANCE SUITE FOR TASK 5
# ============================================================================

def run_task5_verification() -> None:
    print("=" * 88)
    print("RUNNING TASK 5 DIAGNOSTIC METRICS, SLICES & ORDERED ABLATION VERIFICATION")
    print("=" * 88)

    ablation_results, slice_report = execute_ordered_ablation_study()

    header = (
        f"{'Ablation Stage':<30} | {'Acc %':>6} | {'MacF1':>5} | {'TurnF1':>6} | "
        f"{'Err(ms)':>7} | {'FSw/m':>6} | {'Brier':>6}"
    )
    print(header)
    print("-" * 88)
    for r in ablation_results:
        print(
            f"{r.config_name:<30} | {r.frame_accuracy * 100:6.1f} | {r.macro_f1:5.3f} | "
            f"{r.turnover_f1:6.3f} | {r.mean_abs_timing_error_ms:7.1f} | "
            f"{r.false_switches_per_min:6.1f} | {r.brier_score:6.4f}"
        )
    print("-" * 88)

    print("DIAGNOSTIC EDGE-CASE SLICES (FINAL STAGE 6 OOF):")
    for slice_name, acc_val in slice_report.items():
        print(f"  * {slice_name:<28}: {acc_val * 100:6.1f}%")
    print("-" * 88)

    base_res = ablation_results[0]   # Step 1a: Nearest-Player
    hyst_res = ablation_results[1]   # Step 1b: Simple Hysteresis
    brid_res = ablation_results[4]   # Step 4: + Jev Anchors & Bridges
    vit_res = ablation_results[5]    # Step 5: + Soft Viterbi Decoder
    fin_res = ablation_results[-1]   # Step 6: Full MVP

    # 1. Verify elementary hysteresis reduces false switches vs static nearest-player
    assert hyst_res.false_switches_per_min < base_res.false_switches_per_min, (
        "Simple hysteresis should reduce false switches per minute over static proximity."
    )
    print(
        f"[PASS] 1. Baseline & Hysteresis diagnostics verified "
        f"(False switches/min dropped from {base_res.false_switches_per_min:.1f} to {hyst_res.false_switches_per_min:.1f})."
    )

    # 2. Verify Jev Anchors & Bounded Bridges significantly improve OOF accuracy over local proximity
    assert brid_res.frame_accuracy > base_res.frame_accuracy + 0.15, (
        f"Expected Anchor-Bridge OOF accuracy ({brid_res.frame_accuracy:.3f}) to beat baseline ({base_res.frame_accuracy:.3f}) by >15%."
    )
    print(
        f"[PASS] 2. Probabilistic Anchors & Bounded Bridges gain verified "
        f"(OOF Acc: {base_res.frame_accuracy * 100:.1f}% -> {brid_res.frame_accuracy * 100:.1f}%)."
    )

    # 3. Verify Soft Viterbi Decoder improves or maintains Turnover Event F1 and eliminates short flicker
    assert vit_res.false_switches_per_min <= brid_res.false_switches_per_min, (
        "Soft Viterbi decoder must not increase false switches over raw GBDT emissions."
    )
    assert fin_res.turnover_f1 >= 0.85, (
        f"Expected high 1-to-1 Turnover Event F1 (>=0.85), got {fin_res.turnover_f1:.3f}."
    )
    print(
        f"[PASS] 3. Soft Viterbi & 1-to-1 Turnover Event matching verified "
        f"(Turnover F1 = {fin_res.turnover_f1:.3f}, Mean Timing Error = {fin_res.mean_abs_timing_error_ms:.1f} ms)."
    )

    # 4. Verify Match-Level Bootstrap CI and Edge-Case Slice Reporting (RFC Section 6.2)
    assert 0.0 <= fin_res.match_bootstrap_ci_low <= fin_res.match_acc_mean <= fin_res.match_bootstrap_ci_high <= 1.0, (
        "Match-level bootstrap confidence interval bounds are invalid."
    )
    assert slice_report["PassBy_Near_Opponent_Acc"] >= 0.85, (
        f"Pass-by near opponent slice accuracy too low: {slice_report['PassBy_Near_Opponent_Acc']:.3f}"
    )
    print(
        f"[PASS] 4. Match-level bootstrap CI ([{fin_res.match_bootstrap_ci_low * 100:.1f}%, "
        f"{fin_res.match_bootstrap_ci_high * 100:.1f}%]) & edge-case slices verified."
    )

    print("=" * 88)
    print("ALL TASK 5 CHECKS & FULL RFC v2.0 MVP PIPELINE VERIFIED SUCCESSFULLY.")
    print("=" * 88)


if __name__ == "__main__":
    run_task5_verification()