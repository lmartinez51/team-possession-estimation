# Task 1: Canonical Data Contract, Synthetic Diagnostic Generator,
# and Quality-Aware Segmented Preprocessor (RFC v2.0).
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ============================================================================
# 1. CANONICAL DATA SCHEMA & METADATA CONTRACT (RFC v2.0 Section 2 & 3.1)
# ============================================================================

class TeamID(str, Enum):
    HOME = "Home"
    AWAY = "Away"


class PlayerRole(str, Enum):
    OUTFIELD = "Outfield"
    GK = "GK"


class CoordinateSemantics(str, Enum):
    PITCH_METERS = "pitch_meters"
    NORMALIZED_01 = "normalized_01"
    RAW_CAMERA_PIXELS = "raw_camera_px"


@dataclass(frozen=True)
class MatchMetadata:
    match_id: str
    sampling_rate_hz: float
    pitch_length_m: float
    pitch_width_m: float
    coord_semantics: CoordinateSemantics = CoordinateSemantics.PITCH_METERS
    home_attacks_right_period1: bool = True

    def seconds_to_samples(self, duration_s: float, odd_window: bool = False) -> int:
        if duration_s <= 0.0:
            return 1
        intervals = int(np.round(duration_s * self.sampling_rate_hz))
        n_samples = max(2, intervals + 1)
        if odd_window and (n_samples % 2 == 0):
            n_samples += 1
        return n_samples


@dataclass(frozen=True)
class PlayerMeta:
    player_id: str
    team_id: TeamID
    role: PlayerRole
    jersey_number: int

    def __post_init__(self):
        if not isinstance(self.team_id, TeamID):
            object.__setattr__(self, "team_id", TeamID(self.team_id))
        if not isinstance(self.role, PlayerRole):
            object.__setattr__(self, "role", PlayerRole(self.role))


@dataclass
class PreprocessConfig:
    max_interp_gap_s: float = 0.40
    multiscale_short_win_s: float = 0.12
    multiscale_long_win_s: float = 0.24
    player_smooth_win_s: float = 0.28
    ball_smooth_win_s: float = 0.16
    savgol_polyorder: int = 2
    boundary_suppress_s: float = 0.16
    min_candidate_dv_mps: float = 4.0


@dataclass
class ValidatedTrackingDataset:
    metadata: MatchMetadata
    players: Dict[str, PlayerMeta]
    frame_df: pd.DataFrame
    player_tracks_df: pd.DataFrame


# ============================================================================
# 2. SCHEMA VALIDATOR & COORDINATE NORMALIZER
# ============================================================================

class TrackingSchemaValidator:
    REQUIRED_FRAME_COLS = {
        "frame_id", "timestamp_s", "period_id", "ball_x", "ball_y",
        "is_dead_ball", "is_contested", "eval_mask", "camera_cut"
    }
    REQUIRED_PLAYER_COLS = {
        "frame_id", "player_id", "x", "y", "visibility_conf"
    }

    @classmethod
    def validate_and_normalize(
        cls,
        metadata: MatchMetadata,
        players: Dict[str, PlayerMeta],
        frame_df: pd.DataFrame,
        player_tracks_df: pd.DataFrame,
        homography_matrices: Optional[Dict[int, np.ndarray]] = None,
    ) -> ValidatedTrackingDataset:
        missing_f = cls.REQUIRED_FRAME_COLS - set(frame_df.columns)
        if missing_f:
            raise ValueError(f"frame_df missing required columns: {missing_f}")

        missing_p = cls.REQUIRED_PLAYER_COLS - set(player_tracks_df.columns)
        if missing_p:
            raise ValueError(f"player_tracks_df missing required columns: {missing_p}")

        gk_teams = [p.team_id for p in players.values() if p.role == PlayerRole.GK]
        if TeamID.HOME not in gk_teams or TeamID.AWAY not in gk_teams:
            raise ValueError("Both Home and Away goalkeepers must retain explicit TeamIDs.")

        f_df = frame_df.sort_values(["period_id", "timestamp_s"]).reset_index(drop=True).copy()
        p_df = player_tracks_df.copy()

        for period_id, group in f_df.groupby("period_id", sort=False):
            ts = group["timestamp_s"].to_numpy(dtype=float)
            if len(ts) > 1:
                dt = np.diff(ts)
                median_dt = float(np.median(dt))
                expected_dt = 1.0 / metadata.sampling_rate_hz
                if not np.isclose(median_dt, expected_dt, rtol=0.05):
                    raise ValueError(
                        f"Period {period_id} median dt ({median_dt:.4f}s) diverges from "
                        f"metadata sampling_rate_hz ({metadata.sampling_rate_hz} Hz)."
                    )

        if metadata.coord_semantics == CoordinateSemantics.NORMALIZED_01:
            f_df["ball_x"] = (f_df["ball_x"].astype(float) - 0.5) * metadata.pitch_length_m
            f_df["ball_y"] = (f_df["ball_y"].astype(float) - 0.5) * metadata.pitch_width_m
            p_df["x"] = (p_df["x"].astype(float) - 0.5) * metadata.pitch_length_m
            p_df["y"] = (p_df["y"].astype(float) - 0.5) * metadata.pitch_width_m
        elif metadata.coord_semantics == CoordinateSemantics.RAW_CAMERA_PIXELS:
            if homography_matrices is None:
                raise ValueError("Homography matrices required when coord_semantics is RAW_CAMERA_PIXELS.")
            f_df, p_df = cls._apply_planar_homography(f_df, p_df, homography_matrices)
        else:
            f_df["ball_x"] = f_df["ball_x"].astype(float)
            f_df["ball_y"] = f_df["ball_y"].astype(float)
            p_df["x"] = p_df["x"].astype(float)
            p_df["y"] = p_df["y"].astype(float)

        f_df["is_missing_ball"] = f_df["ball_x"].isna() | f_df["ball_y"].isna()

        return ValidatedTrackingDataset(
            metadata=metadata,
            players=players,
            frame_df=f_df,
            player_tracks_df=p_df,
        )

    @staticmethod
    def _apply_planar_homography(
        f_df: pd.DataFrame,
        p_df: pd.DataFrame,
        H_dict: Dict[int, np.ndarray],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bx = f_df["ball_x"].to_numpy(dtype=float).copy()
        by = f_df["ball_y"].to_numpy(dtype=float).copy()
        fids = f_df["frame_id"].to_numpy(dtype=int)

        for i in range(len(f_df)):
            fid = int(fids[i])
            if np.isfinite(bx[i]) and np.isfinite(by[i]):
                if fid in H_dict:
                    H = H_dict[fid]
                    proj = H @ np.array([bx[i], by[i], 1.0], dtype=float)
                    if abs(proj[2]) > 1e-12:
                        bx[i] = proj[0] / proj[2]
                        by[i] = proj[1] / proj[2]
                    else:
                        bx[i] = np.nan
                        by[i] = np.nan
                else:
                    bx[i] = np.nan
                    by[i] = np.nan

        f_df["ball_x"] = bx
        f_df["ball_y"] = by

        px = p_df["x"].to_numpy(dtype=float).copy()
        py = p_df["y"].to_numpy(dtype=float).copy()
        p_fids = p_df["frame_id"].to_numpy(dtype=int)

        for i in range(len(p_df)):
            fid = int(p_fids[i])
            if np.isfinite(px[i]) and np.isfinite(py[i]):
                if fid in H_dict:
                    H = H_dict[fid]
                    proj = H @ np.array([px[i], py[i], 1.0], dtype=float)
                    if abs(proj[2]) > 1e-12:
                        px[i] = proj[0] / proj[2]
                        py[i] = proj[1] / proj[2]
                    else:
                        px[i] = np.nan
                        py[i] = np.nan
                else:
                    px[i] = np.nan
                    py[i] = np.nan

        p_df["x"] = px
        p_df["y"] = py
        return f_df, p_df


# ============================================================================
# 3. QUALITY-AWARE SEGMENTED PREPROCESSOR (RFC v2.0 Section 3.2 & 4.1)
# ============================================================================

class SegmentedPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(self, dataset: ValidatedTrackingDataset) -> ValidatedTrackingDataset:
        meta = dataset.metadata
        f_df = dataset.frame_df.reset_index(drop=True).copy()
        p_df = dataset.player_tracks_df.reset_index(drop=True).copy()

        if len(f_df) == 0:
            return ValidatedTrackingDataset(
                metadata=meta,
                players=dataset.players,
                frame_df=f_df,
                player_tracks_df=p_df,
            )

        period_arr = f_df["period_id"].to_numpy()
        cam_cut_arr = f_df["camera_cut"].to_numpy(dtype=bool)
        period_change = np.zeros(len(f_df), dtype=bool)
        if len(f_df) > 1:
            period_change[1:] = period_arr[1:] != period_arr[:-1]

        structural_break = period_change | cam_cut_arr
        f_df["structural_seg_id"] = np.cumsum(structural_break)

        f_df = self._interpolate_and_segment_ball(f_df, meta)
        f_df = self._compute_multiscale_ball_kinematics(f_df, meta)
        f_df = self._smooth_ball_piecewise(f_df, meta)
        p_df = self._process_player_trajectories(
            p_df, f_df[["frame_id", "timestamp_s", "structural_seg_id"]], meta
        )

        team_map = {pid: pmeta.team_id.value for pid, pmeta in dataset.players.items()}
        role_map = {pid: pmeta.role.value for pid, pmeta in dataset.players.items()}
        p_df["team_id"] = p_df["player_id"].map(team_map)
        p_df["role"] = p_df["player_id"].map(role_map)

        valid_p = p_df[p_df["x"].notna() & p_df["y"].notna()]
        counts = (
            valid_p.groupby(["frame_id", "team_id"])["player_id"]
            .count()
            .unstack(fill_value=0)
        )
        home_col = counts[TeamID.HOME.value] if TeamID.HOME.value in counts.columns else pd.Series(dtype=int)
        away_col = counts[TeamID.AWAY.value] if TeamID.AWAY.value in counts.columns else pd.Series(dtype=int)

        f_df["visible_home_players"] = f_df["frame_id"].map(home_col).fillna(0).astype(int)
        f_df["visible_away_players"] = f_df["frame_id"].map(away_col).fillna(0).astype(int)

        return ValidatedTrackingDataset(
            metadata=meta,
            players=dataset.players,
            frame_df=f_df,
            player_tracks_df=p_df,
        )

    @staticmethod
    def _interpolate_1d_bounded(
        arr: np.ndarray, max_gap_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Pure NumPy bounded interior linear interpolation avoiding Pandas index pitfalls.
        out = arr.astype(float).copy()
        n = len(out)
        is_interp = np.zeros(n, dtype=bool)
        gap_len_samples = np.zeros(n, dtype=int)

        isnan = ~np.isfinite(out)
        i = 0
        while i < n:
            if isnan[i]:
                j = i
                while j < n and isnan[j]:
                    j += 1
                run_len = j - i
                gap_len_samples[i:j] = run_len
                if i > 0 and j < n and run_len <= max_gap_samples:
                    left_val = out[i - 1]
                    right_val = out[j]
                    steps = np.arange(1, run_len + 1, dtype=float) / float(run_len + 1)
                    out[i:j] = left_val + steps * (right_val - left_val)
                    is_interp[i:j] = True
                i = j
            else:
                i += 1
        return out, is_interp, gap_len_samples

    def _interpolate_and_segment_ball(self, f_df: pd.DataFrame, meta: MatchMetadata) -> pd.DataFrame:
        max_gap_samples = int(np.floor(self.config.max_interp_gap_s * meta.sampling_rate_hz))
        dt_s = 1.0 / meta.sampling_rate_hz

        bx_all = f_df["ball_x"].to_numpy(dtype=float).copy()
        by_all = f_df["ball_y"].to_numpy(dtype=float).copy()
        interp_all = np.zeros(len(f_df), dtype=bool)
        gap_dur_all = np.zeros(len(f_df), dtype=float)

        for _, group in f_df.groupby("structural_seg_id", sort=False):
            pos_idx = group.index.to_numpy(dtype=int)
            bx_seg = bx_all[pos_idx]
            by_seg = by_all[pos_idx]

            # Sync NaN mask across x and y
            joint_nan = (~np.isfinite(bx_seg)) | (~np.isfinite(by_seg))
            bx_seg[joint_nan] = np.nan
            by_seg[joint_nan] = np.nan

            bx_new, is_interp_x, gap_samples = self._interpolate_1d_bounded(bx_seg, max_gap_samples)
            by_new, _, _ = self._interpolate_1d_bounded(by_seg, max_gap_samples)

            bx_all[pos_idx] = bx_new
            by_all[pos_idx] = by_new
            interp_all[pos_idx] = is_interp_x
            gap_dur_all[pos_idx] = np.round(gap_samples * dt_s, 6)

        f_df["ball_x"] = bx_all
        f_df["ball_y"] = by_all
        f_df["ball_is_interpolated"] = interp_all
        f_df["ball_gap_duration_s"] = gap_dur_all

        ball_valid = np.isfinite(bx_all) & np.isfinite(by_all)
        f_df["ball_is_valid"] = ball_valid

        seg_ids = f_df["structural_seg_id"].to_numpy(dtype=int)
        seg_break = np.zeros(len(f_df), dtype=bool)
        if len(f_df) > 0:
            seg_break[0] = True
        if len(f_df) > 1:
            seg_break[1:] = (
                (seg_ids[1:] != seg_ids[:-1])
                | (~ball_valid[1:])
                | (~ball_valid[:-1])
            )
        raw_seg_num = np.cumsum(seg_break)
        f_df["ball_valid_seg_id"] = np.where(ball_valid, raw_seg_num, -1)
        return f_df

    @staticmethod
    def _local_ols_velocity(pos: np.ndarray, dt: float, window_samples: int, forward: bool) -> np.ndarray:
        n = len(pos)
        vel = np.full(n, np.nan, dtype=float)
        k = max(2, int(window_samples))
        if n < k:
            return vel

        t_vec = np.arange(k, dtype=float) * dt
        t_centered = t_vec - np.mean(t_vec)
        t_denom = float(np.sum(t_centered ** 2))
        if t_denom <= 1e-12:
            return vel

        # Sliding window OLS slope: sum(t_centered * (y - y_bar)) == sum(t_centered * y) since sum(t_centered) == 0
        for i in range(n):
            if forward:
                if i + k > n:
                    continue
                y_slice = pos[i : i + k]
            else:
                if i - k + 1 < 0:
                    continue
                y_slice = pos[i - k + 1 : i + 1]

            if not np.all(np.isfinite(y_slice)):
                continue
            vel[i] = float(np.dot(t_centered, y_slice) / t_denom)
        return vel

    def _compute_multiscale_ball_kinematics(self, f_df: pd.DataFrame, meta: MatchMetadata) -> pd.DataFrame:
        dt = 1.0 / meta.sampling_rate_hz
        k_short = meta.seconds_to_samples(self.config.multiscale_short_win_s)
        k_long = meta.seconds_to_samples(self.config.multiscale_long_win_s)
        suppress_samples = meta.seconds_to_samples(self.config.boundary_suppress_s)

        n_total = len(f_df)
        dv_out = np.zeros(n_total, dtype=float)
        cons_out = np.zeros(n_total, dtype=float)
        unrel_out = np.ones(n_total, dtype=bool)
        split_out = np.zeros(n_total, dtype=bool)

        for seg_id, group in f_df.groupby("ball_valid_seg_id", sort=False):
            if seg_id < 0 or len(group) < k_long * 2:
                continue

            pos_idx = group.index.to_numpy(dtype=int)
            bx = group["ball_x"].to_numpy(dtype=float)
            by = group["ball_y"].to_numpy(dtype=float)
            interp_flag = group["ball_is_interpolated"].to_numpy(dtype=bool)

            vx_fwd_s = self._local_ols_velocity(bx, dt, k_short, forward=True)
            vy_fwd_s = self._local_ols_velocity(by, dt, k_short, forward=True)
            vx_bwd_s = self._local_ols_velocity(bx, dt, k_short, forward=False)
            vy_bwd_s = self._local_ols_velocity(by, dt, k_short, forward=False)

            vx_fwd_l = self._local_ols_velocity(bx, dt, k_long, forward=True)
            vy_fwd_l = self._local_ols_velocity(by, dt, k_long, forward=True)
            vx_bwd_l = self._local_ols_velocity(bx, dt, k_long, forward=False)
            vy_bwd_l = self._local_ols_velocity(by, dt, k_long, forward=False)

            dv_short = np.hypot(vx_fwd_s - vx_bwd_s, vy_fwd_s - vy_bwd_s)
            dv_long = np.hypot(vx_fwd_l - vx_bwd_l, vy_fwd_l - vy_bwd_l)

            dvx_s, dvy_s = vx_fwd_s - vx_bwd_s, vy_fwd_s - vy_bwd_s
            dvx_l, dvy_l = vx_fwd_l - vx_bwd_l, vy_fwd_l - vy_bwd_l
            dot = dvx_s * dvx_l + dvy_s * dvy_l
            denom = np.maximum(dv_short * dv_long, 1e-6)
            cos_agree = np.clip(dot / denom, -1.0, 1.0)

            ratio = np.minimum(dv_short, dv_long) / np.maximum(np.maximum(dv_short, dv_long), 1e-3)
            consistency = np.where(
                (~np.isfinite(dv_short)) | (~np.isfinite(dv_long)),
                0.0,
                np.clip(0.5 * (cos_agree + 1.0) * ratio, 0.0, 1.0),
            )

            n = len(group)
            unreliable = np.zeros(n, dtype=bool)
            unreliable[:suppress_samples] = True
            unreliable[max(0, n - suppress_samples) :] = True
            if interp_flag.any():
                kernel = np.ones(2 * suppress_samples + 1, dtype=int)
                dilated = np.convolve(interp_flag.astype(int), kernel, mode="same") > 0
                unreliable |= dilated

            dv_clean = np.nan_to_num(dv_short, nan=0.0)
            dv_clean[unreliable] = 0.0
            consistency[unreliable] = 0.0

            is_local_peak = np.zeros(n, dtype=bool)
            for i in range(1, n - 1):
                if (
                    not unreliable[i]
                    and dv_clean[i] >= self.config.min_candidate_dv_mps
                    and consistency[i] >= 0.65
                    and dv_clean[i] >= dv_clean[i - 1]
                    and dv_clean[i] >= dv_clean[i + 1]
                ):
                    is_local_peak[i] = True

            dv_out[pos_idx] = dv_clean
            cons_out[pos_idx] = consistency
            unrel_out[pos_idx] = unreliable
            split_out[pos_idx] = is_local_peak

        f_df["ball_vel_change_proxy_mps"] = dv_out
        f_df["ball_multiscale_consistency"] = cons_out
        f_df["ball_boundary_unreliable"] = unrel_out
        f_df["pre_smooth_contact_split"] = split_out
        return f_df

    @staticmethod
    def _smooth_and_differentiate_1d(
        x: np.ndarray, y: np.ndarray, dt: float, win_len: int, poly: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        if n >= win_len and win_len >= 3 and poly >= 1:
            xs = savgol_filter(x, window_length=win_len, polyorder=poly, mode="interp")
            ys = savgol_filter(y, window_length=win_len, polyorder=poly, mode="interp")
            vx = savgol_filter(x, window_length=win_len, polyorder=poly, deriv=1, delta=dt, mode="interp")
            vy = savgol_filter(y, window_length=win_len, polyorder=poly, deriv=1, delta=dt, mode="interp")
            if poly >= 2:
                ax = savgol_filter(x, window_length=win_len, polyorder=poly, deriv=2, delta=dt, mode="interp")
                ay = savgol_filter(y, window_length=win_len, polyorder=poly, deriv=2, delta=dt, mode="interp")
            else:
                ax = np.gradient(vx, dt)
                ay = np.gradient(vy, dt)
        elif n >= 2:
            xs, ys = x.copy(), y.copy()
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            ax = np.gradient(vx, dt) if n >= 3 else np.zeros(n, dtype=float)
            ay = np.gradient(vy, dt) if n >= 3 else np.zeros(n, dtype=float)
        else:
            xs, ys = x.copy(), y.copy()
            vx = np.zeros(n, dtype=float)
            vy = np.zeros(n, dtype=float)
            ax = np.zeros(n, dtype=float)
            ay = np.zeros(n, dtype=float)
        return xs, ys, vx, vy, ax, ay

    def _smooth_ball_piecewise(self, f_df: pd.DataFrame, meta: MatchMetadata) -> pd.DataFrame:
        dt = 1.0 / meta.sampling_rate_hz
        win_len = max(3, meta.seconds_to_samples(self.config.ball_smooth_win_s, odd_window=True))
        poly = min(self.config.savgol_polyorder, win_len - 1)

        n_total = len(f_df)
        bx_s_all = f_df["ball_x"].to_numpy(dtype=float).copy()
        by_s_all = f_df["ball_y"].to_numpy(dtype=float).copy()
        vx_all = np.full(n_total, np.nan, dtype=float)
        vy_all = np.full(n_total, np.nan, dtype=float)
        ax_all = np.full(n_total, np.nan, dtype=float)
        ay_all = np.full(n_total, np.nan, dtype=float)

        for seg_id, group in f_df.groupby("ball_valid_seg_id", sort=False):
            if seg_id < 0:
                continue

            sub_seg_ids = np.cumsum(group["pre_smooth_contact_split"].to_numpy(dtype=int))
            sub_seg_series = pd.Series(sub_seg_ids, index=group.index)
            for _, sub_group in group.groupby(sub_seg_series, sort=False):
                s_idx = sub_group.index.to_numpy(dtype=int)
                bx = sub_group["ball_x"].to_numpy(dtype=float)
                by = sub_group["ball_y"].to_numpy(dtype=float)

                xs, ys, vx, vy, ax, ay = self._smooth_and_differentiate_1d(bx, by, dt, win_len, poly)
                bx_s_all[s_idx] = xs
                by_s_all[s_idx] = ys
                vx_all[s_idx] = vx
                vy_all[s_idx] = vy
                ax_all[s_idx] = ax
                ay_all[s_idx] = ay

        f_df["ball_x_smooth"] = bx_s_all
        f_df["ball_y_smooth"] = by_s_all
        f_df["ball_vx"] = vx_all
        f_df["ball_vy"] = vy_all
        f_df["ball_speed"] = np.hypot(vx_all, vy_all)
        f_df["ball_ax"] = ax_all
        f_df["ball_ay"] = ay_all
        f_df["ball_accel"] = np.hypot(ax_all, ay_all)
        return f_df

    def _process_player_trajectories(
        self,
        p_df: pd.DataFrame,
        frame_seg_df: pd.DataFrame,
        meta: MatchMetadata,
    ) -> pd.DataFrame:
        dt = 1.0 / meta.sampling_rate_hz
        max_gap_samples = int(np.floor(self.config.max_interp_gap_s * meta.sampling_rate_hz))
        win_len = max(3, meta.seconds_to_samples(self.config.player_smooth_win_s, odd_window=True))
        poly = min(self.config.savgol_polyorder, win_len - 1)

        drop_cols = [c for c in ["timestamp_s", "structural_seg_id"] if c in p_df.columns]
        if drop_cols:
            p_df = p_df.drop(columns=drop_cols)
        p_df = p_df.merge(frame_seg_df, on="frame_id", how="left")
        p_df = p_df.sort_values(["player_id", "timestamp_s"]).reset_index(drop=True)

        n_total = len(p_df)
        px_all = p_df["x"].to_numpy(dtype=float).copy()
        py_all = p_df["y"].to_numpy(dtype=float).copy()
        interp_all = np.zeros(n_total, dtype=bool)
        vx_all = np.full(n_total, np.nan, dtype=float)
        vy_all = np.full(n_total, np.nan, dtype=float)

        for (_, _), group in p_df.groupby(["player_id", "structural_seg_id"], sort=False):
            g_idx = group.index.to_numpy(dtype=int)
            px_seg = px_all[g_idx]
            py_seg = py_all[g_idx]

            joint_nan = (~np.isfinite(px_seg)) | (~np.isfinite(py_seg))
            px_seg[joint_nan] = np.nan
            py_seg[joint_nan] = np.nan

            px_interp, is_interp, _ = self._interpolate_1d_bounded(px_seg, max_gap_samples)
            py_interp, _, _ = self._interpolate_1d_bounded(py_seg, max_gap_samples)

            px_all[g_idx] = px_interp
            py_all[g_idx] = py_interp
            interp_all[g_idx] = is_interp

            valid_mask = np.isfinite(px_interp) & np.isfinite(py_interp)
            n_seg = len(g_idx)
            i = 0
            while i < n_seg:
                if valid_mask[i]:
                    j = i
                    while j < n_seg and valid_mask[j]:
                        j += 1
                    run_indices = g_idx[i:j]
                    xs, ys, vx, vy, _, _ = self._smooth_and_differentiate_1d(
                        px_interp[i:j], py_interp[i:j], dt, win_len, poly
                    )
                    px_all[run_indices] = xs
                    py_all[run_indices] = ys
                    vx_all[run_indices] = vx
                    vy_all[run_indices] = vy
                    i = j
                else:
                    i += 1

        p_df["x"] = px_all
        p_df["y"] = py_all
        p_df["is_interpolated"] = interp_all
        p_df["vx"] = vx_all
        p_df["vy"] = vy_all
        p_df["speed"] = np.hypot(vx_all, vy_all)
        return p_df


# ============================================================================
# 4. METRICA-STYLE SYNTHETIC DIAGNOSTIC GENERATOR
# ============================================================================

def generate_synthetic_diagnostic_match(
    sampling_rate_hz: float = 25.0,
    seed: int = 42,
) -> ValidatedTrackingDataset:
    rng = np.random.default_rng(seed)
    metadata = MatchMetadata(
        match_id="SYNTH_BARCA_001",
        sampling_rate_hz=sampling_rate_hz,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        coord_semantics=CoordinateSemantics.PITCH_METERS,
    )

    players: Dict[str, PlayerMeta] = {}
    for i in range(1, 12):
        role = PlayerRole.GK if i == 1 else PlayerRole.OUTFIELD
        players[f"H_{i:02d}"] = PlayerMeta(f"H_{i:02d}", TeamID.HOME, role, i)
        players[f"A_{i:02d}"] = PlayerMeta(f"A_{i:02d}", TeamID.AWAY, role, i)

    dt = 1.0 / sampling_rate_hz
    n_p1 = int(round(6.0 * sampling_rate_hz))
    n_p2 = int(round(2.0 * sampling_rate_hz))

    frame_rows = []
    player_rows = []

    for step in range(n_p1):
        t = round(step * dt, 6)
        fid = step + 1
        if t < 1.0:
            bx, by = -10.0 + 5.0 * t, 0.0
            gt_team = TeamID.HOME.value
        elif t < 2.8:
            bx = -5.0 + 12.0 * (t - 1.0)
            by = 0.0 + 3.0 * (t - 1.0)
            gt_team = TeamID.HOME.value
        else:
            bx = 16.6 - 10.0 * (t - 2.8)
            by = 5.4 - 6.0 * (t - 2.8)
            gt_team = TeamID.AWAY.value

        bx_obs = bx + rng.normal(0.0, 0.02)
        by_obs = by + rng.normal(0.0, 0.02)

        if 1.80 <= t < 2.00:
            bx_obs, by_obs = np.nan, np.nan

        if 3.80 <= t < 4.60:
            bx_obs, by_obs = np.nan, np.nan

        frame_rows.append({
            "frame_id": fid,
            "timestamp_s": t,
            "period_id": 1,
            "ball_x": bx_obs,
            "ball_y": by_obs,
            "is_dead_ball": False,
            "is_contested": bool(2.75 <= t <= 2.85),
            "eval_mask": True,
            "camera_cut": False,
            "gt_possession_team": gt_team,
        })

        for pid, pmeta in players.items():
            if pid == "H_08":
                px = -10.0 + 5.0 * min(t, 1.2)
                py = 0.0
            elif pid == "A_04":
                px = 16.6 if t >= 2.6 else 4.0 + 4.5 * t
                py = 5.4 if t >= 2.6 else 2.5 + 1.0 * t
            elif pmeta.role == PlayerRole.GK:
                px = -48.0 if pmeta.team_id == TeamID.HOME else 48.0
                py = 0.0
            else:
                idx_num = pmeta.jersey_number
                sign = -1.0 if pmeta.team_id == TeamID.HOME else 1.0
                px = sign * (15.0 + idx_num * 1.5) + 0.5 * t
                py = (idx_num - 6) * 4.0

            player_rows.append({
                "frame_id": fid,
                "player_id": pid,
                "x": px + rng.normal(0.0, 0.015),
                "y": py + rng.normal(0.0, 0.015),
                "visibility_conf": 0.98,
            })

    for step in range(n_p2):
        t = round(2700.0 + step * dt, 6)
        fid = n_p1 + step + 1
        bx_obs = 30.0 - 4.0 * (step * dt)
        by_obs = -20.0 + 1.5 * (step * dt)

        frame_rows.append({
            "frame_id": fid,
            "timestamp_s": t,
            "period_id": 2,
            "ball_x": bx_obs,
            "ball_y": by_obs,
            "is_dead_ball": False,
            "is_contested": False,
            "eval_mask": True,
            "camera_cut": False,
            "gt_possession_team": TeamID.AWAY.value,
        })

        for pid, pmeta in players.items():
            if pid == "A_11":
                continue
            sign = 1.0 if pmeta.team_id == TeamID.HOME else -1.0
            px = sign * (10.0 + pmeta.jersey_number)
            py = (pmeta.jersey_number - 6) * 3.5
            player_rows.append({
                "frame_id": fid,
                "player_id": pid,
                "x": px,
                "y": py,
                "visibility_conf": 0.99,
            })

    frame_df = pd.DataFrame(frame_rows)
    player_tracks_df = pd.DataFrame(player_rows)

    return TrackingSchemaValidator.validate_and_normalize(
        metadata=metadata,
        players=players,
        frame_df=frame_df,
        player_tracks_df=player_tracks_df,
    )


# ============================================================================
# 5. AUTOMATED VERIFICATION & AUDIT COMPLIANCE SUITE
# ============================================================================

def run_task1_verification() -> None:
    print("=" * 72)
    print("RUNNING TASK 1 AUDIT COMPLIANCE & PREPROCESSING VERIFICATION")
    print("=" * 72)

    meta_25 = MatchMetadata("M25", sampling_rate_hz=25.0, pitch_length_m=105.0, pitch_width_m=68.0)
    meta_50 = MatchMetadata("M50", sampling_rate_hz=50.0, pitch_length_m=105.0, pitch_width_m=68.0)
    assert meta_25.seconds_to_samples(3.0) == 76, "3.0s at 25Hz should span 75 intervals (76 samples)."
    assert meta_50.seconds_to_samples(3.0) == 151, "3.0s at 50Hz should span 150 intervals (151 samples)."
    print("[PASS] 1. Temporal parameterization in seconds verified across 25Hz and 50Hz.")

    raw_dataset = generate_synthetic_diagnostic_match(sampling_rate_hz=25.0)
    preprocessor = SegmentedPreprocessor()
    proc_dataset = preprocessor.process(raw_dataset)
    f_df = proc_dataset.frame_df

    p1_away_count = int(f_df.loc[f_df["period_id"] == 1, "visible_away_players"].iloc[0])
    p2_away_count = int(f_df.loc[f_df["period_id"] == 2, "visible_away_players"].iloc[0])
    assert p1_away_count == 11 and p2_away_count == 10, (
        f"Expected 11 Away players in P1 and 10 in P2, got {p1_away_count} and {p2_away_count}."
    )
    print(f"[PASS] 2. Variable player cardinality & decoupled GK roles verified (P1 Away={p1_away_count}, P2 Away={p2_away_count}).")

    short_gap_rows = f_df[(f_df["timestamp_s"] >= 1.80) & (f_df["timestamp_s"] < 2.00)]
    long_gap_rows = f_df[(f_df["timestamp_s"] >= 3.80) & (f_df["timestamp_s"] < 4.60)]
    assert short_gap_rows["ball_is_interpolated"].all(), "Short 0.20s gap should be interpolated."
    assert not short_gap_rows["ball_x"].isna().any(), "Short gap coordinates should not be NaN after interpolation."
    assert np.allclose(short_gap_rows["ball_gap_duration_s"].to_numpy(), 0.20), "Short gap duration must be recorded as 0.20s."

    assert not long_gap_rows["ball_is_interpolated"].any(), "Long 0.80s gap must NOT be interpolated."
    assert long_gap_rows["ball_x"].isna().all(), "Long gap coordinates must remain NaN."
    print("[PASS] 3. Bounded gap interpolation (0.20s interpolated, 0.80s isolated) verified.")

    p2_first_row = f_df[f_df["period_id"] == 2].iloc[0]
    expected_p2_speed = float(np.hypot(-4.0, 1.5))
    assert np.isclose(float(p2_first_row["ball_speed"]), expected_p2_speed, atol=0.5), (
        f"Period 2 initial speed corrupted by Period 1 boundary jump: {p2_first_row['ball_speed']:.2f} m/s"
    )
    print(f"[PASS] 4. Period boundary isolation verified (P2 start speed = {p2_first_row['ball_speed']:.2f} m/s).")

    kick_window = f_df[(f_df["timestamp_s"] >= 0.92) & (f_df["timestamp_s"] <= 1.08)]
    intercept_window = f_df[(f_df["timestamp_s"] >= 2.72) & (f_df["timestamp_s"] <= 2.88)]
    long_gap_edges = f_df[(f_df["timestamp_s"] >= 3.68) & (f_df["timestamp_s"] <= 4.72)]

    assert kick_window["pre_smooth_contact_split"].any(), "True pass kick at t=1.0s should be detected as a pre-smooth contact split."
    assert intercept_window["pre_smooth_contact_split"].any(), "True interception at t=2.8s should be detected as a pre-smooth contact split."
    assert not long_gap_edges["pre_smooth_contact_split"].any(), "Long gap edges must be suppressed and not trigger false contact splits."

    max_kick_dv = float(kick_window["ball_vel_change_proxy_mps"].max())
    max_intercept_dv = float(intercept_window["ball_vel_change_proxy_mps"].max())
    print(
        f"[PASS] 5. Multi-scale velocity-change proxy I_b verified: "
        f"Kick dV = {max_kick_dv:.2f} m/s, Interception dV = {max_intercept_dv:.2f} m/s, "
        f"Gap-edge false splits = 0."
    )
    print("=" * 72)
    print("ALL TASK 1 CHECKS PASSED SUCCESSFULLY.")
    print("=" * 72)


if __name__ == "__main__":
    run_task1_verification()