# Task 2: Local Kinematic Feature Engine & Short-Horizon Interception Search (RFC v2.0).
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import ValidatedTrackingDataset, TeamID, PlayerRole


@dataclass
class FeatureConfig:
    reachability_horizons_s: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    max_player_speed_mps: float = 9.5
    max_player_accel_mps2: float = 7.0
    reaction_time_s: float = 0.30
    touch_proximity_threshold_m: float = 1.20
    carry_proximity_threshold_m: float = 2.50
    congestion_radius_near_m: float = 2.00
    congestion_radius_far_m: float = 5.00


@dataclass
class LocalKinematicFeatureExtractor:
    """
    Computes player-ball relative kinematics, short-horizon reachability search,
    decoupled carry/touch evidence, and team ambiguity margins.
    """
    config: Optional[FeatureConfig] = None

    def __post_init__(self):
        if self.config is None:
            self.config = FeatureConfig()

    def extract_features(self, dataset: ValidatedTrackingDataset) -> pd.DataFrame:
        f_df = dataset.frame_df.copy()
        p_df = dataset.player_tracks_df.copy()

        # Map team_id from dataset.players if not already in player_tracks_df
        if "team_id" not in p_df.columns and dataset.players:
            team_map = {pid: (pmeta.team_id.value if hasattr(pmeta.team_id, "value") else str(pmeta.team_id)) for pid, pmeta in dataset.players.items()}
            p_df["team_id"] = p_df["player_id"].map(team_map)

        if len(f_df) == 0:
            null_dict = self._get_null_features()
            for col, val in null_dict.items():
                f_df[col] = val
            return f_df

        # Merge frame ball state with player tracks per frame
        # Ensure frame_id sorting
        f_df = f_df.sort_values("frame_id").reset_index(drop=True)
        p_df = p_df.sort_values(["frame_id", "player_id"]).reset_index(drop=True)

        n_frames = len(f_df)
        dt = 1.0 / dataset.metadata.sampling_rate_hz

        # Pre-allocate output feature columns
        feat_rows = []

        # Group players by frame for rapid vector operations
        players_by_frame = {fid: group for fid, group in p_df.groupby("frame_id", sort=False)}

        for idx, row in f_df.iterrows():
            fid = int(row["frame_id"])
            t = float(row["timestamp_s"])
            bx = float(row["ball_x"]) if pd.notna(row.get("ball_x")) else np.nan
            by = float(row["ball_y"]) if pd.notna(row.get("ball_y")) else np.nan
            bvx = float(row["ball_vx"]) if pd.notna(row.get("ball_vx")) else 0.0
            bvy = float(row["ball_vy"]) if pd.notna(row.get("ball_vy")) else 0.0
            bspeed = float(row["ball_speed"]) if pd.notna(row.get("ball_speed")) else 0.0
            dv_proxy = float(row["ball_vel_change_proxy_mps"]) if pd.notna(row.get("ball_vel_change_proxy_mps")) else 0.0
            is_valid_ball = bool(row["ball_is_valid"]) if "ball_is_valid" in row else (np.isfinite(bx) and np.isfinite(by))
            ball_ax = float(row["ball_ax"]) if pd.notna(row.get("ball_ax")) else 0.0
            ball_ay = float(row["ball_ay"]) if pd.notna(row.get("ball_ay")) else 0.0

            frame_feats = {"frame_id": fid, "timestamp_s": t}

            if not is_valid_ball or fid not in players_by_frame or not (np.isfinite(bx) and np.isfinite(by)):
                # Default null / safe values when tracking is missing
                frame_feats.update(self._get_null_features())
                feat_rows.append(frame_feats)
                continue

            p_group = players_by_frame[fid]
            valid_p = p_group[p_group["x"].notna() & p_group["y"].notna()]

            if len(valid_p) == 0:
                frame_feats.update(self._get_null_features())
                feat_rows.append(frame_feats)
                continue

            # Extract arrays
            px = valid_p["x"].to_numpy(dtype=float)
            py = valid_p["y"].to_numpy(dtype=float)
            pvx = valid_p["vx"].fillna(0.0).to_numpy(dtype=float) if "vx" in valid_p.columns else np.zeros(len(valid_p), dtype=float)
            pvy = valid_p["vy"].fillna(0.0).to_numpy(dtype=float) if "vy" in valid_p.columns else np.zeros(len(valid_p), dtype=float)
            p_speed = np.hypot(pvx, pvy)
            teams = valid_p["team_id"].to_numpy(dtype=str) if "team_id" in valid_p.columns else np.array([""] * len(valid_p))

            # Distances to ball
            dx = px - bx
            dy = py - by
            dists = np.hypot(dx, dy)

            home_mask = (teams == TeamID.HOME.value)
            away_mask = (teams == TeamID.AWAY.value)

            # 1. Minimum distance to ball per team
            min_dist_home = float(np.min(dists[home_mask])) if home_mask.any() else 999.0
            min_dist_away = float(np.min(dists[away_mask])) if away_mask.any() else 999.0

            frame_feats["min_dist_home"] = min_dist_home
            frame_feats["min_dist_away"] = min_dist_away
            frame_feats["home_away_min_dist_diff"] = min_dist_home - min_dist_away

            # 2. Short-Horizon Interception Reachability Search (RFC v2.0 Section 4.3)
            # Evaluate reachability across horizons tau considering inertial motion and touch radius
            home_min_intercept_tau = 999.0
            away_min_intercept_tau = 999.0

            for tau in self.config.reachability_horizons_s:
                # Predict ball position at t + tau assuming constant acceleration model
                bx_tau = bx + bvx * tau + 0.5 * ball_ax * (tau ** 2)
                by_tau = by + bvy * tau + 0.5 * ball_ay * (tau ** 2)

                t_inertial = min(tau, self.config.reaction_time_s)
                t_accel = max(0.0, tau - self.config.reaction_time_s)

                # Predict player inertial trajectory during reaction time, then acceleration
                px_tau = px + pvx * t_inertial
                py_tau = py + pvy * t_inertial
                p_dist = np.hypot(px_tau - bx_tau, py_tau - by_tau)

                # Max reach includes physical touch margin plus acceleration after reaction time
                max_reach = self.config.touch_proximity_threshold_m + 0.5 * self.config.max_player_accel_mps2 * (t_accel ** 2)
                max_reach = np.minimum(max_reach, self.config.max_player_speed_mps * tau + self.config.touch_proximity_threshold_m)

                can_reach = p_dist <= max_reach
                if can_reach.any():
                    if (can_reach & home_mask).any() and tau < home_min_intercept_tau:
                        home_min_intercept_tau = tau
                    if (can_reach & away_mask).any() and tau < away_min_intercept_tau:
                        away_min_intercept_tau = tau

            frame_feats["home_min_intercept_tau"] = home_min_intercept_tau
            frame_feats["away_min_intercept_tau"] = away_min_intercept_tau

            # 3. Decoupled Touch and Carry Evidence
            # Touch evidence: proximity + velocity-change proxy dv_proxy
            touch_prox_home = np.exp(-((min_dist_home / self.config.touch_proximity_threshold_m) ** 2))
            touch_prox_away = np.exp(-((min_dist_away / self.config.touch_proximity_threshold_m) ** 2))

            frame_feats["home_touch_evidence"] = touch_prox_home * min(dv_proxy / 5.0, 1.0)
            frame_feats["away_touch_evidence"] = touch_prox_away * min(dv_proxy / 5.0, 1.0)

            # Carry evidence: sustained low distance and velocity alignment
            carry_home_mask = (dists <= self.config.carry_proximity_threshold_m) & home_mask
            carry_away_mask = (dists <= self.config.carry_proximity_threshold_m) & away_mask

            home_carry_score = 0.0
            if carry_home_mask.any():
                h_p_spd = p_speed[carry_home_mask]
                h_dists = dists[carry_home_mask]
                h_pvx = pvx[carry_home_mask]
                h_pvy = pvy[carry_home_mask]
                moving = (h_p_spd > 0.5) & (bspeed > 0.5)
                align = np.full(len(h_p_spd), 0.5, dtype=float)
                if moving.any():
                    cos_sim = np.clip(
                        (h_pvx[moving] * bvx + h_pvy[moving] * bvy) / (h_p_spd[moving] * bspeed),
                        -1.0, 1.0
                    )
                    align[moving] = 0.5 * (cos_sim + 1.0)
                scores = (1.0 - (h_dists / self.config.carry_proximity_threshold_m)) * align
                home_carry_score = float(np.max(scores))

            away_carry_score = 0.0
            if carry_away_mask.any():
                a_p_spd = p_speed[carry_away_mask]
                a_dists = dists[carry_away_mask]
                a_pvx = pvx[carry_away_mask]
                a_pvy = pvy[carry_away_mask]
                moving = (a_p_spd > 0.5) & (bspeed > 0.5)
                align = np.full(len(a_p_spd), 0.5, dtype=float)
                if moving.any():
                    cos_sim = np.clip(
                        (a_pvx[moving] * bvx + a_pvy[moving] * bvy) / (a_p_spd[moving] * bspeed),
                        -1.0, 1.0
                    )
                    align[moving] = 0.5 * (cos_sim + 1.0)
                scores = (1.0 - (a_dists / self.config.carry_proximity_threshold_m)) * align
                away_carry_score = float(np.max(scores))

            frame_feats["home_carry_evidence"] = home_carry_score
            frame_feats["away_carry_evidence"] = away_carry_score

            # 4. Team Ambiguity Margins & Congestion Counts
            home_dists_sorted = np.sort(dists[home_mask]) if home_mask.any() else np.array([999.0, 999.0])
            away_dists_sorted = np.sort(dists[away_mask]) if away_mask.any() else np.array([999.0, 999.0])

            h_margin = float(home_dists_sorted[1] - home_dists_sorted[0]) if len(home_dists_sorted) > 1 else 0.0
            a_margin = float(away_dists_sorted[1] - away_dists_sorted[0]) if len(away_dists_sorted) > 1 else 0.0

            frame_feats["home_top1_top2_margin"] = h_margin
            frame_feats["away_top1_top2_margin"] = a_margin

            # Congestion counts (opponents near ball)
            frame_feats["opp_count_near_home_2m"] = int(np.sum((dists <= self.config.congestion_radius_near_m) & away_mask))
            frame_feats["opp_count_near_away_2m"] = int(np.sum((dists <= self.config.congestion_radius_near_m) & home_mask))
            frame_feats["opp_count_near_home_5m"] = int(np.sum((dists <= self.config.congestion_radius_far_m) & away_mask))
            frame_feats["opp_count_near_away_5m"] = int(np.sum((dists <= self.config.congestion_radius_far_m) & home_mask))

            feat_rows.append(frame_feats)

        feat_df = pd.DataFrame(feat_rows)
        return f_df.merge(feat_df, on=["frame_id", "timestamp_s"], how="left")

    @staticmethod
    def _get_null_features() -> Dict[str, float]:
        return {
            "min_dist_home": 999.0,
            "min_dist_away": 999.0,
            "home_away_min_dist_diff": 0.0,
            "home_min_intercept_tau": 999.0,
            "away_min_intercept_tau": 999.0,
            "home_touch_evidence": 0.0,
            "away_touch_evidence": 0.0,
            "home_carry_evidence": 0.0,
            "away_carry_evidence": 0.0,
            "home_top1_top2_margin": 0.0,
            "away_top1_top2_margin": 0.0,
            "opp_count_near_home_2m": 0,
            "opp_count_near_away_2m": 0,
            "opp_count_near_home_5m": 0,
            "opp_count_near_away_5m": 0,
        }


# ============================================================================
# VERIFICATION SUITE FOR TASK 2
# ============================================================================

def run_task2_verification() -> None:
    print("=" * 72)
    print("RUNNING TASK 2 LOCAL KINEMATIC & REACHABILITY VERIFICATION")
    print("=" * 72)

    from task1_schema_and_preprocessing import generate_synthetic_diagnostic_match, SegmentedPreprocessor

    raw = generate_synthetic_diagnostic_match(sampling_rate_hz=25.0)
    prep = SegmentedPreprocessor()
    proc = prep.process(raw)

    extractor = LocalKinematicFeatureExtractor()
    feat_df = extractor.extract_features(proc)

    # Verify key feature columns exist
    assert "home_min_intercept_tau" in feat_df.columns
    assert "home_touch_evidence" in feat_df.columns
    assert "home_carry_evidence" in feat_df.columns
    assert "opp_count_near_home_2m" in feat_df.columns

    # Verify intercept tau during Home carry (t < 1.0s) -> Home tau should be very small
    carry_rows = feat_df[(feat_df["timestamp_s"] >= 0.2) & (feat_df["timestamp_s"] <= 0.8)]
    assert (carry_rows["home_min_intercept_tau"] <= 0.2).all(), "Home player carrying ball should have instant reachability tau <= 0.2s."

    # Verify pass kick touch evidence spike at t = 1.0s
    kick_row = feat_df[np.isclose(feat_df["timestamp_s"], 1.0, atol=0.04)]
    assert not kick_row.empty, "Kick frame should be present."
    home_touch = float(kick_row["home_touch_evidence"].iloc[0])
    assert home_touch > 0.1, f"Home touch evidence should spike during kick, got {home_touch}"

    print(f"[PASS] 1. Short-horizon interception tau verified during carry (tau <= 0.2s).")
    print(f"[PASS] 2. Decoupled touch evidence spike verified at pass kick (touch score = {home_touch:.3f}).")
    print("=" * 72)
    print("ALL TASK 2 CHECKS PASSED SUCCESSFULLY.")
    print("=" * 72)


if __name__ == "__main__":
    run_task2_verification()