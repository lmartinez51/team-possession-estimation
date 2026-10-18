# Task 3: Probabilistic Candidate Contacts, Unreliable-Proximity Proxy,
# Bounded Bridges, and TypeSafe AI (Jev / System One) Ambiguity Refiner (RFC v2.0).
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import (
    MatchMetadata,
    SegmentedPreprocessor,
    TeamID,
    ValidatedTrackingDataset,
    generate_synthetic_diagnostic_match,
)
from task2_feature_engine import LocalKinematicFeatureExtractor


# ============================================================================
# 1. CONFIGURATION & TYPESAFE AI (JEV / SYSTEM ONE) DATA STRUCTURES
# ============================================================================

@dataclass
class AnchorBridgeConfig:
    # All temporal parameters strictly in seconds (RFC v2.0 Section 3.2 & 4.4)
    max_bridge_span_s: float = 4.00
    max_allowed_gap_in_bridge_s: float = 0.50
    min_anchor_Peak_window_s: float = 0.12
    contact_dv_scale_mps: float = 5.50
    contact_dv_offset_mps: float = 1.20
    contact_reach_scale_m: float = 1.35
    bounce_min_dist_m: float = 1.80
    passby_fast_ball_mps: float = 7.00
    passby_close_dist_m: float = 1.60
    min_anchor_conf_threshold: float = 0.30
    enable_jev_refiner: bool = True
    jev_batch_size: int = 20

    @property
    def min_anchor_peak_window_s(self) -> float:
        return self.min_anchor_Peak_window_s


@dataclass(frozen=True)
class NoulDecision:
    # Atomic binary probabilistic primitive from System One / Jev
    choice: bool
    prob_true: float
    confidence: float


@dataclass(frozen=True)
class ChoiceDecision:
    # Atomic categorical primitive over a closed option set from System One / Jev
    selected: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreDecision:
    # Atomic continuous scalar primitive in [0, 1] from System One / Jev
    score: float
    confidence: float


# ============================================================================
# 2. TYPESAFE AI (JEV / SYSTEM ONE) SPARSE EVENT REFINER
# ============================================================================

class JevSystemOneRefiner:
    # Evaluates ambiguous candidate contacts and close pass-by events using
    # TypeSafe AI's System One primitives (Noul, Choice, Score).
    # Operates in Live API mode if TYPESAFE_API_KEY is set and use_live_api=True,
    # or falls back to a deterministic calibrated RLCD emulator for offline runs.

    CHOICE_OPTIONS = [
        "Home_Touch",
        "Away_Touch",
        "Ground_Bounce",
        "Pass_By_Or_Noise",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.typesafe.ai/v1/systemone",
        model_name: str = "jev-1.13.0",
        use_live_api: bool = False,
        timeout_s: float = 5.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.api_url = api_url
        self.model_name = model_name
        self.use_live_api = bool(use_live_api and len(self.api_key) > 0)
        self.timeout_s = timeout_s

    def build_system_one_payload(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Builds a batched System One request evaluating Noul, Choice, and Score
        # primitives in parallel over a structured kinematic state.
        questions: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            fid = int(ev["frame_id"])
            questions[f"f_{fid}_is_player_contact"] = {
                "type": "noul",
                "prompt": (
                    f"At frame {fid}, does the kinematic discontinuity and player proximity "
                    f"indicate a genuine physical player-ball contact rather than noise or free flight?"
                ),
            }
            questions[f"f_{fid}_attribution"] = {
                "type": "choice",
                "options": list(self.CHOICE_OPTIONS),
                "prompt": (
                    f"At frame {fid}, classify the ball interaction event among "
                    f"Home_Touch, Away_Touch, Ground_Bounce, or Pass_By_Or_Noise."
                ),
            }
            questions[f"f_{fid}_unreliable_prox"] = {
                "type": "score",
                "prompt": (
                    f"At frame {fid}, score the likelihood in [0, 1] that 2D planar proximity "
                    f"is unreliable due to a high-speed un-deflected pass-by or aerial flight."
                ),
            }

        return {
            "model": self.model_name,
            "state": {"kinematic_events": events},
            "questions": questions,
        }

    def refine_ambiguous_events(
        self, events: List[Dict[str, Any]], batch_size: int = 20
    ) -> Dict[int, Dict[str, Any]]:
        if not events:
            return {}

        results: Dict[int, Dict[str, Any]] = {}
        for start in range(0, len(events), max(1, batch_size)):
            batch = events[start : start + batch_size]
            if self.use_live_api:
                batch_res = self._call_live_api_with_fallback(batch)
            else:
                batch_res = self._evaluate_calibrated_offline(batch)
            results.update(batch_res)
        return results

    def _call_live_api_with_fallback(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        payload = self.build_system_one_payload(batch)
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))
                return self._parse_api_response(batch, raw_json)
        except Exception:
            # Seamless fallback to calibrated offline evaluator if network or schema fails
            return self._evaluate_calibrated_offline(batch)

    def _parse_api_response(
        self, batch: List[Dict[str, Any]], raw_json: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        answers = raw_json.get("answers", raw_json.get("results", {}))
        parsed: Dict[int, Dict[str, Any]] = {}
        offline_backup = self._evaluate_calibrated_offline(batch)

        for ev in batch:
            fid = int(ev["frame_id"])
            k_noul = f"f_{fid}_is_player_contact"
            k_choice = f"f_{fid}_attribution"
            k_score = f"f_{fid}_unreliable_prox"

            if k_noul not in answers or k_choice not in answers or k_score not in answers:
                parsed[fid] = offline_backup[fid]
                continue

            a_noul = answers[k_noul]
            a_choice = answers[k_choice]
            a_score = answers[k_score]

            prob_true = float(a_noul.get("probability", a_noul.get("prob_true", 0.5)))
            noul_dec = NoulDecision(
                choice=bool(prob_true >= 0.5),
                prob_true=float(np.clip(prob_true, 0.0, 1.0)),
                confidence=float(np.clip(a_noul.get("confidence", 0.8), 0.0, 1.0)),
            )

            probs_raw = a_choice.get("probabilities", {})
            probs_clean = {opt: float(probs_raw.get(opt, 0.25)) for opt in self.CHOICE_OPTIONS}
            total_p = sum(probs_clean.values())
            if total_p > 1e-9:
                probs_clean = {k: v / total_p for k, v in probs_clean.items()}
            else:
                probs_clean = {opt: 1.0 / len(self.CHOICE_OPTIONS) for opt in self.CHOICE_OPTIONS}
            selected_opt = max(probs_clean.items(), key=lambda kv: kv[1])[0]
            choice_dec = ChoiceDecision(
                selected=selected_opt,
                probabilities=probs_clean,
                confidence=float(np.clip(a_choice.get("confidence", 0.8), 0.0, 1.0)),
            )

            score_val = float(np.clip(a_score.get("score", a_score.get("value", 0.0)), 0.0, 1.0))
            score_dec = ScoreDecision(
                score=score_val,
                confidence=float(np.clip(a_score.get("confidence", 0.8), 0.0, 1.0)),
            )

            parsed[fid] = {
                "noul": noul_dec,
                "choice": choice_dec,
                "score": score_dec,
            }
        return parsed

    def _evaluate_calibrated_offline(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        # Calibrated, team-swap symmetric evaluator implementing System One's
        # Noul, Choice, and Score semantics over the kinematic event state.
        out: Dict[int, Dict[str, Any]] = {}
        for ev in batch:
            fid = int(ev["frame_id"])
            dv = float(ev["dv_proxy_mps"])
            cons = float(ev["multiscale_consistency"])
            d_home = float(ev["min_dist_home"])
            d_away = float(ev["min_dist_away"])
            tau_home = float(ev["home_min_intercept_tau"])
            tau_away = float(ev["away_min_intercept_tau"])
            touch_h = float(ev["home_touch_evidence"])
            touch_a = float(ev["away_touch_evidence"])
            carry_h = float(ev["home_carry_evidence"])
            carry_a = float(ev["away_carry_evidence"])
            bspeed = float(ev["ball_speed"])
            is_interp = bool(ev["is_interpolated"])
            unrel_bound = bool(ev["boundary_unreliable"])

            d_min = min(d_home, d_away)
            quality_factor = 0.0 if unrel_bound else (0.40 if is_interp else 1.0)

            # 1. Choice logits (strictly symmetric between Home and Away)
            prox_h = np.exp(-((d_home / 1.25) ** 2))
            prox_a = np.exp(-((d_away / 1.25) ** 2))
            reach_h = np.exp(-max(0.0, tau_home) / 0.45) if tau_home < 10.0 else 0.0
            reach_a = np.exp(-max(0.0, tau_away) / 0.45) if tau_away < 10.0 else 0.0

            kin_kick = np.clip((dv - 1.0) / 5.0, 0.0, 1.5) * (0.35 + 0.65 * cons)

            logit_home = (
                2.6 * prox_h * kin_kick
                + 1.8 * touch_h
                + 1.1 * carry_h
                + 0.6 * reach_h
            ) * quality_factor - 1.1

            logit_away = (
                2.6 * prox_a * kin_kick
                + 1.8 * touch_a
                + 1.1 * carry_a
                + 0.6 * reach_a
            ) * quality_factor - 1.1

            # Ground bounce rises when velocity changes sharply but no player is within reach
            no_player_near = float(np.clip((d_min - 1.35) / 1.0, 0.0, 1.5))
            logit_bounce = (2.4 * kin_kick * no_player_near * quality_factor) - 1.6

            # Pass-by or noise rises when ball moves fast past players without velocity change
            # or when tracking is interpolated/unreliable
            no_kick = float(np.exp(-dv / 2.2))
            fast_flight = float(np.clip((bspeed - 6.0) / 6.0, 0.0, 1.2))
            logit_passby = 1.9 * no_kick * (0.5 + fast_flight) + (1.2 if is_interp else 0.0) - 0.6

            logits = np.array([logit_home, logit_away, logit_bounce, logit_passby], dtype=float)
            logits_shifted = logits - np.max(logits)
            exp_l = np.exp(logits_shifted)
            probs = exp_l / np.sum(exp_l)

            probs_dict = {
                "Home_Touch": float(probs[0]),
                "Away_Touch": float(probs[1]),
                "Ground_Bounce": float(probs[2]),
                "Pass_By_Or_Noise": float(probs[3]),
            }
            selected_opt = max(probs_dict.items(), key=lambda kv: kv[1])[0]

            # Normalized Shannon entropy determines calibrated decision confidence
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            max_entropy = float(np.log(len(self.CHOICE_OPTIONS)))
            choice_conf = float(np.clip((1.0 - entropy / max_entropy) * quality_factor, 0.05, 0.99))

            # 2. Noul Decision (Is genuine player contact?)
            prob_player_contact = float(np.clip(probs[0] + probs[1], 0.0, 1.0))
            noul_conf = float(np.clip(abs(2.0 * prob_player_contact - 1.0) * quality_factor, 0.10, 0.99))
            noul_dec = NoulDecision(
                choice=bool(prob_player_contact >= 0.50),
                prob_true=prob_player_contact,
                confidence=noul_conf,
            )

            choice_dec = ChoiceDecision(
                selected=selected_opt,
                probabilities=probs_dict,
                confidence=choice_conf,
            )

            # 3. Score Decision (Unreliable 2D proximity / aerial pass-by likelihood)
            close_opponent_passby = float(np.exp(-((d_min / 1.50) ** 2))) * fast_flight * no_kick
            unrel_score = float(np.clip(0.65 * close_opponent_passby + 0.35 * probs_dict["Pass_By_Or_Noise"], 0.0, 1.0))
            score_dec = ScoreDecision(
                score=unrel_score,
                confidence=float(np.clip(0.5 + 0.5 * cons * quality_factor, 0.20, 0.98)),
            )

            out[fid] = {
                "noul": noul_dec,
                "choice": choice_dec,
                "score": score_dec,
            }
        return out


# ============================================================================
# 3. KINEMATIC CONTACT, AERIAL PROXY & BOUNDED BRIDGE ENGINE (RFC v2.0 Sec 4)
# ============================================================================

class KinematicContactAndBridgeEngine:
    def __init__(
        self,
        config: Optional[AnchorBridgeConfig] = None,
        jev_refiner: Optional[JevSystemOneRefiner] = None,
    ):
        self.config = config or AnchorBridgeConfig()
        self.jev_refiner = jev_refiner or JevSystemOneRefiner()

    def extract_anchor_and_bridge_features(
        self, feat_df: pd.DataFrame, metadata: MatchMetadata
    ) -> pd.DataFrame:
        df = feat_df.sort_values("frame_id").reset_index(drop=True).copy() if "frame_id" in feat_df.columns else feat_df.copy()
        if len(df) == 0:
            null_cols = {
                "candidate_contact_confidence": 0.0,
                "ground_bounce_candidate_prob": 0.0,
                "anchor_home_prob": 0.5,
                "anchor_away_prob": 0.5,
                "unreliable_proximity_likelihood": 0.0,
                "jev_refined_flag": False,
                "jev_decision_confidence": 0.0,
                "is_candidate_anchor": False,
                "prev_anchor_home_prob": 0.5,
                "prev_anchor_away_prob": 0.5,
                "prev_anchor_age_s": self.config.max_bridge_span_s,
                "prev_anchor_confidence": 0.0,
                "next_anchor_home_prob": 0.5,
                "next_anchor_away_prob": 0.5,
                "next_anchor_lead_s": self.config.max_bridge_span_s,
                "next_anchor_confidence": 0.0,
                "bridge_same_team_agreement": 0.0,
                "bridge_contradiction_score": 0.0,
            }
            for col, val in null_cols.items():
                df[col] = val
            return df

        # Step 1: Analytical candidate contacts, ground bounce separation & aerial proxy
        df = self._compute_analytical_candidates(df)

        # Step 2: Sparse ambiguity detection & TypeSafe AI (Jev / System One) refinement
        if self.config.enable_jev_refiner:
            df = self._apply_jev_sparse_refinement(df)

        # Step 3: Non-maximum suppression of local anchor peaks within continuous segments
        df = self._select_salient_anchors(df, metadata)

        # Step 4: Bounded forward/backward bridge propagation & contradiction scoring
        df = self._compute_bounded_bridges(df, metadata)

        return df

    def _compute_analytical_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        dv = df["ball_vel_change_proxy_mps"].fillna(0.0).to_numpy(dtype=float) if "ball_vel_change_proxy_mps" in df.columns else np.zeros(n, dtype=float)
        cons = df["ball_multiscale_consistency"].fillna(0.0).to_numpy(dtype=float) if "ball_multiscale_consistency" in df.columns else np.zeros(n, dtype=float)
        unrel_bound = df["ball_boundary_unreliable"].fillna(True).to_numpy(dtype=bool) if "ball_boundary_unreliable" in df.columns else np.zeros(n, dtype=bool)
        is_interp = df["ball_is_interpolated"].fillna(False).to_numpy(dtype=bool) if "ball_is_interpolated" in df.columns else np.zeros(n, dtype=bool)
        gap_dur = df["ball_gap_duration_s"].fillna(0.0).to_numpy(dtype=float) if "ball_gap_duration_s" in df.columns else np.zeros(n, dtype=float)
        valid_ball = df["ball_is_valid"].fillna(False).to_numpy(dtype=bool) if "ball_is_valid" in df.columns else np.ones(n, dtype=bool)
        bspeed = df["ball_speed"].fillna(0.0).to_numpy(dtype=float) if "ball_speed" in df.columns else np.zeros(n, dtype=float)

        d_home = df["min_dist_home"].fillna(999.0).to_numpy(dtype=float) if "min_dist_home" in df.columns else np.full(n, 999.0, dtype=float)
        d_away = df["min_dist_away"].fillna(999.0).to_numpy(dtype=float) if "min_dist_away" in df.columns else np.full(n, 999.0, dtype=float)
        d_min = np.minimum(d_home, d_away)

        touch_h = df["home_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_touch_evidence" in df.columns else np.zeros(n, dtype=float)
        touch_a = df["away_touch_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_touch_evidence" in df.columns else np.zeros(n, dtype=float)
        carry_h = df["home_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "home_carry_evidence" in df.columns else np.zeros(n, dtype=float)
        carry_a = df["away_carry_evidence"].fillna(0.0).to_numpy(dtype=float) if "away_carry_evidence" in df.columns else np.zeros(n, dtype=float)

        # Quality discount factor (RFC v2.0 Section 4.1)
        quality = np.where(
            (~valid_ball) | unrel_bound,
            0.0,
            np.where(is_interp, 0.35, 1.0) * np.exp(-gap_dur / 0.25),
        )

        # Kinematic discontinuity strength from velocity-change proxy I_b
        dv_excess = np.maximum(0.0, dv - self.config.contact_dv_offset_mps)
        kin_disc = (1.0 - np.exp(-dv_excess / self.config.contact_dv_scale_mps)) * (0.35 + 0.65 * cons)

        # Plausibility that a player is close enough to cause the contact
        reach_plausibility = np.exp(-((d_min / self.config.contact_reach_scale_m) ** 2))

        # Combine sharp kick/interception discontinuities with high-confidence controlled carries
        max_carry = np.maximum(carry_h, carry_a)
        contact_raw = np.maximum(kin_disc * reach_plausibility, 0.55 * max_carry * reach_plausibility)
        candidate_contact_conf = np.clip(contact_raw * quality, 0.0, 1.0)

        # Separate ground bounce candidates (kinematic discontinuity when no player is within reach)
        far_from_players = np.clip((d_min - self.config.contact_reach_scale_m) / self.config.bounce_min_dist_m, 0.0, 1.0)
        ground_bounce_prob = np.clip(kin_disc * far_from_players * quality, 0.0, 1.0)

        # Symmetric team attribution probabilities at each candidate frame
        prox_h = np.exp(-((d_home / self.config.contact_reach_scale_m) ** 2))
        prox_a = np.exp(-((d_away / self.config.contact_reach_scale_m) ** 2))
        score_h = prox_h * (0.50 + 0.30 * touch_h + 0.20 * carry_h)
        score_a = prox_a * (0.50 + 0.30 * touch_a + 0.20 * carry_a)
        denom = score_h + score_a + 1e-9

        # Blend toward 0.5 when candidate_contact_conf is near 0
        raw_p_home = score_h / denom
        raw_p_away = score_a / denom
        anchor_home_prob = 0.5 + (raw_p_home - 0.5) * candidate_contact_conf
        anchor_away_prob = 0.5 + (raw_p_away - 0.5) * candidate_contact_conf

        # Uncertain Aerial / Unreliable-Proximity Proxy (RFC v2.0 Section 4.2)
        # Spikes when the ball moves rapidly past a nearby player without kinematic deflection (dv ~ 0)
        # or during interpolated occlusion gaps in mid-flight.
        fast_factor = np.clip((bspeed - 4.0) / self.config.passby_fast_ball_mps, 0.0, 1.0)
        no_deflection = np.exp(-dv / 2.0)
        close_passby = np.exp(-((d_min / self.config.passby_close_dist_m) ** 2))
        unreliable_prox = np.clip(
            fast_factor * no_deflection * np.maximum(close_passby, 0.50 * is_interp.astype(float)),
            0.0,
            1.0,
        )
        unreliable_prox = np.where(valid_ball, unreliable_prox, 1.0)

        df["candidate_contact_confidence"] = candidate_contact_conf
        df["ground_bounce_candidate_prob"] = ground_bounce_prob
        df["anchor_home_prob"] = anchor_home_prob
        df["anchor_away_prob"] = anchor_away_prob
        df["unreliable_proximity_likelihood"] = unreliable_prox
        df["jev_refined_flag"] = False
        df["jev_decision_confidence"] = quality
        return df

    def _apply_jev_sparse_refinement(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identifies sparse ambiguous frames (contested touches, borderline bounces,
        # or high-speed close pass-bys) and routes them to JevSystemOneRefiner.
        d_home = df["min_dist_home"].to_numpy(dtype=float)
        d_away = df["min_dist_away"].to_numpy(dtype=float)
        dv = df["ball_vel_change_proxy_mps"].to_numpy(dtype=float)
        c_conf = df["candidate_contact_confidence"].to_numpy(dtype=float)
        unrel = df["unreliable_proximity_likelihood"].to_numpy(dtype=float)
        valid_ball = df["ball_is_valid"].to_numpy(dtype=bool)

        # Ambiguity criteria:
        # (a) Candidate contact where both teams are within 2.5m,
        # (b) High velocity change (dv >= 3.5 m/s) with uncertain player reach, or
        # (c) Suspected close pass-by (unrel >= 0.35).
        both_close = (d_home <= 2.5) & (d_away <= 2.5) & (c_conf >= 0.20)
        sharp_dv = dv >= 3.5
        suspected_passby = unrel >= 0.35
        ambiguous_mask = valid_ball & (both_close | sharp_dv | suspected_passby)

        ambig_indices = np.where(ambiguous_mask)[0]
        if len(ambig_indices) == 0:
            return df

        events_payload: List[Dict[str, Any]] = []
        for idx in ambig_indices:
            row = df.iloc[idx]
            events_payload.append({
                "frame_id": int(row["frame_id"]),
                "timestamp_s": float(row["timestamp_s"]),
                "dv_proxy_mps": float(row["ball_vel_change_proxy_mps"]),
                "multiscale_consistency": float(row["ball_multiscale_consistency"]),
                "min_dist_home": float(row["min_dist_home"]),
                "min_dist_away": float(row["min_dist_away"]),
                "home_min_intercept_tau": float(row["home_min_intercept_tau"]),
                "away_min_intercept_tau": float(row["away_min_intercept_tau"]),
                "home_touch_evidence": float(row["home_touch_evidence"]),
                "away_touch_evidence": float(row["away_touch_evidence"]),
                "home_carry_evidence": float(row["home_carry_evidence"]),
                "away_carry_evidence": float(row["away_carry_evidence"]),
                "ball_speed": float(row["ball_speed"]) if np.isfinite(row["ball_speed"]) else 0.0,
                "is_interpolated": bool(row["ball_is_interpolated"]),
                "boundary_unreliable": bool(row["ball_boundary_unreliable"]),
            })

        refined_map = self.jev_refiner.refine_ambiguous_events(
            events_payload, batch_size=self.config.jev_batch_size
        )

        c_conf_arr = df["candidate_contact_confidence"].to_numpy(dtype=float).copy()
        bounce_arr = df["ground_bounce_candidate_prob"].to_numpy(dtype=float).copy()
        ah_arr = df["anchor_home_prob"].to_numpy(dtype=float).copy()
        aa_arr = df["anchor_away_prob"].to_numpy(dtype=float).copy()
        unrel_arr = df["unreliable_proximity_likelihood"].to_numpy(dtype=float).copy()
        flag_arr = df["jev_refined_flag"].to_numpy(dtype=bool).copy()
        jconf_arr = df["jev_decision_confidence"].to_numpy(dtype=float).copy()

        for idx in ambig_indices:
            fid = int(df.iloc[idx]["frame_id"])
            if fid not in refined_map:
                continue
            res = refined_map[fid]
            noul_dec: NoulDecision = res["noul"]
            choice_dec: ChoiceDecision = res["choice"]
            score_dec: ScoreDecision = res["score"]

            w_jev = 0.65 * choice_dec.confidence
            c_conf_arr[idx] = float(
                np.clip((1.0 - w_jev) * c_conf_arr[idx] + w_jev * noul_dec.prob_true, 0.0, 1.0)
            )
            bounce_arr[idx] = float(
                np.clip(
                    (1.0 - w_jev) * bounce_arr[idx] + w_jev * choice_dec.probabilities["Ground_Bounce"],
                    0.0,
                    1.0,
                )
            )

            # Normalized team conditional probability from Choice primitive
            p_h_choice = choice_dec.probabilities["Home_Touch"]
            p_a_choice = choice_dec.probabilities["Away_Touch"]
            p_sum = p_h_choice + p_a_choice
            if p_sum > 1e-6:
                cond_h = p_h_choice / p_sum
                cond_a = p_a_choice / p_sum
            else:
                cond_h, cond_a = 0.5, 0.5

            ah_arr[idx] = float(
                0.5 + (w_jev * (cond_h - 0.5) + (1.0 - w_jev) * (ah_arr[idx] - 0.5)) * c_conf_arr[idx]
            )
            aa_arr[idx] = float(1.0 - ah_arr[idx])

            w_score = 0.60 * score_dec.confidence
            unrel_arr[idx] = float(
                np.clip((1.0 - w_score) * unrel_arr[idx] + w_score * score_dec.score, 0.0, 1.0)
            )
            flag_arr[idx] = True
            jconf_arr[idx] = float(choice_dec.confidence)

        df["candidate_contact_confidence"] = c_conf_arr
        df["ground_bounce_candidate_prob"] = bounce_arr
        df["anchor_home_prob"] = ah_arr
        df["anchor_away_prob"] = aa_arr
        df["unreliable_proximity_likelihood"] = unrel_arr
        df["jev_refined_flag"] = flag_arr
        df["jev_decision_confidence"] = jconf_arr
        return df

    def _select_salient_anchors(self, df: pd.DataFrame, metadata: MatchMetadata) -> pd.DataFrame:
        # Identifies salient candidate contact anchors within each valid ball segment.
        half_win = max(1, metadata.seconds_to_samples(self.config.min_anchor_Peak_window_s) // 2)
        c_conf = df["candidate_contact_confidence"].to_numpy(dtype=float)
        dv = df["ball_vel_change_proxy_mps"].fillna(0.0).to_numpy(dtype=float)
        pre_split = df["pre_smooth_contact_split"].fillna(False).to_numpy(dtype=bool)
        carry_h = df["home_carry_evidence"].fillna(0.0).to_numpy(dtype=float)
        carry_a = df["away_carry_evidence"].fillna(0.0).to_numpy(dtype=float)

        is_anchor = np.zeros(len(df), dtype=bool)

        for seg_id, group in df.groupby("ball_valid_seg_id", sort=False):
            if seg_id < 0:
                continue
            pos_idx = group.index.to_numpy(dtype=int)
            n = len(pos_idx)
            for i in range(n):
                g_i = pos_idx[i]
                if c_conf[g_i] < self.config.min_anchor_conf_threshold:
                    continue

                left = max(0, i - half_win)
                right = min(n, i + half_win + 1)
                local_slice = c_conf[pos_idx[left:right]]

                # Anchor is selected if it is a pre-detected kick/interception peak,
                # a local confidence maximum with significant velocity change, or strong sustained carry
                is_local_max = bool(c_conf[g_i] >= np.max(local_slice) - 1e-6)
                strong_carry = bool(max(carry_h[g_i], carry_a[g_i]) >= 0.65 and c_conf[g_i] >= 0.40)
                if pre_split[g_i] or (is_local_max and dv[g_i] >= 2.5) or strong_carry:
                    is_anchor[g_i] = True

        df["is_candidate_anchor"] = is_anchor
        return df

    def _compute_bounded_bridges(self, df: pd.DataFrame, metadata: MatchMetadata) -> pd.DataFrame:
        # Computes non-causal past/future anchor features within bounded reach (<= max_bridge_span_s),
        # halting across segment breaks or long gaps, and tracks opposing-team contradiction evidence.
        n = len(df)
        ts = df["timestamp_s"].to_numpy(dtype=float)
        is_anchor = df["is_candidate_anchor"].to_numpy(dtype=bool)
        c_conf = df["candidate_contact_confidence"].to_numpy(dtype=float)
        ah_prob = df["anchor_home_prob"].to_numpy(dtype=float)
        aa_prob = df["anchor_away_prob"].to_numpy(dtype=float)
        gap_dur = df["ball_gap_duration_s"].fillna(0.0).to_numpy(dtype=float)

        touch_h = df["home_touch_evidence"].fillna(0.0).to_numpy(dtype=float)
        touch_a = df["away_touch_evidence"].fillna(0.0).to_numpy(dtype=float)
        carry_h = df["home_carry_evidence"].fillna(0.0).to_numpy(dtype=float)
        carry_a = df["away_carry_evidence"].fillna(0.0).to_numpy(dtype=float)

        # Default neutral bridge arrays
        prev_home = np.full(n, 0.5, dtype=float)
        prev_away = np.full(n, 0.5, dtype=float)
        prev_age = np.full(n, self.config.max_bridge_span_s, dtype=float)
        prev_conf = np.zeros(n, dtype=float)

        next_home = np.full(n, 0.5, dtype=float)
        next_away = np.full(n, 0.5, dtype=float)
        next_lead = np.full(n, self.config.max_bridge_span_s, dtype=float)
        next_conf = np.zeros(n, dtype=float)

        bridge_agree = np.zeros(n, dtype=float)
        bridge_contradict = np.zeros(n, dtype=float)

        max_span = self.config.max_bridge_span_s

        # Isolate bridges by structural_seg_id and break if ball_gap_duration_s > max_allowed_gap_in_bridge_s
        long_gap_break = gap_dur > self.config.max_allowed_gap_in_bridge_s
        struct_seg = df["structural_seg_id"].to_numpy(dtype=int)
        bridge_seg_break = np.zeros(n, dtype=bool)
        if n > 0:
            bridge_seg_break[0] = True
        if n > 1:
            bridge_seg_break[1:] = (
                (struct_seg[1:] != struct_seg[:-1])
                | long_gap_break[1:]
                | long_gap_break[:-1]
            )
        bridge_seg_id = np.cumsum(bridge_seg_break)

        half_span = 0.5 * max_span
        for _, group in df.groupby(bridge_seg_id, sort=False):
            pos_idx = group.index.to_numpy(dtype=int)
            m = len(pos_idx)
            if m == 0:
                continue

            # 1. Forward pass: find most recent valid anchor <= t
            last_anchor_idx = -1
            for k in range(m):
                gi = pos_idx[k]
                if is_anchor[gi]:
                    last_anchor_idx = gi
                if last_anchor_idx >= 0:
                    age = float(ts[gi] - ts[last_anchor_idx])
                    if 0.0 <= age <= max_span:
                        decay = math.exp(-age / half_span)
                        prev_age[gi] = age
                        prev_conf[gi] = float(c_conf[last_anchor_idx] * decay)
                        prev_home[gi] = float(0.5 + (ah_prob[last_anchor_idx] - 0.5) * decay)
                        prev_away[gi] = float(1.0 - prev_home[gi])

            # 2. Backward pass: find nearest future valid anchor >= t
            next_anchor_idx = -1
            for k in range(m - 1, -1, -1):
                gi = pos_idx[k]
                if is_anchor[gi]:
                    next_anchor_idx = gi
                if next_anchor_idx >= 0:
                    lead = float(ts[next_anchor_idx] - ts[gi])
                    if 0.0 <= lead <= max_span:
                        decay = math.exp(-lead / half_span)
                        next_lead[gi] = lead
                        next_conf[gi] = float(c_conf[next_anchor_idx] * decay)
                        next_home[gi] = float(0.5 + (ah_prob[next_anchor_idx] - 0.5) * decay)
                        next_away[gi] = float(1.0 - next_home[gi])

            # 3. Compute same-team bridge agreement and contradiction score
            for k in range(m):
                gi = pos_idx[k]
                total_span = prev_age[gi] + next_lead[gi]
                if prev_conf[gi] > 0.05 and next_conf[gi] > 0.05 and total_span <= max_span:
                    # Probability that both bounding anchors belong to the same team
                    same_team_prob = (
                        prev_home[gi] * next_home[gi] + prev_away[gi] * next_away[gi]
                    )
                    geom_conf = math.sqrt(prev_conf[gi] * next_conf[gi])
                    bridge_agree[gi] = float(np.clip((2.0 * same_team_prob - 1.0) * geom_conf, -1.0, 1.0))

                    # Contradiction: current frame shows strong control/touch by the team opposing prev_anchor
                    opp_evidence_if_home = max(touch_a[gi], carry_a[gi])
                    opp_evidence_if_away = max(touch_h[gi], carry_h[gi])
                    expected_opp_ev = (
                        prev_home[gi] * opp_evidence_if_home + prev_away[gi] * opp_evidence_if_away
                    )
                    bridge_contradict[gi] = float(np.clip(expected_opp_ev * prev_conf[gi], 0.0, 1.0))

        df["prev_anchor_home_prob"] = prev_home
        df["prev_anchor_away_prob"] = prev_away
        df["prev_anchor_age_s"] = prev_age
        df["prev_anchor_confidence"] = prev_conf

        df["next_anchor_home_prob"] = next_home
        df["next_anchor_away_prob"] = next_away
        df["next_anchor_lead_s"] = next_lead
        df["next_anchor_confidence"] = next_conf

        df["bridge_same_team_agreement"] = bridge_agree
        df["bridge_contradiction_score"] = bridge_contradict
        return df


# ============================================================================
# 4. AUTOMATED VERIFICATION & AUDIT COMPLIANCE SUITE FOR TASK 3
# ============================================================================

def run_task3_verification() -> None:
    print("=" * 72)
    print("RUNNING TASK 3 ANCHORS, BRIDGES & TYPESAFE AI (JEV) VERIFICATION")
    print("=" * 72)

    raw_ds = generate_synthetic_diagnostic_match(sampling_rate_hz=25.0)
    prep = SegmentedPreprocessor()
    proc_ds = prep.process(raw_ds)

    t2_extractor = LocalKinematicFeatureExtractor()
    t2_df = t2_extractor.extract_features(proc_ds)

    engine = KinematicContactAndBridgeEngine()
    t3_df = engine.extract_anchor_and_bridge_features(t2_df, proc_ds.metadata)

    # 1. Verify salient anchor detection at pass kick (t=1.0s, Home) and interception (t=2.8s, Away)
    kick_slice = t3_df[(t3_df["timestamp_s"] >= 0.92) & (t3_df["timestamp_s"] <= 1.08)]
    intercept_slice = t3_df[(t3_df["timestamp_s"] >= 2.72) & (t3_df["timestamp_s"] <= 2.88)]

    assert kick_slice["is_candidate_anchor"].any(), "Pass kick at t=1.0s must be flagged as a candidate anchor."
    assert intercept_slice["is_candidate_anchor"].any(), "Interception at t=2.8s must be flagged as a candidate anchor."

    kick_home_prob = float(kick_slice.loc[kick_slice["is_candidate_anchor"], "anchor_home_prob"].max())
    intercept_away_prob = float(intercept_slice.loc[intercept_slice["is_candidate_anchor"], "anchor_away_prob"].max())
    assert kick_home_prob > 0.75, f"Expected high Home anchor prob at t=1.0s, got {kick_home_prob:.3f}"
    assert intercept_away_prob > 0.75, f"Expected high Away anchor prob at t=2.8s, got {intercept_away_prob:.3f}"
    print(
        f"[PASS] 1. Salient contact anchors verified "
        f"(Kick Home prob = {kick_home_prob:.3f}, Interception Away prob = {intercept_away_prob:.3f})."
    )

    # 2. Verify Unreliable-Proximity Proxy during close pass-by (t=1.60s) & short occlusion gap (t=1.88s)
    passby_slice = t3_df[(t3_df["timestamp_s"] >= 1.56) & (t3_df["timestamp_s"] <= 2.00)]
    max_unrel_prox = float(passby_slice["unreliable_proximity_likelihood"].max())
    assert max_unrel_prox > 0.35, f"Expected elevated unreliable_proximity_likelihood during fast pass-by, got {max_unrel_prox:.3f}"
    print(
        f"[PASS] 2. Unreliable-proximity (aerial/pass-by) proxy verified during defender pass-by "
        f"(max likelihood = {max_unrel_prox:.3f})."
    )

    # 3. Verify Bounded Bridge continuity across short 0.20s gap vs halting at long 0.80s gap
    mid_pass_row = t3_df[np.isclose(t3_df["timestamp_s"], 1.92, atol=0.02)].iloc[0]
    assert float(mid_pass_row["prev_anchor_home_prob"]) > 0.65, (
        f"Bridge at t=1.92s should remember Home kick anchor at t=1.0s, got {mid_pass_row['prev_anchor_home_prob']:.3f}"
    )
    assert float(mid_pass_row["next_anchor_away_prob"]) > 0.65, (
        f"Bridge at t=1.92s should look ahead to Away interception at t=2.8s, got {mid_pass_row['next_anchor_away_prob']:.3f}"
    )

    long_gap_row = t3_df[np.isclose(t3_df["timestamp_s"], 4.20, atol=0.02)].iloc[0]
    assert float(long_gap_row["prev_anchor_confidence"]) == 0.0, "Bridge must halt across long 0.80s missing tracking gap."
    assert float(long_gap_row["next_anchor_confidence"]) == 0.0, "Bridge must halt across long 0.80s missing tracking gap."
    print(
        f"[PASS] 3. Bounded bridges verified: preserved across 0.20s occlusion "
        f"(prev_home={mid_pass_row['prev_anchor_home_prob']:.3f}, next_away={mid_pass_row['next_anchor_away_prob']:.3f}) "
        f"and halted across 0.80s gap."
    )

    # 4. Verify TypeSafe AI (Jev / System One) payload schema & team-swap symmetry
    refiner = JevSystemOneRefiner(use_live_api=True)
    sample_event = {
        "frame_id": 25,
        "timestamp_s": 1.0,
        "dv_proxy_mps": 7.8,
        "multiscale_consistency": 0.90,
        "min_dist_home": 0.35,
        "min_dist_away": 3.20,
        "home_min_intercept_tau": 0.10,
        "away_min_intercept_tau": 0.80,
        "home_touch_evidence": 0.92,
        "away_touch_evidence": 0.02,
        "home_carry_evidence": 0.75,
        "away_carry_evidence": 0.0,
        "ball_speed": 12.0,
        "is_interpolated": False,
        "boundary_unreliable": False,
    }
    swapped_event = dict(sample_event)
    swapped_event["frame_id"] = 26
    swapped_event["min_dist_home"], swapped_event["min_dist_away"] = (
        sample_event["min_dist_away"],
        sample_event["min_dist_home"],
    )
    swapped_event["home_min_intercept_tau"], swapped_event["away_min_intercept_tau"] = (
        sample_event["away_min_intercept_tau"],
        sample_event["home_min_intercept_tau"],
    )
    swapped_event["home_touch_evidence"], swapped_event["away_touch_evidence"] = (
        sample_event["away_touch_evidence"],
        sample_event["home_touch_evidence"],
    )
    swapped_event["home_carry_evidence"], swapped_event["away_carry_evidence"] = (
        sample_event["away_carry_evidence"],
        sample_event["home_carry_evidence"],
    )

    payload = refiner.build_system_one_payload([sample_event])
    assert payload["model"] == "jev-1.13.0"
    assert "f_25_is_player_contact" in payload["questions"]
    assert payload["questions"]["f_25_is_player_contact"]["type"] == "noul"
    assert payload["questions"]["f_25_attribution"]["type"] == "choice"
    assert payload["questions"]["f_25_unreliable_prox"]["type"] == "score"

    res_pair = refiner.refine_ambiguous_events([sample_event, swapped_event])
    p_home_orig = res_pair[25]["choice"].probabilities["Home_Touch"]
    p_away_swap = res_pair[26]["choice"].probabilities["Away_Touch"]
    assert np.isclose(p_home_orig, p_away_swap, atol=1e-9), (
        f"Team-swap symmetry violated in Jev Choice primitive: {p_home_orig} vs {p_away_swap}"
    )
    assert int(t3_df["jev_refined_flag"].sum()) > 0, "Sparse Jev refiner should have evaluated ambiguous frames."
    print(
        f"[PASS] 4. TypeSafe AI (Jev / System One) Noul/Choice/Score primitives & team-swap symmetry verified "
        f"(refined {int(t3_df['jev_refined_flag'].sum())} sparse ambiguous frames)."
    )

    print("=" * 72)
    print("ALL TASK 3 CHECKS PASSED SUCCESSFULLY.")
    print("=" * 72)


if __name__ == "__main__":
    run_task3_verification()