# Unified CLI Submission & Reproducibility Pipeline (RFC v2.0 Section 5, 6 & 8).
# Connects Tasks 1-5 into an end-to-end train, OOF-calibrate, cache, and predict workflow.
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import (
    CoordinateSemantics,
    MatchMetadata,
    PlayerMeta,
    PlayerRole,
    PreprocessConfig,
    SegmentedPreprocessor,
    TeamID,
    TrackingSchemaValidator,
    ValidatedTrackingDataset,
)
from task2_feature_engine import FeatureConfig, LocalKinematicFeatureExtractor
from task3_anchors_and_bridges import (
    AnchorBridgeConfig,
    JevSystemOneRefiner,
    KinematicContactAndBridgeEngine,
)
from task4_classifier_and_decoder import (
    DecoderConfig,
    DependencyPurgeCalculator,
    SoftViterbiDecoder,
    SymmetricFeatureBuilder,
    TemporalWindowConfig,
)
from task5_evaluation_and_ablation import (
    EvaluationMetrics,
    PossessionDiagnosticEvaluator,
    SubsetSymmetricGBDT,
    TacticalGeometryExtractor,
    generate_challenging_ablation_benchmark,
)


# ============================================================================
# 1. PIPELINE VERSIONING & DETERMINISTIC FEATURE CACHE (RFC v2.0 Section 5)
# ============================================================================

SCHEMA_VERSION = "RFC_v2.0.0_MVP"


@dataclass
class FullPipelineConfig:
    schema_version: str = SCHEMA_VERSION
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    local_features: FeatureConfig = field(default_factory=FeatureConfig)
    anchors_bridges: AnchorBridgeConfig = field(default_factory=AnchorBridgeConfig)
    temporal_windows: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    gbdt_n_estimators: int = 60
    gbdt_learning_rate: float = 0.08
    use_live_jev_api: bool = False

    def compute_config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class FeatureCacheManager:
    # Caches extracted match feature DataFrames tagged with schema and config hashes
    # so iterative decoder or model experiments never recompute unchanged features.

    def __init__(self, cache_dir: str = ".feature_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, match_id: str, config_hash: str) -> str:
        safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in match_id)
        filename = f"{safe_id}_{SCHEMA_VERSION}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def load_or_compute(
        self,
        dataset: ValidatedTrackingDataset,
        config: FullPipelineConfig,
        builder_fn: Any,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        cfg_hash = config.compute_config_hash()
        cache_path = self._get_cache_path(dataset.metadata.match_id, cfg_hash)

        if not force_recompute and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f_in:
                    cached_obj = pickle.load(f_in)
                if (
                    isinstance(cached_obj, dict)
                    and cached_obj.get("schema_version") == SCHEMA_VERSION
                    and cached_obj.get("config_hash") == cfg_hash
                    and len(cached_obj.get("feature_df", [])) == len(dataset.frame_df)
                ):
                    return cached_obj["feature_df"]
            except Exception:
                # Invalidate corrupt or truncated cache file and fall back to recomputing
                pass

        feature_df = builder_fn(dataset, config)
        with open(cache_path, "wb") as f_out:
            pickle.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_hash": cfg_hash,
                    "match_id": dataset.metadata.match_id,
                    "feature_df": feature_df,
                },
                f_out,
            )
        return feature_df


# ============================================================================
# 2. METRICA SPORTS CSV / PARQUET INGESTION ADAPTER (RFC v2.0 Section 4.1)
# ============================================================================

class MetricaFormatAdapter:
    # Ingests either canonical (frame_df + player_tracks_df) files or wide-format
    # Metrica Sports tracking tables and normalizes them via TrackingSchemaValidator.

    @staticmethod
    def load_table(path: str) -> pd.DataFrame:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)

    @staticmethod
    def _is_raw_metrica_csv(path: str) -> bool:
        if not path or not os.path.exists(path) or not path.lower().endswith(".csv"):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                return ("Home" in first_line or "Away" in first_line) and first_line.startswith(",")
        except Exception:
            return False

    @staticmethod
    def parse_metrica_tracking_csv(
        file_path: str, team_prefix: str
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
        with open(file_path, "r", encoding="utf-8") as f:
            header_row0 = f.readline().strip().split(",")
            header_row1 = f.readline().strip().split(",")
            header_row2 = f.readline().strip().split(",")

        col_names = ["period_id", "frame_id", "timestamp_s"]
        player_col_map: Dict[str, Tuple[str, str]] = {}

        idx = 3
        num_cols = len(header_row2)
        while idx < num_cols:
            h1 = header_row1[idx].strip() if idx < len(header_row1) else ""
            h2 = header_row2[idx].strip() if idx < len(header_row2) else ""

            if "ball" in h2.lower() or "ball" in h1.lower():
                col_x = "ball_x"
                col_y = "ball_y"
                col_names.extend([col_x, col_y])
                idx += 2
            elif h2 or h1:
                player_label = h2 if h2 else f"Player{h1}"
                pid = f"{team_prefix}_{player_label}"
                col_x = f"{pid}_x"
                col_y = f"{pid}_y"
                col_names.extend([col_x, col_y])
                player_col_map[pid] = (col_x, col_y)
                idx += 2
            else:
                col_names.append(f"extra_{idx}")
                idx += 1

        df = pd.read_csv(file_path, skiprows=3, header=None, low_memory=False)
        if len(df.columns) > len(col_names):
            col_names.extend([f"col_{i}" for i in range(len(col_names), len(df.columns))])
        df.columns = col_names[: len(df.columns)]

        df["period_id"] = pd.to_numeric(df["period_id"], errors="coerce").fillna(1).astype(int)
        df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").fillna(0).astype(int)
        df["timestamp_s"] = pd.to_numeric(df["timestamp_s"], errors="coerce").fillna(0.0).astype(float)
        return df, player_col_map

    @classmethod
    def from_metrica_raw_files(
        cls,
        file1_path: str,
        file2_path: str,
        metadata_json_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> ValidatedTrackingDataset:
        with open(file1_path, "r", encoding="utf-8") as f:
            line_0 = f.readline()
        if "Away" in line_0 and "Home" not in line_0:
            home_path, away_path = file2_path, file1_path
        else:
            home_path, away_path = file1_path, file2_path

        home_df, home_players = cls.parse_metrica_tracking_csv(home_path, "H")
        away_df, away_players = cls.parse_metrica_tracking_csv(away_path, "A")

        if max_frames and len(home_df) > max_frames:
            home_df = home_df.iloc[:max_frames].copy()
            away_df = away_df.iloc[:max_frames].copy()

        frame_ids = home_df["frame_id"].to_numpy(dtype=int)
        period_ids = home_df["period_id"].to_numpy(dtype=int)
        timestamps = home_df["timestamp_s"].to_numpy(dtype=float)

        if "ball_x" in home_df.columns and "ball_y" in home_df.columns:
            ball_x_raw = pd.to_numeric(home_df["ball_x"], errors="coerce").to_numpy(dtype=float)
            ball_y_raw = pd.to_numeric(home_df["ball_y"], errors="coerce").to_numpy(dtype=float)
        else:
            ball_x_raw = pd.to_numeric(away_df["ball_x"], errors="coerce").to_numpy(dtype=float)
            ball_y_raw = pd.to_numeric(away_df["ball_y"], errors="coerce").to_numpy(dtype=float)

        frame_df = pd.DataFrame({
            "frame_id": frame_ids,
            "timestamp_s": timestamps,
            "period_id": period_ids,
            "ball_x": ball_x_raw,
            "ball_y": ball_y_raw,
            "is_dead_ball": False,
            "is_contested": False,
            "eval_mask": True,
            "camera_cut": False,
        })

        players: Dict[str, PlayerMeta] = {}
        player_frames_list: List[pd.DataFrame] = []

        for pid, (xcol, ycol) in home_players.items():
            digits = "".join(c for c in pid if c.isdigit())
            j_num = int(digits) if digits else 99
            role = PlayerRole.GK if j_num == 11 else PlayerRole.OUTFIELD
            players[pid] = PlayerMeta(pid, TeamID.HOME, role, j_num)

            xs = pd.to_numeric(home_df[xcol], errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(home_df[ycol], errors="coerce").to_numpy(dtype=float)
            player_frames_list.append(pd.DataFrame({
                "frame_id": frame_ids,
                "player_id": pid,
                "x": xs,
                "y": ys,
                "visibility_conf": np.where(np.isfinite(xs), 1.0, 0.0),
            }))

        for pid, (xcol, ycol) in away_players.items():
            digits = "".join(c for c in pid if c.isdigit())
            j_num = int(digits) if digits else 99
            role = PlayerRole.GK if j_num == 25 else PlayerRole.OUTFIELD
            players[pid] = PlayerMeta(pid, TeamID.AWAY, role, j_num)

            xs = pd.to_numeric(away_df[xcol], errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(away_df[ycol], errors="coerce").to_numpy(dtype=float)
            player_frames_list.append(pd.DataFrame({
                "frame_id": frame_ids,
                "player_id": pid,
                "x": xs,
                "y": ys,
                "visibility_conf": np.where(np.isfinite(xs), 1.0, 0.0),
            }))

        player_tracks_df = pd.concat(player_frames_list, ignore_index=True)

        if metadata_json_path and os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                m_dict = json.load(f)
            metadata = MatchMetadata(
                match_id=str(m_dict.get("match_id", "METRICA_SAMPLE_GAME")),
                sampling_rate_hz=float(m_dict.get("sampling_rate_hz", 25.0)),
                pitch_length_m=float(m_dict.get("pitch_length_m", 105.0)),
                pitch_width_m=float(m_dict.get("pitch_width_m", 68.0)),
                coord_semantics=CoordinateSemantics(
                    m_dict.get("coord_semantics", CoordinateSemantics.NORMALIZED_01.value)
                ),
                home_attacks_right_period1=bool(m_dict.get("home_attacks_right_period1", True)),
            )
        else:
            metadata = MatchMetadata(
                match_id="METRICA_SAMPLE_GAME",
                sampling_rate_hz=25.0,
                pitch_length_m=105.0,
                pitch_width_m=68.0,
                coord_semantics=CoordinateSemantics.NORMALIZED_01,
                home_attacks_right_period1=True,
            )

        return TrackingSchemaValidator.validate_and_normalize(
            metadata=metadata,
            players=players,
            frame_df=frame_df,
            player_tracks_df=player_tracks_df,
        )

    @classmethod
    def from_canonical_files(
        cls,
        frame_path: str,
        players_path: str,
        metadata_json_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> ValidatedTrackingDataset:
        if cls._is_raw_metrica_csv(frame_path) or cls._is_raw_metrica_csv(players_path):
            print(f"[INFO] Auto-detected raw Metrica Sports split Home/Away tracking tables. Ingesting multi-header format...")
            return cls.from_metrica_raw_files(frame_path, players_path, metadata_json_path, max_frames=max_frames)

        f_df = cls.load_table(frame_path)
        p_df = cls.load_table(players_path)
        if max_frames and len(f_df) > max_frames:
            fids = set(f_df["frame_id"].iloc[:max_frames])
            f_df = f_df.iloc[:max_frames].copy()
            p_df = p_df[p_df["frame_id"].isin(fids)].copy()
        return cls._build_validated_dataset(f_df, p_df, metadata_json_path)

    @classmethod
    def normalize_frame_df(cls, df: pd.DataFrame, default_fps: float = 25.0) -> pd.DataFrame:
        out = df.copy()
        col_map = {
            "period": "period_id", "Period": "period_id", "half": "period_id", "Half": "period_id",
            "frame": "frame_id", "Frame": "frame_id", "step": "frame_id", "Step": "frame_id",
            "Time [s]": "timestamp_s", "time [s]": "timestamp_s", "time_s": "timestamp_s",
            "Time": "timestamp_s", "time": "timestamp_s", "timestamp": "timestamp_s",
            "Ball_X": "ball_x", "Ball_x": "ball_x", "ballx": "ball_x",
            "Ball_Y": "ball_y", "Ball_y": "ball_y", "bally": "ball_y",
        }
        for old_c, new_c in col_map.items():
            if old_c in out.columns and new_c not in out.columns:
                out[new_c] = out[old_c]

        if "period_id" not in out.columns:
            out["period_id"] = 1
        if "frame_id" not in out.columns:
            out["frame_id"] = np.arange(1, len(out) + 1, dtype=int)
        if "timestamp_s" not in out.columns:
            fids = out["frame_id"].to_numpy(dtype=float)
            out["timestamp_s"] = (fids - fids[0]) / max(1.0, default_fps)

        # Inject default competition masks if not present in raw tracking CSVs
        if "eval_mask" not in out.columns:
            out["eval_mask"] = True
        else:
            out["eval_mask"] = out["eval_mask"].fillna(True).astype(bool)

        if "is_dead_ball" not in out.columns:
            out["is_dead_ball"] = False
        else:
            out["is_dead_ball"] = out["is_dead_ball"].fillna(False).astype(bool)

        if "is_contested" not in out.columns:
            out["is_contested"] = False
        else:
            out["is_contested"] = out["is_contested"].fillna(False).astype(bool)

        if "camera_cut" not in out.columns:
            out["camera_cut"] = False
        else:
            out["camera_cut"] = out["camera_cut"].fillna(False).astype(bool)

        return out

    @classmethod
    def normalize_player_tracks_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col_map = {
            "frame": "frame_id", "Frame": "frame_id", "step": "frame_id", "Step": "frame_id",
            "player": "player_id", "Player": "player_id", "track_id": "player_id",
            "X": "x", "player_x": "x", "x_m": "x",
            "Y": "y", "player_y": "y", "y_m": "y",
            "conf": "visibility_conf", "confidence": "visibility_conf",
        }
        for old_c, new_c in col_map.items():
            if old_c in out.columns and new_c not in out.columns:
                out[new_c] = out[old_c]

        if "visibility_conf" not in out.columns:
            out["visibility_conf"] = 1.0
        else:
            out["visibility_conf"] = out["visibility_conf"].fillna(1.0).astype(float)
        return out

    @classmethod
    def _build_validated_dataset(
        cls,
        f_df: pd.DataFrame,
        p_df: pd.DataFrame,
        metadata_json_path: Optional[str] = None,
        default_fps: float = 25.0,
    ) -> ValidatedTrackingDataset:
        f_norm = cls.normalize_frame_df(f_df, default_fps=default_fps)
        p_norm = cls.normalize_player_tracks_df(p_df)

        if metadata_json_path and os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                m_dict = json.load(f)
            meta = MatchMetadata(
                match_id=str(m_dict.get("match_id", "METRICA_MATCH")),
                sampling_rate_hz=float(m_dict.get("sampling_rate_hz", default_fps)),
                pitch_length_m=float(m_dict.get("pitch_length_m", 105.0)),
                pitch_width_m=float(m_dict.get("pitch_width_m", 68.0)),
                coord_semantics=CoordinateSemantics(
                    m_dict.get("coord_semantics", CoordinateSemantics.PITCH_METERS.value)
                ),
                home_attacks_right_period1=bool(m_dict.get("home_attacks_right_period1", True)),
            )
        else:
            ts = f_norm["timestamp_s"].to_numpy(dtype=float)
            dt = float(np.median(np.diff(ts))) if len(ts) > 1 else (1.0 / default_fps)
            fps = float(np.round(1.0 / max(1e-4, dt)))
            meta = MatchMetadata(
                match_id="METRICA_CANONICAL_MATCH",
                sampling_rate_hz=fps,
                pitch_length_m=105.0,
                pitch_width_m=68.0,
                coord_semantics=CoordinateSemantics.PITCH_METERS,
            )

        # Build player registry ensuring both Home and Away have explicit GKs
        players: Dict[str, PlayerMeta] = {}
        unique_pids = p_norm["player_id"].astype(str).unique()
        has_team_col = "team_id" in p_norm.columns or "team" in p_norm.columns
        team_col_name = "team_id" if "team_id" in p_norm.columns else ("team" if "team" in p_norm.columns else None)
        has_role_col = "role" in p_norm.columns

        home_players: List[str] = []
        away_players: List[str] = []

        for pid in unique_pids:
            if team_col_name:
                t_val = str(p_norm.loc[p_norm["player_id"] == pid, team_col_name].iloc[0]).upper()
                is_home = "HOME" in t_val or t_val.startswith("H")
            else:
                is_home = pid.upper().startswith("H") or "HOME" in pid.upper()
            if is_home:
                home_players.append(pid)
            else:
                away_players.append(pid)

        # Fallback if team prefixing was not used (e.g. Player1..Player22)
        if len(home_players) == 0 or len(away_players) == 0:
            half = max(1, len(unique_pids) // 2)
            home_players = list(unique_pids[:half])
            away_players = list(unique_pids[half:])

        def assign_team_players(pids: List[str], team_id: TeamID) -> None:
            gk_assigned = False
            for pid in pids:
                digits = "".join(ch for ch in pid if ch.isdigit())
                j_num = int(digits) if digits else 99
                if has_role_col:
                    r_val = str(p_norm.loc[p_norm["player_id"] == pid, "role"].iloc[0]).upper()
                    is_gk = "GK" in r_val or "GOALKEEPER" in r_val
                else:
                    is_gk = "GK" in pid.upper() or j_num == 1
                if is_gk and not gk_assigned:
                    role = PlayerRole.GK
                    gk_assigned = True
                else:
                    role = PlayerRole.OUTFIELD
                players[pid] = PlayerMeta(pid, team_id, role, j_num)

            # Guarantee at least one GK for each team to fulfill schema validation
            if not gk_assigned and pids:
                first_pid = pids[0]
                orig = players[first_pid]
                players[first_pid] = PlayerMeta(orig.player_id, orig.team_id, PlayerRole.GK, orig.jersey_number)

        assign_team_players(home_players, TeamID.HOME)
        assign_team_players(away_players, TeamID.AWAY)

        return TrackingSchemaValidator.validate_and_normalize(
            metadata=meta,
            players=players,
            frame_df=f_norm,
            player_tracks_df=p_norm,
        )

    @classmethod
    def from_wide_table(
        cls,
        wide_df: pd.DataFrame,
        metadata_json_path: Optional[str] = None,
        default_fps: float = 25.0,
    ) -> ValidatedTrackingDataset:
        norm_wide = cls.normalize_frame_df(wide_df, default_fps=default_fps)
        player_cols: Dict[str, Tuple[str, str]] = {}
        for col in norm_wide.columns:
            if col.endswith("_x") and not col.lower().startswith("ball"):
                base_pid = col[:-2]
                y_col = base_pid + "_y"
                if y_col in norm_wide.columns:
                    player_cols[base_pid] = (col, y_col)

        player_rows: List[Dict[str, Any]] = []
        fids = norm_wide["frame_id"].to_numpy(dtype=int)

        for pid, (x_c, y_c) in player_cols.items():
            px_arr = norm_wide[x_c].to_numpy(dtype=float)
            py_arr = norm_wide[y_c].to_numpy(dtype=float)
            for i in range(len(fids)):
                player_rows.append({
                    "frame_id": fids[i],
                    "player_id": pid,
                    "x": px_arr[i],
                    "y": py_arr[i],
                    "visibility_conf": 1.0,
                })

        p_df = pd.DataFrame(player_rows)
        drop_cols = [c for pair in player_cols.values() for c in pair]
        f_df = norm_wide.drop(columns=[c for c in drop_cols if c in norm_wide.columns])
        return cls._build_validated_dataset(f_df, p_df, metadata_json_path, default_fps)


# ============================================================================
# 3. COMPLETE STAGE-6 FEATURE PIPELINE & ARTIFACT BUNDLE
# ============================================================================

def get_stage6_feature_columns() -> Tuple[List[Tuple[str, str]], List[str]]:
    paired_cols = SymmetricFeatureBuilder.PAIRED_COLUMNS + TacticalGeometryExtractor.PAIRED_TACTICAL_COLUMNS
    invariant_cols = list(SymmetricFeatureBuilder.INVARIANT_COLUMNS)
    return paired_cols, invariant_cols


def extract_all_match_features(
    raw_ds: ValidatedTrackingDataset,
    config: FullPipelineConfig,
) -> pd.DataFrame:
    prep = SegmentedPreprocessor(config=config.preprocess)
    proc_ds = prep.process(raw_ds)

    t2_ext = LocalKinematicFeatureExtractor(config=config.local_features)
    df2 = t2_ext.extract_features(proc_ds)

    jev_ref = JevSystemOneRefiner(use_live_api=config.use_live_jev_api)
    t3_eng = KinematicContactAndBridgeEngine(
        config=config.anchors_bridges,
        jev_refiner=jev_ref,
    )
    df3 = t3_eng.extract_anchor_and_bridge_features(df2, proc_ds.metadata)

    sym_builder = SymmetricFeatureBuilder(win_config=config.temporal_windows)
    df4 = sym_builder.add_segmented_temporal_windows(df3, proc_ds.metadata)

    df5 = TacticalGeometryExtractor.extract_tactical_features(proc_ds, df4)
    df5["match_id"] = proc_ds.metadata.match_id
    return df5


class PossessionSubmissionPipeline:
    # End-to-end trainable and serializable pipeline combining:
    #   - Stage 6 Feature Extractor + Cache
    #   - Team-Swap Symmetric GBDT Classifier (OOF Platt-calibrated)
    #   - OOF-Tuned Soft Discriminative Viterbi Decoder
    #   - Post-Decoding Posterior Calibrator & Binary Submission Formatter

    def __init__(
        self,
        config: Optional[FullPipelineConfig] = None,
        cache_dir: str = ".feature_cache",
    ):
        self.config = config or FullPipelineConfig()
        self.cache_mgr = FeatureCacheManager(cache_dir=cache_dir)
        paired_cols, inv_cols = get_stage6_feature_columns()
        self.emission_model = SubsetSymmetricGBDT(
            paired_cols=paired_cols,
            invariant_cols=inv_cols,
            n_estimators=self.config.gbdt_n_estimators,
            learning_rate=self.config.gbdt_learning_rate,
        )
        self.decoder = SoftViterbiDecoder(config=self.config.decoder)
        self.post_viterbi_platt_a: float = 1.0
        self.purge_buffer_s: float = DependencyPurgeCalculator(
            prep_cfg=self.config.preprocess,
            feat_cfg=self.config.local_features,
            bridge_cfg=self.config.anchors_bridges,
            win_cfg=self.config.temporal_windows,
        ).compute_purge_buffer_seconds()

    def fit_with_oof_validation(
        self,
        raw_matches: List[ValidatedTrackingDataset],
        force_recompute_features: bool = False,
    ) -> Tuple[EvaluationMetrics, Dict[str, float], List[pd.DataFrame]]:
        if len(raw_matches) < 2:
            raise ValueError("At least 2 independent matches are required for Leave-One-Match-Out OOF.")

        match_dfs: List[pd.DataFrame] = []
        for ds in raw_matches:
            feat_df = self.cache_mgr.load_or_compute(
                dataset=ds,
                config=self.config,
                builder_fn=extract_all_match_features,
                force_recompute=force_recompute_features,
            )
            match_dfs.append(feat_df)

        paired_cols, inv_cols = get_stage6_feature_columns()
        oof_probs_list: List[np.ndarray] = []
        all_oof_p: List[np.ndarray] = []
        all_oof_y: List[np.ndarray] = []
        all_oof_m: List[np.ndarray] = []

        # 1. Leave-One-Match-Out cross-validation for emission probabilities
        for i in range(len(match_dfs)):
            train_df = pd.concat(
                [match_dfs[j] for j in range(len(match_dfs)) if j != i],
                ignore_index=True,
            )
            val_df = match_dfs[i]

            fold_gbdt = SubsetSymmetricGBDT(
                paired_cols=paired_cols,
                invariant_cols=inv_cols,
                n_estimators=self.config.gbdt_n_estimators,
                learning_rate=self.config.gbdt_learning_rate,
            )
            fold_gbdt.fit(train_df)
            p_val = fold_gbdt.predict_raw_home_prob(val_df)

            oof_probs_list.append(p_val)
            all_oof_p.append(p_val)
            all_oof_y.append(
                (val_df["gt_possession_team"].astype(str).to_numpy() == TeamID.HOME.value).astype(int)
            )
            all_oof_m.append(
                val_df["eval_mask"].to_numpy(dtype=bool) & val_df["ball_is_valid"].to_numpy(dtype=bool)
            )

        # 2. Fit final emission model on full dataset and calibrate emission scale on OOF
        self.emission_model.fit(pd.concat(match_dfs, ignore_index=True))
        self.emission_model.fit_oof_calibration(
            np.concatenate(all_oof_p),
            np.concatenate(all_oof_y),
            np.concatenate(all_oof_m),
        )

        # 3. Tune Soft Viterbi switch penalty strictly on OOF emission predictions
        self.decoder.tune_on_oof(match_dfs, oof_probs_list)

        # 4. Decode OOF matches and calibrate post-Viterbi sequence margin probabilities (RFC Sec 4.7)
        decoded_oof_dfs: List[pd.DataFrame] = []
        post_logits_all: List[np.ndarray] = []

        for df_m, p_m in zip(match_dfs, oof_probs_list):
            dec_df = self.decoder.decode(df_m, p_m)
            decoded_oof_dfs.append(dec_df)
            p_post = np.clip(dec_df["decoded_home_prob"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            post_logits_all.append(np.log(p_post / (1.0 - p_post)))

        self.post_viterbi_platt_a = self._fit_post_decoder_calibration(
            np.concatenate(post_logits_all),
            np.concatenate(all_oof_y),
            np.concatenate(all_oof_m),
        )

        # Apply post-decoder calibration and format final submission columns on OOF
        formatted_oof_dfs = [self._format_submission_dataframe(d) for d in decoded_oof_dfs]

        evaluator = PossessionDiagnosticEvaluator()
        fps = raw_matches[0].metadata.sampling_rate_hz
        oof_metrics = evaluator.evaluate_oof_matches(
            "RFC_v2.0_Full_Submission_Pipeline",
            formatted_oof_dfs,
            prob_col="calibrated_posterior_home_prob",
            pred_col="submission_possession_team",
            sampling_rate_hz=fps,
        )
        slice_report = evaluator.evaluate_diagnostic_slices(
            formatted_oof_dfs,
            pred_col="submission_possession_team",
        )
        return oof_metrics, slice_report, formatted_oof_dfs

    @staticmethod
    def _fit_post_decoder_calibration(
        logits: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray
    ) -> float:
        l_val = logits[valid_mask]
        y_val = y_true[valid_mask]
        if len(l_val) == 0:
            return 1.0

        best_scale = 1.0
        best_brier = 1e9
        for scale in np.linspace(0.25, 2.50, 46):
            p = 1.0 / (1.0 + np.exp(-np.clip(scale * l_val, -20.0, 20.0)))
            brier = float(np.mean((p - y_val) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_scale = float(scale)
        return best_scale

    def _format_submission_dataframe(self, decoded_df: pd.DataFrame) -> pd.DataFrame:
        # Separates internal latent state semantics from the required binary submission output
        # so missing-ball or dead-ball intervals are never silently conflated (RFC Sec 3 P0).
        out = decoded_df.reset_index(drop=True).copy()
        p_raw_post = np.clip(out["decoded_home_prob"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        logit = np.log(p_raw_post / (1.0 - p_raw_post))
        p_cal = 1.0 / (1.0 + np.exp(-np.clip(self.post_viterbi_platt_a * logit, -20.0, 20.0)))
        p_cal = np.clip(p_cal, 0.0, 1.0)

        valid_ball = (
            out["ball_is_valid"].fillna(False).to_numpy(dtype=bool)
            if "ball_is_valid" in out.columns
            else np.ones(len(out), dtype=bool)
        )
        is_dead = (
            out["is_dead_ball"].fillna(False).to_numpy(dtype=bool)
            if "is_dead_ball" in out.columns
            else np.zeros(len(out), dtype=bool)
        )
        is_contest = (
            out["is_contested"].fillna(False).to_numpy(dtype=bool)
            if "is_contested" in out.columns
            else np.zeros(len(out), dtype=bool)
        )

        # Propagate last valid posterior probability across uninterpolated gaps within each period
        # for binary submission completeness while keeping latent state explicit.
        p_filled = np.where(valid_ball, p_cal, np.nan)
        for _, grp in out.groupby("period_id", sort=False):
            idx = grp.index.to_numpy(dtype=int)
            s = pd.Series(p_filled[idx]).ffill().bfill().fillna(0.5)
            p_filled[idx] = s.to_numpy(dtype=float)

        binary_team = np.where(p_filled >= 0.5, TeamID.HOME.value, TeamID.AWAY.value)

        # Vectorized string addition to avoid stringifying the entire NumPy array
        cont_str = np.char.add("Contested_", binary_team.astype(str))
        latent_state = np.where(
            is_dead,
            "Dead_Ball",
            np.where(
                ~valid_ball,
                "Unknown_Missing_Ball",
                np.where(is_contest, cont_str, binary_team),
            ),
        )

        out["calibrated_posterior_home_prob"] = p_filled
        out["calibrated_posterior_away_prob"] = 1.0 - p_filled
        out["submission_possession_team"] = binary_team
        out["latent_possession_state"] = latent_state
        if "diagnostic_confidence" not in out.columns:
            out["diagnostic_confidence"] = np.abs(p_filled - 0.5) * 2.0
        return out

    def predict_match(
        self,
        raw_ds: ValidatedTrackingDataset,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        t0 = time.perf_counter()
        if use_cache:
            feat_df = self.cache_mgr.load_or_compute(
                dataset=raw_ds,
                config=self.config,
                builder_fn=extract_all_match_features,
                force_recompute=False,
            )
        else:
            feat_df = extract_all_match_features(raw_ds, self.config)

        p_emit = self.emission_model.predict_raw_home_prob(feat_df)
        dec_df = self.decoder.decode(feat_df, p_emit)
        final_df = self._format_submission_dataframe(dec_df)
        elapsed_s = max(1e-6, time.perf_counter() - t0)

        # Pre-submission audit & team-swap verification gate
        swapped_df = SymmetricFeatureBuilder.swap_dataframe_teams(feat_df)
        for h_c, a_c in TacticalGeometryExtractor.PAIRED_TACTICAL_COLUMNS:
            swapped_df[h_c] = feat_df[a_c].copy()
            swapped_df[a_c] = feat_df[h_c].copy()

        p_emit_swap = self.emission_model.predict_raw_home_prob(swapped_df)
        max_sym_error = float(np.max(np.abs(p_emit - (1.0 - p_emit_swap))))

        diagnostics = {
            "match_id": raw_ds.metadata.match_id,
            "num_frames": len(final_df),
            "runtime_s": round(elapsed_s, 4),
            "throughput_fps": round(len(final_df) / elapsed_s, 1),
            "max_team_swap_asymmetry": max_sym_error,
            "missing_ball_frames": int((~final_df["ball_is_valid"]).sum()),
            "jev_refined_frames": int(final_df["jev_refined_flag"].sum()) if "jev_refined_flag" in final_df.columns else 0,
        }
        return final_df, diagnostics

    @staticmethod
    def export_submission_csv(df: pd.DataFrame, output_csv_path: str) -> pd.DataFrame:
        submission_cols = [
            "match_id",
            "period_id",
            "frame_id",
            "timestamp_s",
            "eval_mask",
            "submission_possession_team",
            "calibrated_posterior_home_prob",
            "calibrated_posterior_away_prob",
            "latent_possession_state",
            "diagnostic_confidence",
        ]
        sub_df = pd.DataFrame()
        for col in submission_cols:
            if col in df.columns:
                sub_df[col] = df[col].copy()
            elif col == "match_id":
                sub_df[col] = "MATCH"
            elif col == "diagnostic_confidence":
                sub_df[col] = 1.0
            else:
                sub_df[col] = np.nan

        # Strict schema integrity checks (RFC v2.0 Section 8 Week 4 Gate)
        assert not sub_df["submission_possession_team"].isna().any(), "Submission contains NaN team labels!"
        assert set(sub_df["submission_possession_team"].unique()).issubset(
            {TeamID.HOME.value, TeamID.AWAY.value}
        ), "Invalid class names in submission_possession_team!"
        probs = np.clip(sub_df["calibrated_posterior_home_prob"].to_numpy(dtype=float), 0.0, 1.0)
        sub_df["calibrated_posterior_home_prob"] = probs
        sub_df["calibrated_posterior_away_prob"] = 1.0 - probs
        assert np.all(np.isfinite(probs)) and np.all((probs >= 0.0) & (probs <= 1.0)), (
            "Posterior probabilities out of bounds [0, 1]!"
        )

        out_dir = os.path.dirname(os.path.abspath(output_csv_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sub_df.to_csv(output_csv_path, index=False)
        return sub_df

    def save_model_bundle(self, bundle_path: str) -> None:
        out_dir = os.path.dirname(os.path.abspath(bundle_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "emission_model": self.emission_model,
            "decoder": self.decoder,
            "post_viterbi_platt_a": self.post_viterbi_platt_a,
            "purge_buffer_s": self.purge_buffer_s,
        }
        with open(bundle_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_model_bundle(cls, bundle_path: str, cache_dir: str = ".feature_cache") -> "PossessionSubmissionPipeline":
        class _MainRedirectUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "__main__":
                    module = "run_submission_pipeline"
                return super().find_class(module, name)

        with open(bundle_path, "rb") as f:
            payload = _MainRedirectUnpickler(f).load()
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Model bundle schema ({payload.get('schema_version')}) != {SCHEMA_VERSION}"
            )
        pipe = cls(config=payload["config"], cache_dir=cache_dir)
        pipe.emission_model = payload["emission_model"]
        pipe.decoder = payload["decoder"]
        pipe.post_viterbi_platt_a = float(payload["post_viterbi_platt_a"])
        pipe.purge_buffer_s = float(payload["purge_buffer_s"])
        return pipe


# ============================================================================
# 4. END-TO-END VERIFICATION & CLI ENTRYPOINT
# ============================================================================

def run_end_to_end_submission_verification() -> None:
    print("=" * 88)
    print("RUNNING END-TO-END SUBMISSION PIPELINE & REPRODUCIBILITY GATE (RFC v2.0)")
    print("=" * 88)

    # 1. Generate 4 benchmark training matches + 1 unseen held-out test match
    all_matches = generate_challenging_ablation_benchmark(n_matches=5, sampling_rate_hz=25.0)
    train_matches = all_matches[:4]
    unseen_test_match = all_matches[4]

    pipeline = PossessionSubmissionPipeline()
    oof_metrics, slice_report, _ = pipeline.fit_with_oof_validation(
        train_matches, force_recompute_features=False
    )
    print(
        f"[PASS] 1. OOF Training & Post-Decoder Calibration complete: "
        f"OOF Acc = {oof_metrics.frame_accuracy * 100:.1f}%, "
        f"Turnover F1 = {oof_metrics.turnover_f1:.3f}, "
        f"Brier = {oof_metrics.brier_score:.4f}, "
        f"Purge Buffer = {pipeline.purge_buffer_s:.2f}s."
    )

    # 2. Verify model bundle serialization & reload
    bundle_path = os.path.join("artifacts", "barca_possession_mvp_v2.pkl")
    pipeline.save_model_bundle(bundle_path)
    loaded_pipeline = PossessionSubmissionPipeline.load_model_bundle(bundle_path)
    print(f"[PASS] 2. Model artifact serialized & reloaded from '{bundle_path}'.")

    # 3. Run unseen test match inference, cache verification, and pre-submission checks
    pred_df, diag = loaded_pipeline.predict_match(unseen_test_match, use_cache=False)
    assert diag["max_team_swap_asymmetry"] < 1e-9, (
        f"Team-swap symmetry check failed on unseen match: {diag['max_team_swap_asymmetry']}"
    )

    # Measure unseen match accuracy on valid evaluated frames
    valid_mask = pred_df["eval_mask"].to_numpy(dtype=bool) & pred_df["ball_is_valid"].to_numpy(dtype=bool)
    unseen_acc = float(
        np.mean(
            pred_df.loc[valid_mask, "submission_possession_team"].to_numpy(dtype=str)
            == pred_df.loc[valid_mask, "gt_possession_team"].to_numpy(dtype=str)
        )
    )
    print(
        f"[PASS] 3. Unseen Test Match ({diag['match_id']}) inference verified: "
        f"Acc = {unseen_acc * 100:.1f}%, Throughput = {diag['throughput_fps']:.1f} fps, "
        f"Team-Swap Asymmetry = {diag['max_team_swap_asymmetry']:.2e}."
    )

    # 4. Export final submission CSV and verify contract schema
    sub_csv_path = os.path.join("artifacts", "submission_BENCH_MATCH_05.csv")
    sub_df = loaded_pipeline.export_submission_csv(pred_df, sub_csv_path)
    assert os.path.exists(sub_csv_path), "Submission CSV was not written to disk."
    assert len(sub_df) == diag["num_frames"], "Submission row count mismatch."
    assert "Unknown_Missing_Ball" in set(sub_df["latent_possession_state"].unique()), (
        "Latent state column must preserve Unknown_Missing_Ball during 0.80s uninterpolated gap."
    )
    print(
        f"[PASS] 4. Submission CSV exported & schema-verified at '{sub_csv_path}' "
        f"({len(sub_df)} rows, 0 NaNs, latent states preserved)."
    )
    print("=" * 88)
    print("FULL MORE THAN A HACK 2027 SUBMISSION PIPELINE READY FOR PRODUCTION.")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="More than a Hack 2027 — Frame-Level Possession Estimation CLI Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "predict", "train"],
        default="verify",
        help="Run self-contained end-to-end verification ('verify'), train model bundle ('train'), or predict on CSV/Parquet files ('predict').",
    )
    parser.add_argument("--frames", type=str, help="Path to input frame_df (or single wide tracking table) CSV/Parquet file.")
    parser.add_argument("--players", type=str, default=None, help="Optional path to input player_tracks_df CSV/Parquet file.")
    parser.add_argument("--meta", type=str, default=None, help="Optional path to match metadata JSON.")
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=os.path.join("artifacts", "barca_possession_mvp_v2.pkl"),
        help="Path to trained model .pkl bundle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("artifacts", "submission_output.csv"),
        help="Path to output submission CSV.",
    )
    parser.add_argument(
        "--live-jev",
        action="store_true",
        help="Enable live TypeSafe AI (Jev / System One) API calls using TYPESAFE_API_KEY.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to load for rapid testing.",
    )

    args = parser.parse_args()

    if args.mode == "verify":
        run_end_to_end_submission_verification()
    elif args.mode == "train":
        print("Training pipeline on 4 challenging benchmark matches and saving model bundle...")
        all_matches = generate_challenging_ablation_benchmark(n_matches=4, sampling_rate_hz=25.0)
        pipeline = PossessionSubmissionPipeline()
        pipeline.config.use_live_jev_api = bool(args.live_jev)
        metrics, slice_rep, _ = pipeline.fit_with_oof_validation(all_matches, force_recompute_features=True)
        pipeline.save_model_bundle(args.model_bundle)
        print(f"Model bundle trained (OOF Acc={metrics.frame_accuracy*100:.1f}%, F1={metrics.macro_f1:.3f}) and saved to '{args.model_bundle}'.")
    elif args.mode == "predict":
        if not args.frames:
            raise ValueError("--frames path is required in 'predict' mode.")
        if not os.path.exists(args.model_bundle):
            print("Model bundle not found; running verification first to initialize baseline artifact...")
            run_end_to_end_submission_verification()

        pipeline = PossessionSubmissionPipeline.load_model_bundle(args.model_bundle)
        pipeline.config.use_live_jev_api = bool(args.live_jev)

        if args.players and os.path.exists(args.players):
            dataset = MetricaFormatAdapter.from_canonical_files(
                args.frames, args.players, args.meta, max_frames=args.max_frames
            )
        else:
            # Single-table wide format support
            f_table = MetricaFormatAdapter.load_table(args.frames)
            if args.max_frames and len(f_table) > args.max_frames:
                f_table = f_table.iloc[:args.max_frames].copy()
            dataset = MetricaFormatAdapter.from_wide_table(f_table, args.meta)

        pred_df, diag = pipeline.predict_match(dataset, use_cache=True)
        pipeline.export_submission_csv(pred_df, args.output)
        print(f"Prediction complete: {diag} -> Saved to '{args.output}'")


if __name__ == "__main__":
    main()