# 3D Interactive Football Broadcast Replay Generator (Three.js / WebGL).
# Translates probabilistic possession, bounded bridges, and Jev System One decisions
# into an intuitive FC Barcelona themed 3D broadcast experience for any football fan.
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English (Code) / Mexican Spanish (Visual UI for Fans)

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import TeamID, ValidatedTrackingDataset
from task5_evaluation_and_ablation import generate_challenging_ablation_benchmark

try:
    from run_submission_pipeline import PossessionSubmissionPipeline
except ImportError:
    from task4_classifier_and_decoder import PossessionSubmissionPipeline


def load_match_for_3d_visualization(
    max_frames: int = 1500,
    frames_path: Optional[str] = None,
    players_path: Optional[str] = None,
    data_dir: str = "data",
    match_id: Optional[str] = None,
    model_bundle_path: Optional[str] = None,
) -> Tuple[ValidatedTrackingDataset, pd.DataFrame]:
    # Priority 0: Explicit command-line file paths (--frames, --players)
    # Priority 1: Official competition files in 'data/' (or specified --data-dir)
    # Priority 2: Real Metrica Sports Sample Game 1 in 'data_metrica/'
    # Priority 3: Synthetic diagnostic benchmark fallback
    bundle_path = model_bundle_path or os.path.join("artifacts", "barca_possession_mvp_v2.pkl")

    if not os.path.exists(bundle_path):
        print("[INFO] Model bundle not found. Initializing baseline artifact...")
        from run_submission_pipeline import run_end_to_end_submission_verification
        run_end_to_end_submission_verification()

    pipeline = PossessionSubmissionPipeline.load_model_bundle(bundle_path)

    # 0. Check explicit files
    if frames_path and os.path.exists(frames_path):
        from run_submission_pipeline import MetricaFormatAdapter
        print(f"[INFO] Priority 0: Loading explicit files '{frames_path}'...")
        if players_path and os.path.exists(players_path):
            dataset = MetricaFormatAdapter.from_canonical_files(
                frames_path, players_path, max_frames=max_frames
            )
        else:
            f_table = MetricaFormatAdapter.load_table(frames_path)
            if max_frames and len(f_table) > max_frames:
                f_table = f_table.iloc[:max_frames].copy()
            dataset = MetricaFormatAdapter.from_wide_table(f_table)

    # 1. Check data_dir for official competition files
    elif os.path.exists(data_dir):
        comp_matches = []
        try:
            from auto_process_competition import AutoMatchDetector, load_dataset_from_match_info
            comp_matches = AutoMatchDetector.scan_directory(data_dir)
        except ImportError:
            comp_matches = []

        if comp_matches:
            target_match = comp_matches[0]
            if match_id:
                for m in comp_matches:
                    if m["match_id"].lower() == match_id.lower():
                        target_match = m
                        break
            print(f"[INFO] Priority 1: Loading official competition match '{target_match['match_id']}' from '{data_dir}'...")
            full_ds = load_dataset_from_match_info(target_match)
            f_sub = full_ds.frame_df.iloc[:max_frames].copy()
            valid_fids = set(f_sub["frame_id"].to_numpy(dtype=int))
            p_sub = full_ds.player_tracks_df[full_ds.player_tracks_df["frame_id"].isin(valid_fids)].copy()
            dataset = ValidatedTrackingDataset(full_ds.metadata, full_ds.players, f_sub, p_sub)
        else:
            dataset = None

    else:
        dataset = None

    # 2. Check data_metrica
    metrica_home = os.path.join("data_metrica", "Sample_Game_1_RawTrackingData_Home_Team.csv")
    if dataset is None and os.path.exists(metrica_home):
        print(f"[INFO] Priority 2: Loading real Metrica Sports sample match from '{metrica_home}' ({max_frames} frames)...")
        from ingest_and_test_metrica_sample import load_metrica_sample_game_1
        dataset, _ = load_metrica_sample_game_1(max_frames=max_frames)

    # 3. Synthetic benchmark fallback
    if dataset is None:
        print(f"[INFO] Priority 3: No real data files found; loading synthetic benchmark match (capped to {max_frames} frames)...")
        full_ds = generate_challenging_ablation_benchmark(n_matches=1, sampling_rate_hz=25.0)[0]
        if max_frames and len(full_ds.frame_df) > max_frames:
            f_sub = full_ds.frame_df.iloc[:max_frames].copy()
            valid_fids = set(f_sub["frame_id"].to_numpy(dtype=int))
            p_sub = full_ds.player_tracks_df[full_ds.player_tracks_df["frame_id"].isin(valid_fids)].copy()
            dataset = ValidatedTrackingDataset(full_ds.metadata, full_ds.players, f_sub, p_sub)
        else:
            dataset = full_ds

    pred_df, diag = pipeline.predict_match(dataset, use_cache=True)
    print(
        f"[INFO] Match predicted for 3D export: {diag['num_frames']} frames | "
        f"Throughput: {diag['throughput_fps']} fps"
    )
    return dataset, pred_df


def build_fan_friendly_3d_payload(
    dataset: ValidatedTrackingDataset,
    pred_df: pd.DataFrame,
) -> Dict[str, Any]:
    # Precomputes 3D coordinates, simulated aerial ball height, active ball controller,
    # pass-shield activation, and plain-Spanish TV commentary per frame.
    df = pred_df.sort_values("frame_id").reset_index(drop=True).copy()
    p_df = dataset.player_tracks_df.copy()
    n = len(df)

    if n == 0:
        raise ValueError("Cannot build 3D replay payload: prediction DataFrame contains 0 frames.")

    # Map player metadata safely (handles non-numeric or missing jersey numbers)
    roster: List[Dict[str, Any]] = []
    pid_to_idx: Dict[str, int] = {}
    for pid, pmeta in sorted(dataset.players.items()):
        idx = len(roster)
        pid_to_idx[pid] = idx
        raw_num_str = "".join(c for c in str(pmeta.jersey_number) if c.isdigit())
        j_num = int(raw_num_str) if raw_num_str else 99
        roster.append({
            "idx": idx,
            "id": pid,
            "num": j_num,
            "team": "H" if pmeta.team_id == TeamID.HOME else "A",
            "gk": 1 if pmeta.role.value == "GK" else 0,
        })

    # Interpolate missing ball coordinates exclusively for smooth 3D camera & hologram rendering
    bx_raw = df["ball_x"].to_numpy(dtype=float) if "ball_x" in df.columns else np.zeros(n, dtype=float)
    by_raw = df["ball_y"].to_numpy(dtype=float) if "ball_y" in df.columns else np.zeros(n, dtype=float)
    bx_vis = pd.Series(bx_raw).interpolate(method="linear").bfill().ffill().fillna(0.0).to_numpy(dtype=float)
    by_vis = pd.Series(by_raw).interpolate(method="linear").bfill().ffill().fillna(0.0).to_numpy(dtype=float)

    ball_valid = df["ball_is_valid"].fillna(False).to_numpy(dtype=bool) if "ball_is_valid" in df.columns else np.ones(n, dtype=bool)
    ball_speed = df["ball_speed"].fillna(0.0).to_numpy(dtype=float) if "ball_speed" in df.columns else np.zeros(n, dtype=float)
    unrel_prox = df["unreliable_proximity_likelihood"].fillna(0.0).to_numpy(dtype=float) if "unreliable_proximity_likelihood" in df.columns else np.zeros(n, dtype=float)
    c_conf = df["candidate_contact_confidence"].fillna(0.0).to_numpy(dtype=float) if "candidate_contact_confidence" in df.columns else np.zeros(n, dtype=float)
    is_anchor = df["is_candidate_anchor"].fillna(False).to_numpy(dtype=bool) if "is_candidate_anchor" in df.columns else np.zeros(n, dtype=bool)
    jev_flag = df["jev_refined_flag"].fillna(False).to_numpy(dtype=bool) if "jev_refined_flag" in df.columns else np.zeros(n, dtype=bool)

    team_col = "submission_possession_team" if "submission_possession_team" in df.columns else (
        "possession_team" if "possession_team" in df.columns else None
    )
    if team_col is not None:
        our_team_arr = np.where(df[team_col].astype(str).to_numpy() == TeamID.HOME.value, "H", "A")
    else:
        our_team_arr = np.full(n, "H")

    prob_col = "calibrated_posterior_home_prob" if "calibrated_posterior_home_prob" in df.columns else (
        "home_prob" if "home_prob" in df.columns else None
    )
    if prob_col is not None:
        our_prob_arr = np.round(df[prob_col].fillna(0.5).to_numpy(dtype=float), 3)
    else:
        our_prob_arr = np.full(n, 0.5)

    # Naive Nearest-Player baseline for the interactive fan comparison toggle
    d_home = df["min_dist_home"].fillna(99.0).to_numpy(dtype=float) if "min_dist_home" in df.columns else np.full(n, 99.0)
    d_away = df["min_dist_away"].fillna(99.0).to_numpy(dtype=float) if "min_dist_away" in df.columns else np.full(n, 99.0)
    naive_team_arr = np.where(d_home <= d_away, "H", "A")

    # High-Performance pre-indexing of player trajectories (vectorized, avoiding pandas iterrows overhead)
    valid_p = p_df.dropna(subset=["x", "y"]).copy()
    valid_p_fids = valid_p["frame_id"].to_numpy(dtype=int)
    valid_p_pids = valid_p["player_id"].astype(str).to_numpy()
    valid_p_xs = np.round(valid_p["x"].to_numpy(dtype=float), 2)
    valid_p_ys = np.round(valid_p["y"].to_numpy(dtype=float), 2)

    tracks_by_fid: Dict[int, List[Tuple[int, float, float]]] = {}
    for k in range(len(valid_p)):
        fid_k = int(valid_p_fids[k])
        p_idx = pid_to_idx.get(valid_p_pids[k], -1)
        if p_idx >= 0:
            if fid_k not in tracks_by_fid:
                tracks_by_fid[fid_k] = []
            tracks_by_fid[fid_k].append((p_idx, float(valid_p_xs[k]), float(valid_p_ys[k])))

    frames_payload: List[Dict[str, Any]] = []
    turnover_markers: List[Dict[str, Any]] = []

    our_switches = 0
    naive_switches = 0
    last_owner_idx = -1

    for i in range(n):
        fid = int(df["frame_id"].iloc[i])
        ts = round(float(df["timestamp_s"].iloc[i]), 2)
        b_val = bool(ball_valid[i])
        bx = round(float(bx_vis[i]), 2)
        by = round(float(by_vis[i]), 2)

        # Simulated 3D ball height (z in meters): lifts the ball during fast passes / aerial proxies
        raw_z = 0.22
        if b_val:
            flight_boost = max(0.0, (ball_speed[i] - 7.5) * 0.16) + unrel_prox[i] * 2.4
            if c_conf[i] > 0.55:
                flight_boost *= 0.25
            raw_z = float(np.clip(0.22 + flight_boost, 0.22, 3.60))
        bz = round(raw_z, 2)

        our_t = str(our_team_arr[i])
        naive_t = str(naive_team_arr[i])

        if i > 0 and b_val and ball_valid[i - 1]:
            if our_t != str(our_team_arr[i - 1]):
                our_switches += 1
            if naive_t != str(naive_team_arr[i - 1]):
                naive_switches += 1

        # Extract active players in this frame and find closest player of the possessing team
        p_list: List[List[float]] = []
        best_owner_dist = 999.0
        best_owner_idx = last_owner_idx
        nearest_opp_dist = float(d_away[i] if our_t == "H" else d_home[i])

        if fid in tracks_by_fid:
            for p_idx, px_f, py_f in tracks_by_fid[fid]:
                p_list.append([p_idx, px_f, py_f])
                p_team = roster[p_idx]["team"]
                dist_b = float(np.hypot(px_f - bx, py_f - by))
                if p_team == our_t and dist_b < best_owner_dist:
                    best_owner_dist = dist_b
                    best_owner_idx = p_idx

        if best_owner_dist <= 3.2 or last_owner_idx < 0:
            last_owner_idx = best_owner_idx

        owner_num = roster[last_owner_idx]["num"] if last_owner_idx >= 0 else 10
        team_label = "FC Barcelona (Local)" if our_t == "H" else "Rival (Visitante)"

        # Determine visual event & Pass-Shield state
        is_turnover = (i > 0 and b_val and ball_valid[i - 1] and our_t != str(our_team_arr[i - 1]))
        shield_active = bool(
            b_val
            and (unrel_prox[i] >= 0.20 or (our_t != naive_t and ball_speed[i] >= 5.0))
        )
        false_flicker = bool(b_val and (our_t != naive_t))

        ev_code = "NONE"
        if not b_val:
            ev_code = "MISSING"
            commentary = (
                "📡 Balón fuera de cuadro o en saque — "
                "Nuestra IA congela la memoria táctica sin inventar robos falsos."
            )
        elif is_turnover:
            ev_code = f"TURNOVER_{our_t}"
            commentary = (
                f"⚡ ¡RECUPERACIÓN REAL! El dorsal #{owner_num} de {team_label} "
                f"corta la jugada y toma el control del balón."
            )
            turnover_markers.append({
                "frame_idx": i,
                "timestamp_s": ts,
                "team": our_t,
                "label": f"{ts:.1f}s — Robo #{owner_num} ({'Barça' if our_t == 'H' else 'Rival'})",
            })
        elif shield_active:
            ev_code = "SHIELD"
            commentary = (
                f"🛡️ ESCUDO DE PASE ACTIVO: El balón pasa a {nearest_opp_dist:.1f}m de un rival "
                f"a {ball_speed[i]:.1f} m/s, pero nuestra IA sabe que es un pase de {team_label}."
            )
        elif is_anchor[i] and c_conf[i] >= 0.45:
            ev_code = f"TOUCH_{our_t}"
            commentary = (
                f"👟 Toque firme del dorsal #{owner_num} ({team_label}) "
                f"— Hilo de control asegurado."
            )
        elif jev_flag[i]:
            ev_code = "JEV"
            commentary = (
                f"🧠 Jev (System One) analizando jugada dividida — "
                f"Control asignado a {team_label} (#{owner_num})."
            )
        else:
            commentary = (
                f"⚽ {team_label} domina la posesión con el dorsal #{owner_num} "
                f"(Velocidad del balón: {ball_speed[i]:.1f} m/s)."
            )

        frames_payload.append({
            "fid": fid,
            "t": ts,
            "v": 1 if b_val else 0,
            "bx": bx,
            "by": by,
            "bz": bz,
            "spd": round(float(ball_speed[i]), 1),
            "ot": our_t,
            "nt": naive_t,
            "op": float(our_prob_arr[i]),
            "own": int(last_owner_idx),
            "sh": 1 if shield_active else 0,
            "flk": 1 if false_flicker else 0,
            "jev": 1 if jev_flag[i] else 0,
            "ev": ev_code,
            "our_sw": our_switches,
            "nai_sw": naive_switches,
            "msg": commentary,
            "p": p_list,
        })

    return {
        "match_id": dataset.metadata.match_id,
        "fps": float(dataset.metadata.sampling_rate_hz),
        "pitch_length": float(dataset.metadata.pitch_length_m),
        "pitch_width": float(dataset.metadata.pitch_width_m),
        "roster": roster,
        "turnovers": turnover_markers,
        "frames": frames_payload,
    }


BASELINE_VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FC Barcelona 3D Tactical Possession Experience — More than a Hack 2027</title>
  
  <!-- Zero-CORS Match Payload Script: Loads window.MATCH_DATA without local fetch() blocking -->
  <script id="match-payload-script" src="match_payload.js"></script>

  <style>
    :root {
      --barca-blue: #004D98;
      --barca-garnet: #A50044;
      --barca-gold: #EDBB00;
      --barca-dark: #070B19;
      --rival-cyan: #00E5FF;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
    body { background: var(--barca-dark); color: #F8FAFC; overflow: hidden; width: 100vw; height: 100vh; }
    #canvas-container { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 1; }

    /* TOP BROADCAST SCOREBOARD HUD */
    .hud-top {
      position: absolute; top: 14px; left: 50%; transform: translateX(-50%); z-index: 10;
      width: min(96vw, 1180px);
      background: linear-gradient(135deg, rgba(7,11,25,0.92) 0%, rgba(0,45,98,0.88) 55%, rgba(125,0,50,0.88) 100%);
      border: 2px solid rgba(237, 187, 0, 0.65);
      border-radius: 16px; padding: 10px 18px;
      box-shadow: 0 12px 35px rgba(0,0,0,0.65);
      backdrop-filter: blur(10px);
    }
    .hud-header-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
    .brand-badge { display: flex; align-items: center; gap: 10px; }
    .crest-circle {
      width: 38px; height: 38px; border-radius: 50%;
      background: conic-gradient(#004D98 0deg 180deg, #A50044 180deg 360deg);
      border: 2px solid var(--barca-gold); display: flex; align-items: center; justify-content: center;
      font-weight: 900; font-size: 13px; color: var(--barca-gold); box-shadow: 0 0 12px rgba(237,187,0,0.5);
    }
    .brand-title { font-size: 14px; font-weight: 800; letter-spacing: 0.6px; color: var(--barca-gold); text-transform: uppercase; }
    .brand-sub { font-size: 11px; color: #CBD5E1; }

    /* AI BRAIN COMPARISON SWITCH */
    .mode-switch-box { display: flex; background: rgba(0,0,0,0.55); border-radius: 10px; padding: 4px; border: 1px solid rgba(255,255,255,0.18); }
    .mode-btn {
      border: none; background: transparent; color: #94A3B8; padding: 7px 14px;
      border-radius: 7px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s;
    }
    .mode-btn.active-our { background: linear-gradient(90deg, var(--barca-blue), var(--barca-garnet)); color: #FFF; box-shadow: 0 0 12px rgba(237,187,0,0.45); }
    .mode-btn.active-naive { background: #DC2626; color: #FFF; box-shadow: 0 0 14px rgba(220,38,38,0.75); }

    /* TUG OF WAR POSSESSION BAR */
    .tug-container { margin-top: 8px; }
    .tug-labels { display: flex; justify-content: space-between; font-size: 12px; font-weight: 800; margin-bottom: 4px; }
    .team-home-lbl { color: #60A5FA; display: flex; align-items: center; gap: 6px; }
    .team-away-lbl { color: var(--rival-cyan); display: flex; align-items: center; gap: 6px; }
    .tug-bar-bg { width: 100%; height: 12px; background: rgba(255,255,255,0.15); border-radius: 99px; overflow: hidden; display: flex; position: relative; border: 1px solid rgba(237,187,0,0.4); }
    #tug-fill-home { height: 100%; background: linear-gradient(90deg, #004D98, #A50044, #EDBB00); width: 50%; transition: width 0.12s ease-out; }
    #tug-fill-away { height: 100%; background: linear-gradient(90deg, #0284C7, #00E5FF); flex: 1; }

    /* LIVE METRICS PILLS FOR NORMAL FANS */
    .fan-stats-row { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.12); font-size: 12px; flex-wrap: wrap; gap: 8px; }
    .stat-pill { background: rgba(0,0,0,0.42); padding: 4px 10px; border-radius: 99px; border: 1px solid rgba(255,255,255,0.14); }
    .stat-pill strong { color: var(--barca-gold); }
    #flicker-alert {
      display: none; background: rgba(220, 38, 38, 0.92); color: white; font-weight: 800;
      padding: 4px 12px; border-radius: 99px; font-size: 12px; animation: pulseAlert 0.5s infinite alternate;
    }
    @keyframes pulseAlert { from { transform: scale(1); } to { transform: scale(1.05); } }

    /* CENTER FLOATING EVENT BANNER */
    #event-toast {
      position: absolute; top: 155px; left: 50%; transform: translateX(-50%); z-index: 12;
      padding: 10px 24px; border-radius: 99px; font-size: 15px; font-weight: 900;
      letter-spacing: 0.5px; pointer-events: none; opacity: 0; transition: opacity 0.25s, transform 0.25s;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }

    /* LEFT VISUAL LEGEND FOR NON-TECHNICAL FANS */
    .legend-card {
      position: absolute; left: 16px; top: 165px; z-index: 10; width: 265px;
      background: rgba(7, 11, 25, 0.84); border-left: 4px solid var(--barca-gold);
      border-radius: 12px; padding: 12px 14px; backdrop-filter: blur(8px);
      box-shadow: 0 8px 25px rgba(0,0,0,0.5); font-size: 12px;
    }
    .legend-title { font-weight: 800; color: var(--barca-gold); margin-bottom: 8px; font-size: 13px; text-transform: uppercase; }
    .legend-item { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; line-height: 1.25; color: #E2E8F0; }
    .dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }

    /* TV-STYLE ACTIVE CARRIER HUD CARD */
    #carrier-hud-card {
      position: absolute; right: 16px; top: 165px; z-index: 10; width: 285px;
      background: rgba(7, 11, 25, 0.88); border-radius: 12px;
      padding: 12px 14px; backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.14);
      border-left: 5px solid var(--barca-gold);
      transition: all 0.25s ease-out;
    }
    .carrier-card-title {
      font-size: 10px; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.8px; color: #94A3B8; margin-bottom: 6px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .carrier-main-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
    }
    .carrier-badge-wrap {
      display: flex; align-items: center; gap: 10px;
    }
    .carrier-crest {
      width: 40px; height: 40px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 900; color: #FFF; flex-shrink: 0;
      border: 2px solid var(--barca-gold);
      box-shadow: 0 0 12px rgba(237,187,0,0.4);
      background: conic-gradient(#004D98 0deg 180deg, #A50044 180deg 360deg);
      transition: all 0.25s;
    }
    .carrier-text-info { display: flex; flex-direction: column; }
    .carrier-player-num { font-size: 15px; font-weight: 900; color: #FFF; }
    .carrier-team-lbl { font-size: 11px; color: #CBD5E1; font-weight: 600; }
    .carrier-jersey-circle {
      width: 42px; height: 42px; border-radius: 10px;
      background: rgba(255,255,255,0.08); border: 2px solid var(--barca-gold);
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; font-weight: 900; color: var(--barca-gold);
      box-shadow: 0 0 10px rgba(237,187,0,0.3);
      transition: all 0.25s;
    }
    .carrier-accent-bar {
      width: 100%; height: 3px; border-radius: 2px; margin: 8px 0;
      background: linear-gradient(90deg, #004D98, #A50044, #EDBB00);
      transition: background 0.25s;
    }
    .carrier-status-row {
      display: flex; justify-content: space-between; align-items: center; font-size: 11px;
    }
    .carrier-status-badge {
      font-weight: 800; font-size: 11px; padding: 3px 9px; border-radius: 99px;
      letter-spacing: 0.5px; text-transform: uppercase;
      transition: all 0.2s;
    }

    /* TIMELINE OVERLAY PIPS TRACK */
    .timeline-container {
      position: relative;
      flex: 1;
      display: flex;
      align-items: center;
      min-width: 180px;
      height: 32px;
    }
    #timeline-pips-track {
      position: absolute;
      left: 0;
      right: 0;
      height: 100%;
      pointer-events: none;
      z-index: 5;
    }
    .timeline-pip {
      position: absolute;
      top: 50%;
      transform: translate(-50%, -50%);
      width: 4px;
      height: 18px;
      border-radius: 2px;
      pointer-events: auto;
      cursor: pointer;
      transition: transform 0.15s, height 0.15s;
    }
    .timeline-pip:hover {
      transform: translate(-50%, -50%) scale(1.5);
      height: 24px;
      z-index: 10;
    }
    .pip-home {
      background: var(--barca-gold);
      box-shadow: 0 0 8px rgba(237, 187, 0, 0.95);
    }
    .pip-away {
      background: var(--rival-cyan);
      box-shadow: 0 0 8px rgba(0, 229, 255, 0.95);
    }
    .pip-jev {
      background: #A855F7;
      box-shadow: 0 0 10px rgba(168, 85, 247, 1.0);
      width: 5px;
      height: 20px;
    }
    .chip-btn.chip-jev {
      background: rgba(168, 85, 247, 0.45);
      border-color: #A855F7;
    }
    .chip-btn.chip-jev:hover {
      background: #A855F7;
      color: #FFF;
      box-shadow: 0 0 10px rgba(168, 85, 247, 0.7);
    }

    @media (max-width: 900px) {
      .legend-card { display: none; }
      #carrier-hud-card {
        top: auto; bottom: 135px; right: 16px; width: min(92vw, 290px);
      }
    }

    /* BOTTOM PLAYBACK & CAMERA DOCK */
    .hud-bottom {
      position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%); z-index: 10;
      width: min(96vw, 1180px);
      background: rgba(7, 11, 25, 0.92); border: 1px solid rgba(237, 187, 0, 0.5);
      border-radius: 16px; padding: 12px 18px; backdrop-filter: blur(10px);
      box-shadow: 0 10px 35px rgba(0,0,0,0.7);
    }
    #commentary-bar {
      background: linear-gradient(90deg, rgba(0,77,152,0.45), rgba(165,0,68,0.45));
      border-left: 4px solid var(--barca-gold); padding: 8px 14px; border-radius: 8px;
      font-size: 14px; font-weight: 600; color: #FFF; margin-bottom: 10px;
      display: flex; justify-content: space-between; align-items: center; gap: 10px;
    }
    .controls-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
    .btn-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .ctrl-btn {
      background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #FFF;
      padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; transition: 0.15s;
    }
    .ctrl-btn:hover { background: rgba(237,187,0,0.28); border-color: var(--barca-gold); }
    .ctrl-btn.active { background: var(--barca-gold); color: #000; border-color: var(--barca-gold); }
    .ctrl-btn-reload {
      background: linear-gradient(90deg, rgba(0,77,152,0.7), rgba(165,0,68,0.7));
      border: 1px solid var(--barca-gold); color: #FFF;
    }
    .ctrl-btn-reload:hover {
      background: linear-gradient(90deg, var(--barca-blue), var(--barca-garnet));
      box-shadow: 0 0 10px rgba(237,187,0,0.6);
    }
    #timeline-slider { flex: 1; min-width: 180px; accent-color: var(--barca-gold); cursor: pointer; }
    .turnover-chips { display: flex; gap: 6px; margin-top: 8px; align-items: center; flex-wrap: wrap; font-size: 11px; color: #CBD5E1; }
    .chip-btn {
      background: rgba(165,0,68,0.55); border: 1px solid var(--barca-gold); color: #FFF;
      padding: 3px 9px; border-radius: 99px; font-size: 11px; cursor: pointer; font-weight: 700;
    }
    .chip-btn:hover { background: var(--barca-gold); color: #000; }

    /* DATA MISSING OVERLAY */
    #data-missing-overlay {
      position: fixed; inset: 0; background: rgba(7,11,25,0.95); z-index: 9999;
      display: none; flex-direction: column; align-items: center; justify-content: center;
      text-align: center; padding: 20px;
    }
    .missing-box {
      background: rgba(0,45,98,0.35); border: 2px solid var(--barca-gold);
      border-radius: 18px; padding: 32px 40px; max-width: 580px; box-shadow: 0 15px 40px rgba(0,0,0,0.85);
    }
  </style>
</head>
<body>

<div id="canvas-container"></div>

<!-- DATA MISSING OVERLAY -->
<div id="data-missing-overlay">
  <div class="missing-box">
    <div style="font-size: 46px; margin-bottom: 12px;">⚽ 📡</div>
    <h2 style="color: var(--barca-gold); font-size: 22px; font-weight: 800; margin-bottom: 12px;">DATOS DEL PARTIDO NO DETECTADOS</h2>
    <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6; margin-bottom: 20px;">
      El visor 3D requiere el archivo <code>match_payload.js</code> en el mismo directorio.<br>
      Ejecuta en tu terminal el generador de repeticiones para exportar la jugada:
    </p>
    <pre style="background: #070B19; border: 1px solid rgba(237,187,0,0.4); border-radius: 8px; padding: 12px; font-size: 13px; color: #60A5FA; overflow-x: auto; margin-bottom: 24px; text-align: left;">.\.venv\Scripts\python.exe generate_3d_replay_html.py</pre>
    <button class="ctrl-btn ctrl-btn-reload" style="padding: 10px 24px; font-size: 14px; border-radius: 99px;" onclick="reloadDataPayload()">
      🔄 Reintentar Carga
    </button>
  </div>
</div>

<!-- TOP SCOREBOARD HUD -->
<div class="hud-top">
  <div class="hud-header-row">
    <div class="brand-badge">
      <div class="crest-circle">FCB</div>
      <div>
        <div class="brand-title">Barça Innovation Hub × Metrica Sports 3D</div>
        <div class="brand-sub">More than a Hack 2027 • Explicador Visual de Posesión en Tiempo Real</div>
      </div>
    </div>
    <div class="mode-switch-box">
      <button id="btn-mode-our" class="mode-btn active-our" onclick="setBrainMode('OUR')">🧠 NUESTRA IA (Barça + Viterbi + Jev)</button>
      <button id="btn-mode-naive" class="mode-btn" onclick="setBrainMode('NAIVE')">🤖 IA COMÚN (Solo Jugador Más Cercano)</button>
    </div>
  </div>

  <div class="tug-container">
    <div class="tug-labels">
      <span class="team-home-lbl">🔵🔴 FC BARCELONA (LOCAL): <span id="lbl-prob-home">50%</span></span>
      <span id="lbl-clock" style="color:#EDBB00; font-size:14px;">⏱️ 00:00.0</span>
      <span class="team-away-lbl"><span id="lbl-prob-away">50%</span> :RIVAL (VISITANTE) ⚪🔵</span>
    </div>
    <div class="tug-bar-bg">
      <div id="tug-fill-home"></div>
      <div id="tug-fill-away"></div>
    </div>
  </div>

  <div class="fan-stats-row">
    <div class="stat-pill">Cambios de Posesión (Nuestra IA): <strong id="stat-our-sw">0</strong></div>
    <div class="stat-pill">Cambios Falsos (IA Común): <strong id="stat-naive-sw" style="color:#F87171;">0</strong></div>
    <div class="stat-pill">Parpadeos Evitados: <strong id="stat-saved">0</strong></div>
    <div id="flicker-alert">⚠️ ¡ERROR DE IA COMÚN! Falso robo por simple cercanía</div>
  </div>
</div>

<!-- FLOATING EVENT BANNER -->
<div id="event-toast">⚡ ¡RECUPERACIÓN DE BALÓN!</div>

<!-- LEFT FAN LEGEND -->
<div class="legend-card">
  <div class="legend-title">Guía Rápida para el Aficionado</div>
  <div class="legend-item"><span class="dot" style="background:#A50044; border:2px solid #EDBB00;"></span><span><b>FC Barcelona (Azulgrana):</b> Equipo local atacando.</span></div>
  <div class="legend-item"><span class="dot" style="background:#00E5FF; border:2px solid #FFF;"></span><span><b>Equipo Rival (Blanco/Cian):</b> Equipo visitante.</span></div>
  <div class="legend-item"><span class="dot" style="background:#EDBB00; box-shadow:0 0 8px #EDBB00;"></span><span><b>Baliza 3D y Esferas de Energía:</b> Identifica al poseedor activo y el hilo hacia el balón.</span></div>
  <div class="legend-item"><span class="dot" style="background:transparent; border:2px dashed #EDBB00;"></span><span><b>Escudo 3D Anti-Robo:</b> Protege los pases que pasan cerca de rivales sin ser tocados.</span></div>
  <div class="legend-item"><span class="dot" style="background:#A855F7;"></span><span><b>Destello Violeta (Jev AI):</b> Desempate inteligente en jugadas divididas.</span></div>
</div>

<!-- TV-STYLE ACTIVE CARRIER HUD CARD -->
<div id="carrier-hud-card">
  <div class="carrier-card-title">
    <span>📡 Poseedor Activo del Balón</span>
    <span id="carrier-brain-tag" style="color:var(--barca-gold); font-size:9px; font-weight:800;">NUESTRA IA</span>
  </div>
  <div class="carrier-main-row">
    <div class="carrier-badge-wrap">
      <div id="carrier-crest" class="carrier-crest">FCB</div>
      <div class="carrier-text-info">
        <div id="carrier-player-title" class="carrier-player-num">Dorsal #10</div>
        <div id="carrier-team-name" class="carrier-team-lbl">FC Barcelona (Local)</div>
      </div>
    </div>
    <div id="carrier-jersey-box" class="carrier-jersey-circle">#10</div>
  </div>
  <div id="carrier-accent-bar" class="carrier-accent-bar"></div>
  <div class="carrier-status-row">
    <span style="color:#94A3B8; font-weight:700;">ESTADO DE CONTROL:</span>
    <span id="carrier-status-badge" class="carrier-status-badge" style="background:rgba(34,197,94,0.25); border:1px solid #22C55E; color:#4ADE80;">EN CONTROL</span>
  </div>
</div>

<!-- BOTTOM CONTROLS & NARRATOR -->
<div class="hud-bottom">
  <div id="commentary-bar">
    <span id="commentary-text">⚽ Cargando repetición 3D interactiva...</span>
    <span id="ball-speed-badge" style="font-size:12px; color:#EDBB00; white-space:nowrap;">Vel. Balón: 0.0 m/s</span>
  </div>
  <div class="controls-row">
    <div class="btn-group">
      <button id="btn-play" class="ctrl-btn active" onclick="togglePlay()">⏸️ Pausa</button>
      <button class="ctrl-btn" onclick="stepFrames(-25)">⏪ -1s</button>
      <button class="ctrl-btn" onclick="stepFrames(25)">⏩ +1s</button>
    </div>
    <div class="timeline-container">
      <div id="timeline-pips-track"></div>
      <input id="timeline-slider" type="range" min="0" max="100" value="0" step="0.1" oninput="onSliderInput(this.value)">
    </div>
    <div class="btn-group">
      <button id="spd-025" class="ctrl-btn" onclick="setSpeed(0.25)">0.25x</button>
      <button id="spd-05" class="ctrl-btn" onclick="setSpeed(0.5)">0.5x</button>
      <button id="spd-1" class="ctrl-btn active" onclick="setSpeed(1.0)">1x</button>
      <button id="spd-2" class="ctrl-btn" onclick="setSpeed(2.0)">2x</button>
    </div>
    <div class="btn-group">
      <button id="cam-tv" class="ctrl-btn active" onclick="setCameraMode('TV')">🎥 TV</button>
      <button id="cam-drone" class="ctrl-btn" onclick="setCameraMode('DRONE')">🚁 Dron</button>
      <button id="cam-top" class="ctrl-btn" onclick="setCameraMode('TOP')">🏟️ Aérea</button>
      <button id="cam-free" class="ctrl-btn" onclick="setCameraMode('FREE')">🕹️ Libre</button>
    </div>
    <button id="btn-reload" class="ctrl-btn ctrl-btn-reload" onclick="reloadDataPayload()">🔄 Recargar Datos</button>
  </div>
  <div id="turnover-chips-row" class="turnover-chips">
    <span>⚡ Saltar a momentos clave:</span>
  </div>
</div>

<!-- THREE.JS CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
// --- GLOBAL APPLICATION STATE ---
let brainMode = 'OUR'; // 'OUR' vs 'NAIVE'
let cameraMode = 'TV'; // 'TV', 'DRONE', 'TOP', 'FREE'
let isPlaying = true;
let playbackSpeed = 1.0;
let currentFrameFloat = 0.0;
let totalFrames = 0;
let fps = 25.0;
let playerMeshes = {};

const slider = document.getElementById('timeline-slider');

// --- THREE.JS SCENE SETUP ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050814);
scene.fog = new THREE.FogExp2(0x050814, 0.0045);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.5, 400);
camera.position.set(0, 46, 62);
const cameraTarget = new THREE.Vector3(0, 0, 0);

// --- ZERO-GC VECTOR & COLOR SCRATCHPAD FOR 60 FPS RENDER LOOP ---
const _scratchV3 = new THREE.Vector3();
const _scratchTarget = new THREE.Vector3();
const _scratchColorJev = new THREE.Color(0xA855F7);
const _scratchP1Map = {};

// Renderer with preserveDrawingBuffer enabled for canvas toDataURL captures
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: true
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// --- STADIUM LIGHTING ---
const ambientLight = new THREE.AmbientLight(0xD8E6FF, 0.65);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xFFFDF5, 0.95);
sunLight.position.set(35, 75, 45);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
sunLight.shadow.camera.left = -65;
sunLight.shadow.camera.right = 65;
sunLight.shadow.camera.top = 45;
sunLight.shadow.camera.bottom = -45;
scene.add(sunLight);

// --- PROCEDURAL SOCCER PITCH TEXTURE WITH MOWED STRIPES & LINES ---
function createPitchTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 2048; canvas.height = 1360;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // Mowed grass stripes
  const numStripes = 18;
  const stripeW = w / numStripes;
  for (let i = 0; i < numStripes; i++) {
    ctx.fillStyle = (i % 2 === 0) ? '#1B5E20' : '#236B28';
    ctx.fillRect(i * stripeW, 0, stripeW + 1, h);
  }

  // Pitch white markings
  const padX = 90, padY = 80;
  const pw = w - 2 * padX, ph = h - 2 * padY;
  ctx.strokeStyle = 'rgba(255,255,255,0.92)';
  ctx.lineWidth = 6;
  ctx.strokeRect(padX, padY, pw, ph);

  // Halfway line & center circle
  ctx.beginPath(); ctx.moveTo(w / 2, padY); ctx.lineTo(w / 2, h - padY); ctx.stroke();
  ctx.beginPath(); ctx.arc(w / 2, h / 2, ph * (9.15 / 68.0), 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = '#FFF'; ctx.beginPath(); ctx.arc(w / 2, h / 2, 8, 0, Math.PI * 2); ctx.fill();

  // Penalty areas
  const penL = pw * (16.5 / 105.0), penW = ph * (40.3 / 68.0);
  ctx.strokeRect(padX, (h - penW) / 2, penL, penW);
  ctx.strokeRect(w - padX - penL, (h - penW) / 2, penL, penW);

  // 6-yard goal boxes
  const goalL = pw * (5.5 / 105.0), goalW = ph * (18.32 / 68.0);
  ctx.strokeRect(padX, (h - goalW) / 2, goalL, goalW);
  ctx.strokeRect(w - padX - goalL, (h - goalW) / 2, goalL, goalW);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 8;
  return tex;
}

let pitchL = 105.0;
let pitchW = 68.0;
const pitchGeo = new THREE.PlaneGeometry(pitchL + 10, pitchW + 10);
const pitchMat = new THREE.MeshStandardMaterial({ map: createPitchTexture(), roughness: 0.85 });
const pitchMesh = new THREE.Mesh(pitchGeo, pitchMat);
pitchMesh.rotation.x = -Math.PI / 2;
pitchMesh.receiveShadow = true;
scene.add(pitchMesh);

// --- BLAUGRANA LED PERIMETER BOARDS & 3D GOALS ---
function buildStadiumSurroundings() {
  const boardMat = new THREE.MeshBasicMaterial({ color: 0x004D98 });
  const goldMat = new THREE.MeshBasicMaterial({ color: 0xEDBB00 });

  const backBoard = new THREE.Mesh(new THREE.BoxGeometry(pitchL + 4, 1.3, 0.4), boardMat);
  backBoard.position.set(0, 0.65, -pitchW / 2 - 2.6);
  scene.add(backBoard);

  const goldTrim = new THREE.Mesh(new THREE.BoxGeometry(pitchL + 4, 0.18, 0.45), goldMat);
  goldTrim.position.set(0, 1.32, -pitchW / 2 - 2.6);
  scene.add(goldTrim);

  // 3D Goalposts
  const postMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, metalness: 0.4, roughness: 0.2 });
  [-pitchL / 2, pitchL / 2].forEach((gx) => {
    const goalGrp = new THREE.Group();
    const p1 = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 2.44, 12), postMat);
    p1.position.set(gx, 1.22, -3.66);
    const p2 = p1.clone(); p2.position.set(gx, 1.22, 3.66);
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 7.32, 12), postMat);
    bar.rotation.x = Math.PI / 2; bar.position.set(gx, 2.44, 0);
    goalGrp.add(p1, p2, bar);
    scene.add(goalGrp);
  });
}
buildStadiumSurroundings();

// --- PLAYER 3D AVATARS WITH FLOATING JERSEY NUMBERS ---
function makeNumberSprite(numStr, teamCode, isGK) {
  const c = document.createElement('canvas'); c.width = 128; c.height = 128;
  const ctx = c.getContext('2d');
  ctx.fillStyle = isGK ? '#F59E0B' : (teamCode === 'H' ? '#A50044' : '#0F172A');
  ctx.beginPath(); ctx.arc(64, 64, 48, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = 7;
  ctx.strokeStyle = (teamCode === 'H') ? '#EDBB00' : '#00E5FF';
  ctx.stroke();
  ctx.fillStyle = '#FFFFFF'; ctx.font = 'bold 52px Segoe UI, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(String(numStr), 64, 66);
  const tex = new THREE.CanvasTexture(c);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
  spr.scale.set(1.8, 1.8, 1);
  spr.position.set(0, 2.85, 0);
  return spr;
}

// --- 3D BALL, PASS SHIELD, SHADOW, TRAIL & CONTROL TETHER ---
const ballGroup = new THREE.Group();
const ballMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.38, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xFFFFFF, emissive: 0xEDBB00, emissiveIntensity: 0.35, roughness: 0.2 })
);
ballMesh.castShadow = true;
ballGroup.add(ballMesh);

// Anti-theft Pass Shield ('Escudo de Pase Filtrado')
const shieldMesh = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.15, 1),
  new THREE.MeshBasicMaterial({ color: 0xEDBB00, wireframe: true, transparent: true, opacity: 0.65 })
);
shieldMesh.visible = false;
ballGroup.add(shieldMesh);

const ballLight = new THREE.PointLight(0xEDBB00, 2.2, 14);
ballLight.position.set(0, 0.8, 0);
ballGroup.add(ballLight);

// --- JEV AI (SYSTEM ONE) 3D VIOLET NEURAL GYROSCOPE ---
const jevPulseGroup = new THREE.Group();
const jevTorusGeo = new THREE.TorusGeometry(0.92, 0.045, 10, 28);
const jevRing1Mat = new THREE.MeshBasicMaterial({ color: 0xA855F7, transparent: true, opacity: 0.0, depthWrite: false });
const jevRing1 = new THREE.Mesh(jevTorusGeo, jevRing1Mat);
const jevRing2Mat = new THREE.MeshBasicMaterial({ color: 0xA855F7, transparent: true, opacity: 0.0, depthWrite: false });
const jevRing2 = new THREE.Mesh(jevTorusGeo, jevRing2Mat);
jevRing2.rotation.y = Math.PI / 2;
jevPulseGroup.add(jevRing1);
jevPulseGroup.add(jevRing2);

const jevGroundHexGeo = new THREE.RingGeometry(1.55, 1.95, 6);
const jevGroundHexMat = new THREE.MeshBasicMaterial({ color: 0xA855F7, side: THREE.DoubleSide, transparent: true, opacity: 0.0, depthWrite: false });
const jevGroundHex = new THREE.Mesh(jevGroundHexGeo, jevGroundHexMat);
jevGroundHex.rotation.x = -Math.PI / 2;
jevPulseGroup.add(jevGroundHex);

jevPulseGroup.visible = false;
ballGroup.add(jevPulseGroup);
let jevVisualAlpha = 0.0;

scene.add(ballGroup);

// Ground drop shadow directly below the elevated ball
const ballShadow = new THREE.Mesh(
  new THREE.CircleGeometry(0.65, 24),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.55 })
);
ballShadow.rotation.x = -Math.PI / 2; ballShadow.position.y = 0.03;
scene.add(ballShadow);

// Glowing Ground Possession Halo on the turf directly beneath the ball (complementing ballShadow)
const ballGroundHalo = new THREE.Mesh(
  new THREE.RingGeometry(0.68, 1.55, 32),
  new THREE.MeshBasicMaterial({ color: 0xEDBB00, side: THREE.DoubleSide, transparent: true, opacity: 0.55, depthWrite: false })
);
ballGroundHalo.rotation.x = -Math.PI / 2;
ballGroundHalo.position.y = 0.035;
scene.add(ballGroundHalo);

// --- SCRUB-SAFE 3D BALL COMET MOTION TRAIL ---
const BALL_TRAIL_COUNT = 18;
const ballTrailGroup = new THREE.Group();
const trailGeo = new THREE.SphereGeometry(0.32, 12, 12);
const trailSpheres = [];
for (let k = 0; k < BALL_TRAIL_COUNT; k++) {
  const trailMat = new THREE.MeshBasicMaterial({
    color: 0xEDBB00,
    transparent: true,
    opacity: 0.0,
    depthWrite: false
  });
  const trailMesh = new THREE.Mesh(trailGeo, trailMat);
  trailMesh.visible = false;
  ballTrailGroup.add(trailMesh);
  trailSpheres.push(trailMesh);
}
scene.add(ballTrailGroup);

// Curved 3D Control Tether ('Hilo Dorado de Control') from owner to ball
const tetherPoints = [];
for (let i = 0; i <= 20; i++) tetherPoints.push(new THREE.Vector3());
const tetherGeo = new THREE.BufferGeometry().setFromPoints(tetherPoints);
const tetherMat = new THREE.LineBasicMaterial({ color: 0xEDBB00, linewidth: 3, transparent: true, opacity: 0.35 });
const tetherLine = new THREE.Line(tetherGeo, tetherMat);
scene.add(tetherLine);

// --- 3D VOLUMETRIC ANIMATED ENERGY BEAD CHAIN TETHER ---
const TETHER_BEADS_COUNT = 16;
const tetherBeadsGroup = new THREE.Group();
const tetherBeads = [];
const beadGeo = new THREE.SphereGeometry(0.16, 10, 10);
for (let i = 0; i < TETHER_BEADS_COUNT; i++) {
  const beadMat = new THREE.MeshStandardMaterial({
    color: 0xEDBB00,
    emissive: 0xEDBB00,
    emissiveIntensity: 0.85,
    roughness: 0.15
  });
  const beadMesh = new THREE.Mesh(beadGeo, beadMat);
  beadMesh.visible = false;
  tetherBeadsGroup.add(beadMesh);
  tetherBeads.push(beadMesh);
}
tetherBeadsGroup.visible = false;
scene.add(tetherBeadsGroup);

// --- EA SPORTS FC STYLE 3D ACTIVE CARRIER BEACON ---
const ownerBeaconGroup = new THREE.Group();

// 1. Outer Pulsing Ground Ring
const beaconRingGeo = new THREE.RingGeometry(0.95, 1.35, 32);
const beaconRingMat = new THREE.MeshBasicMaterial({
  color: 0xEDBB00,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.9
});
const beaconRing = new THREE.Mesh(beaconRingGeo, beaconRingMat);
beaconRing.rotation.x = -Math.PI / 2;
beaconRing.position.y = 0.05;
ownerBeaconGroup.add(beaconRing);

// 2. Vertical Spotlight Aura Cylinder (open-ended)
const beaconAuraGeo = new THREE.CylinderGeometry(0.85, 0.85, 3.2, 24, 1, true);
const beaconAuraMat = new THREE.MeshBasicMaterial({
  color: 0xEDBB00,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.18,
  depthWrite: false
});
const beaconAura = new THREE.Mesh(beaconAuraGeo, beaconAuraMat);
beaconAura.position.y = 1.6;
ownerBeaconGroup.add(beaconAura);

// 3. Floating 3D Diamond / Chevron Marker Above Head
const beaconDiamondGeo = new THREE.OctahedronGeometry(0.48, 0);
const beaconDiamondMat = new THREE.MeshStandardMaterial({
  color: 0xEDBB00,
  emissive: 0xEDBB00,
  emissiveIntensity: 0.8,
  roughness: 0.1
});
const beaconDiamond = new THREE.Mesh(beaconDiamondGeo, beaconDiamondMat);
beaconDiamond.position.y = 4.1;
beaconDiamond.castShadow = true;
ownerBeaconGroup.add(beaconDiamond);

ownerBeaconGroup.visible = false;
scene.add(ownerBeaconGroup);

// Expanding Shockwave Ring on Touches & Turnovers
const shockRing = new THREE.Mesh(
  new THREE.RingGeometry(0.5, 1.15, 32),
  new THREE.MeshBasicMaterial({ color: 0xEDBB00, side: THREE.DoubleSide, transparent: true, opacity: 0.0 })
);
shockRing.rotation.x = -Math.PI / 2; shockRing.position.y = 0.06;
scene.add(shockRing);
let shockScale = 1.0, shockAlpha = 0.0;

// --- DYNAMIC DATA INITIALIZATION / HOT-RELOAD ---
function initMatchData(data) {
  const overlay = document.getElementById('data-missing-overlay');
  if (!data || !data.frames || data.frames.length === 0) {
    if (overlay) overlay.style.display = 'flex';
    totalFrames = 0;
    return;
  }
  if (overlay) overlay.style.display = 'none';

  window.MATCH_DATA = data;
  totalFrames = data.frames.length;
  fps = data.fps || 25.0;
  pitchL = data.pitch_length || 105.0;
  pitchW = data.pitch_width || 68.0;

  // Initialize or update timeline slider bounds
  if (slider) {
    slider.max = Math.max(0, totalFrames - 1);
    if (currentFrameFloat > totalFrames - 1) {
      currentFrameFloat = 0;
      slider.value = 0;
    }
  }

  // Populate Visual Timeline Turnover Markers on Slider Track & Key Event Chips
  const pipsContainer = document.getElementById('timeline-pips-track');
  const chipsRow = document.getElementById('turnover-chips-row');
  if (pipsContainer) pipsContainer.innerHTML = '';
  if (chipsRow) chipsRow.innerHTML = '<span>⚡ Saltar a momentos clave:</span>';

  const timelineEvents = [];
  const seenFrames = new Set();

  if (data.turnovers && data.turnovers.length > 0) {
    data.turnovers.forEach((to) => {
      timelineEvents.push({
        frame_idx: to.frame_idx,
        type: 'TURNOVER',
        team: to.team,
        label: to.label || `${(to.timestamp_s || to.frame_idx / fps).toFixed(1)}s — Robo de Balón (${to.team === 'H' ? 'Barça' : 'Rival'})`
      });
      seenFrames.add(to.frame_idx);
    });
  }

  // Scan frames for key transitions and Jev AI activations
  if (data.frames) {
    let prevJev = 0;
    data.frames.forEach((f, idx) => {
      if (f.ev && f.ev.startsWith('TURNOVER') && !seenFrames.has(idx)) {
        timelineEvents.push({
          frame_idx: idx,
          type: 'TURNOVER',
          team: f.ot,
          label: `${f.t.toFixed(1)}s — ⚡ Recuperación (${f.ot === 'H' ? 'Barça' : 'Rival'})`
        });
        seenFrames.add(idx);
      } else if (f.jev === 1 && prevJev === 0 && !seenFrames.has(idx)) {
        timelineEvents.push({
          frame_idx: idx,
          type: 'JEV',
          team: f.ot,
          label: `${f.t.toFixed(1)}s — 🧠 Jev AI: Jugada Dividida`
        });
        seenFrames.add(idx);
      }
      prevJev = f.jev;
    });
  }

  timelineEvents.sort((a, b) => a.frame_idx - b.frame_idx);

  // Overlay pips on slider track
  if (pipsContainer && totalFrames > 1) {
    timelineEvents.forEach((evt) => {
      const pct = (evt.frame_idx / (totalFrames - 1)) * 100.0;
      const pip = document.createElement('div');
      pip.className = `timeline-pip ${evt.type === 'JEV' ? 'pip-jev' : (evt.team === 'H' ? 'pip-home' : 'pip-away')}`;
      pip.style.left = `${pct}%`;
      pip.title = evt.label;
      pip.onclick = (e) => {
        e.stopPropagation();
        currentFrameFloat = evt.frame_idx;
        if (slider) slider.value = currentFrameFloat;
        triggerEventToast(
          evt.label,
          evt.type === 'JEV' ? 'linear-gradient(90deg,#9333EA,#A855F7)' : (evt.team === 'H' ? 'linear-gradient(90deg,#004D98,#A50044)' : '#0284C7')
        );
      };
      pipsContainer.appendChild(pip);
    });
  }

  // Populate interactive chips
  if (chipsRow) {
    if (timelineEvents.length > 0) {
      timelineEvents.forEach((evt) => {
        const btn = document.createElement('button');
        btn.className = `chip-btn ${evt.type === 'JEV' ? 'chip-jev' : ''}`;
        btn.innerText = evt.label;
        btn.onclick = () => {
          currentFrameFloat = Math.max(0, evt.frame_idx - (evt.type === 'JEV' ? 8 : 18));
          if (slider) slider.value = currentFrameFloat;
        };
        chipsRow.appendChild(btn);
      });
    } else {
      const btn = document.createElement('button');
      btn.className = 'chip-btn';
      btn.innerText = 'Inicio del Partido (0.0s)';
      btn.onclick = () => {
        currentFrameFloat = 0;
        if (slider) slider.value = 0;
      };
      chipsRow.appendChild(btn);
    }
  }

  // Clear existing players and build new roster meshes with thorough GPU resource disposal
  for (const idx in playerMeshes) {
    const pm = playerMeshes[idx];
    if (pm && pm.group) {
      pm.group.traverse((child) => {
        if (child.isMesh || child.isSprite) {
          if (child.geometry) {
            child.geometry.dispose();
          }
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach((mat) => {
                if (mat.map) mat.map.dispose();
                mat.dispose();
              });
            } else {
              if (child.material.map) child.material.map.dispose();
              child.material.dispose();
            }
          }
        }
      });
      scene.remove(pm.group);
    }
  }
  playerMeshes = {};

  if (data.roster && data.roster.length > 0) {
    data.roster.forEach((p) => {
      const grp = new THREE.Group();
      const isHome = (p.team === 'H');
      const bodyColor = p.gk ? 0xEAB308 : (isHome ? 0x004D98 : 0xF8FAFC);
      const trimColor = isHome ? 0xA50044 : 0x00E5FF;

      const torso = new THREE.Mesh(
        new THREE.CylinderGeometry(0.45, 0.36, 1.35, 16),
        new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 0.35 })
      );
      torso.position.y = 0.85; torso.castShadow = true;
      grp.add(torso);

      const stripe = new THREE.Mesh(
        new THREE.CylinderGeometry(0.47, 0.44, 0.45, 16),
        new THREE.MeshStandardMaterial({ color: trimColor, roughness: 0.4 })
      );
      stripe.position.y = 1.05;
      grp.add(stripe);

      const head = new THREE.Mesh(
        new THREE.SphereGeometry(0.32, 16, 16),
        new THREE.MeshStandardMaterial({ color: 0xFDE68A, roughness: 0.5 })
      );
      head.position.y = 1.82; head.castShadow = true;
      grp.add(head);

      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.55, 0.78, 24),
        new THREE.MeshBasicMaterial({
          color: isHome ? 0xEDBB00 : 0x00E5FF,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.22
        })
      );
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.04;
      grp.add(ring);

      grp.add(makeNumberSprite(p.num, p.team, p.gk));
      grp.visible = false;
      scene.add(grp);
      playerMeshes[p.idx] = { group: grp, ring: ring, torso: torso, team: p.team, num: p.num };
    });
  }
}

// In-App Hot-Reload Handler: Dynamically reloads match_payload.js without losing camera or speed
function reloadDataPayload(callback) {
  const isFile = (window.location.protocol === 'file:');
  const oldScript = document.getElementById('match-payload-script');
  if (oldScript) {
    oldScript.remove();
  }
  const script = document.createElement('script');
  script.id = 'match-payload-script';
  // Avoid query string params on local file:// URLs so Windows paths resolve safely
  script.src = isFile ? 'match_payload.js' : `match_payload.js?t=${Date.now()}`;
  script.onload = () => {
    if (window.MATCH_DATA && window.MATCH_DATA.frames) {
      initMatchData(window.MATCH_DATA);
      triggerEventToast('🔄 Datos del partido actualizados en vivo', 'linear-gradient(90deg,#004D98,#EDBB00)');
      if (typeof callback === 'function') callback(null, window.MATCH_DATA);
    } else {
      triggerEventToast('❌ Error: MATCH_DATA no encontrado', '#DC2626');
      if (typeof callback === 'function') callback(new Error('MATCH_DATA missing'));
    }
  };
  script.onerror = (err) => {
    triggerEventToast('❌ Error al cargar match_payload.js', '#DC2626');
    if (typeof callback === 'function') callback(err);
  };
  document.head.appendChild(script);
}

// --- INTERACTIVE UI HANDLERS ---
function setBrainMode(mode) {
  brainMode = mode;
  document.getElementById('btn-mode-our').className = (mode === 'OUR') ? 'mode-btn active-our' : 'mode-btn';
  document.getElementById('btn-mode-naive').className = (mode === 'NAIVE') ? 'mode-btn active-naive' : 'mode-btn';
}
function togglePlay() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('btn-play');
  btn.innerText = isPlaying ? '⏸️ Pausa' : '▶️ Reproducir';
  btn.classList.toggle('active', isPlaying);
}
function stepFrames(delta) {
  if (isNaN(currentFrameFloat)) currentFrameFloat = 0.0;
  if (totalFrames > 0) {
    currentFrameFloat = Math.min(totalFrames - 1, Math.max(0, currentFrameFloat + delta));
    if (slider) slider.value = currentFrameFloat;
  }
}
function onSliderInput(val) {
  const parsed = parseFloat(val);
  if (!isNaN(parsed) && totalFrames > 0) {
    currentFrameFloat = Math.max(0, Math.min(totalFrames - 1, parsed));
  }
}
function setSpeed(spd) {
  playbackSpeed = spd;
  ['025', '05', '1', '2'].forEach(k => document.getElementById('spd-' + k).classList.remove('active'));
  const idMap = { 0.25: '025', 0.5: '05', 1.0: '1', 2.0: '2' };
  if (idMap[spd]) document.getElementById('spd-' + idMap[spd]).classList.add('active');
}
function setCameraMode(mode) {
  cameraMode = mode;
  ['tv', 'drone', 'top', 'free'].forEach(k => {
    const el = document.getElementById('cam-' + k);
    if (el) el.classList.remove('active');
  });
  const activeEl = document.getElementById('cam-' + mode.toLowerCase());
  if (activeEl) activeEl.classList.add('active');
}

// Mouse Drag and Wheel Zoom for Free Orbit Camera
let isDragging = false;
let prevMouseX = 0, prevMouseY = 0;
let freeCamRotY = 0.0, freeCamPitch = 0.58;
let freeCamDist = 65.0;

container.addEventListener('mousedown', (e) => {
  if (cameraMode === 'FREE') {
    isDragging = true;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
  }
});
window.addEventListener('mousemove', (e) => {
  if (isDragging && cameraMode === 'FREE') {
    const dx = e.clientX - prevMouseX;
    const dy = e.clientY - prevMouseY;
    prevMouseX = e.clientX;
    prevMouseY = e.clientY;
    freeCamRotY += dx * 0.007;
    freeCamPitch = Math.max(0.08, Math.min(1.45, freeCamPitch + dy * 0.007));
  }
});
window.addEventListener('mouseup', () => { isDragging = false; });
container.addEventListener('wheel', (e) => {
  if (cameraMode === 'FREE') {
    freeCamDist = Math.max(12.0, Math.min(140.0, freeCamDist + e.deltaY * 0.06));
  }
}, { passive: true });

// Keyboard navigation shortcuts
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); stepFrames(-25); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); stepFrames(25); }
  else if (e.key === '1') { setCameraMode('TV'); }
  else if (e.key === '2') { setCameraMode('DRONE'); }
  else if (e.key === '3') { setCameraMode('TOP'); }
  else if (e.key === '4') { setCameraMode('FREE'); }
  else if (e.key === 'm' || e.key === 'M') { setBrainMode(brainMode === 'OUR' ? 'NAIVE' : 'OUR'); }
});

let lastEventCode = 'NONE';
function triggerEventToast(text, bgColor) {
  const toast = document.getElementById('event-toast');
  toast.innerText = text;
  toast.style.background = bgColor;
  toast.style.opacity = '1';
  toast.style.transform = 'translateX(-50%) scale(1.06)';
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) scale(0.95)';
  }, 1400);
}

// --- GLOBAL CONTROLLER API FOR FUTURE REALTIME AI (GEMINI LIVE / MULTIMODAL HOOKS) ---
window.PossessionViewerAPI = {
  getCurrentFrameState: function() {
    if (!window.MATCH_DATA || !window.MATCH_DATA.frames || window.MATCH_DATA.frames.length === 0) {
      return null;
    }
    const idx0 = Math.max(0, Math.min(totalFrames - 1, Math.floor(currentFrameFloat)));
    const idx1 = Math.max(0, Math.min(totalFrames - 1, idx0 + 1));
    const alpha = Math.max(0.0, Math.min(1.0, currentFrameFloat - idx0));
    const f0 = window.MATCH_DATA.frames[idx0] || {};
    const f1 = window.MATCH_DATA.frames[idx1] || f0;
    const bx = f0.bx != null && f1.bx != null ? f0.bx + (f1.bx - f0.bx) * alpha : (f0.bx || 0);
    const by = f0.by != null && f1.by != null ? f0.by + (f1.by - f0.by) * alpha : (f0.by || 0);
    const bz = f0.bz != null && f1.bz != null ? f0.bz + (f1.bz - f0.bz) * alpha : (f0.bz || 0);
    const activeTeam = (brainMode === 'OUR') ? f0.ot : f0.nt;

    return {
      frameIndex: idx0,
      currentFrameFloat: currentFrameFloat,
      timestamp_s: f0.t,
      brainMode: brainMode, // 'OUR' vs 'NAIVE'
      activePossessionTeam: activeTeam, // 'H' vs 'A'
      cameraMode: cameraMode,
      playbackSpeed: playbackSpeed,
      isPlaying: isPlaying,
      ball: {
        x: bx,
        y: by,
        z: bz,
        speed: f0.spd,
        valid: f0.v === 1
      },
      event: f0.ev,
      passShieldActive: f0.sh === 1,
      falseFlicker: f0.flk === 1,
      jevActive: f0.jev === 1,
      rawFrame: f0,
      interpolated: {
        ball_x: bx,
        ball_y: by,
        ball_z: bz,
        alpha: alpha
      }
    };
  },
  captureCanvasBase64: function() {
    if (!renderer || !renderer.domElement) return null;
    try {
      return renderer.domElement.toDataURL('image/jpeg', 0.8);
    } catch (err) {
      console.error('Error capturing canvas Base64:', err);
      return null;
    }
  },
  setCameraMode: function(mode) {
    setCameraMode(mode);
  },
  setBrainMode: function(mode) {
    setBrainMode(mode);
  },
  setSpeed: function(speed) {
    setSpeed(speed);
  },
  seekToSecond: function(sec) {
    if (!window.MATCH_DATA || !window.MATCH_DATA.frames || totalFrames === 0) return;
    const targetFrame = Math.max(0, Math.min(totalFrames - 1, sec * fps));
    currentFrameFloat = targetFrame;
    if (slider) slider.value = targetFrame;
  },
  togglePlay: function() {
    togglePlay();
  },
  reloadData: function(cb) {
    reloadDataPayload(cb);
  }
};

// --- MAIN 60FPS RENDER & INTERPOLATION LOOP ---
let lastTime = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - lastTime) / 1000.0, 0.1);
  lastTime = now;

  if (!window.MATCH_DATA || !window.MATCH_DATA.frames || totalFrames === 0) {
    renderer.render(scene, camera);
    return;
  }

  if (isPlaying && totalFrames > 1) {
    currentFrameFloat += dt * fps * playbackSpeed;
    if (currentFrameFloat >= totalFrames - 1) currentFrameFloat = 0.0;
    if (slider) slider.value = currentFrameFloat;
  }

  if (isNaN(currentFrameFloat) || currentFrameFloat < 0) currentFrameFloat = 0.0;
  if (currentFrameFloat >= totalFrames - 1) {
    currentFrameFloat = isPlaying ? 0.0 : totalFrames - 1;
  }

  const idx0 = Math.max(0, Math.min(totalFrames - 1, Math.floor(currentFrameFloat)));
  const idx1 = Math.max(0, Math.min(totalFrames - 1, idx0 + 1));
  const alpha = Math.max(0.0, Math.min(1.0, currentFrameFloat - idx0));
  const f0 = window.MATCH_DATA.frames[idx0];
  const f1 = window.MATCH_DATA.frames[idx1] || f0;
  if (!f0) return;

  // Coordinate mapping: Pitch X -> Three.js X, Pitch Y -> Three.js -Z, Elevation -> Three.js Y
  const bx = f0.bx + (f1.bx - f0.bx) * alpha;
  const bz_pitch = -(f0.by + (f1.by - f0.by) * alpha);
  const by_elev = f0.bz + (f1.bz - f0.bz) * alpha;

  ballGroup.position.set(bx, by_elev, bz_pitch);
  ballShadow.position.set(bx, 0.03, bz_pitch);
  const shadowScale = Math.max(0.4, 1.1 - by_elev * 0.18);
  ballShadow.scale.set(shadowScale, shadowScale, 1);

  // Determine active possession team based on selected Brain Mode ('OUR' vs 'NAIVE')
  const activeTeam = (brainMode === 'OUR') ? f0.ot : f0.nt;
  const teamColorHex = (activeTeam === 'H') ? 0xA50044 : 0x00E5FF;
  const goldOrCyanHex = (activeTeam === 'H') ? 0xEDBB00 : 0x00E5FF;

  ballMesh.material.emissive.setHex(goldOrCyanHex);
  ballLight.color.setHex(teamColorHex);

  // Glowing Ground Possession Halo on the turf directly beneath the ball (complementing ballShadow)
  if (f0.v === 1) {
    ballGroundHalo.visible = true;
    ballGroundHalo.position.set(bx, 0.035, bz_pitch);
    ballGroundHalo.material.color.setHex(teamColorHex);
    const haloPulse = 1.0 + 0.12 * Math.sin(now * 0.007);
    ballGroundHalo.scale.set(haloPulse, haloPulse, 1);
  } else {
    ballGroundHalo.visible = false;
  }

  // Jev AI (System One) 3D Violet Neural Gyroscope with persistence (~0.45s decay)
  if (brainMode === 'OUR' && f0.jev === 1 && f0.v === 1) {
    jevVisualAlpha = 1.0;
  } else {
    jevVisualAlpha = Math.max(0.0, jevVisualAlpha - dt * 2.2);
  }

  if (jevVisualAlpha > 0.01) {
    jevPulseGroup.visible = true;
    jevRing1.rotation.x += dt * 5.0;
    jevRing1.rotation.y += dt * 3.5;
    jevRing2.rotation.y += dt * 5.0;
    jevRing2.rotation.z += dt * 2.8;
    jevGroundHex.rotation.z += dt * 2.0;
    jevRing1Mat.opacity = jevVisualAlpha * 0.9;
    jevRing2Mat.opacity = jevVisualAlpha * 0.9;
    jevGroundHexMat.opacity = jevVisualAlpha * 0.75;
    jevGroundHex.position.y = 0.04 - by_elev;
    ballLight.color.lerp(_scratchColorJev, Math.min(1.0, jevVisualAlpha * 1.2));
  } else {
    jevPulseGroup.visible = false;
  }

  // Scrub-Safe 3D Comet Motion Trail (Anti-Glitch Timeline Sampling)
  if (f0.v === 1) {
    ballTrailGroup.visible = true;
    const speedFactor = Math.min(1.0, (f0.spd || 0) / 10.0);
    const trailColorHex = (jevVisualAlpha > 0.05) ? 0xA855F7 : goldOrCyanHex;
    for (let k = 1; k <= BALL_TRAIL_COUNT; k++) {
      const trailFloat = Math.max(0.0, currentFrameFloat - k * 0.45);
      const t0 = Math.floor(trailFloat);
      const t1 = Math.min(totalFrames - 1, t0 + 1);
      const tAlpha = trailFloat - t0;
      const tf0 = window.MATCH_DATA.frames[t0];
      const tf1 = window.MATCH_DATA.frames[t1] || tf0;
      const tx = tf0.bx + (tf1.bx - tf0.bx) * tAlpha;
      const tz = -(tf0.by + (tf1.by - tf0.by) * tAlpha);
      const ty = tf0.bz + (tf1.bz - tf0.bz) * tAlpha;

      const trailNode = trailSpheres[k - 1];
      trailNode.position.set(tx, ty, tz);
      const taper = 1.0 - k / (BALL_TRAIL_COUNT + 1);
      trailNode.scale.set(taper, taper, taper);
      trailNode.material.opacity = taper * speedFactor * 0.75;
      trailNode.material.color.setHex(trailColorHex);
      trailNode.visible = (speedFactor > 0.02);
    }
  } else {
    ballTrailGroup.visible = false;
  }

  // Pass Shield animation
  shieldMesh.visible = (brainMode === 'OUR' && f0.sh === 1);
  if (shieldMesh.visible) {
    shieldMesh.rotation.y += dt * 3.2;
    shieldMesh.rotation.x += dt * 1.8;
  }

  // Update players & interpolate positions (Zero-GC using reusable _scratchP1Map)
  Object.values(playerMeshes).forEach(pm => { pm.group.visible = false; });
  for (const k in _scratchP1Map) delete _scratchP1Map[k];
  f1.p.forEach(item => { _scratchP1Map[item[0]] = item; });

  // Dynamic Owner Resolution for OUR vs NAIVE brain modes
  let activeOwnerIdx = f0.own;
  if (brainMode === 'NAIVE') {
    let minBallDist = Infinity;
    let closestPIdx = -1;
    f0.p.forEach(item => {
      const pIdx = item[0];
      let px = item[1], py = item[2];
      if (_scratchP1Map[pIdx]) {
        px += (_scratchP1Map[pIdx][1] - px) * alpha;
        py += (_scratchP1Map[pIdx][2] - py) * alpha;
      }
      const dist = Math.hypot(px - bx, -py - bz_pitch);
      if (dist < minBallDist) {
        minBallDist = dist;
        closestPIdx = pIdx;
      }
    });
    if (closestPIdx >= 0) {
      activeOwnerIdx = closestPIdx;
    }
  }

  const activeOwnerTeam = (playerMeshes[activeOwnerIdx] && brainMode === 'NAIVE') ?
    playerMeshes[activeOwnerIdx].team : activeTeam;
  const activeOwnerColorHex = (activeOwnerTeam === 'H') ? 0xEDBB00 : 0x00E5FF;

  let ownerPos = null;
  f0.p.forEach(item => {
    const pIdx = item[0];
    const pm = playerMeshes[pIdx];
    if (!pm) return;
    let px = item[1], py = item[2];
    if (_scratchP1Map[pIdx]) {
      px += (_scratchP1Map[pIdx][1] - px) * alpha;
      py += (_scratchP1Map[pIdx][2] - py) * alpha;
    }
    pm.group.position.set(px, 0, -py);
    pm.group.visible = true;

    if (pIdx === activeOwnerIdx) {
      ownerPos = pm.group.position;
      pm.ring.material.opacity = 0.95;
      pm.ring.scale.set(1.5, 1.5, 1);
    } else {
      pm.ring.material.opacity = 0.22;
      pm.ring.scale.set(1.0, 1.0, 1);
    }
  });

  // Update EA Sports FC Style 3D Active Carrier Beacon
  if (ownerPos && f0.v === 1) {
    ownerBeaconGroup.visible = true;
    ownerBeaconGroup.position.copy(ownerPos);

    // Dynamic Team Colors
    beaconRingMat.color.setHex(activeOwnerColorHex);
    beaconAuraMat.color.setHex(activeOwnerColorHex);
    beaconDiamondMat.color.setHex(activeOwnerColorHex);
    beaconDiamondMat.emissive.setHex(activeOwnerColorHex);

    // Floating diamond rotation & vertical bobbing
    beaconDiamond.rotation.y += dt * 3.5;
    beaconDiamond.position.y = 4.1 + Math.sin(now * 0.006) * 0.22;

    // Ground ring pulsing
    const ringPulse = 1.0 + 0.18 * Math.sin(now * 0.008);
    beaconRing.scale.set(ringPulse, ringPulse, 1);
  } else {
    ownerBeaconGroup.visible = false;
  }

  // Update 3D Volumetric Animated Control Tether (Energy Bead Chain + guide line)
  if (ownerPos && f0.v === 1) {
    tetherLine.visible = true;
    tetherMat.color.setHex(activeOwnerColorHex);
    const posAttr = tetherGeo.attributes.position;
    for (let i = 0; i <= 20; i++) {
      const u = i / 20.0;
      const tx = ownerPos.x + (bx - ownerPos.x) * u;
      const tz = ownerPos.z + (bz_pitch - ownerPos.z) * u;
      const ty = 0.25 + (by_elev - 0.25) * u + Math.sin(u * Math.PI) * 1.35;
      posAttr.setXYZ(i, tx, ty, tz);
    }
    posAttr.needsUpdate = true;

    // Animated volumetric energy bead chain
    tetherBeadsGroup.visible = true;
    const flowPhase = (now * 0.0025) % 1.0;
    for (let i = 0; i < TETHER_BEADS_COUNT; i++) {
      const bead = tetherBeads[i];
      bead.visible = true;
      bead.material.color.setHex(activeOwnerColorHex);
      bead.material.emissive.setHex(activeOwnerColorHex);

      const u = (i / TETHER_BEADS_COUNT + flowPhase) % 1.0;
      const tx = ownerPos.x + (bx - ownerPos.x) * u;
      const tz = ownerPos.z + (bz_pitch - ownerPos.z) * u;
      const ty = 0.25 + (by_elev - 0.25) * u + Math.sin(u * Math.PI) * 1.35;
      bead.position.set(tx, ty, tz);

      const arcPulse = 1.0 + 0.55 * Math.sin(u * Math.PI);
      bead.scale.set(arcPulse, arcPulse, arcPulse);
    }
  } else {
    tetherLine.visible = false;
    tetherBeadsGroup.visible = false;
    tetherBeads.forEach(b => { b.visible = false; });
  }

  // Shockwave effect on Touches & Turnovers
  if (f0.ev !== lastEventCode) {
    lastEventCode = f0.ev;
    if (f0.ev.startsWith('TURNOVER')) {
      shockRing.position.set(bx, 0.06, bz_pitch);
      shockRing.material.color.setHex(goldOrCyanHex);
      shockScale = 1.0; shockAlpha = 1.0;
      triggerEventToast(
        activeTeam === 'H' ? '⚡ ¡RECUPERACIÓN DEL FC BARCELONA!' : '⚡ ¡ROBO DE BALÓN DEL RIVAL!',
        activeTeam === 'H' ? 'linear-gradient(90deg,#004D98,#A50044)' : '#0284C7'
      );
    } else if (f0.ev === 'SHIELD' && brainMode === 'OUR') {
      triggerEventToast('🛡️ ESCUDO DE PASE EN TRÁNSITO (Ignora cercanía rival)', 'rgba(237,187,0,0.92)');
    } else if (f0.ev.startsWith('TOUCH')) {
      shockRing.position.set(bx, 0.06, bz_pitch);
      shockRing.material.color.setHex(0xFFFFFF);
      shockScale = 1.0; shockAlpha = 0.75;
    }
  }
  if (shockAlpha > 0.01) {
    shockScale += dt * 8.5;
    shockAlpha = Math.max(0.0, shockAlpha - dt * 1.6);
    shockRing.scale.set(shockScale, shockScale, 1);
    shockRing.material.opacity = shockAlpha;
  } else {
    shockRing.material.opacity = 0.0;
  }

  // Smooth Camera Controller (Zero-GC with scratch vectors)
  if (cameraMode === 'TV') {
    _scratchV3.set(bx * 0.45, 38, 54);
    _scratchTarget.set(bx * 0.65, 0.5, bz_pitch * 0.45);
    camera.position.lerp(_scratchV3, 0.06);
    cameraTarget.lerp(_scratchTarget, 0.08);
  } else if (cameraMode === 'DRONE') {
    _scratchV3.set(bx - 14, 15, bz_pitch + 18);
    _scratchTarget.set(bx, by_elev, bz_pitch);
    camera.position.lerp(_scratchV3, 0.08);
    cameraTarget.lerp(_scratchTarget, 0.12);
  } else if (cameraMode === 'TOP') {
    _scratchV3.set(0, 88, 0.01);
    _scratchTarget.set(0, 0, 0);
    camera.position.lerp(_scratchV3, 0.08);
    cameraTarget.lerp(_scratchTarget, 0.08);
  } else if (cameraMode === 'FREE') {
    const camX = bx + freeCamDist * Math.sin(freeCamRotY) * Math.cos(freeCamPitch);
    const camY = Math.max(2.0, freeCamDist * Math.sin(freeCamPitch));
    const camZ = bz_pitch + freeCamDist * Math.cos(freeCamRotY) * Math.cos(freeCamPitch);
    _scratchV3.set(camX, camY, camZ);
    _scratchTarget.set(bx, by_elev, bz_pitch);
    camera.position.lerp(_scratchV3, 0.12);
    cameraTarget.lerp(_scratchTarget, 0.12);
  }
  camera.lookAt(cameraTarget);

  // Update HUD Scoreboard & Narrator
  const probHomePct = (brainMode === 'OUR') ? Math.round(f0.op * 100) : (f0.nt === 'H' ? 92 : 8);
  const probAwayPct = 100 - probHomePct;
  document.getElementById('lbl-prob-home').innerText = probHomePct + '%';
  document.getElementById('lbl-prob-away').innerText = probAwayPct + '%';
  document.getElementById('tug-fill-home').style.width = probHomePct + '%';

  const mins = Math.floor(f0.t / 60);
  const secs = (f0.t % 60).toFixed(1).padStart(4, '0');
  document.getElementById('lbl-clock').innerText = `⏱️ 0${mins}:${secs} (Frame ${f0.fid})`;
  document.getElementById('stat-our-sw').innerText = f0.our_sw;
  document.getElementById('stat-naive-sw').innerText = f0.nai_sw;
  document.getElementById('stat-saved').innerText = Math.max(0, f0.nai_sw - f0.our_sw);
  document.getElementById('ball-speed-badge').innerText = `Vel. Balón: ${f0.spd.toFixed(1)} m/s | Altura 3D: ${by_elev.toFixed(1)}m`;

  const flkAlert = document.getElementById('flicker-alert');
  if (brainMode === 'NAIVE' && f0.flk === 1) {
    flkAlert.style.display = 'inline-block';
    document.getElementById('commentary-text').innerText = '⚠️ ¡ERROR EN IA COMÚN! Como el balón pasó cerca de un rival, la IA común cree erróneamente que cambió de dueño.';
  } else {
    flkAlert.style.display = 'none';
    document.getElementById('commentary-text').innerText = f0.msg;
  }

  // Update TV-Style Active Carrier HUD Card
  const carrierCard = document.getElementById('carrier-hud-card');
  if (carrierCard && window.MATCH_DATA && window.MATCH_DATA.roster) {
    const rItem = window.MATCH_DATA.roster[activeOwnerIdx];
    const pNum = rItem ? rItem.num : (f0.own >= 0 && window.MATCH_DATA.roster[f0.own] ? window.MATCH_DATA.roster[f0.own].num : 10);
    const pTeam = (rItem && brainMode === 'NAIVE') ? rItem.team : activeTeam;
    const isHome = (pTeam === 'H');

    const crestEl = document.getElementById('carrier-crest');
    const titleEl = document.getElementById('carrier-player-title');
    const teamEl = document.getElementById('carrier-team-name');
    const jerseyEl = document.getElementById('carrier-jersey-box');
    const accentEl = document.getElementById('carrier-accent-bar');
    const statusEl = document.getElementById('carrier-status-badge');
    const brainTagEl = document.getElementById('carrier-brain-tag');

    if (brainTagEl) {
      brainTagEl.innerText = (brainMode === 'OUR') ? 'NUESTRA IA' : 'IA COMÚN';
      brainTagEl.style.color = (brainMode === 'OUR') ? 'var(--barca-gold)' : '#F87171';
    }

    if (isHome) {
      carrierCard.style.borderLeftColor = 'var(--barca-gold)';
      if (crestEl) {
        crestEl.innerText = 'FCB';
        crestEl.style.background = 'conic-gradient(#004D98 0deg 180deg, #A50044 180deg 360deg)';
        crestEl.style.borderColor = 'var(--barca-gold)';
        crestEl.style.boxShadow = '0 0 12px rgba(237,187,0,0.45)';
      }
      if (teamEl) teamEl.innerText = 'FC Barcelona (Local)';
      if (jerseyEl) {
        jerseyEl.innerText = `#${pNum}`;
        jerseyEl.style.borderColor = 'var(--barca-gold)';
        jerseyEl.style.color = 'var(--barca-gold)';
      }
      if (titleEl) titleEl.innerText = `Dorsal #${pNum}${rItem && rItem.gk ? ' (Portero)' : ''}`;
      if (accentEl) accentEl.style.background = 'linear-gradient(90deg, #004D98, #A50044, #EDBB00)';
    } else {
      carrierCard.style.borderLeftColor = 'var(--rival-cyan)';
      if (crestEl) {
        crestEl.innerText = 'RIV';
        crestEl.style.background = 'linear-gradient(135deg, #0284C7, #00E5FF)';
        crestEl.style.borderColor = 'var(--rival-cyan)';
        crestEl.style.boxShadow = '0 0 12px rgba(0,229,255,0.45)';
      }
      if (teamEl) teamEl.innerText = 'Rival (Visitante)';
      if (jerseyEl) {
        jerseyEl.innerText = `#${pNum}`;
        jerseyEl.style.borderColor = 'var(--rival-cyan)';
        jerseyEl.style.color = 'var(--rival-cyan)';
      }
      if (titleEl) titleEl.innerText = `Dorsal #${pNum}${rItem && rItem.gk ? ' (Portero)' : ''}`;
      if (accentEl) accentEl.style.background = 'linear-gradient(90deg, #0284C7, #00E5FF)';
    }

    // Dynamic Role / Status Text
    if (statusEl) {
      if (f0.v === 0) {
        statusEl.innerText = 'SIN CONTROL (BALÓN FUERA)';
        statusEl.style.background = 'rgba(239, 68, 68, 0.25)';
        statusEl.style.borderColor = '#EF4444';
        statusEl.style.color = '#F87171';
        statusEl.style.boxShadow = 'none';
      } else if (brainMode === 'OUR' && (f0.jev === 1 || jevVisualAlpha > 0.05)) {
        statusEl.innerText = 'BALÓN DIVIDIDO (JEV AI)';
        statusEl.style.background = 'rgba(168, 85, 247, 0.3)';
        statusEl.style.borderColor = '#A855F7';
        statusEl.style.color = '#C084FC';
        statusEl.style.boxShadow = '0 0 10px rgba(168, 85, 247, 0.5)';
      } else if (brainMode === 'OUR' && (f0.sh === 1 || f0.ev === 'SHIELD')) {
        statusEl.innerText = 'PASE EN TRÁNSITO';
        statusEl.style.background = 'rgba(237, 187, 0, 0.25)';
        statusEl.style.borderColor = '#EDBB00';
        statusEl.style.color = '#EDBB00';
        statusEl.style.boxShadow = 'none';
      } else {
        statusEl.innerText = 'EN CONTROL';
        statusEl.style.background = 'rgba(34, 197, 94, 0.25)';
        statusEl.style.borderColor = '#22C55E';
        statusEl.style.color = '#4ADE80';
        statusEl.style.boxShadow = 'none';
      }
    }
  }

  renderer.render(scene, camera);
}

// Window resize listener
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initial boot: initialize with window.MATCH_DATA if already present in DOM
if (window.MATCH_DATA && window.MATCH_DATA.frames) {
  initMatchData(window.MATCH_DATA);
} else {
  initMatchData(null);
}

requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def get_baseline_viewer_html(template_path: Optional[str] = None) -> str:
    """Returns the baseline viewer HTML, preferring disk template if present."""
    if template_path and os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    viewer_disk_path = os.path.join("artifacts", "visualizations", "viewer.html")
    if os.path.exists(viewer_disk_path):
        with open(viewer_disk_path, "r", encoding="utf-8") as f:
            return f.read()
    return BASELINE_VIEWER_HTML


def build_html_document(payload: Dict[str, Any], template_html: Optional[str] = None) -> str:
    """Builds a portable self-contained HTML replay document by embedding the JSON payload inline."""
    base_html = template_html or get_baseline_viewer_html()
    json_compact = json.dumps(payload, separators=(",", ":"))
    inline_script = (
        '<script id="match-payload-script">\n'
        f"window.MATCH_DATA = {json_compact};\n"
        "</script>"
    )
    if '<script id="match-payload-script" src="match_payload.js"></script>' in base_html:
        return base_html.replace(
            '<script id="match-payload-script" src="match_payload.js"></script>',
            inline_script,
        )
    elif 'src="match_payload.js"' in base_html:
        import re
        return re.sub(
            r'<script[^>]*src=["\']match_payload\.js["\'][^>]*></script>',
            inline_script,
            base_html,
        )
    else:
        return base_html.replace(
            "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>",
            f"{inline_script}\n<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decoupled 3D FC Barcelona Broadcast match payload and viewer (Three.js / WebGL)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1500,
        help="Number of frames to export into the 3D replay (default: 1500 = 60s at 25Hz).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Optional path to custom frames CSV/Parquet or Metrica Home tracking table.",
    )
    parser.add_argument(
        "--players",
        type=str,
        default=None,
        help="Optional path to custom players CSV/Parquet or Metrica Away tracking table.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to scan for official competition matches (default: 'data').",
    )
    parser.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Specific match ID to load from --data-dir if multiple matches exist.",
    )
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=None,
        help="Path to trained model bundle .pkl (default: artifacts/barca_possession_mvp_v2.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("artifacts", "visualizations"),
        help="Directory to write decoupled 3D viewer and match data (default: artifacts/visualizations).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path for standalone single-file HTML replay.",
    )
    parser.add_argument(
        "--reset-viewer",
        action="store_true",
        default=False,
        help="Force overwrite artifacts/visualizations/viewer.html with baseline template.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=False,
        help="Also compile a self-contained portable HTML file (barca_3d_interactive_replay.html).",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("GENERATING FC BARCELONA 3D INTERACTIVE BROADCAST REPLAY (WEBGL / THREE.JS)")
    print("=" * 88)

    dataset, pred_df = load_match_for_3d_visualization(
        max_frames=args.max_frames,
        frames_path=args.frames,
        players_path=args.players,
        data_dir=args.data_dir,
        match_id=args.match_id,
        model_bundle_path=args.model_bundle,
    )
    payload = build_fan_friendly_3d_payload(dataset, pred_df)

    out_dir = os.path.abspath(args.output_dir)
    if args.output is not None and not args.output.endswith(os.sep) and os.path.splitext(args.output)[1]:
        out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Export Zero-CORS match_payload.js
    js_path = os.path.join(out_dir, "match_payload.js")
    json_compact = json.dumps(payload, separators=(",", ":"))
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(f"window.MATCH_DATA = {json_compact};\n")

    # 2. Export raw match_payload.json for external tooling/APIs
    json_path = os.path.join(out_dir, "match_payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 3. Hot-Edit Protection (Anti-Overwrite Guard for viewer.html)
    viewer_path = os.path.join(out_dir, "viewer.html")
    if not os.path.exists(viewer_path) or args.reset_viewer:
        viewer_html = get_baseline_viewer_html()
        with open(viewer_path, "w", encoding="utf-8") as f:
            f.write(viewer_html)
        print(f"[PASS] 1. Visual template written: '{viewer_path}'")
    else:
        print(f"[SKIP] viewer.html already exists (hot-edit protection active). Use --reset-viewer to overwrite.")

    print(f"[PASS] 2. Zero-CORS payload exported: '{js_path}' ({len(payload['frames'])} frames, {len(payload['roster'])} players)")
    print(f"[PASS] 3. Raw JSON payload exported: '{json_path}'")
    print(f"[PASS] 4. Detected {len(payload['turnovers'])} real turnovers vs {payload['frames'][-1]['nai_sw']} naive false switches.")

    # 4. Optional Standalone Bundle
    if args.standalone or args.output is not None:
        standalone_path = args.output if args.output else os.path.join(out_dir, "barca_3d_interactive_replay.html")
        current_viewer_template = None
        if os.path.exists(viewer_path):
            with open(viewer_path, "r", encoding="utf-8") as f:
                current_viewer_template = f.read()
        standalone_html = build_html_document(payload, template_html=current_viewer_template)
        with open(standalone_path, "w", encoding="utf-8") as f:
            f.write(standalone_html)
        print(f"[PASS] 5. Standalone self-contained HTML replay written to: '{standalone_path}'")

    print("=" * 88)


if __name__ == "__main__":
    main()