# Task 6: Match Visualizer & Diagnostic Coordinate Plotter (RFC v2.0 Section 3 P0).
# Generates 2D tactical pitch plots and multi-panel temporal probability dashboards
# to visually inspect possession switches, anchor bridges, and Jev refiner decisions.
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

import os
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task1_schema_and_preprocessing import ValidatedTrackingDataset, TeamID
from run_submission_pipeline import PossessionSubmissionPipeline
from task5_evaluation_and_ablation import generate_challenging_ablation_benchmark


class MatchVisualizer:
    def __init__(self, output_dir: str = "artifacts/visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_match_dashboard(
        self,
        dataset: ValidatedTrackingDataset,
        pred_df: pd.DataFrame,
        target_frame_id: Optional[int] = None,
        time_window_s: Optional[Tuple[float, float]] = None,
        save_filename: str = "match_diagnostic_dashboard.png",
    ) -> str:
        # Creates a 3-panel professional dashboard:
        # Panel 1: 2D Pitch View at a specific critical frame (e.g., turnover event).
        # Panel 2: Temporal Posterior Probability Curve vs Ground Truth over Match Duration.
        # Panel 3: Kinematic Proxies, Bridge Agreement, and Jev Refiner Activations.
        if pred_df.empty:
            raise ValueError("Input pred_df cannot be empty for dashboard visualization.")

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.0, 0.8])

        # --------------------------------------------------------------------
        # PANEL 1: 2D Tactical Pitch View
        # --------------------------------------------------------------------
        ax_pitch = fig.add_subplot(gs[0])
        pitch_l = float(dataset.metadata.pitch_length_m) if dataset and dataset.metadata else 105.0
        pitch_w = float(dataset.metadata.pitch_width_m) if dataset and dataset.metadata else 68.0
        self._draw_soccer_pitch(ax_pitch, pitch_l, pitch_w)

        # Select target frame (default to first detected turnover or midpoint)
        ball_valid_mask = (
            pred_df["ball_is_valid"].fillna(False).astype(bool)
            if "ball_is_valid" in pred_df.columns
            else pd.Series(True, index=pred_df.index)
        )
        valid_sub = pred_df[ball_valid_mask]

        if target_frame_id is not None:
            target_fid = int(target_frame_id)
        elif not valid_sub.empty and "submission_possession_team" in valid_sub.columns:
            # Find first possession switch
            teams = valid_sub["submission_possession_team"].astype(str).to_numpy()
            fids = valid_sub["frame_id"].to_numpy(dtype=int)
            switches = np.where(teams[1:] != teams[:-1])[0]
            target_fid = int(fids[switches[0] + 1]) if len(switches) > 0 else int(fids[len(fids) // 2])
        elif not valid_sub.empty:
            target_fid = int(valid_sub["frame_id"].iloc[len(valid_sub) // 2])
        else:
            target_fid = int(pred_df["frame_id"].iloc[0])

        f_row = pred_df[pred_df["frame_id"] == target_fid]
        if f_row.empty:
            target_fid = int(pred_df["frame_id"].iloc[0])
            f_row = pred_df[pred_df["frame_id"] == target_fid]

        # Robust ball coordinate extraction
        bx, by = np.nan, np.nan
        if not f_row.empty:
            fr = f_row.iloc[0]
            bx = float(fr["ball_x"]) if "ball_x" in fr and pd.notna(fr["ball_x"]) else np.nan
            by = float(fr["ball_y"]) if "ball_y" in fr and pd.notna(fr["ball_y"]) else np.nan

        # Fallback to dataset.frame_df if ball coordinates were not retained in pred_df
        if (np.isnan(bx) or np.isnan(by)) and hasattr(dataset, "frame_df") and not dataset.frame_df.empty:
            f_raw = dataset.frame_df[dataset.frame_df["frame_id"] == target_fid]
            if not f_raw.empty:
                bx = float(f_raw.iloc[0].get("ball_x", np.nan))
                by = float(f_raw.iloc[0].get("ball_y", np.nan))

        if np.isfinite(bx) and np.isfinite(by):
            ax_pitch.scatter([bx], [by], color="black", s=140, zorder=6, edgecolors="yellow", linewidths=1.5)

        # Plot player tracking coordinates
        p_tracks = (
            dataset.player_tracks_df[dataset.player_tracks_df["frame_id"] == target_fid]
            if hasattr(dataset, "player_tracks_df") and not dataset.player_tracks_df.empty
            else pd.DataFrame()
        )

        if not p_tracks.empty:
            for _, pr in p_tracks.iterrows():
                px_val = pr.get("x", np.nan)
                py_val = pr.get("y", np.nan)
                if pd.isna(px_val) or pd.isna(py_val):
                    continue

                pid_str = str(pr.get("player_id", ""))
                p_meta = dataset.players.get(pid_str) if hasattr(dataset, "players") else None
                team_val = p_meta.team_id if p_meta is not None else pr.get("team_id", None)

                if team_val == TeamID.HOME or (isinstance(team_val, str) and team_val == TeamID.HOME.value):
                    color = "#004D98"  # Barça Blau
                elif team_val == TeamID.AWAY or (isinstance(team_val, str) and team_val == TeamID.AWAY.value):
                    color = "#A50044"  # Barça Grana / Away
                else:
                    color = "#7F8C8D"  # Neutral / Referee

                ax_pitch.scatter([px_val], [py_val], color=color, s=85, edgecolors="white", linewidths=1.2, zorder=4)
                ax_pitch.text(px_val, py_val + 1.2, pid_str, fontsize=8, ha="center", color="black", fontweight="bold")

        t_cur = float(f_row["timestamp_s"].iloc[0]) if (not f_row.empty and "timestamp_s" in f_row.columns) else 0.0
        poss_cur = str(f_row["submission_possession_team"].iloc[0]) if (not f_row.empty and "submission_possession_team" in f_row.columns) else "N/A"
        latent_cur = str(f_row["latent_possession_state"].iloc[0]) if (not f_row.empty and "latent_possession_state" in f_row.columns) else "N/A"
        match_id_str = str(dataset.metadata.match_id) if hasattr(dataset, "metadata") else "Match"

        ax_pitch.set_title(
            f"Match: {match_id_str} | Frame: {target_fid} (t = {t_cur:.2f}s) | Predicted: {poss_cur} (State: {latent_cur})",
            fontsize=12,
            fontweight="bold",
        )

        home_handle = mlines.Line2D([], [], color="#004D98", marker="o", linestyle="None", markersize=8, label="Home Player")
        away_handle = mlines.Line2D([], [], color="#A50044", marker="o", linestyle="None", markersize=8, label="Away Player")
        ball_handle = mlines.Line2D([], [], color="black", markeredgecolor="yellow", marker="o", linestyle="None", markersize=9, label="Ball")
        ax_pitch.legend(handles=[home_handle, away_handle, ball_handle], loc="upper right", framealpha=0.9)

        # --------------------------------------------------------------------
        # PANEL 2: Temporal Posterior Probability Curve vs Ground Truth
        # --------------------------------------------------------------------
        ax_prob = fig.add_subplot(gs[1])
        valid_eval = pred_df["eval_mask"].fillna(True).astype(bool) if "eval_mask" in pred_df.columns else pd.Series(True, index=pred_df.index)
        v_df = pred_df[valid_eval & ball_valid_mask].sort_values("timestamp_s").copy()

        if time_window_s is not None:
            t_min, t_max = time_window_s
            v_df = v_df[(v_df["timestamp_s"] >= t_min) & (v_df["timestamp_s"] <= t_max)]

        if not v_df.empty:
            ts = v_df["timestamp_s"].to_numpy(dtype=float)

            # Resolve Home Posterior Probability
            if "calibrated_posterior_home_prob" in v_df.columns:
                p_home = v_df["calibrated_posterior_home_prob"].to_numpy(dtype=float)
            elif "decoded_home_prob" in v_df.columns:
                p_home = v_df["decoded_home_prob"].to_numpy(dtype=float)
            elif "submission_possession_team" in v_df.columns:
                p_home = (v_df["submission_possession_team"].astype(str).to_numpy() == TeamID.HOME.value).astype(float)
            else:
                p_home = np.full(len(v_df), 0.5)

            ax_prob.plot(ts, p_home, color="#004D98", lw=2.2, label="Calibrated Posterior P(Home)")
            ax_prob.fill_between(ts, 0.5, p_home, where=(p_home >= 0.5), color="#004D98", alpha=0.15, label="Home Possession")
            ax_prob.fill_between(ts, 0.5, p_home, where=(p_home < 0.5), color="#A50044", alpha=0.15, label="Away Possession")
            ax_prob.axhline(0.5, color="black", linestyle=":", alpha=0.6)

            # Plot Ground Truth if available (competition benchmark evaluation)
            if "gt_possession_team" in v_df.columns:
                gt_home = (v_df["gt_possession_team"].astype(str).to_numpy() == TeamID.HOME.value).astype(float)
                ax_prob.step(ts, gt_home, color="gray", lw=1.5, where="mid", linestyle="--", label="Ground Truth (1=Home, 0=Away)")

            # Highlight candidate contact anchors
            if "is_candidate_anchor" in v_df.columns:
                anchor_mask = v_df["is_candidate_anchor"].fillna(False).astype(bool)
                anchors = v_df[anchor_mask]
            elif "candidate_contact_confidence" in v_df.columns:
                anchor_mask = v_df["candidate_contact_confidence"].fillna(0.0) >= 0.5
                anchors = v_df[anchor_mask]
            else:
                anchors = pd.DataFrame()

            if not anchors.empty:
                anc_ts = anchors["timestamp_s"].to_numpy(dtype=float)
                anc_p = p_home[anchor_mask.to_numpy()]
                ax_prob.scatter(
                    anc_ts,
                    anc_p,
                    color="#EDBB00",
                    s=60,
                    zorder=5,
                    edgecolors="black",
                    label="Candidate Contact Anchor",
                )

            # Indicate target frame with a vertical marker
            if t_cur >= ts[0] and t_cur <= ts[-1]:
                ax_prob.axvline(t_cur, color="crimson", linestyle="-", alpha=0.7, lw=1.5, label=f"Selected Frame (t={t_cur:.2f}s)")

        ax_prob.set_ylabel("Possession Probability", fontsize=10, fontweight="bold")
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.set_title("Sequence-Level Posterior Probability & Contact Anchors (Soft Viterbi)", fontsize=11, fontweight="bold")
        ax_prob.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax_prob.grid(True, alpha=0.3)

        # --------------------------------------------------------------------
        # PANEL 3: Kinematic Proxies, Bounded Bridges & Jev Refiner Activations
        # --------------------------------------------------------------------
        ax_aux = fig.add_subplot(gs[2])
        if not v_df.empty:
            ts = v_df["timestamp_s"].to_numpy(dtype=float)
            has_signals = False

            if "ball_vel_change_proxy_mps" in v_df.columns:
                dv_proxy = v_df["ball_vel_change_proxy_mps"].fillna(0.0).to_numpy(dtype=float)
                ax_aux.plot(ts, np.clip(dv_proxy / 10.0, 0.0, 1.0), color="#FF5733", lw=1.5, label="Velocity-Change Proxy I_b (scaled)")
                has_signals = True

            if "unreliable_proximity_likelihood" in v_df.columns:
                unrel = v_df["unreliable_proximity_likelihood"].fillna(0.0).to_numpy(dtype=float)
                ax_aux.plot(ts, unrel, color="#900C3F", lw=1.5, linestyle="-.", label="Unreliable Proximity (Aerial/Pass-By)")
                has_signals = True

            if "bridge_same_team_agreement" in v_df.columns:
                bridge_agree = v_df["bridge_same_team_agreement"].fillna(0.0).to_numpy(dtype=float)
                ax_aux.plot(ts, bridge_agree, color="#2E86C1", lw=1.5, label="Bridge Same-Team Agreement")
                has_signals = True

            if "jev_refined_flag" in v_df.columns:
                jev_act = v_df[v_df["jev_refined_flag"].fillna(False).astype(bool)]
                if not jev_act.empty:
                    ax_aux.scatter(
                        jev_act["timestamp_s"],
                        np.ones(len(jev_act)) * 0.95,
                        color="purple",
                        marker="v",
                        s=50,
                        zorder=6,
                        label="Jev (System One) Refined Event",
                    )
                    has_signals = True

            if not has_signals and "diagnostic_confidence" in v_df.columns:
                diag_conf = v_df["diagnostic_confidence"].fillna(1.0).to_numpy(dtype=float)
                ax_aux.plot(ts, diag_conf, color="#2E86C1", lw=1.5, label="Decoder Diagnostic Confidence")

            if t_cur >= ts[0] and t_cur <= ts[-1]:
                ax_aux.axvline(t_cur, color="crimson", linestyle="-", alpha=0.7, lw=1.5)

        ax_aux.set_xlabel("Match Timestamp (seconds)", fontsize=10, fontweight="bold")
        ax_aux.set_ylabel("Normalized Metrics", fontsize=10, fontweight="bold")
        ax_aux.set_ylim(-0.05, 1.05)
        ax_aux.set_title("Kinematic Discontinuities, Bounded Bridges & Jev Sparse Activations", fontsize=11, fontweight="bold")
        ax_aux.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_aux.grid(True, alpha=0.3)

        out_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _draw_soccer_pitch(ax: plt.Axes, length: float, width: float) -> None:
        # Draws a standard soccer pitch representation in Matplotlib with canonical metric coordinates
        pitch_color = "#2D682A"
        line_color = "white"

        ax.set_facecolor(pitch_color)
        ax.set_xlim(-length / 2 - 4, length / 2 + 4)
        ax.set_ylim(-width / 2 - 4, width / 2 + 4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        hl = length / 2.0
        hw = width / 2.0

        # Outer boundary
        ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color=line_color, lw=2.0)
        # Halfway line
        ax.plot([0, 0], [-hw, hw], color=line_color, lw=2.0)
        # Center circle & spot
        center_circle = plt.Circle((0, 0), 9.15, color=line_color, fill=False, lw=1.8)
        ax.add_patch(center_circle)
        ax.scatter([0], [0], color=line_color, s=25)

        # Penalty boxes (16.5m depth, 40.32m width)
        box_w, box_l = 40.32, 16.5
        ax.plot([-hl, -hl + box_l, -hl + box_l, -hl], [-box_w / 2, -box_w / 2, box_w / 2, box_w / 2], color=line_color, lw=1.8)
        ax.plot([hl, hl - box_l, hl - box_l, hl], [-box_w / 2, -box_w / 2, box_w / 2, box_w / 2], color=line_color, lw=1.8)

        # Goal boxes (5.5m depth, 18.32m width)
        gbox_w, gbox_l = 18.32, 5.5
        ax.plot([-hl, -hl + gbox_l, -hl + gbox_l, -hl], [-gbox_w / 2, -gbox_w / 2, gbox_w / 2, gbox_w / 2], color=line_color, lw=1.5)
        ax.plot([hl, hl - gbox_l, hl - gbox_l, hl], [-gbox_w / 2, -gbox_w / 2, gbox_w / 2, gbox_w / 2], color=line_color, lw=1.5)

        # Penalty spots (11.0m from goal lines)
        ax.scatter([-hl + 11.0, hl - 11.0], [0, 0], color=line_color, s=20)


def run_task6_verification() -> None:
    print("=" * 72)
    print("RUNNING TASK 6 MATCH VISUALIZER & DIAGNOSTIC PLOTTER VERIFICATION")
    print("=" * 72)

    model_path = "artifacts/barca_possession_mvp_v2.pkl"
    matches = generate_challenging_ablation_benchmark(n_matches=2, sampling_rate_hz=25.0)
    test_match = matches[0]

    if os.path.exists(model_path):
        print(f"[INFO] Loading pre-trained submission pipeline bundle from '{model_path}'...")
        pipeline = PossessionSubmissionPipeline.load_model_bundle(model_path)
    else:
        print("[INFO] No pre-trained bundle found. Fitting pipeline with OOF validation across synthetic matches...")
        pipeline = PossessionSubmissionPipeline()
        pipeline.fit_with_oof_validation(matches, force_recompute_features=False)

    # Predict
    pred_df, diag = pipeline.predict_match(test_match, use_cache=True)
    print(f"[INFO] Prediction complete for match '{diag['match_id']}': {diag['num_frames']} frames in {diag['runtime_s']}s.")

    viz = MatchVisualizer(output_dir="artifacts/visualizations")
    img_path = viz.plot_match_dashboard(test_match, pred_df, save_filename="test_match_diagnostic.png")

    assert os.path.exists(img_path), "Visualization dashboard image was not generated."
    file_size_kb = os.path.getsize(img_path) / 1024.0
    print(f"[PASS] Match diagnostic visualization dashboard successfully generated and saved at '{img_path}' ({file_size_kb:.1f} KB).")
    print("=" * 72)


if __name__ == "__main__":
    run_task6_verification()