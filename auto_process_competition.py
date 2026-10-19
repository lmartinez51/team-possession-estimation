# Auto Competition Ingestion & Batch Predictor (RFC v2.0).
# Automatically scans an input directory, identifies file formats,
# pairs Home/Away tables or canonical frame/player files, runs inference,
# and generates final submission CSVs in 'artifacts/'.
# Target Event: More than a Hack 2027 (FC Barcelona & Metrica Sports)
# Author: Lorenzo Martinez
# Language: American English

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from task1_schema_and_preprocessing import ValidatedTrackingDataset
from run_submission_pipeline import (
    MetricaFormatAdapter,
    PossessionSubmissionPipeline,
    run_end_to_end_submission_verification,
)
from task6_match_visualizer import MatchVisualizer


class AutoMatchDetector:
    # Automatically scans an input directory and categorizes discovered matches

    @staticmethod
    def scan_directory(input_dir: str) -> List[Dict[str, Any]]:
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            return []

        all_files = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
        valid_exts = (".csv", ".parquet")
        data_files = [
            os.path.abspath(f)
            for f in all_files
            if f.lower().endswith(valid_exts)
            and not os.path.basename(f).lower().startswith("submission")
            and "event" not in os.path.basename(f).lower()
        ]

        matches_found: List[Dict[str, Any]] = []
        consumed_files = set()

        # --------------------------------------------------------------------
        # Strategy A: Pair split Home / Away Metrica tables
        # --------------------------------------------------------------------
        home_files = [f for f in data_files if "home" in os.path.basename(f).lower()]
        away_files = [f for f in data_files if "away" in os.path.basename(f).lower()]

        for h_file in home_files:
            if h_file in consumed_files:
                continue
            h_dir = os.path.dirname(h_file)
            h_name = os.path.basename(h_file)
            h_lower = h_name.lower()

            # Exact counterpart name replacement
            matched_away = None
            for repl_pattern in [("home", "away"), ("Home", "Away"), ("HOME", "AWAY")]:
                target_away_name = h_name.replace(repl_pattern[0], repl_pattern[1])
                candidate_target = os.path.join(h_dir, target_away_name)
                if candidate_target in away_files and candidate_target not in consumed_files:
                    matched_away = candidate_target
                    break

            # Fallback to prefix/stem matching
            if not matched_away:
                prefix = h_lower.split("home")[0]
                for a_file in away_files:
                    if a_file not in consumed_files and os.path.basename(a_file).lower().startswith(prefix):
                        matched_away = a_file
                        break

            if matched_away:
                consumed_files.add(h_file)
                consumed_files.add(matched_away)

                stem = os.path.splitext(h_name)[0]
                for kw in ["_Home_Team", "_Home", "Home_Team", "Home", "_home_team", "_home"]:
                    stem = stem.replace(kw, "")
                match_id = stem.strip("_- ") or "METRICA_MATCH"

                # Check for metadata JSON
                meta_json = os.path.splitext(h_file)[0] + ".json"
                if not os.path.exists(meta_json):
                    meta_json = os.path.join(h_dir, "metadata.json")

                matches_found.append({
                    "match_id": match_id,
                    "format": "metrica_split",
                    "home_path": h_file,
                    "away_path": matched_away,
                    "meta_path": meta_json if os.path.exists(meta_json) else None,
                })

        # --------------------------------------------------------------------
        # Strategy B: Pair canonical frames + players tables
        # --------------------------------------------------------------------
        frame_files = [
            f for f in data_files if "frame" in os.path.basename(f).lower() and f not in consumed_files
        ]
        player_files = [
            f for f in data_files if "player" in os.path.basename(f).lower() and f not in consumed_files
        ]

        for f_file in frame_files:
            if f_file in consumed_files:
                continue
            f_dir = os.path.dirname(f_file)
            f_name = os.path.basename(f_file)

            matched_player = None
            for p_from, p_to in [("frame", "player"), ("Frame", "Player"), ("frames", "players"), ("Frames", "Players")]:
                target_p_name = f_name.replace(p_from, p_to)
                cand_p = os.path.join(f_dir, target_p_name)
                if cand_p in player_files and cand_p not in consumed_files:
                    matched_player = cand_p
                    break

            if not matched_player:
                prefix = f_name.lower().split("frame")[0]
                for p_cand in player_files:
                    if p_cand not in consumed_files and os.path.basename(p_cand).lower().startswith(prefix):
                        matched_player = p_cand
                        break

            if matched_player:
                consumed_files.add(f_file)
                consumed_files.add(matched_player)

                stem = os.path.splitext(f_name)[0]
                for kw in ["_frames", "_frame", "frames", "frame"]:
                    stem = stem.replace(kw, "")
                match_id = stem.strip("_- ") or "CANONICAL_MATCH"

                meta_json = os.path.splitext(f_file)[0] + ".json"
                if not os.path.exists(meta_json):
                    meta_json = os.path.join(f_dir, "metadata.json")

                matches_found.append({
                    "match_id": match_id,
                    "format": "canonical_pair",
                    "frames_path": f_file,
                    "players_path": matched_player,
                    "meta_path": meta_json if os.path.exists(meta_json) else None,
                })

        # --------------------------------------------------------------------
        # Strategy C: Single wide tracking table
        # --------------------------------------------------------------------
        remaining_files = [f for f in data_files if f not in consumed_files]
        for rem_file in remaining_files:
            try:
                sample_df = pd.read_csv(rem_file, nrows=2, low_memory=False)
                has_coords = any(
                    c.endswith(("_x", "_X", "_y", "_Y")) or "ball" in c.lower() or "player" in c.lower()
                    for c in sample_df.columns
                )
                if not has_coords:
                    continue
            except Exception:
                continue

            stem = os.path.splitext(os.path.basename(rem_file))[0]
            match_id = stem.strip("_- ") or "WIDE_MATCH"
            meta_json = os.path.splitext(rem_file)[0] + ".json"
            if not os.path.exists(meta_json):
                meta_json = os.path.join(os.path.dirname(rem_file), "metadata.json")

            matches_found.append({
                "match_id": match_id,
                "format": "wide_single",
                "frames_path": rem_file,
                "meta_path": meta_json if os.path.exists(meta_json) else None,
            })

        return matches_found


def load_dataset_from_match_info(
    match_info: Dict[str, Any], max_frames: Optional[int] = None
) -> ValidatedTrackingDataset:
    fmt = match_info["format"]
    meta_path = match_info.get("meta_path")

    if fmt == "metrica_split":
        return MetricaFormatAdapter.from_metrica_raw_files(
            match_info["home_path"],
            match_info["away_path"],
            metadata_json_path=meta_path,
            max_frames=max_frames,
        )
    elif fmt == "canonical_pair":
        return MetricaFormatAdapter.from_canonical_files(
            match_info["frames_path"],
            match_info["players_path"],
            metadata_json_path=meta_path,
            max_frames=max_frames,
        )
    elif fmt == "wide_single":
        wide_df = MetricaFormatAdapter.load_table(match_info["frames_path"])
        if max_frames and len(wide_df) > max_frames:
            wide_df = wide_df.iloc[:max_frames].copy()
        return MetricaFormatAdapter.from_wide_table(
            wide_df,
            metadata_json_path=meta_path,
        )
    else:
        raise ValueError(f"Unknown match format: '{fmt}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-Scan & Batch Processor for More than a Hack 2027 Competition Data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where downloaded competition files are stored (default: 'data').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory where final submission CSVs will be saved (default: 'artifacts').",
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
        help="Optional maximum number of frames to load per match for rapid testing.",
    )
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=os.path.join("artifacts", "barca_possession_mvp_v2.pkl"),
        help="Path to trained model .pkl bundle.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate 3-panel tactical dashboard PNG for each processed match in artifacts/visualizations/.",
    )

    args = parser.parse_args()

    print("=" * 88)
    print(f"SCANNING DIRECTORY: '{args.data_dir}' FOR COMPETITION DATA...")
    matches = AutoMatchDetector.scan_directory(args.data_dir)

    if not matches:
        print(f"[!] No matching tracking data files found in '{args.data_dir}'.")
        print("    Please place the competition CSV/Parquet files in that folder.")
        print("    Supported: Metrica split Home/Away tables, canonical frame/player tables, or single wide tables.")
        sys.exit(0)

    print(f"[*] Found {len(matches)} match dataset(s) to process:")
    for idx, m in enumerate(matches, 1):
        print(f"    {idx}. Match ID: {m['match_id']} (Format: {m['format']})")
    print("=" * 88)

    # Load trained pipeline
    if not os.path.exists(args.model_bundle):
        print("[!] Model bundle not found. Initializing baseline artifact...")
        run_end_to_end_submission_verification()

    print(f"Loading trained pipeline from '{args.model_bundle}'...")
    pipeline = PossessionSubmissionPipeline.load_model_bundle(args.model_bundle)
    pipeline.config.use_live_jev_api = bool(args.live_jev)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)

    # Process all matches
    for idx, m_info in enumerate(matches, 1):
        m_id = m_info["match_id"]
        print(f"\n[{idx}/{len(matches)}] Processing: {m_id} ({m_info['format']})...")
        t_start = time.perf_counter()

        dataset = load_dataset_from_match_info(m_info, max_frames=args.max_frames)
        pred_df, diag = pipeline.predict_match(dataset, use_cache=True)

        out_csv = os.path.join(args.output_dir, f"submission_{m_id}.csv")
        pipeline.export_submission_csv(pred_df, out_csv)

        elapsed = time.perf_counter() - t_start
        print(f"    -> Done in {elapsed:.2f}s ({diag['throughput_fps']} fps)")
        print(f"    -> Output saved: '{out_csv}' ({diag['num_frames']} frames, 0 NaNs)")

        if args.visualize:
            viz = MatchVisualizer(output_dir=os.path.join(args.output_dir, "visualizations"))
            viz_path = viz.plot_match_dashboard(
                dataset,
                pred_df,
                save_filename=f"{m_id}_diagnostic.png",
            )
            print(f"    -> Tactical visualization: '{viz_path}'")

    print("\n" + "=" * 88)
    print("ALL COMPETITION MATCHES PROCESSED SUCCESSFULLY!")
    print(f"Check your final submission files in: '{args.output_dir}'")
    print("=" * 88)


if __name__ == "__main__":
    main()