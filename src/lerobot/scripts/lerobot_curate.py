#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""lerobot-curate: CLI for Robotic Dataset Observability, Trajectory Anomaly Detection,
Interactive Trimming, and Clean Sub-Dataset Export.

Usage Examples:
    # Run full trajectory similarity & outlier detection:
    lerobot-curate analyze --repo_id hubnemo/cube_out_of_box_dataset --output_dir ./curation_out

    # Include CPU visual feature extraction (DINOv2 / MobileNet):
    lerobot-curate analyze --repo_id hubnemo/cube_out_of_box_dataset --visual --output_dir ./curation_out

    # Launch interactive Web UI for video scrubbing, trimming, and curation:
    lerobot-curate ui --repo_id hubnemo/cube_out_of_box_dataset --port 8080

    # Export clean dataset based on curated manifest:
    lerobot-curate export --repo_id hubnemo/cube_out_of_box_dataset \
        --manifest ./curation_out/curation_manifest.json \
        --output_root ./clean_dataset

    # Print summary of a manifest:
    lerobot-curate summary --manifest ./curation_out/curation_manifest.json
"""

import argparse
import os
import sys
from pathlib import Path

# Force CPU execution to prevent interference with ongoing GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lerobot.datasets.curation import (
    CurationManifest,
    DatasetCurator,
    export_full_analysis,
)
from lerobot.datasets.curation.rerun_visualizer import visualize_in_rerun
from lerobot.datasets.curation.server import start_curation_server


def print_banner() -> None:
    print(
        r"""
  _       _____       _           _        _____                 _
 | |     |  __ \     | |         | |      / ____|               | |
 | |     | |__) |___ | |__   ___ | |_ ___| |    _   _ _ __ __ _| |_ ___  _ __
 | |     |  _  // _ \| '_ \ / _ \| __/ __| |   | | | | '__/ _` | __/ _ \| '__|
 | |____ | | \ \ (_) | |_) | (_) | |_\__ \ |___| |_| | | | (_| | || (_) | |
 |______||_|  \_\___/|_.__/ \___/ \__|___/\_____\__,_|_|  \__,_|\__\___/|_|
                     Observability & Curation Toolkit
"""
    )


def parse_group_specs(groups_str: str | None) -> list[tuple[int, int, str]] | None:
    if not groups_str:
        return None
    specs = []
    for item in groups_str.split(","):
        item = item.strip()
        if not item:
            continue
        range_part, name = item.split(":", 1)
        s_str, e_str = range_part.split("-", 1)
        specs.append((int(s_str.strip()), int(e_str.strip()), name.strip()))
    return specs


def cmd_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "curation_manifest.json"

    print(f"📊 Loading and analyzing dataset: {args.repo_id} (CPU mode)...")
    curator = DatasetCurator(repo_id=args.repo_id, root=args.root)

    manifest = curator.analyze(
        extract_visual=args.visual,
        visual_backbone=args.visual_backbone,
        sample_keyframes=args.sample_keyframes,
        dtw_window=args.dtw_window,
        outlier_percentile=args.outlier_percentile,
        n_clusters=args.n_clusters,
    )

    manifest.save_json(manifest_path)
    split_path = output_dir / "clean_split.json"
    curator.export_split_manifest(manifest, split_path)

    group_specs = parse_group_specs(args.groups)
    if args.plot or group_specs is not None:
        print("📈 Generating high-resolution analysis plots and group comparison report...")
        export_full_analysis(curator, manifest, output_dir, group_specs=group_specs)

    # Print summary report
    print_manifest_summary(manifest)
    print(f"\n✅ Curation Manifest saved to: {manifest_path}")
    print(f"✅ Split manifest saved to:     {split_path}")
    if args.plot or group_specs is not None:
        print(f"✅ Analysis plots & report:    {output_dir}")
    print(
        f"Run `lerobot-curate ui --repo_id {args.repo_id} --manifest {manifest_path}` to review and trim in browser!"
    )


def print_manifest_summary(manifest: CurationManifest) -> None:
    s = manifest.summary()
    print("\n" + "=" * 75)
    print(f" Dataset Curation Summary: {s['repo_id']}")
    print("=" * 75)
    print(f" Total Episodes:        {s['total_episodes']}")
    print(f" Kept Episodes:         {s['kept_episodes']} ({s['retention_rate_pct']}%)")
    print(f" Review Needed:         {s['review_episodes']}")
    print(f" Dropped Episodes:      {s['dropped_episodes']}")
    print(f" Frames Saved (Trim):   {s['frames_saved_by_trimming']}")
    print(f" Central Medoid Ep:     {manifest.metadata.get('medoid_episode_index')}")

    dups = manifest.metadata.get("detected_duplicates", [])
    if dups:
        print(f" Duplicate Pairs:       {len(dups)} pairs detected (e.g. {dups[:3]})")

    print("\n Top Outlier Episodes (Highest Anomaly Score):")
    print("-" * 75)
    print(
        f" {'Ep':<4} | {'Score':<6} | {'Status':<6} | {'DTW':<5} | {'Jerk':<8} | {'Suggested Trim':<14} | Reasons"
    )
    print("-" * 75)

    sorted_items = sorted(manifest.episodes.values(), key=lambda x: x.anomaly_score, reverse=True)
    for item in sorted_items[:8]:
        reasons_str = "; ".join(item.anomaly_reasons) if item.anomaly_reasons else "Nominal"
        trim_str = f"{item.trim_start} -> {item.trim_end}"
        smoothness = item.kinematics.get("smoothness_score", 0.0)
        print(
            f" {item.episode_index:<4} | {item.anomaly_score:<6.1f} | {item.status.upper():<6} | "
            f"{item.proprio_anomaly_score:<5.2f} | {smoothness:<8.2f} | {trim_str:<14} | {reasons_str[:30]}"
        )
    print("-" * 75)


def cmd_ui(args: argparse.Namespace) -> None:
    curator = DatasetCurator(repo_id=args.repo_id, root=args.root)

    manifest_path = Path(args.manifest) if args.manifest else Path("./curation_manifest.json")
    if manifest_path.exists():
        print(f"Loading existing manifest from {manifest_path}...")
        manifest = CurationManifest.load_json(manifest_path)
    else:
        print("No manifest provided/found. Running initial dataset analysis...")
        manifest = curator.analyze(extract_visual=False)
        manifest.save_json(manifest_path)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    start_curation_server(
        curator=curator,
        manifest=manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        port=args.port,
        open_browser=args.open_browser,
    )


def cmd_export(args: argparse.Namespace) -> None:
    if not args.manifest or not Path(args.manifest).exists():
        print(f"Error: Manifest file '{args.manifest}' does not exist.")
        sys.exit(1)

    manifest = CurationManifest.load_json(args.manifest)
    curator = DatasetCurator(repo_id=args.repo_id, root=args.root)

    out_repo = args.output_repo_id if args.output_repo_id else f"{args.repo_id}_curated"
    out_path = curator.export_clean_dataset(
        manifest=manifest,
        output_repo_id=out_repo,
        output_root=args.output_root,
        slice_frames=args.slice_frames,
        push_to_hub=args.push_to_hub,
    )
    print(f"🎉 Clean curated dataset successfully exported to: {out_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    if not args.manifest or not Path(args.manifest).exists():
        print(f"Error: Manifest file '{args.manifest}' does not exist.")
        sys.exit(1)

    manifest = CurationManifest.load_json(args.manifest)
    print_manifest_summary(manifest)


def cmd_rerun(args: argparse.Namespace) -> None:
    manifest = CurationManifest.load_json(args.manifest)
    curator = DatasetCurator(repo_id=args.repo_id, root=args.root)
    visualize_in_rerun(
        curator=curator,
        manifest=manifest,
        episode_index=args.episode_index,
        compare_with_medoid=not args.no_medoid,
        save_path=args.save_rrd,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-curate",
        description="Dataset Observability, Trajectory Anomaly Detection, Trimming & Curation Tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze trajectory similarity, kinematics, and visual anomalies"
    )
    p_analyze.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    p_analyze.add_argument("--root", type=str, default=None, help="Local dataset root directory")
    p_analyze.add_argument(
        "--output_dir", type=str, default="./curation_results", help="Directory to save manifest"
    )
    p_analyze.add_argument(
        "--visual", action="store_true", help="Extract visual features and detect visual outliers"
    )
    p_analyze.add_argument(
        "--visual_backbone", type=str, default="auto", choices=["auto", "mobilenet", "dinov2", "stats"]
    )
    p_analyze.add_argument(
        "--sample_keyframes", type=int, default=5, help="Number of keyframes to evaluate per episode"
    )
    p_analyze.add_argument("--dtw_window", type=int, default=20, help="Sakoe-Chiba constraint window")
    p_analyze.add_argument(
        "--outlier_percentile", type=float, default=85.0, help="Percentile threshold to flag outliers"
    )
    p_analyze.add_argument("--n_clusters", type=int, default=3, help="Number of behavioral clusters")
    p_analyze.add_argument(
        "--plot", action="store_true", help="Generate observability and analysis plots (PNG)"
    )
    p_analyze.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Cohort group ranges, e.g. '0-20:Anton (V1-Setup),20-40:Kumpel (V1-Setup),40-100:Anton (Variations)'",
    )

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive browser-based curation & trimming interface")
    p_ui.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    p_ui.add_argument("--root", type=str, default=None, help="Local dataset root directory")
    p_ui.add_argument("--manifest", type=str, default=None, help="Existing curation manifest path")
    p_ui.add_argument("--output_dir", type=str, default=None, help="Output directory for exports")
    p_ui.add_argument("--port", type=int, default=8080, help="HTTP server port")
    p_ui.add_argument("--open_browser", action="store_true", help="Open browser automatically")

    # export
    p_export = subparsers.add_parser("export", help="Export clean subset or trimmed LeRobotDataset")
    p_export.add_argument("--repo_id", type=str, required=True, help="Source dataset repo id")
    p_export.add_argument("--root", type=str, default=None, help="Local source dataset root directory")
    p_export.add_argument("--manifest", type=str, required=True, help="Path to curation manifest JSON")
    p_export.add_argument(
        "--output_root", type=str, required=True, help="Destination directory for clean dataset"
    )
    p_export.add_argument("--output_repo_id", type=str, default=None, help="Repo id for new dataset")
    p_export.add_argument(
        "--slice_frames", action="store_true", default=True, help="Apply frame-level trimming"
    )
    p_export.add_argument(
        "--no_slice",
        action="store_false",
        dest="slice_frames",
        help="Keep full episodes without frame trimming",
    )
    p_export.add_argument(
        "--push_to_hub", action="store_true", help="Push exported dataset to Hugging Face Hub"
    )

    # summary
    p_summary = subparsers.add_parser("summary", help="Print summary of a curation manifest")
    p_summary.add_argument("--manifest", type=str, required=True, help="Path to curation manifest JSON")

    # rerun
    p_rerun = subparsers.add_parser("rerun", help="Visualize outlier vs medoid in Rerun")
    p_rerun.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    p_rerun.add_argument("--root", type=str, default=None, help="Local dataset root directory")
    p_rerun.add_argument("--manifest", type=str, required=True, help="Path to curation manifest JSON")
    p_rerun.add_argument("--episode_index", type=int, required=True, help="Episode index to visualize")
    p_rerun.add_argument(
        "--no_medoid", action="store_true", help="Do not overlay medoid reference trajectory"
    )
    p_rerun.add_argument(
        "--save_rrd", type=str, default=None, help="Save to .rrd file instead of spawning viewer"
    )

    args = parser.parse_args()

    print_banner()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "ui":
        cmd_ui(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "rerun":
        cmd_rerun(args)


if __name__ == "__main__":
    main()
