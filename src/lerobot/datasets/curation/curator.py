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

"""DatasetCurator: High-level engine for dataset observability, trajectory anomaly detection,
interactive trimming, and clean dataset export.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.curation.anomaly import (
    detect_duplicate_episodes,
    rank_anomalies,
)
from lerobot.datasets.curation.dtw import (
    compute_dtw_anomaly_scores,
    compute_pairwise_dtw_matrix,
    find_medoid_trajectory,
)
from lerobot.datasets.curation.kinematics import compute_kinematics
from lerobot.datasets.curation.manifest import CurationManifest, EpisodeCurationItem
from lerobot.datasets.curation.visual import (
    EpisodeVisualProfile,
    VisualFeatureExtractor,
    compute_visual_anomaly_scores,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class DatasetCurator:
    """Orchestrates trajectory similarity, multi-modal outlier detection, and clean dataset export.

    Runs entirely on CPU to avoid interfering with concurrent GPU model training.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        camera_key: str | None = None,
    ):
        # Force CPU device for safety
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.device = torch.device("cpu")
        self.repo_id = repo_id
        self.root = Path(root) if root else None
        self.state_key = state_key
        self.action_key = action_key
        self.camera_key = camera_key

        logger.info(f"Loading LeRobotDataset {repo_id} on CPU...")
        self.dataset = LeRobotDataset(repo_id=repo_id, root=self.root, download_videos=False)
        self.total_episodes = self.dataset.meta.total_episodes
        self.fps = float(self.dataset.meta.fps)

        # Auto-detect camera key if not provided
        if self.camera_key is None and len(self.dataset.meta.camera_keys) > 0:
            self.camera_key = self.dataset.meta.camera_keys[0]

    def get_episode_slice(self, episode_index: int) -> tuple[int, int]:
        """Returns the start and end row indices for an episode in the dataset."""
        ep_meta = self.dataset.meta.episodes
        from_idx = int(ep_meta["dataset_from_index"][episode_index])
        to_idx = int(ep_meta["dataset_to_index"][episode_index])
        return from_idx, to_idx

    def get_episode_trajectories(self, episode_index: int) -> dict[str, np.ndarray]:
        """Loads state, action, and timestamp vectors for a single episode."""
        from_idx, to_idx = self.get_episode_slice(episode_index)
        hf = self.dataset.hf_dataset

        states = np.array(hf[self.state_key][from_idx:to_idx], dtype=np.float32)
        actions = np.array(hf[self.action_key][from_idx:to_idx], dtype=np.float32)

        if "timestamp" in hf.column_names:
            timestamps = np.array(hf["timestamp"][from_idx:to_idx], dtype=np.float64).flatten()
        else:
            timestamps = np.arange(len(states), dtype=np.float64) / self.fps

        return {
            "states": states,
            "actions": actions,
            "timestamps": timestamps,
        }

    def analyze(
        self,
        extract_visual: bool = True,
        visual_backbone: str = "auto",
        sample_keyframes: int = 5,
        dtw_window: int | None = 20,
        outlier_percentile: float = 85.0,
        n_clusters: int = 3,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> CurationManifest:
        """Runs multi-modal trajectory similarity, outlier detection, and trimming analysis.

        Args:
            extract_visual: Whether to extract visual embeddings from video observations.
            visual_backbone: Visual encoder ('mobilenet', 'dinov2', 'stats', or 'auto').
            sample_keyframes: Number of keyframes per episode to evaluate.
            dtw_window: Sakoe-Chiba constraint band for DTW.
            outlier_percentile: Score threshold percentile to flag as review/drop.
            n_clusters: Number of behavioral trajectory clusters to fit.
            progress_callback: Optional callable(fraction, message) for progress updates.

        Returns:
            Populated CurationManifest.
        """
        n_eps = self.total_episodes
        logger.info(f"Analyzing {n_eps} episodes from {self.repo_id}...")

        # 1. Kinematic Profiling & Trajectory Extraction
        trajectories = []
        kinematic_profiles = []
        trajectory_lengths = []

        for ep_idx in range(n_eps):
            if progress_callback:
                progress_callback(0.2 * (ep_idx / n_eps), f"Computing kinematics for episode {ep_idx}...")
            data = self.get_episode_trajectories(ep_idx)
            states = data["states"]
            trajectories.append(states)
            trajectory_lengths.append(len(states))

            profile = compute_kinematics(
                positions=states,
                fps=self.fps,
                timestamps=data["timestamps"],
            )
            kinematic_profiles.append(profile)

        # 2. Proprioceptive DTW Distance Matrix
        if progress_callback:
            progress_callback(0.3, "Computing pairwise Dynamic Time Warping (DTW) matrix...")
        dtw_matrix = compute_pairwise_dtw_matrix(trajectories, window=dtw_window)
        medoid_idx = find_medoid_trajectory(dtw_matrix)
        dtw_scores = compute_dtw_anomaly_scores(dtw_matrix, k_neighbors=min(5, n_eps - 1))

        # 3. Visual Feature Extraction (CPU-only)
        visual_anomaly_scores = None
        visual_profiles: list[EpisodeVisualProfile] = []

        if extract_visual and self.camera_key:
            if progress_callback:
                progress_callback(0.5, f"Extracting visual features via {visual_backbone} on CPU...")
            try:
                # Re-instantiate dataset with video loading for keyframes
                vis_dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=self.root,
                    download_videos=True,
                )
                extractor = VisualFeatureExtractor(backbone=visual_backbone, device="cpu")

                for ep_idx in range(n_eps):
                    from_idx, to_idx = self.get_episode_slice(ep_idx)
                    ep_len = to_idx - from_idx
                    if ep_len == 0:
                        continue

                    # Sample evenly spaced keyframes
                    step_indices = np.linspace(from_idx, to_idx - 1, min(sample_keyframes, ep_len), dtype=int)
                    frames = []
                    for idx in step_indices:
                        frame_data = vis_dataset[int(idx)]
                        if self.camera_key in frame_data:
                            frames.append(frame_data[self.camera_key])

                    if frames:
                        prof = extractor.analyze_episode_frames(ep_idx, frames)
                        visual_profiles.append(prof)

                if visual_profiles:
                    vis_anomalies = compute_visual_anomaly_scores(visual_profiles)
                    visual_anomaly_scores = vis_anomalies["composite_visual_score"]
            except Exception as e:
                logger.warning(f"Visual extraction skipped due to: {e}")
                visual_anomaly_scores = None

        # 4. Multimodal Anomaly Ranking & Clustering
        if progress_callback:
            progress_callback(0.8, "Fusing multimodal anomaly scores and clustering...")

        smoothness_scores = np.array([p.smoothness_score for p in kinematic_profiles])
        jerk_scores = np.array([p.jerk_metric for p in kinematic_profiles])
        lengths = np.array(trajectory_lengths)

        anomaly_reports = rank_anomalies(
            episode_indices=list(range(n_eps)),
            dtw_anomaly_scores=dtw_scores,
            kinematic_smoothness_scores=smoothness_scores,
            kinematic_jerk_scores=jerk_scores,
            trajectory_lengths=lengths,
            visual_anomaly_scores=visual_anomaly_scores,
            outlier_percentile=outlier_percentile,
            n_clusters=n_clusters,
        )

        duplicates = detect_duplicate_episodes(dtw_matrix, dtw_threshold=0.03)

        # 5. Populate Curation Manifest
        report_map = {r.episode_index: r for r in anomaly_reports}

        manifest = CurationManifest(
            repo_id=self.repo_id,
            total_episodes=n_eps,
            fps=self.fps,
            metadata={
                "medoid_episode_index": medoid_idx,
                "detected_duplicates": duplicates,
                "outlier_percentile_cutoff": outlier_percentile,
                "visual_evaluated": bool(visual_anomaly_scores is not None),
            },
        )

        for ep_idx in range(n_eps):
            rep = report_map[ep_idx]
            kin = kinematic_profiles[ep_idx]

            # Default status: 'review' if flagged as outlier, otherwise 'keep'
            status = "review" if rep.is_outlier else "keep"

            tags = []
            if ep_idx == medoid_idx:
                tags.append("medoid")
            if rep.is_outlier:
                tags.append("outlier")
            if rep.proprio_score > 0.7:
                tags.append("high_dtw")
            if rep.kinematic_score > 0.6:
                tags.append("jittery")
            if rep.visual_score > 0.6:
                tags.append("visual_anomaly")

            # Check if part of duplicate pair
            dup_matches = [d for d in duplicates if ep_idx in (d[0], d[1])]
            if dup_matches:
                tags.append("duplicate")

            manifest.episodes[str(ep_idx)] = EpisodeCurationItem(
                episode_index=ep_idx,
                status=status,
                trim_start=kin.suggested_trim_start,
                trim_end=kin.suggested_trim_end,
                original_length=len(trajectories[ep_idx]),
                tags=tags,
                notes=" / ".join(rep.reasons) if rep.reasons else "Nominal demonstration",
                anomaly_score=rep.composite_score,
                anomaly_reasons=rep.reasons,
                cluster_id=rep.cluster_id,
                kinematics=kin.to_dict(),
                visual_anomaly_score=rep.visual_score,
                proprio_anomaly_score=rep.proprio_score,
            )

        # Store computed artifacts on curator instance for downstream visualization / analysis
        self.trajectories = trajectories
        self.kinematic_profiles = kinematic_profiles
        self.dtw_matrix = dtw_matrix
        self.visual_profiles = visual_profiles
        self.visual_anomaly_scores = visual_anomaly_scores
        self.anomaly_reports = anomaly_reports

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return manifest

    def export_split_manifest(
        self,
        manifest: CurationManifest,
        output_path: Path | str,
        val_episodes: list[int] | None = None,
    ) -> Path:
        """Exports a clean split manifest with kept episodes (usable directly with LeRobotDataset)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clean_episodes = manifest.get_clean_episode_indices()

        if val_episodes is not None:
            train_episodes = [ep for ep in clean_episodes if ep not in val_episodes]
            val_eps = [ep for ep in clean_episodes if ep in val_episodes]
        else:
            # Standard 80/20 split on clean episodes
            n_clean = len(clean_episodes)
            n_train = max(1, int(0.8 * n_clean))
            train_episodes = clean_episodes[:n_train]
            val_eps = clean_episodes[n_train:]

        split_dict = {
            "source_repo_id": manifest.repo_id,
            "total_original_episodes": manifest.total_episodes,
            "total_clean_episodes": len(clean_episodes),
            "splits": {
                "train": train_episodes,
                "val": val_eps,
            },
            "dropped_episodes": manifest.get_dropped_episode_indices(),
            "review_episodes": manifest.get_review_episode_indices(),
        }

        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)

        logger.info(f"Exported clean split manifest to {output_path}")
        return output_path

    def export_clean_dataset(
        self,
        manifest: CurationManifest,
        output_repo_id: str,
        output_root: Path | str,
        slice_frames: bool = True,
        push_to_hub: bool = False,
    ) -> Path:
        """Materializes a new LeRobotDataset containing only the curated episodes.

        Args:
            manifest: CurationManifest containing status and trimming ranges.
            output_repo_id: Repository identifier for the new dataset.
            output_root: Destination directory path.
            slice_frames: If True, slices each episode from trim_start to trim_end.
                If False, retains the full length of kept episodes.
            push_to_hub: Whether to upload to Hugging Face Hub when done.

        Returns:
            Path to the newly exported dataset root.
        """
        output_root = Path(output_root)
        clean_ep_indices = manifest.get_clean_episode_indices()

        if len(clean_ep_indices) == 0:
            raise ValueError("No episodes marked with status 'keep' in manifest.")

        logger.info(
            f"Exporting {len(clean_ep_indices)} kept episodes to {output_root} (slice_frames={slice_frames})..."
        )

        # If not slicing frames, we can use the existing delete_episodes tool
        if not slice_frames:
            from lerobot.datasets.dataset_tools import delete_episodes

            dropped_indices = [i for i in range(self.total_episodes) if i not in clean_ep_indices]
            if dropped_indices:
                delete_episodes(
                    dataset=self.dataset,
                    episode_indices=dropped_indices,
                    output_dir=output_root,
                    repo_id=output_repo_id,
                )
                logger.info(f"Dataset exported cleanly to {output_root}")
                return output_root
            else:
                logger.info("All episodes kept, nothing to delete.")
                return output_root

        # Slicing frames: Build new dataset using LeRobotDataset.create()
        src_ds = LeRobotDataset(repo_id=self.repo_id, root=self.root, download_videos=True)

        features = dict(self.dataset.features)
        fps = int(self.dataset.meta.fps)
        robot_type = self.dataset.meta.robot_type
        use_videos = len(self.dataset.meta.video_keys) > 0

        clean_ds = LeRobotDataset.create(
            repo_id=output_repo_id,
            root=output_root,
            fps=fps,
            features=features,
            robot_type=robot_type,
            use_videos=use_videos,
        )

        for ep_idx in tqdm(clean_ep_indices, desc="Exporting curated episodes"):
            item = manifest.episodes[str(ep_idx)]
            from_idx, to_idx = self.get_episode_slice(ep_idx)

            start_f = max(from_idx, from_idx + item.trim_start)
            end_f = min(to_idx, from_idx + item.trim_end)

            if end_f <= start_f:
                start_f = from_idx
                end_f = to_idx

            for global_idx in range(start_f, end_f):
                frame_data = src_ds[global_idx]
                clean_frame = {}
                for k, v in frame_data.items():
                    if k in ("timestamp", "index", "task_index", "frame_index", "episode_index"):
                        continue
                    if isinstance(v, torch.Tensor):
                        if k in clean_ds.meta.video_keys or k in clean_ds.meta.image_keys:
                            if v.ndim == 3 and v.shape[0] in (1, 3):
                                v = v.permute(1, 2, 0)
                            if v.dtype != torch.uint8 and v.max() <= 1.0:
                                v = (v * 255.0).to(torch.uint8)
                            clean_frame[k] = v.cpu().numpy()
                        else:
                            clean_frame[k] = v.cpu().numpy()
                    else:
                        clean_frame[k] = v

                if "task" not in clean_frame:
                    clean_frame["task"] = frame_data.get("task", "default_task")

                clean_ds.add_frame(clean_frame)

            clean_ds.save_episode()

        clean_ds.finalize()
        logger.info(f"Successfully finalized clean curated dataset at {output_root}")

        if push_to_hub:
            logger.info(f"Pushing {output_repo_id} to Hugging Face Hub...")
            clean_ds.push_to_hub()

        return output_root
