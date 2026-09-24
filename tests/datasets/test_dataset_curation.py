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

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.utils.import_utils import _matplotlib_available

# Enforce CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lerobot.datasets.curation import (
    CurationManifest,
    DatasetCurator,
    EpisodeCurationItem,
    VisualFeatureExtractor,
    compute_dtw_anomaly_scores,
    compute_kinematics,
    compute_pairwise_dtw_matrix,
    compute_visual_anomaly_scores,
    detect_duplicate_episodes,
    dtw_distance,
    find_medoid_trajectory,
    kmeans_clustering,
)


def test_dtw_identical_and_warped():
    # 1. Identical sequences
    t = np.linspace(0, 4 * np.pi, 50)
    seq_a = np.column_stack([np.sin(t), np.cos(t)])
    seq_b = seq_a.copy()

    dist, path = dtw_distance(seq_a, seq_b, return_path=True)
    assert np.isclose(dist, 0.0, atol=1e-6)
    assert len(path) == 50
    assert path[0] == (0, 0)
    assert path[-1] == (49, 49)

    # 2. Time-warped sequence (same trajectory shape, slower speed)
    t_warped = np.linspace(0, 4 * np.pi, 30)
    seq_warped = np.column_stack([np.sin(t_warped), np.cos(t_warped)])
    dist_warped = dtw_distance(seq_a, seq_warped)
    assert dist_warped < 0.2

    # 3. Completely different sequence
    seq_diff = np.column_stack([np.ones(50) * 10.0, np.ones(50) * -5.0])
    dist_diff = dtw_distance(seq_a, seq_diff)
    assert dist_diff > 5.0


def test_pairwise_dtw_and_medoid():
    t = np.linspace(0, np.pi, 30)
    nominal_1 = np.sin(t)[:, None]
    nominal_2 = (np.sin(t) + 0.05)[:, None]
    nominal_3 = (np.sin(t) - 0.05)[:, None]
    outlier = (np.ones_like(t) * 5.0)[:, None]

    trajectories = [nominal_1, nominal_2, nominal_3, outlier]
    mat = compute_pairwise_dtw_matrix(trajectories, window=15)

    assert mat.shape == (4, 4)
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)

    medoid = find_medoid_trajectory(mat)
    assert medoid in (0, 1, 2)  # Should not be outlier
    assert medoid != 3

    scores = compute_dtw_anomaly_scores(mat, k_neighbors=2)
    assert len(scores) == 4
    # Outlier must have highest anomaly score
    assert np.argmax(scores) == 3
    assert scores[3] > scores[0]


def test_kinematics_and_idle_trimming():
    t_steps = 100
    positions = np.zeros((t_steps, 6))

    # Dead idle from t=0..19, motion from 20..79, dead idle from 80..99
    positions[20:80, :5] = np.linspace(0, 20, 60)[:, None]
    positions[40:60, 5] = 100.0  # Gripper actuates mid-way

    profile = compute_kinematics(positions, fps=10.0, gripper_index=5)

    assert profile.num_frames == 100
    assert 18 <= profile.suggested_trim_start <= 22
    assert 78 <= profile.suggested_trim_end <= 82
    assert profile.gripper_actuations >= 1
    assert profile.path_length > 0.0
    assert profile.max_velocity > 0.0
    assert profile.smoothness_score > 0.0


def test_visual_feature_extractor_cpu():
    # Test fallback stats extractor
    extractor_stats = VisualFeatureExtractor(backbone="stats", device="cpu")
    dummy_frames = [torch.rand(3, 128, 128) for _ in range(4)]
    profile_stats = extractor_stats.analyze_episode_frames(episode_index=0, frames=dummy_frames)

    assert profile_stats.mean_embedding.ndim == 1
    assert len(profile_stats.mean_embedding) > 0
    assert profile_stats.brightness_mean > 0.0

    # Test mobilenet extractor on CPU
    extractor_nn = VisualFeatureExtractor(backbone="mobilenet", device="cpu")
    profile_nn = extractor_nn.analyze_episode_frames(episode_index=1, frames=dummy_frames)
    assert len(profile_nn.mean_embedding) == 576

    # Test anomaly detection on profiles
    dark_frame = [torch.zeros(3, 128, 128) for _ in range(4)]
    dark_profile = extractor_stats.analyze_episode_frames(episode_index=2, frames=dark_frame)

    scores = compute_visual_anomaly_scores([profile_stats, profile_stats, dark_profile])
    assert len(scores["composite_visual_score"]) == 3
    # Dark scene should be flagged with higher lighting anomaly
    assert scores["lighting_anomaly"][2] > scores["lighting_anomaly"][0]


def test_kmeans_and_duplicate_detection():
    # 2 distinct clusters
    c1 = np.random.randn(10, 4) + 5.0
    c2 = np.random.randn(10, 4) - 5.0
    data = np.vstack([c1, c2])

    labels = kmeans_clustering(data, n_clusters=2, random_state=42)
    assert len(labels) == 20
    assert len(np.unique(labels)) == 2
    # Check that points in first cluster share the same label
    assert len(np.unique(labels[:10])) == 1
    assert len(np.unique(labels[10:])) == 1

    # Duplicate detection
    dtw_mat = np.zeros((3, 3))
    dtw_mat[0, 1] = dtw_mat[1, 0] = 0.01  # Near identical
    dtw_mat[0, 2] = dtw_mat[2, 0] = 10.0
    dtw_mat[1, 2] = dtw_mat[2, 1] = 10.0

    dups = detect_duplicate_episodes(dtw_mat, dtw_threshold=0.05)
    assert len(dups) == 1
    assert dups[0][0] == 0 and dups[0][1] == 1


def test_curation_manifest_lifecycle(tmp_path: Path):
    manifest = CurationManifest(repo_id="test/sample_dataset", total_episodes=3, fps=10.0)
    manifest.episodes["0"] = EpisodeCurationItem(
        episode_index=0, status="keep", trim_start=5, trim_end=95, original_length=100
    )
    manifest.episodes["1"] = EpisodeCurationItem(
        episode_index=1, status="drop", trim_start=0, trim_end=50, original_length=50
    )
    manifest.episodes["2"] = EpisodeCurationItem(
        episode_index=2, status="review", trim_start=0, trim_end=120, original_length=120
    )

    assert manifest.get_clean_episode_indices() == [0]
    assert manifest.get_dropped_episode_indices() == [1]
    assert manifest.get_review_episode_indices() == [2]

    manifest.mark_episode(2, status="keep", trim_start=10, trim_end=110)
    assert manifest.get_clean_episode_indices() == [0, 2]

    # Save & reload JSON
    json_path = tmp_path / "manifest.json"
    manifest.save_json(json_path)
    loaded = CurationManifest.load_json(json_path)

    assert loaded.repo_id == "test/sample_dataset"
    assert loaded.total_episodes == 3
    assert loaded.episodes["2"].status == "keep"
    assert loaded.episodes["2"].trim_start == 10
    assert loaded.episodes["2"].trim_end == 110


def test_curator_on_cube_dataset(tmp_path: Path):
    """End-to-end integration test with locally cached hubnemo/cube_out_of_box_dataset."""
    repo_id = "hubnemo/cube_out_of_box_dataset"
    curator = DatasetCurator(repo_id=repo_id)

    assert curator.total_episodes == 40
    assert curator.fps == 10.0

    # Analyze trajectory dynamics on CPU
    manifest = curator.analyze(
        extract_visual=False,
        dtw_window=15,
        n_clusters=3,
    )

    summary = manifest.summary()
    assert summary["total_episodes"] == 40
    assert summary["kept_episodes"] > 0
    assert manifest.metadata.get("medoid_episode_index") is not None

    # Export split manifest
    split_file = tmp_path / "split.json"
    curator.export_split_manifest(manifest, split_file)
    assert split_file.exists()

    with open(split_file) as f:
        split_data = json.load(f)
    assert "train" in split_data["splits"]
    assert len(split_data["splits"]["train"]) > 0


@pytest.mark.skipif(not _matplotlib_available, reason="matplotlib not installed")
def test_visualizations_and_export(tmp_path: Path):
    """Tests plotting and analytical export utilities."""
    from lerobot.datasets.curation.visualizations import (
        plot_anomaly_ranking,
        plot_dtw_distance_matrix,
        plot_group_metrics_comparison,
        plot_trajectory_kinematics_2d,
        plot_visual_embeddings_2d,
    )

    n_eps = 10
    dtw_mat = np.random.rand(n_eps, n_eps)
    dtw_mat = (dtw_mat + dtw_mat.T) / 2.0
    np.fill_diagonal(dtw_mat, 0.0)

    group_specs = [(0, 4, "Group A"), (4, 7, "Group B"), (7, 10, "Group C")]

    dtw_img = tmp_path / "dtw.png"
    plot_dtw_distance_matrix(dtw_mat, group_specs, dtw_img)
    assert dtw_img.exists()
    assert dtw_img.stat().st_size > 1000

    kin_feats = np.random.randn(n_eps, 8)
    kin_img = tmp_path / "kin.png"
    plot_trajectory_kinematics_2d(
        kin_feats,
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        ["Group A", "Group B", "Group C"],
        [False] * 9 + [True],
        list(range(n_eps)),
        kin_img,
    )
    assert kin_img.exists()
    assert kin_img.stat().st_size > 1000

    vis_embeds = np.random.randn(n_eps, 64)
    vis_img = tmp_path / "vis.png"
    plot_visual_embeddings_2d(
        vis_embeds,
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        ["Group A", "Group B", "Group C"],
        [False] * 9 + [True],
        list(range(n_eps)),
        vis_img,
    )
    assert vis_img.exists()
    assert vis_img.stat().st_size > 1000

    metrics_img = tmp_path / "metrics.png"
    plot_group_metrics_comparison(
        durations=[10.0] * n_eps,
        mean_velocities=[20.0] * n_eps,
        jerk_metrics=[1e5] * n_eps,
        smoothness_scores=[0.5] * n_eps,
        dtw_mean_distances=[50.0] * n_eps,
        brightnesses=[70.0] * n_eps,
        group_slices=[slice(0, 4), slice(4, 7), slice(7, 10)],
        group_names=["Group A", "Group B", "Group C"],
        output_path=metrics_img,
    )
    assert metrics_img.exists()
    assert metrics_img.stat().st_size > 1000

    anomaly_img = tmp_path / "anomaly.png"
    sample_episodes_data = [
        {
            "episode_index": i,
            "composite_score": float(i * 10),
            "group_id": i % 3,
            "reasons": ["Test anomaly reason"],
        }
        for i in range(n_eps)
    ]
    plot_anomaly_ranking(
        episodes_data=sample_episodes_data,
        group_names=["Group A", "Group B", "Group C"],
        cutoff_threshold=50.0,
        output_path=anomaly_img,
    )
    assert anomaly_img.exists()
    assert anomaly_img.stat().st_size > 1000
