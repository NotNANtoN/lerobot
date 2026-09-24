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

"""Multi-modal anomaly detection, ranking, and trajectory clustering.

Fuses proprioceptive (DTW), visual, and kinematic metrics into a calibrated
composite anomaly score, assigns behavioral clusters, and detects duplicate demonstrations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EpisodeAnomalyDetail:
    episode_index: int
    composite_score: float  # [0, 100]
    proprio_score: float  # [0, 1]
    visual_score: float  # [0, 1]
    kinematic_score: float  # [0, 1]
    length_score: float  # [0, 1]
    cluster_id: int
    is_outlier: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """Vectorized K-means clustering on CPU without scikit-learn dependency.

    Args:
        embeddings: Array of shape (N, D).
        n_clusters: Number of clusters (clamped to N).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for centroid movement.
        random_state: Random seed for centroid initialization.

    Returns:
        Array of shape (N,) with integer cluster labels.
    """
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    n_samples, n_features = embeddings.shape
    if n_samples <= n_clusters:
        return np.arange(n_samples, dtype=int)

    rng = np.random.default_rng(random_state)

    # K-means++ initialization
    centers = np.empty((n_clusters, n_features), dtype=embeddings.dtype)
    first_idx = rng.integers(0, n_samples)
    centers[0] = embeddings[first_idx]

    for c_idx in range(1, n_clusters):
        # Distances from all points to existing centers
        dists = np.min(
            np.linalg.norm(embeddings[:, None, :] - centers[:c_idx, :], axis=2) ** 2,
            axis=1,
        )
        dists = np.nan_to_num(dists, nan=0.0)
        sum_dists = np.sum(dists)
        if sum_dists <= 1e-9:
            probs = np.full(n_samples, 1.0 / n_samples)
        else:
            probs = dists / sum_dists
            probs = np.nan_to_num(probs, nan=0.0)
            p_sum = np.sum(probs)
            probs = np.full(n_samples, 1.0 / n_samples) if p_sum <= 1e-9 else probs / p_sum

        next_idx = rng.choice(n_samples, p=probs)
        centers[c_idx] = embeddings[next_idx]

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign points to closest center
        dists = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        # Recompute centers
        new_centers = np.empty_like(centers)
        for k in range(n_clusters):
            members = embeddings[new_labels == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = embeddings[rng.integers(0, n_samples)]

        center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        labels = new_labels

        if center_shift < tol:
            break

    return labels


def detect_duplicate_episodes(
    dtw_matrix: np.ndarray,
    visual_similarity_matrix: np.ndarray | None = None,
    dtw_threshold: float = 0.05,
    visual_sim_threshold: float = 0.98,
) -> list[tuple[int, int, float]]:
    """Detects duplicate or near-identical trajectory pairs.

    Returns list of (ep_i, ep_j, similarity) tuples.
    """
    n = dtw_matrix.shape[0]
    duplicates = []

    # Normalize DTW distance to similarity in [0, 1]
    safe_mat = np.nan_to_num(dtw_matrix, nan=0.0)
    max_dtw = float(np.max(safe_mat)) if safe_mat.size > 0 and np.max(safe_mat) > 0 else 1.0
    dtw_sim = np.ones_like(safe_mat) if max_dtw <= 1e-9 else 1.0 - (safe_mat / max_dtw)
    dtw_sim = np.clip(np.nan_to_num(dtw_sim, nan=1.0), 0.0, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            is_dup = False
            sim_score = dtw_sim[i, j]
            if dtw_matrix[i, j] <= dtw_threshold:
                if visual_similarity_matrix is not None:
                    if visual_similarity_matrix[i, j] >= visual_sim_threshold:
                        is_dup = True
                else:
                    is_dup = True

            if is_dup:
                duplicates.append((i, j, float(sim_score)))

    return duplicates


def rank_anomalies(
    episode_indices: Sequence[int],
    dtw_anomaly_scores: np.ndarray,
    kinematic_smoothness_scores: np.ndarray,
    kinematic_jerk_scores: np.ndarray,
    trajectory_lengths: np.ndarray,
    visual_anomaly_scores: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
    outlier_percentile: float = 85.0,
    n_clusters: int = 3,
    feature_embeddings: np.ndarray | None = None,
) -> list[EpisodeAnomalyDetail]:
    """Combines all multimodal signals into a ranked list of anomaly reports.

    Args:
        episode_indices: List of episode index numbers.
        dtw_anomaly_scores: Array of shape (N,) in [0, 1].
        kinematic_smoothness_scores: Smoothness in [0, 1] (lower = more jerky/anomalous).
        kinematic_jerk_scores: Raw jerk metric.
        trajectory_lengths: Length of trajectories in frames.
        visual_anomaly_scores: Optional visual anomaly scores in [0, 1].
        weights: Custom weight dictionary for 'proprio', 'visual', 'kinematic', 'length'.
        outlier_percentile: Threshold percentile above which an episode is flagged.
        n_clusters: Number of behavioral clusters to find.
        feature_embeddings: Optional embeddings for clustering.

    Returns:
        List of EpisodeAnomalyDetail objects sorted by descending composite anomaly score.
    """
    n = len(episode_indices)
    if n == 0:
        return []

    w = {
        "proprio": 0.40,
        "visual": 0.25 if visual_anomaly_scores is not None else 0.0,
        "kinematic": 0.25,
        "length": 0.10,
    }
    if weights:
        w.update(weights)

    # Normalize weights
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    # Proprioceptive score [0, 1]
    p_score = np.clip(dtw_anomaly_scores, 0.0, 1.0)

    # Kinematic score: combine jerk and lack of smoothness
    j_med = np.median(kinematic_jerk_scores)
    j_mad = np.median(np.abs(kinematic_jerk_scores - j_med)) + 1e-6
    jerk_norm = np.clip((kinematic_jerk_scores - j_med) / (3.0 * j_mad), 0.0, 1.0)
    roughness = 1.0 - np.clip(kinematic_smoothness_scores, 0.0, 1.0)
    k_score = 0.6 * jerk_norm + 0.4 * roughness

    # Trajectory length anomaly score
    len_med = np.median(trajectory_lengths)
    len_mad = np.median(np.abs(trajectory_lengths - len_med)) + 1e-6
    l_score = np.clip(np.abs(trajectory_lengths - len_med) / (3.0 * len_mad), 0.0, 1.0)

    # Visual score [0, 1]
    if visual_anomaly_scores is not None and len(visual_anomaly_scores) == n:
        v_score = np.clip(visual_anomaly_scores, 0.0, 1.0)
    else:
        v_score = np.zeros(n)

    # Composite score [0, 100]
    raw_composite = (
        w["proprio"] * p_score + w["visual"] * v_score + w["kinematic"] * k_score + w["length"] * l_score
    )
    composite_100 = raw_composite * 100.0

    # Determine outlier cutoff
    cutoff = np.percentile(composite_100, outlier_percentile) if n > 3 else 50.0

    # Cluster assignments
    if feature_embeddings is not None and feature_embeddings.shape[0] == n:
        cluster_labels = kmeans_clustering(feature_embeddings, n_clusters=min(n_clusters, n))
    else:
        # Cluster on normalized metrics
        combined_feats = np.column_stack([p_score, k_score, l_score, v_score])
        combined_feats = np.nan_to_num(combined_feats, nan=0.0)
        cluster_labels = kmeans_clustering(combined_feats, n_clusters=min(n_clusters, n))

    results: list[EpisodeAnomalyDetail] = []
    for idx in range(n):
        ep = episode_indices[idx]
        score = float(composite_100[idx])
        is_outlier = bool(score >= cutoff)

        reasons = []
        if p_score[idx] > 0.7:
            reasons.append("Unusual joint path (High DTW distance)")
        if k_score[idx] > 0.6:
            reasons.append("High jerk / teleop chatter")
        if l_score[idx] > 0.7:
            reasons.append(f"Abnormal duration ({trajectory_lengths[idx]} frames vs median {int(len_med)})")
        if v_score[idx] > 0.6:
            reasons.append("Visual anomaly (lighting, occlusion, or novel scene)")

        if not reasons and is_outlier:
            reasons.append("Composite score above outlier threshold")

        results.append(
            EpisodeAnomalyDetail(
                episode_index=ep,
                composite_score=round(score, 2),
                proprio_score=round(float(p_score[idx]), 3),
                visual_score=round(float(v_score[idx]), 3),
                kinematic_score=round(float(k_score[idx]), 3),
                length_score=round(float(l_score[idx]), 3),
                cluster_id=int(cluster_labels[idx]),
                is_outlier=is_outlier,
                reasons=reasons,
            )
        )

    # Sort descending by composite score
    results.sort(key=lambda x: x.composite_score, reverse=True)
    return results
