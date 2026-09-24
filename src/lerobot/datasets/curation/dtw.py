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

"""Dynamic Time Warping (DTW) algorithms for robotic trajectory comparison.

Provides vectorized, CPU-efficient Dynamic Time Warping distance computation,
warping path extraction, Sakoe-Chiba band constraints, and pairwise similarity matrices
without requiring external C/Fortran libraries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def dtw_distance(
    seq_a: np.ndarray | torch.Tensor,
    seq_b: np.ndarray | torch.Tensor,
    window: int | None = None,
    weights: np.ndarray | torch.Tensor | None = None,
    return_path: bool = False,
) -> tuple[float, list[tuple[int, int]] | None] | float:
    """Computes multidimensional Dynamic Time Warping (DTW) distance between two trajectories.

    Args:
        seq_a: Array of shape (T_a, D) or (T_a,).
        seq_b: Array of shape (T_b, D) or (T_b,).
        window: Sakoe-Chiba constraint window size. If None, full warping is allowed.
        weights: Optional feature weights of shape (D,) to emphasize specific joints.
        return_path: If True, returns (normalized_distance, warping_path).
            Otherwise, returns only normalized_distance.

    Returns:
        Normalized DTW distance (total cost / path length), and optionally the warping path.
    """
    if isinstance(seq_a, torch.Tensor):
        seq_a = seq_a.detach().cpu().numpy()
    if isinstance(seq_b, torch.Tensor):
        seq_b = seq_b.detach().cpu().numpy()

    a = np.atleast_2d(seq_a).astype(np.float64)
    b = np.atleast_2d(seq_b).astype(np.float64)

    t_a, d_a = a.shape
    t_b, d_b = b.shape

    if d_a != d_b:
        raise ValueError(f"Feature dimension mismatch: seq_a has {d_a}, seq_b has {d_b}")

    if t_a == 0 or t_b == 0:
        dist = float("inf")
        return (dist, []) if return_path else dist

    if weights is not None:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != d_a:
            raise ValueError(f"Weights length {w.shape[0]} does not match feature dimension {d_a}")
        sqrt_w = np.sqrt(np.maximum(w, 0.0))
        a = a * sqrt_w
        b = b * sqrt_w

    # Distance matrix cost computation
    # cost_mat[i, j] = ||a[i] - b[j]||_2
    # Vectorized computation of pairwise squared Euclidean distances:
    # ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j
    a_sq = np.sum(a**2, axis=1, keepdims=True)
    b_sq = np.sum(b**2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, a_sq + b_sq.T - 2.0 * np.dot(a, b.T))
    pointwise_dist = np.sqrt(dist_sq)

    # Dynamic programming accumulated cost matrix
    cost = np.full((t_a + 1, t_b + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    # Ensure constraint window is at least |t_a - t_b| so the end cell (t_a, t_b) is always reachable
    w_size = max(window, abs(t_a - t_b)) if window is not None else max(t_a, t_b)

    for i in range(1, t_a + 1):
        # Apply Sakoe-Chiba band constraint if window is given
        j_min = max(1, i - w_size)
        j_max = min(t_b + 1, i + w_size + 1)
        for j in range(j_min, j_max):
            c = pointwise_dist[i - 1, j - 1]
            prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = c + prev

    total_cost = cost[t_a, t_b]

    # Backtrack warping path
    path: list[tuple[int, int]] = []
    curr_i, curr_j = t_a, t_b
    while curr_i > 0 or curr_j > 0:
        path.append((curr_i - 1, curr_j - 1))
        if curr_i == 0:
            curr_j -= 1
        elif curr_j == 0:
            curr_i -= 1
        else:
            diag = cost[curr_i - 1, curr_j - 1]
            left = cost[curr_i, curr_j - 1]
            up = cost[curr_i - 1, curr_j]
            min_val = min(diag, left, up)
            if min_val == diag:
                curr_i -= 1
                curr_j -= 1
            elif min_val == up:
                curr_i -= 1
            else:
                curr_j -= 1

    path.reverse()
    path_len = len(path) if len(path) > 0 else 1
    normalized_dist = float(total_cost / path_len)

    if return_path:
        return normalized_dist, path
    return normalized_dist


def compute_pairwise_dtw_matrix(
    trajectories: Sequence[np.ndarray | torch.Tensor],
    window: int | None = None,
    weights: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Computes the symmetric pairwise DTW distance matrix across a list of trajectories.

    Args:
        trajectories: List of N trajectories, each of shape (T_i, D).
        window: Sakoe-Chiba constraint window size.
        weights: Optional feature weights.

    Returns:
        Symmetric matrix of shape (N, N) where M[i, j] is the normalized DTW distance.
    """
    n = len(trajectories)
    matrix = np.zeros((n, n), dtype=np.float64)

    # Pre-convert to numpy arrays
    processed = []
    for traj in trajectories:
        if isinstance(traj, torch.Tensor):
            traj = traj.detach().cpu().numpy()
        processed.append(np.atleast_2d(traj).astype(np.float64))

    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(processed[i], processed[j], window=window, weights=weights)
            matrix[i, j] = dist
            matrix[j, i] = dist

    return matrix


def find_medoid_trajectory(distance_matrix: np.ndarray) -> int:
    """Finds the index of the medoid trajectory (minimum average distance to all others).

    Args:
        distance_matrix: Square distance matrix of shape (N, N).

    Returns:
        Index of the medoid trajectory.
    """
    if distance_matrix.shape[0] == 0:
        return -1
    avg_distances = np.mean(distance_matrix, axis=1)
    return int(np.argmin(avg_distances))


def compute_dtw_anomaly_scores(
    distance_matrix: np.ndarray,
    k_neighbors: int = 5,
) -> np.ndarray:
    """Computes anomaly scores based on k-nearest neighbor DTW distances.

    Episodes that have high distance to their k nearest neighbors are outliers.

    Args:
        distance_matrix: Square distance matrix of shape (N, N).
        k_neighbors: Number of nearest neighbors to consider.

    Returns:
        Normalized anomaly scores in [0, 1] of shape (N,).
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    k = min(k_neighbors, n - 1)
    # Sort distances along rows (ignoring self at index 0)
    sorted_dists = np.sort(distance_matrix, axis=1)
    knn_dists = np.mean(sorted_dists[:, 1 : k + 1], axis=1)

    knn_dists = np.nan_to_num(knn_dists, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(knn_dists))
    max_val = float(np.max(knn_dists))
    denom = max_val - min_val
    scores = (knn_dists - min_val) / denom if denom > 1e-9 else np.zeros(n, dtype=np.float64)

    return np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)
