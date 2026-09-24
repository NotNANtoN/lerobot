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

"""Dataset Observability, Trajectory Anomaly Detection, and Curation Toolkit."""

from lerobot.datasets.curation.anomaly import (
    EpisodeAnomalyDetail,
    detect_duplicate_episodes,
    kmeans_clustering,
    rank_anomalies,
)
from lerobot.datasets.curation.curator import DatasetCurator
from lerobot.datasets.curation.dtw import (
    compute_dtw_anomaly_scores,
    compute_pairwise_dtw_matrix,
    dtw_distance,
    find_medoid_trajectory,
)
from lerobot.datasets.curation.kinematics import KinematicProfile, compute_kinematics
from lerobot.datasets.curation.manifest import CurationManifest, EpisodeCurationItem
from lerobot.datasets.curation.visual import (
    EpisodeVisualProfile,
    VisualFeatureExtractor,
    compute_visual_anomaly_scores,
)
from lerobot.datasets.curation.visualizations import (
    export_full_analysis,
    plot_anomaly_ranking,
    plot_dtw_distance_matrix,
    plot_group_metrics_comparison,
    plot_trajectory_kinematics_2d,
    plot_visual_embeddings_2d,
)

__all__ = [
    "DatasetCurator",
    "CurationManifest",
    "EpisodeCurationItem",
    "dtw_distance",
    "compute_pairwise_dtw_matrix",
    "compute_dtw_anomaly_scores",
    "find_medoid_trajectory",
    "compute_kinematics",
    "KinematicProfile",
    "VisualFeatureExtractor",
    "EpisodeVisualProfile",
    "compute_visual_anomaly_scores",
    "rank_anomalies",
    "kmeans_clustering",
    "detect_duplicate_episodes",
    "EpisodeAnomalyDetail",
    "export_full_analysis",
    "plot_dtw_distance_matrix",
    "plot_trajectory_kinematics_2d",
    "plot_visual_embeddings_2d",
    "plot_group_metrics_comparison",
    "plot_anomaly_ranking",
]
