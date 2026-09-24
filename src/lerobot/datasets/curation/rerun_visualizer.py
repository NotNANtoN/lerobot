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

"""Rerun-based comparative visualizer for trajectory outliers vs medoid."""

from __future__ import annotations

import logging
from pathlib import Path

from lerobot.datasets.curation.curator import DatasetCurator
from lerobot.datasets.curation.manifest import CurationManifest

logger = logging.getLogger(__name__)


def visualize_in_rerun(
    curator: DatasetCurator,
    manifest: CurationManifest,
    episode_index: int,
    compare_with_medoid: bool = True,
    save_path: Path | str | None = None,
) -> None:
    """Visualizes an episode alongside the dataset medoid in Rerun."""
    try:
        import rerun as rr
    except ImportError:
        logger.error(
            "rerun-sdk is not installed. Install with `uv add rerun-sdk` or `pip install 'lerobot[viz]'`."
        )
        print("Please install rerun-sdk: pip install 'lerobot[viz]'")
        return

    ep_key = str(episode_index)
    item = manifest.episodes.get(ep_key)
    if not item:
        raise ValueError(f"Episode {episode_index} not found in manifest.")

    rr.init(f"curation/{curator.repo_id}/ep_{episode_index}", spawn=(save_path is None))

    data = curator.get_episode_trajectories(episode_index)
    states = data["states"]
    timestamps = data["timestamps"]

    medoid_states = None
    medoid_idx = manifest.metadata.get("medoid_episode_index")
    if compare_with_medoid and medoid_idx is not None and medoid_idx != episode_index:
        med_data = curator.get_episode_trajectories(medoid_idx)
        medoid_states = med_data["states"]

    # Log trajectory points and time-series
    for t_idx, t in enumerate(timestamps):
        rr.set_time_seconds("sim_time", float(t))
        rr.set_time_sequence("frame_idx", t_idx)

        # Log joint angles
        for d in range(states.shape[1]):
            rr.log(f"episode_{episode_index}/joint_{d}", rr.Scalar(float(states[t_idx, d])))

        # Log medoid reference
        if medoid_states is not None and t_idx < len(medoid_states):
            for d in range(medoid_states.shape[1]):
                rr.log(f"medoid_ep_{medoid_idx}/joint_{d}", rr.Scalar(float(medoid_states[t_idx, d])))

        # Log trim indicator
        in_trimmed_range = item.trim_start <= t_idx <= item.trim_end
        rr.log("curation/kept_in_trim", rr.Scalar(1.0 if in_trimmed_range else 0.0))

    # Log summary text
    status_text = f"Status: {item.status.upper()} | Anomaly Score: {item.anomaly_score:.1f}"
    if item.anomaly_reasons:
        status_text += f"\nReasons: {', '.join(item.anomaly_reasons)}"
    rr.log("curation/info", rr.TextDocument(status_text))

    if save_path:
        rr.save(str(save_path))
        logger.info(f"Saved Rerun recording to {save_path}")
