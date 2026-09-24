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

"""Curation Manifest data structures for dataset observability, trimming, and filtering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CurationStatus = Literal["keep", "drop", "review"]


@dataclass
class EpisodeCurationItem:
    episode_index: int
    status: CurationStatus = "keep"
    trim_start: int = 0
    trim_end: int = 0
    original_length: int = 0
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    anomaly_score: float = 0.0
    anomaly_reasons: list[str] = field(default_factory=list)
    cluster_id: int = 0
    kinematics: dict[str, Any] = field(default_factory=dict)
    visual_anomaly_score: float = 0.0
    proprio_anomaly_score: float = 0.0

    @property
    def trimmed_length(self) -> int:
        return max(0, self.trim_end - self.trim_start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trimmed_length"] = self.trimmed_length
        return d


@dataclass
class CurationManifest:
    """Stores full curation status, anomaly rankings, trimming bounds, and tags for a dataset."""

    repo_id: str
    total_episodes: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    fps: float = 10.0
    episodes: dict[str, EpisodeCurationItem] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_episode(
        self,
        episode_index: int,
        status: CurationStatus | None = None,
        trim_start: int | None = None,
        trim_end: int | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        key = str(episode_index)
        if key not in self.episodes:
            raise KeyError(f"Episode {episode_index} not found in curation manifest.")

        item = self.episodes[key]
        if status is not None:
            item.status = status
        if trim_start is not None:
            item.trim_start = max(0, min(trim_start, item.original_length))
        if trim_end is not None:
            item.trim_end = max(item.trim_start, min(trim_end, item.original_length))
        if tags is not None:
            item.tags = tags
        if notes is not None:
            item.notes = notes

        self.updated_at = datetime.now(UTC).isoformat()

    def get_clean_episode_indices(self) -> list[int]:
        """Returns sorted list of episode indices marked as 'keep'."""
        clean = [item.episode_index for item in self.episodes.values() if item.status == "keep"]
        return sorted(clean)

    def get_dropped_episode_indices(self) -> list[int]:
        """Returns sorted list of episode indices marked as 'drop'."""
        dropped = [item.episode_index for item in self.episodes.values() if item.status == "drop"]
        return sorted(dropped)

    def get_review_episode_indices(self) -> list[int]:
        """Returns sorted list of episode indices marked as 'review'."""
        review = [item.episode_index for item in self.episodes.values() if item.status == "review"]
        return sorted(review)

    def summary(self) -> dict[str, Any]:
        """Generates summary statistics of the curation state."""
        items = list(self.episodes.values())
        n = len(items)
        if n == 0:
            return {}

        n_keep = sum(1 for x in items if x.status == "keep")
        n_drop = sum(1 for x in items if x.status == "drop")
        n_review = sum(1 for x in items if x.status == "review")

        total_orig_frames = sum(x.original_length for x in items)
        kept_trimmed_frames = sum(x.trimmed_length for x in items if x.status == "keep")
        trimmed_diff = sum((x.original_length - x.trimmed_length) for x in items if x.status == "keep")

        return {
            "repo_id": self.repo_id,
            "total_episodes": self.total_episodes,
            "kept_episodes": n_keep,
            "dropped_episodes": n_drop,
            "review_episodes": n_review,
            "retention_rate_pct": round(100.0 * n_keep / max(1, n), 2),
            "original_total_frames": total_orig_frames,
            "curated_total_frames": kept_trimmed_frames,
            "frames_saved_by_trimming": trimmed_diff,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "total_episodes": self.total_episodes,
            "fps": self.fps,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary(),
            "metadata": self.metadata,
            "episodes": {k: v.to_dict() for k, v in self.episodes.items()},
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> CurationManifest:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls(
            repo_id=data["repo_id"],
            total_episodes=data["total_episodes"],
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            fps=data.get("fps", 10.0),
            metadata=data.get("metadata", {}),
        )

        for k, v in data.get("episodes", {}).items():
            manifest.episodes[k] = EpisodeCurationItem(
                episode_index=v["episode_index"],
                status=v.get("status", "keep"),
                trim_start=v.get("trim_start", 0),
                trim_end=v.get("trim_end", v.get("original_length", 0)),
                original_length=v.get("original_length", 0),
                tags=v.get("tags", []),
                notes=v.get("notes", ""),
                anomaly_score=v.get("anomaly_score", 0.0),
                anomaly_reasons=v.get("anomaly_reasons", []),
                cluster_id=v.get("cluster_id", 0),
                kinematics=v.get("kinematics", {}),
                visual_anomaly_score=v.get("visual_anomaly_score", 0.0),
                proprio_anomaly_score=v.get("proprio_anomaly_score", 0.0),
            )

        return manifest
