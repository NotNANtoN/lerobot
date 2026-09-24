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

"""CPU-efficient visual feature extraction and visual anomaly detection.

Extracts compact visual embeddings (DINOv2, MobileNetV3, or statistical color/spatial moments),
computes scene variation, brightness/contrast shifts, blur/smear anomalies, and ranks
visual outliers across robotic demonstrations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class EpisodeVisualProfile:
    """Visual characteristics and anomaly indicators for a single episode."""

    episode_index: int
    mean_embedding: np.ndarray  # (D,)
    brightness_mean: float
    brightness_std: float
    sharpness_score: float  # Laplacian variance (blur check)
    visual_dispersion: float  # Frame-to-frame visual variance within episode
    num_sampled_frames: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_embedding"] = self.mean_embedding.tolist()
        return d


class VisualFeatureExtractor:
    """Extracts compact visual embeddings from video frames on CPU.

    Supports:
        - 'mobilenet' (MobileNetV3-Small, fast and lightweight, 576-dim)
        - 'dinov2' (DINOv2 ViT-S/14, self-supervised semantic vision tokens, 384-dim)
        - 'stats' (fast spatial color moments + gradient entropy, zero neural network overhead)
        - 'auto' (tries mobilenet -> dinov2 -> stats)
    """

    def __init__(
        self,
        backbone: Literal["auto", "mobilenet", "dinov2", "stats"] = "mobilenet",
        device: str | torch.device = "cpu",
    ):
        # Force CPU device for safety
        self.device = torch.device("cpu")
        self.backbone_name = backbone
        self.model = None
        self._init_model(backbone)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _init_model(self, backbone: str) -> None:
        if backbone == "stats":
            self.model = None
            self.backbone_name = "stats"
            return

        if backbone in ("mobilenet", "auto"):
            try:
                import torchvision.models as models

                m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                m.classifier = torch.nn.Identity()
                m.eval()
                m.to(self.device)
                for param in m.parameters():
                    param.requires_grad = False
                self.model = m
                self.backbone_name = "mobilenet"
                return
            except Exception as e:
                logger.warning(f"Could not load MobileNetV3: {e}. Trying fallback.")
                if backbone == "mobilenet":
                    self.backbone_name = "stats"
                    return

        if backbone in ("dinov2", "auto"):
            try:
                dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # nosec B614
                dino.eval()
                dino.to(self.device)
                for param in dino.parameters():
                    param.requires_grad = False
                self.model = dino
                self.backbone_name = "dinov2"
                return
            except Exception as e:
                logger.warning(f"Could not load DINOv2: {e}. Falling back to stats.")
                self.model = None
                self.backbone_name = "stats"
                return

        self.model = None
        self.backbone_name = "stats"

    def _extract_stats_features(self, frames: torch.Tensor) -> np.ndarray:
        """Extracts spatial color moments + Laplacian sharpness without a neural network."""
        # frames: (B, C, H, W) in [0, 1] or [0, 255]
        b, c, h, w = frames.shape
        features = []
        for i in range(b):
            img = frames[i].numpy()
            if img.max() > 1.0:
                img = img / 255.0

            # 4x4 spatial grid color means & stds
            gh, gw = h // 4, w // 4
            grid_feats = []
            for r in range(4):
                for col in range(4):
                    patch = img[:, r * gh : (r + 1) * gh, col * gw : (col + 1) * gw]
                    grid_feats.extend([patch.mean(axis=(1, 2)), patch.std(axis=(1, 2))])
            flat_grid = np.concatenate(grid_feats)  # 16 * 6 = 96 dims

            # Global color histogram
            hist_feats = []
            for ch in range(c):
                h_vals, _ = np.histogram(img[ch], bins=16, range=(0.0, 1.0), density=True)
                hist_feats.append(h_vals)
            flat_hist = np.concatenate(hist_feats)  # 48 dims

            combined = np.concatenate([flat_grid, flat_hist])
            norm = np.linalg.norm(combined) + 1e-7
            features.append(combined / norm)

        return np.array(features, dtype=np.float32)

    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor | np.ndarray) -> np.ndarray:
        """Extracts normalized visual embeddings for a batch of frames on CPU.

        Args:
            frames: Tensor of shape (B, C, H, W) or (C, H, W), values in [0, 1] or [0, 255].

        Returns:
            Numpy array of shape (B, D) with unit-normalized visual embeddings.
        """
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        if frames.ndim == 3:
            frames = frames.unsqueeze(0)

        # Ensure (B, C, H, W)
        if frames.shape[1] not in (1, 3) and frames.shape[-1] in (1, 3):
            # (B, H, W, C) -> (B, C, H, W)
            frames = frames.permute(0, 3, 1, 2)

        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.max() > 1.0:
            frames = frames / 255.0

        if self.model is None or self.backbone_name == "stats":
            return self._extract_stats_features(frames.cpu())

        # Preprocess for neural backbone
        preprocessed = self.transform(frames.cpu())
        embeds = self.model(preprocessed)
        if isinstance(embeds, dict):
            embeds = embeds.get("x_norm_clstoken", embeds.get("last_hidden_state", embeds))

        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds.cpu().numpy()

    def compute_frame_quality(self, frame: torch.Tensor | np.ndarray) -> tuple[float, float, float]:
        """Computes basic image quality metrics: (brightness_mean, brightness_std, sharpness)."""
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype != np.uint8 and frame.max() <= 1.0:
            frame = (frame * 255.0).astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[-1] == 3 else frame.squeeze()

        mean_bright = float(np.mean(gray))
        std_bright = float(np.std(gray))
        # Laplacian variance measures edge sharpness (low value indicates blur)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))

        return mean_bright, std_bright, sharpness

    def analyze_episode_frames(
        self,
        episode_index: int,
        frames: Sequence[torch.Tensor | np.ndarray],
    ) -> EpisodeVisualProfile:
        """Analyzes a sequence of sampled frames from an episode."""
        if len(frames) == 0:
            raise ValueError("frames sequence cannot be empty.")

        # Extract neural / statistical embeddings
        stacked = torch.stack([f if isinstance(f, torch.Tensor) else torch.from_numpy(f) for f in frames])
        embeds = self.extract_features(stacked)  # (K, D)

        mean_embed = np.mean(embeds, axis=0)
        norm = np.linalg.norm(mean_embed) + 1e-7
        mean_embed = mean_embed / norm

        # Dispersion across keyframes (measures visual movement / dynamics)
        dispersion = float(np.mean(np.linalg.norm(embeds - mean_embed, axis=1))) if len(embeds) > 1 else 0.0

        # Quality metrics across sampled frames
        b_means, b_stds, sharps = [], [], []
        for f in frames:
            bm, bs, sh = self.compute_frame_quality(f)
            b_means.append(bm)
            b_stds.append(bs)
            sharps.append(sh)

        return EpisodeVisualProfile(
            episode_index=episode_index,
            mean_embedding=mean_embed,
            brightness_mean=float(np.mean(b_means)),
            brightness_std=float(np.mean(b_stds)),
            sharpness_score=float(np.mean(sharps)),
            visual_dispersion=dispersion,
            num_sampled_frames=len(frames),
        )


def compute_visual_anomaly_scores(
    profiles: Sequence[EpisodeVisualProfile],
    k_neighbors: int = 5,
) -> dict[str, np.ndarray]:
    """Computes multidimensional visual anomaly scores across episodes.

    Returns dict containing:
        - 'embedding_anomaly': Distance to k-NN or dataset centroid in embedding space.
        - 'lighting_anomaly': Deviation in brightness/lighting from the dataset norm.
        - 'sharpness_anomaly': Low-sharpness outlier score (blurry or camera smudge).
        - 'composite_visual_score': Weighted normalized composite visual anomaly score [0, 1].
    """
    n = len(profiles)
    if n == 0:
        return {
            "embedding_anomaly": np.array([]),
            "lighting_anomaly": np.array([]),
            "sharpness_anomaly": np.array([]),
            "composite_visual_score": np.array([]),
        }

    embeddings = np.array([p.mean_embedding for p in profiles], dtype=np.float64)  # (N, D)
    brightness = np.array([p.brightness_mean for p in profiles], dtype=np.float64)
    sharpness = np.array([p.sharpness_score for p in profiles], dtype=np.float64)

    # 1. Embedding anomaly via distance to dataset median & k-NN
    # Compute pairwise cosine distance: 1 - <u, v>
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7
    normed_emb = embeddings / norms
    sim_matrix = np.dot(normed_emb, normed_emb.T)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    if n > 1:
        k = min(k_neighbors, n - 1)
        sorted_dists = np.sort(dist_matrix, axis=1)
        knn_dist = np.mean(sorted_dists[:, 1 : k + 1], axis=1)
        emb_min, emb_max = np.min(knn_dist), np.max(knn_dist)
        emb_score = (knn_dist - emb_min) / (emb_max - emb_min) if emb_max > emb_min else np.zeros(n)
    else:
        emb_score = np.zeros(n)

    # 2. Lighting anomaly: absolute z-score from median brightness
    b_med = np.median(brightness)
    b_mad = np.median(np.abs(brightness - b_med)) + 1e-5
    b_score = np.clip(np.abs(brightness - b_med) / (3.0 * b_mad), 0.0, 1.0)

    # 3. Blur / sharpness anomaly: lower sharpness than normal
    sh_med = np.median(sharpness)
    sh_mad = np.median(np.abs(sharpness - sh_med)) + 1e-5
    # Only penalize unusually low sharpness (blur)
    sh_score = np.clip((sh_med - sharpness) / (3.0 * sh_mad), 0.0, 1.0)

    # Composite visual anomaly score
    composite = 0.5 * emb_score + 0.3 * b_score + 0.2 * sh_score

    return {
        "embedding_anomaly": emb_score,
        "lighting_anomaly": b_score,
        "sharpness_anomaly": sh_score,
        "composite_visual_score": composite,
    }
