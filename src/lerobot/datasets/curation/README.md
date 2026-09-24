# LeRobot Dataset Observability & Curation Toolkit

A CPU-only, production-grade toolkit for inspecting robotic demonstration datasets, calculating multi-modal trajectory similarity, detecting outliers (proprioceptive, kinematic, and visual), interactive trimming of idle lead-in/tail frames, and exporting clean, curated sub-datasets.

Designed for `LeRobotDataset` (v2 and v3.0), including datasets like `hubnemo/cube_out_of_box_dataset` and `Orellius/cube_out_of_box_v2`.

---

## Key Features

1. **Proprioceptive Trajectory Similarity & DTW:**
   - Multidimensional Dynamic Time Warping (`dtw_distance`) with Sakoe-Chiba constraint band.
   - Pairwise trajectory similarity matrix computation across entire datasets.
   - Dataset Medoid identification (most central, representative demonstration).
   - $k$-Nearest-Neighbor DTW anomaly scoring.

2. **Kinematic & Smoothness Profiling:**
   - Joint velocity, acceleration, and jerk computation.
   - Dimensionless jerk and smoothness scoring.
   - Automated idle detection (`suggested_trim_start` and `suggested_trim_end`).
   - Gripper state transition and actuation tracking.

3. **CPU Visual Outlier Detection:**
   - Lightweight visual embedding extractors running 100% on CPU (`mobilenet`, `dinov2`, or zero-neural-network spatial color moments `stats`).
   - Visual outlier scoring: distance to dataset centroid, lighting/brightness shifts, and blur/sharpness anomaly detection.

4. **Clustering & Composite Outlier Ranking:**
   - Multimodal score fusion: calibrated composite anomaly score in $[0, 100]$.
   - Vectorized K-Means trajectory clustering without scikit-learn dependency.
   - Automated duplicate / near-identical trajectory detection.

5. **Interactive Browser-Based Curation UI:**
   - Zero front-end build step or third-party web framework needed (served via Python built-in HTTP server).
   - Video player with frame scrubber.
   - Interactive dual range slider for frame-level start/end trimming.
   - Real-time synchronized 2D multi-channel joint angle time series canvas.
   - One-click Keep (`K`), Drop (`D`), Review (`R`), and Auto-Trim (`A`).
   - In-UI buttons to save `curation_manifest.json`, export split JSON, or materialize clean datasets.

6. **Clean Sub-Dataset & Split Export:**
   - **Split Export:** JSON manifest with clean `train` and `val` episode index lists for immediate use with `LeRobotDataset(..., episodes=clean_episodes)`.
   - **Materialized Export:** Generates a new `LeRobotDataset` where dropped episodes are excluded, and frames are sliced to `[trim_start, trim_end]` with re-indexed timestamps and re-encoded video clips.

---

## CLI Usage

The tool is accessible via `lerobot-curate` (or `python -m lerobot.scripts.lerobot_curate`).

### 1. Analyze Trajectory Similarities & Outliers

```bash
# Kinematic + Proprioceptive DTW analysis:
lerobot-curate analyze \
    --repo_id hubnemo/cube_out_of_box_dataset \
    --output_dir ./curation_results

# Include CPU visual outlier detection (MobileNet/DINOv2):
lerobot-curate analyze \
    --repo_id hubnemo/cube_out_of_box_dataset \
    --visual \
    --visual_backbone auto \
    --output_dir ./curation_results
```

### 2. Launch Interactive Curation UI

```bash
# Opens interactive web interface at http://localhost:8080:
lerobot-curate ui \
    --repo_id hubnemo/cube_out_of_box_dataset \
    --manifest ./curation_results/curation_manifest.json \
    --port 8080
```

### 3. Export Clean Curated Dataset

```bash
# Materialize a sliced, clean LeRobot dataset:
lerobot-curate export \
    --repo_id hubnemo/cube_out_of_box_dataset \
    --manifest ./curation_results/curation_manifest.json \
    --output_root /path/to/clean_cube_dataset \
    --slice_frames
```

### 4. Print Summary

```bash
lerobot-curate summary --manifest ./curation_results/curation_manifest.json
```

---

## Python API Example

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Pure CPU execution

from lerobot.datasets.curation import DatasetCurator

# 1. Initialize curator
curator = DatasetCurator("hubnemo/cube_out_of_box_dataset")

# 2. Run analysis
manifest = curator.analyze(
    extract_visual=True,
    visual_backbone="auto",
    dtw_window=20,
    outlier_percentile=85.0,
    n_clusters=3,
)

# 3. Inspect results
print("Medoid Episode:", manifest.metadata["medoid_episode_index"])
print("Summary:", manifest.summary())

# 4. Modify / review curation flags
manifest.mark_episode(episode_index=19, status="drop", notes="Abnormal hesitation and teleop chatter")

# 5. Export clean split or materialized dataset
curator.export_split_manifest(manifest, "./clean_split.json")
curator.export_clean_dataset(manifest, "cube_clean", "./clean_cube_dataset", slice_frames=True)
```
