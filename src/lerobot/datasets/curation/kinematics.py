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

"""Kinematic analysis and trajectory quality metrics for robotic episodes.

Computes joint velocities, accelerations, jerk, trajectory smoothness, path lengths,
gripper event transitions, and automated idle trim suggestions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class KinematicProfile:
    """Summary of kinematic metrics for an individual trajectory."""

    duration_seconds: float
    num_frames: int
    fps: float
    path_length: float
    mean_velocity: float
    max_velocity: float
    mean_acceleration: float
    max_acceleration: float
    jerk_metric: float  # Mean squared jerk
    smoothness_score: float  # Log dimensionless jerk or normalized smoothness in [0, 1]
    suggested_trim_start: int
    suggested_trim_end: int
    idle_start_duration_s: float
    idle_end_duration_s: float
    gripper_actuations: int
    is_jerk_outlier: bool = False
    is_velocity_outlier: bool = False
    is_length_outlier: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def compute_kinematics(
    positions: np.ndarray | torch.Tensor,
    fps: float = 10.0,
    timestamps: np.ndarray | torch.Tensor | None = None,
    gripper_index: int | None = -1,
    velocity_threshold: float = 0.05,
    min_moving_frames: int = 3,
) -> KinematicProfile:
    """Computes comprehensive kinematic features and suggested trimming bounds.

    Args:
        positions: Array of shape (T, D) representing joint/state positions over time.
        fps: Recording frames per second.
        timestamps: Optional 1D array of shape (T,) with physical timestamps.
        gripper_index: Index of the gripper dimension in positions, if present (default -1).
        velocity_threshold: Threshold below which the arm is considered stationary.
        min_moving_frames: Number of consecutive moving frames required to declare motion start.

    Returns:
        KinematicProfile dataclass with calculated metrics.
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.detach().cpu().numpy()

    q = np.atleast_2d(positions).astype(np.float64)
    t_steps, d_dims = q.shape

    if t_steps < 2:
        return KinematicProfile(
            duration_seconds=0.0,
            num_frames=t_steps,
            fps=fps,
            path_length=0.0,
            mean_velocity=0.0,
            max_velocity=0.0,
            mean_acceleration=0.0,
            max_acceleration=0.0,
            jerk_metric=0.0,
            smoothness_score=1.0,
            suggested_trim_start=0,
            suggested_trim_end=t_steps,
            idle_start_duration_s=0.0,
            idle_end_duration_s=0.0,
            gripper_actuations=0,
        )

    # Calculate delta t
    if timestamps is not None and len(timestamps) == t_steps:
        dt = np.diff(timestamps)
        # Avoid division by zero
        dt = np.where(dt <= 1e-6, 1.0 / fps, dt)
    else:
        dt = np.full(t_steps - 1, 1.0 / fps, dtype=np.float64)

    # Separate arm joints from gripper if gripper_index is specified
    if gripper_index is not None and 0 <= (gripper_index % d_dims) < d_dims:
        g_idx = gripper_index % d_dims
        arm_dims = [i for i in range(d_dims) if i != g_idx]
        q_arm = q[:, arm_dims]
        q_grip = q[:, g_idx]
    else:
        q_arm = q
        q_grip = None

    # Step-wise displacement in arm space
    diff_arm = np.diff(q_arm, axis=0)  # (T-1, D_arm)
    dt_arm = dt[:, None]

    # Velocities
    vel_arm = diff_arm / dt_arm  # (T-1, D_arm)
    vel_norm = np.linalg.norm(vel_arm, axis=1)  # (T-1,)

    total_path_length = float(np.sum(np.linalg.norm(diff_arm, axis=1)))
    mean_vel = float(np.mean(vel_norm)) if len(vel_norm) > 0 else 0.0
    max_vel = float(np.max(vel_norm)) if len(vel_norm) > 0 else 0.0

    # Accelerations
    if len(vel_arm) >= 2:
        dt_acc = 0.5 * (dt[:-1] + dt[1:])[:, None]
        dt_acc = np.where(dt_acc <= 1e-6, 1.0 / fps, dt_acc)
        acc_arm = np.diff(vel_arm, axis=0) / dt_acc  # (T-2, D_arm)
        acc_norm = np.linalg.norm(acc_arm, axis=1)
        mean_acc = float(np.mean(acc_norm))
        max_acc = float(np.max(acc_norm))
    else:
        acc_norm = np.zeros(1)
        mean_acc = 0.0
        max_acc = 0.0

    # Jerk
    if len(vel_arm) >= 3:
        dt_jerk = dt[1:-1, None]
        dt_jerk = np.where(dt_jerk <= 1e-6, 1.0 / fps, dt_jerk)
        jerk_arm = np.diff(acc_arm, axis=0) / dt_jerk  # (T-3, D_arm)
        jerk_norm_sq = np.sum(jerk_arm**2, axis=1)
        mean_sq_jerk = float(np.mean(jerk_norm_sq))

        # Log dimensionless jerk (smoothness index, higher = smoother)
        # Normalized by duration^3 / path_length^2
        duration = float(np.sum(dt))
        if total_path_length > 1e-5 and duration > 1e-5:
            # Standard logarithmic dimensionless jerk
            raw_dj = float(np.sum(jerk_norm_sq * dt[1:-1]) * (duration**3) / (total_path_length**2 + 1e-6))
            smoothness = float(1.0 / (1.0 + np.log1p(max(0.0, raw_dj))))
        else:
            smoothness = 1.0
    else:
        mean_sq_jerk = 0.0
        smoothness = 1.0

    # Idle Lead-in & Trailing Detection
    is_moving = vel_norm > velocity_threshold
    trim_start = 0
    trim_end = t_steps

    # Find sustained start of motion
    for i in range(len(is_moving) - min_moving_frames + 1):
        if np.all(is_moving[i : i + min_moving_frames]):
            trim_start = max(0, i)
            break

    # Find cessation of motion from the end
    for i in range(len(is_moving) - 1, min_moving_frames - 1, -1):
        if np.all(is_moving[i - min_moving_frames + 1 : i + 1]):
            trim_end = min(t_steps, i + 1)
            break

    if trim_end <= trim_start:
        trim_start = 0
        trim_end = t_steps

    idle_start_s = float(trim_start / fps)
    idle_end_s = float((t_steps - trim_end) / fps)

    # Gripper actuations detection
    gripper_actuations = 0
    if q_grip is not None and len(q_grip) > 1:
        # Check sign changes of gripper movement or crossing midpoint
        g_min, g_max = np.min(q_grip), np.max(q_grip)
        if (g_max - g_min) > 10.0:  # Significant gripper movement
            midpoint = 0.5 * (g_min + g_max)
            binary_state = (q_grip > midpoint).astype(int)
            transitions = np.sum(np.abs(np.diff(binary_state)))
            gripper_actuations = int(transitions)

    duration_s = float(np.sum(dt))

    return KinematicProfile(
        duration_seconds=duration_s,
        num_frames=t_steps,
        fps=fps,
        path_length=total_path_length,
        mean_velocity=mean_vel,
        max_velocity=max_vel,
        mean_acceleration=mean_acc,
        max_acceleration=max_acc,
        jerk_metric=mean_sq_jerk,
        smoothness_score=smoothness,
        suggested_trim_start=int(trim_start),
        suggested_trim_end=int(trim_end),
        idle_start_duration_s=idle_start_s,
        idle_end_duration_s=idle_end_s,
        gripper_actuations=gripper_actuations,
    )
