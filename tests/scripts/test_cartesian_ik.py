"""Targeted unit tests for Cartesian action conversion and differential IK roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.utils.import_utils import _placo_available

pytestmark = pytest.mark.skipif(not _placo_available, reason="placo is required for kinematics tests")


@pytest.fixture
def kinematics():
    from lerobot.model.kinematics import RobotKinematics

    repo_root = Path(__file__).resolve().parents[2]
    urdf_path = repo_root / "assets" / "so101_kinematics.urdf"
    if not urdf_path.exists():
        pytest.skip(f"URDF not found at {urdf_path}")
    kin = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link")
    return kin


def test_joints_to_cartesian_and_ik_roundtrip(kinematics):
    from scripts.video_vam.train_smolexpert import (
        cartesian_to_joints_ik,
        joints_to_cartesian,
    )

    # Sample realistic SO-101 joint configurations (in degrees)
    joints_np = np.array(
        [
            [2.0, -98.0, 98.4, 63.0, -55.0, 4.9],
            [-1.5, -98.4, 97.7, 56.2, -55.6, 4.5],
            [15.0, -85.0, 90.0, 45.0, -30.0, 50.0],
            [-25.0, -90.0, 95.0, 50.0, -70.0, 10.0],
        ],
        dtype=np.float32,
    )
    joints = torch.from_numpy(joints_np).unsqueeze(0)  # [1, 4, 6]
    q_start = joints[:, 0, :]

    # Forward kinematics
    cart = joints_to_cartesian(joints, kinematics)
    assert cart.shape == (1, 4, 7), f"Expected 7-dim Cartesian targets, got {cart.shape}"

    # Inverse kinematics reconstruction
    rec_joints = cartesian_to_joints_ik(q_start, cart, kinematics, num_iters=5)
    assert rec_joints.shape == (1, 4, 6)

    # Check reconstruction error
    err = (rec_joints - joints).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()

    assert max_err < 0.05, f"Expected max reconstruction error < 0.05 deg, got {max_err:.4f} deg"
    assert mean_err < 0.01, f"Expected mean reconstruction error < 0.01 deg, got {mean_err:.4f} deg"


def test_yaw_and_roll_preservation_in_cartesian(kinematics):
    from lerobot.utils.rotation import Rotation
    from scripts.video_vam.train_smolexpert import joints_to_cartesian

    # Test that different base yaw (joint 0) correctly reflects in the rotation vector
    q1 = torch.tensor([[[0.0, -90.0, 90.0, 45.0, 0.0, 0.0]]], dtype=torch.float32)
    q2 = torch.tensor([[[45.0, -90.0, 90.0, 45.0, 0.0, 0.0]]], dtype=torch.float32)

    cart1 = joints_to_cartesian(q1, kinematics)[0, 0]
    cart2 = joints_to_cartesian(q2, kinematics)[0, 0]

    rotvec1 = cart1[3:6].numpy()
    rotvec2 = cart2[3:6].numpy()

    # Rotation vectors must be substantially distinct
    rot_diff = np.linalg.norm(rotvec1 - rotvec2)
    assert rot_diff > 0.3, f"Yaw rotation must alter rotvec; diff was only {rot_diff:.4f}"

    # Check that rotation matrices are orthogonal and proper SO(3)
    rot1 = Rotation.from_rotvec(rotvec1).as_matrix()
    rot2 = Rotation.from_rotvec(rotvec2).as_matrix()
    assert np.allclose(rot1 @ rot1.T, np.eye(3), atol=1e-6)
    assert np.allclose(rot2 @ rot2.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rot1), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.det(rot2), 1.0, atol=1e-6)
