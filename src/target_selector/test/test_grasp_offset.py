"""Unit tests for the marker-to-grasp offset math.

Pure geometry, no ROS runtime or camera needed. Verifies the rigid
composition is interpreted in the marker's local frame.
"""

import math
import os

import numpy as np
import pytest
from geometry_msgs.msg import Pose

from target_selector.selector_node import (
    apply_grasp_offset,
    load_grasp_offsets,
    matrix_to_quaternion,
    offset_for_marker,
    quaternion_to_matrix,
    rpy_to_matrix,
)

IDENTITY_OFFSET = {
    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation_rpy": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
}


def make_pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def test_identity_offset_is_noop():
    marker = make_pose(0.1, -0.2, 0.4)
    grasp = apply_grasp_offset(marker, IDENTITY_OFFSET)
    assert grasp.position.x == pytest.approx(0.1)
    assert grasp.position.y == pytest.approx(-0.2)
    assert grasp.position.z == pytest.approx(0.4)


def test_translation_offset_is_local_when_marker_unrotated():
    marker = make_pose(0.0, 0.0, 0.5)
    offset = {
        "translation": {"x": 0.0, "y": 0.0, "z": -0.03},
        "rotation_rpy": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }
    grasp = apply_grasp_offset(marker, offset)
    # Marker unrotated, so local -z equals world -z.
    assert grasp.position.z == pytest.approx(0.47)


def test_offset_follows_marker_orientation():
    # Marker yawed 90 deg about z: its local +x now points along world +y.
    qx, qy, qz, qw = matrix_to_quaternion(rpy_to_matrix(0.0, 0.0, math.pi / 2))
    marker = make_pose(0.0, 0.0, 0.0, qx, qy, qz, qw)
    offset = {
        "translation": {"x": 0.1, "y": 0.0, "z": 0.0},
        "rotation_rpy": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }
    grasp = apply_grasp_offset(marker, offset)
    # Local +x (0.1) maps to world +y.
    assert grasp.position.x == pytest.approx(0.0, abs=1e-9)
    assert grasp.position.y == pytest.approx(0.1)
    assert grasp.position.z == pytest.approx(0.0, abs=1e-9)


def test_rotation_offset_composes():
    marker = make_pose(0.0, 0.0, 0.0)
    offset = {
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation_rpy": {"roll": math.pi, "pitch": 0.0, "yaw": 0.0},
    }
    grasp = apply_grasp_offset(marker, offset)
    rot = quaternion_to_matrix(
        grasp.orientation.x,
        grasp.orientation.y,
        grasp.orientation.z,
        grasp.orientation.w,
    )
    expected = rpy_to_matrix(math.pi, 0.0, 0.0)
    np.testing.assert_allclose(rot, expected, atol=1e-9)


def test_load_and_select_offsets_from_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "grasp_offsets.yaml"
    )
    offsets = load_grasp_offsets(config_path)
    assert 0 in offsets["markers"]
    marker0 = offset_for_marker(offsets, 0)
    assert marker0["translation"]["z"] == pytest.approx(-0.03)
    # Unknown ID falls back to default (identity).
    unknown = offset_for_marker(offsets, 999)
    assert unknown["translation"]["z"] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
