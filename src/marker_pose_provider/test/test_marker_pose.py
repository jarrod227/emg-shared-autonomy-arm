"""Synthetic-image tests for the marker pose estimation pipeline.

A marker is rendered at an analytically known pose through a virtual
camera, then run through the same detection and PnP code the node uses.
Errors beyond sub-pixel rendering noise can only come from the code, not
from calibration or printing — this is what separates "algorithm wrong"
from "calibration off" when real-camera numbers look bad.
"""

import math

import cv2
import numpy as np
import pytest

from marker_pose_provider.marker_node import (
    detect_marker_ids,
    estimate_marker_pose,
    marker_object_points,
    pick_consistent_solution,
    rotation_angle_between,
    rvec_tvec_to_pose,
)

MARKER_LENGTH = 0.051
CAMERA_MATRIX = np.array(
    [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
)
NO_DISTORTION = np.zeros(5)


def render_marker(rvec, tvec, marker_id=0, image_size=(480, 640)):
    """Render a marker at a known pose through the virtual camera."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_px = 400
    marker_img = cv2.aruco.drawMarker(dictionary, marker_id, marker_px)
    corners_3d = marker_object_points(MARKER_LENGTH)
    projected, _ = cv2.projectPoints(
        corners_3d, rvec, tvec, CAMERA_MATRIX, NO_DISTORTION
    )
    dst_quad = projected.reshape(4, 2).astype(np.float32)
    src_quad = np.array(
        [[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
    canvas = cv2.warpPerspective(
        marker_img,
        transform,
        (image_size[1], image_size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return canvas


def facing_camera_rvec(tilt_x=0.0, tilt_y=0.0):
    """Rotation vector for a marker facing the camera, with optional tilts.

    A marker seen face-on has its +z axis pointing at the camera, i.e. a
    180-degree flip about x relative to the camera frame. rvec = 0 would
    mean "viewed from behind", which renders a mirrored, undetectable
    pattern.
    """
    flip_x, _ = cv2.Rodrigues(np.array([math.pi + tilt_x, 0.0, 0.0]))
    tilt, _ = cv2.Rodrigues(np.array([0.0, tilt_y, 0.0]))
    rvec, _ = cv2.Rodrigues(tilt @ flip_x)
    return rvec


# A clear tilt keeps rotation observable. Near-frontal markers sit in the
# planar-PnP ambiguity valley where sub-pixel corner noise legitimately
# swings the orientation by degrees — that regime is measured in M6, not
# asserted here.
GROUND_TRUTH_RVEC = facing_camera_rvec(tilt_x=0.2, tilt_y=0.5)
GROUND_TRUTH_TVEC = np.array([[0.05], [-0.02], [0.4]])


def test_marker_object_points_shape_and_order():
    points = marker_object_points(MARKER_LENGTH)
    assert points.shape == (4, 3)
    half = MARKER_LENGTH / 2.0
    np.testing.assert_allclose(points[0], [-half, half, 0.0])
    np.testing.assert_allclose(points[2], [half, -half, 0.0])
    assert np.all(points[:, 2] == 0.0)


def test_quaternion_matches_rodrigues():
    rvec = np.array([[0.3], [-0.5], [0.2]])
    tvec = np.array([[0.1], [0.2], [0.3]])
    pose = rvec_tvec_to_pose(rvec, tvec)
    q = pose.orientation
    rot_from_quat = np.array(
        [
            [
                1 - 2 * (q.y**2 + q.z**2),
                2 * (q.x * q.y - q.z * q.w),
                2 * (q.x * q.z + q.y * q.w),
            ],
            [
                2 * (q.x * q.y + q.z * q.w),
                1 - 2 * (q.x**2 + q.z**2),
                2 * (q.y * q.z - q.x * q.w),
            ],
            [
                2 * (q.x * q.z - q.y * q.w),
                2 * (q.y * q.z + q.x * q.w),
                1 - 2 * (q.x**2 + q.y**2),
            ],
        ]
    )
    rot_expected, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(rot_from_quat, rot_expected, atol=1e-9)


def test_detects_expected_id_on_synthetic_image():
    image = render_marker(GROUND_TRUTH_RVEC, GROUND_TRUTH_TVEC, marker_id=7)
    corners, ids = detect_marker_ids(image)
    assert ids is not None
    assert list(ids.flatten()) == [7]
    assert len(corners) == 1


def test_pose_estimate_matches_ground_truth():
    image = render_marker(GROUND_TRUTH_RVEC, GROUND_TRUTH_TVEC)
    corners, ids = detect_marker_ids(image)
    assert ids is not None
    solutions = estimate_marker_pose(
        corners[0], MARKER_LENGTH, CAMERA_MATRIX, NO_DISTORTION
    )
    rvec, tvec, _error = pick_consistent_solution(solutions, None)
    translation_error = np.linalg.norm(tvec - GROUND_TRUTH_TVEC)
    rotation_error = rotation_angle_between(rvec, GROUND_TRUTH_RVEC)
    assert translation_error < 0.001, f"translation off by {translation_error} m"
    assert rotation_error < math.radians(1.0), (
        f"rotation off by {math.degrees(rotation_error)} deg"
    )


def test_pick_consistent_solution_prefers_previous_orientation():
    solution_a = (np.array([[0.1], [0.2], [0.0]]), np.zeros((3, 1)), 0.1)
    solution_b = (np.array([[-0.1], [-0.2], [0.0]]), np.zeros((3, 1)), 0.05)
    previous = np.array([[0.11], [0.19], [0.0]])
    picked = pick_consistent_solution([solution_a, solution_b], previous)
    assert picked is solution_a
    # Without history the lower-error (first-listed) solution wins.
    picked_no_history = pick_consistent_solution([solution_b, solution_a], None)
    assert picked_no_history is solution_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
