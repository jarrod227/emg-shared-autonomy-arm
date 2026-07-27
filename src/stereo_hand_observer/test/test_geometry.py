"""Synthetic calibrated-camera tests for stereo hand-keypoint geometry."""

import numpy as np
import pytest

from stereo_hand_observer.geometry import (
    StereoGeometryError,
    epipolar_error,
    fundamental_from_projections,
    project_point,
    triangulate_keypoint,
)


FOCAL_LENGTH_PX = 800.0
PRINCIPAL_POINT = np.array([320.0, 240.0])
BASELINE_M = 0.12
INTRINSICS = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, PRINCIPAL_POINT[0]],
        [0.0, FOCAL_LENGTH_PX, PRINCIPAL_POINT[1]],
        [0.0, 0.0, 1.0],
    ]
)
LEFT_PROJECTION = INTRINSICS @ np.hstack(
    (np.eye(3), np.zeros((3, 1)))
)
RIGHT_PROJECTION = INTRINSICS @ np.hstack(
    (
        np.eye(3),
        np.array([[-BASELINE_M], [0.0], [0.0]]),
    )
)
# For rectified horizontal stereo, corresponding pixels have the same y.
FUNDAMENTAL_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)


def independently_project(point, right_camera=False):
    """Project with the pinhole equations, independently of production code."""
    camera_x = point[0] - (BASELINE_M if right_camera else 0.0)
    return np.array(
        [
            FOCAL_LENGTH_PX * camera_x / point[2] + PRINCIPAL_POINT[0],
            FOCAL_LENGTH_PX * point[1] / point[2] + PRINCIPAL_POINT[1],
        ]
    )


def test_recovers_known_3d_point_from_corresponding_pixels():
    expected = np.array([0.12, -0.04, 1.2])
    left_pixel = independently_project(expected)
    right_pixel = independently_project(expected, right_camera=True)

    result = triangulate_keypoint(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        FUNDAMENTAL_MATRIX,
        left_pixel,
        right_pixel,
    )

    np.testing.assert_allclose(result.point, expected, atol=1e-10)
    assert result.epipolar_error_px == pytest.approx(0.0, abs=1e-12)
    assert result.max_reprojection_error_px == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_project_point_matches_independent_pinhole_equations():
    point = np.array([-0.08, 0.06, 0.9])
    np.testing.assert_allclose(
        project_point(LEFT_PROJECTION, point),
        independently_project(point),
    )
    np.testing.assert_allclose(
        project_point(RIGHT_PROJECTION, point),
        independently_project(point, right_camera=True),
    )


def test_fundamental_matrix_is_derived_from_camera_info_projections():
    fundamental = fundamental_from_projections(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
    )
    point = np.array([0.05, 0.02, 1.0])

    error = epipolar_error(
        fundamental,
        independently_project(point),
        independently_project(point, right_camera=True),
    )

    assert error == pytest.approx(0.0, abs=1e-10)


def test_identical_projection_matrices_have_no_stereo_baseline():
    with pytest.raises(StereoGeometryError, match="baseline"):
        fundamental_from_projections(
            LEFT_PROJECTION,
            LEFT_PROJECTION,
        )


def test_epipolar_check_rejects_vertical_keypoint_mismatch():
    point = np.array([0.05, 0.02, 1.0])
    left_pixel = independently_project(point)
    right_pixel = independently_project(point, right_camera=True)
    right_pixel[1] += 8.0

    assert epipolar_error(
        FUNDAMENTAL_MATRIX,
        left_pixel,
        right_pixel,
    ) == pytest.approx(8.0)
    with pytest.raises(StereoGeometryError, match="epipolar-error"):
        triangulate_keypoint(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            left_pixel,
            right_pixel,
            max_epipolar_error_px=1.0,
        )


def test_reprojection_check_can_reject_a_bad_pair():
    point = np.array([0.05, 0.02, 1.0])
    left_pixel = independently_project(point)
    right_pixel = independently_project(point, right_camera=True)
    right_pixel[1] += 8.0

    with pytest.raises(StereoGeometryError, match="reprojection-error"):
        triangulate_keypoint(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            left_pixel,
            right_pixel,
            max_epipolar_error_px=10.0,
            max_reprojection_error_px=1.0,
        )


def test_zero_disparity_is_rejected_as_point_at_infinity():
    pixel = np.array([400.0, 240.0])

    with pytest.raises(StereoGeometryError, match="finite 3D point"):
        triangulate_keypoint(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            pixel,
            pixel,
        )


def test_point_behind_cameras_is_rejected():
    point = np.array([0.05, 0.02, -1.0])
    left_pixel = independently_project(point)
    right_pixel = independently_project(point, right_camera=True)

    with pytest.raises(StereoGeometryError, match="behind"):
        triangulate_keypoint(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            left_pixel,
            right_pixel,
        )


@pytest.mark.parametrize(
    "bad_pixel",
    (
        np.array([1.0, 2.0, 3.0]),
        np.array([np.nan, 2.0]),
    ),
)
def test_invalid_pixel_input_is_rejected(bad_pixel):
    with pytest.raises(ValueError, match="left_pixel"):
        triangulate_keypoint(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            bad_pixel,
            np.array([1.0, 2.0]),
        )
