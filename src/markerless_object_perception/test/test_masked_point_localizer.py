"""Tests for mask-filtered, ROS-independent stereo point localization."""

from markerless_object_perception.masked_point_localizer import (
    MaskedPointLocalizer,
    MaskedPointLocalizerConfig,
)
import numpy as np
import pytest


def make_localizer(**overrides):
    """Create a small-test configuration with explicit quality limits."""
    values = {
        'min_depth_m': 0.1,
        'max_depth_m': 2.0,
        'min_valid_points': 5,
        'outlier_mad_scale': 3.5,
        'min_outlier_scale_m': 1.0e-4,
        'max_spread_m': 0.05,
    }
    values.update(overrides)
    return MaskedPointLocalizer(MaskedPointLocalizerConfig(**values))


def symmetric_cluster(center, height=5, width=5, spacing=0.002):
    """Create an HxWx3 point cluster whose component medians equal center."""
    row, column = np.mgrid[:height, :width]
    x_offset = (column - (width - 1) / 2.0) * spacing
    y_offset = (row - (height - 1) / 2.0) * spacing
    points = np.empty((height, width, 3), dtype=np.float64)
    points[:, :, 0] = center[0] + x_offset
    points[:, :, 1] = center[1] + y_offset
    points[:, :, 2] = center[2]
    return points


def test_synthetic_cluster_recovers_known_3d_center():
    expected = np.array([0.35, -0.08, 0.75])
    points = symmetric_cluster(expected)
    mask = np.ones(points.shape[:2], dtype=bool)

    result = make_localizer().localize(mask, points)

    assert result.valid
    assert result.reason == 'localized'
    np.testing.assert_allclose(result.point, expected, atol=1.0e-12)
    assert result.masked_point_count == 25
    assert result.valid_point_count == 25
    assert result.inlier_count == 25
    assert result.spread_m < 0.01


def test_mask_excludes_background_points():
    expected = np.array([0.2, 0.1, 0.6])
    points = np.full((5, 5, 3), [1.5, -1.0, 1.8], dtype=np.float64)
    points[1:4, 1:4] = symmetric_cluster(expected, 3, 3)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    result = make_localizer().localize(mask, points)

    assert result.valid
    np.testing.assert_allclose(result.point, expected, atol=1.0e-12)
    assert result.masked_point_count == 9


def test_float32_masked_points_keep_localization_precision():
    expected = np.array([0.2, 0.1, 0.6], dtype=np.float32)
    points = np.full((5, 5, 3), [9.0, 9.0, 9.0], dtype=np.float32)
    points[1:4, 1:4] = expected
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    result = make_localizer().localize(mask, points)

    assert result.valid
    assert result.point == pytest.approx(
        tuple(float(value) for value in expected)
    )
    assert result.valid_point_count == 9


def test_invalid_depth_and_far_outlier_do_not_pollute_center():
    expected = np.array([0.1, -0.2, 0.8])
    points = np.broadcast_to(expected, (6, 6, 3)).copy()
    mask = np.ones((6, 6), dtype=bool)
    points[0, 0] = [np.nan, 0.0, 0.8]
    points[0, 1] = [0.0, np.inf, 0.8]
    points[0, 2] = [0.0, 0.0, -0.5]
    points[0, 3] = [10.0, 10.0, 0.8]

    result = make_localizer().localize(mask, points)

    assert result.valid
    np.testing.assert_allclose(result.point, expected, atol=1.0e-12)
    assert result.valid_point_count == 33
    assert result.inlier_count == 32


def test_too_few_valid_points_fails_closed():
    points = np.full((3, 3, 3), [0.0, 0.0, 0.5], dtype=np.float64)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, :2] = True

    result = make_localizer(min_valid_points=3).localize(mask, points)

    assert not result.valid
    assert result.point is None
    assert result.reason == 'insufficient_valid_points'
    assert result.masked_point_count == 2
    assert result.valid_point_count == 2


def test_two_separated_clusters_are_rejected_as_excessively_spread():
    points = np.empty((4, 5, 3), dtype=np.float64)
    points[:2, :, :] = [0.0, 0.0, 0.8]
    points[2:, :, :] = [0.25, 0.0, 0.8]
    mask = np.ones((4, 5), dtype=bool)

    result = make_localizer(
        min_valid_points=10,
        max_spread_m=0.05,
    ).localize(mask, points)

    assert not result.valid
    assert result.reason == 'excessive_spread'
    assert result.inlier_count == 20
    assert result.spread_m == pytest.approx(0.125)


@pytest.mark.parametrize(
    ('mask', 'points', 'message'),
    [
        (
            np.ones((2, 2, 1), dtype=bool),
            np.zeros((2, 2, 3)),
            'mask must have shape HxW',
        ),
        (
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2, 2)),
            'xyz_points must have shape HxWx3',
        ),
        (
            np.ones((2, 3), dtype=bool),
            np.zeros((2, 2, 3)),
            'image shapes must match',
        ),
        (
            np.ones((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 3)),
            'mask must be a boolean array',
        ),
    ],
)
def test_invalid_input_shapes_or_mask_type_are_rejected(mask, points, message):
    with pytest.raises(ValueError, match=message):
        make_localizer().localize(mask, points)


@pytest.mark.parametrize(
    'overrides',
    [
        {'min_depth_m': -0.1},
        {'max_depth_m': 0.1},
        {'min_valid_points': 0},
        {'outlier_mad_scale': 0.0},
        {'min_outlier_scale_m': float('nan')},
        {'max_spread_m': -1.0},
    ],
)
def test_invalid_quality_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_localizer(**overrides)
