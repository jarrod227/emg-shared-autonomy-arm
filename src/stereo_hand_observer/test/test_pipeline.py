"""End-to-end synthetic-keypoint tests for the Objective 4.2 pipeline."""

import numpy as np
import pytest

from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointSet,
)


FOCAL_LENGTH_PX = 800.0
BASELINE_M = 0.12
INTRINSICS = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, 320.0],
        [0.0, FOCAL_LENGTH_PX, 240.0],
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
FUNDAMENTAL_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)
GROUND_TRUTH = np.array([0.4, 0.3, 1.0])
# Palm-knuckle offsets chosen so every per-axis median equals GROUND_TRUTH.
KNUCKLE_POINTS = {
    5: GROUND_TRUTH + np.array([-0.03, 0.0, 0.0]),
    9: GROUND_TRUTH + np.array([-0.01, 0.01, 0.0]),
    13: GROUND_TRUTH + np.array([0.01, -0.01, 0.0]),
    17: GROUND_TRUTH + np.array([0.03, 0.0, 0.0]),
}


def project_independently(point, right_camera=False):
    """Project a synthetic 3D point without calling production code."""
    camera_x = point[0] - (BASELINE_M if right_camera else 0.0)
    return (
        FOCAL_LENGTH_PX * camera_x / point[2] + 320.0,
        FOCAL_LENGTH_PX * point[1] / point[2] + 240.0,
    )


def make_pipeline(required_frames=3):
    """Create a pipeline whose delivery volume contains the ground truth."""
    return StereoHandPipeline(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        FUNDAMENTAL_MATRIX,
        DeliveryVolume(center=tuple(GROUND_TRUTH), radius_m=0.2),
        gate_config=StabilityGateConfig(
            required_frames=required_frames,
            min_confidence=0.7,
            max_pair_skew_sec=0.02,
            max_reprojection_error_px=1.5,
            max_age_sec=0.2,
            max_point_step_m=0.05,
        ),
    )


def make_keypoint_set(
    points=None,
    *,
    left_time=10.0,
    right_time=10.01,
    left_confidence=0.9,
    right_confidence=0.8,
    right_pixel_overrides=None,
):
    """Build corresponding multi-landmark pixel sets from 3D points."""
    if points is None:
        points = KNUCKLE_POINTS
    left_pixels = {
        index: project_independently(point)
        for index, point in points.items()
    }
    right_pixels = {
        index: project_independently(point, right_camera=True)
        for index, point in points.items()
    }
    if right_pixel_overrides:
        right_pixels.update(right_pixel_overrides)
    return StereoKeypointSet(
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        left_source_time_sec=left_time,
        right_source_time_sec=right_time,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
    )


def shifted_right_pixel(index, vertical_px):
    """Project one knuckle into the right view and break its epipolar row."""
    pixel = np.asarray(
        project_independently(KNUCKLE_POINTS[index], right_camera=True)
    )
    pixel[1] += vertical_px
    return tuple(pixel)


def test_consensus_median_point_becomes_valid_on_third_set():
    pipeline = make_pipeline()

    results = []
    for index in range(3):
        left_time = 10.0 + index * 0.03
        keypoint_set = make_keypoint_set(
            left_time=left_time,
            right_time=left_time + 0.01,
        )
        results.append(
            pipeline.process_set(keypoint_set, now_sec=left_time + 0.02)
        )

    assert not results[0].valid
    assert not results[1].valid
    assert results[2].valid
    np.testing.assert_allclose(results[2].point, GROUND_TRUTH, atol=1e-9)
    assert results[2].confidence == 0.8
    assert "consensus=4/4" in results[2].diagnostic
    assert "xyz=(0.400, 0.300, 1.000)m" in results[2].diagnostic


def test_single_bad_landmark_is_outvoted_by_consensus():
    pipeline = make_pipeline(required_frames=1)
    keypoint_set = make_keypoint_set(
        right_pixel_overrides={9: shifted_right_pixel(9, 12.6)},
    )

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert result.valid
    np.testing.assert_allclose(result.point, (0.41, 0.3, 1.0), atol=1e-9)
    assert "consensus=3/4" in result.diagnostic


def test_too_few_geometric_survivors_fail_closed():
    pipeline = make_pipeline(required_frames=1)
    keypoint_set = make_keypoint_set(
        right_pixel_overrides={
            9: shifted_right_pixel(9, 8.0),
            13: shifted_right_pixel(13, 8.0),
        },
    )

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert not result.valid
    assert result.reason == "insufficient_consensus"
    assert "2/3 landmarks passed" in result.diagnostic
    assert "epipolar-error" in result.diagnostic


def test_consistent_wrong_depth_landmark_is_dropped_by_palm_span():
    points = dict(KNUCKLE_POINTS)
    points[17] = GROUND_TRUTH + np.array([0.03, 0.0, 0.3])
    pipeline = make_pipeline(required_frames=1)

    result = pipeline.process_set(
        make_keypoint_set(points),
        now_sec=10.02,
    )

    assert result.valid
    np.testing.assert_allclose(result.point, (0.39, 0.3, 1.0), atol=1e-9)
    assert "consensus=3/4" in result.diagnostic


def test_scattered_palm_cluster_fails_closed():
    points = dict(KNUCKLE_POINTS)
    points[13] = GROUND_TRUTH + np.array([0.01, -0.01, 0.3])
    points[17] = GROUND_TRUTH + np.array([0.03, 0.0, 0.3])
    pipeline = make_pipeline(required_frames=1)

    result = pipeline.process_set(
        make_keypoint_set(points),
        now_sec=10.02,
    )

    assert not result.valid
    assert result.reason == "palm_cluster_rejected"


@pytest.mark.parametrize("empty", (None, {}))
def test_missing_side_pixel_set_fails_closed(empty):
    pipeline = make_pipeline(required_frames=1)
    base = make_keypoint_set()
    keypoint_set = StereoKeypointSet(
        left_pixels=base.left_pixels,
        right_pixels=empty,
        left_source_time_sec=base.left_source_time_sec,
        right_source_time_sec=base.right_source_time_sec,
        left_confidence=base.left_confidence,
        right_confidence=base.right_confidence,
    )

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert not result.valid
    assert result.point is None
    assert result.reason == "missing_keypoint"


def test_disjoint_landmark_indices_cannot_reach_consensus():
    pipeline = make_pipeline(required_frames=1)
    base = make_keypoint_set()
    keypoint_set = StereoKeypointSet(
        left_pixels={5: base.left_pixels[5]},
        right_pixels={17: base.right_pixels[17]},
        left_source_time_sec=base.left_source_time_sec,
        right_source_time_sec=base.right_source_time_sec,
        left_confidence=base.left_confidence,
        right_confidence=base.right_confidence,
    )

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert not result.valid
    assert result.reason == "insufficient_consensus"
    assert "0/3 landmarks passed" in result.diagnostic


def test_set_lowest_view_confidence_controls_output_and_validity():
    pipeline = make_pipeline(required_frames=1)
    keypoint_set = make_keypoint_set(
        left_confidence=0.95,
        right_confidence=0.6,
    )

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert not result.valid
    assert result.reason == "low_confidence"
    assert result.confidence == 0.6


def test_set_pair_skew_is_rejected_before_geometry():
    pipeline = make_pipeline(required_frames=1)
    keypoint_set = make_keypoint_set(left_time=10.0, right_time=10.03)

    result = pipeline.process_set(keypoint_set, now_sec=10.04)

    assert not result.valid
    assert result.reason == "excessive_pair_skew"


def test_set_with_non_finite_metadata_is_invalid():
    pipeline = make_pipeline(required_frames=1)
    keypoint_set = make_keypoint_set(left_time=float("nan"))

    result = pipeline.process_set(keypoint_set, now_sec=10.02)

    assert not result.valid
    assert result.reason == "invalid_pair_metadata"


def test_process_set_rejects_a_foreign_payload_type():
    pipeline = make_pipeline()

    with pytest.raises(TypeError, match="StereoKeypointSet"):
        pipeline.process_set(
            {"left_pixels": {}, "right_pixels": {}},
            now_sec=10.0,
        )


@pytest.mark.parametrize(
    "options",
    (
        {"min_consensus_points": 0},
        {"min_consensus_points": 2.5},
        {"max_palm_span_m": 0.0},
        {"max_palm_span_m": float("nan")},
    ),
)
def test_invalid_consensus_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        StereoHandPipeline(
            LEFT_PROJECTION,
            RIGHT_PROJECTION,
            FUNDAMENTAL_MATRIX,
            DeliveryVolume(center=tuple(GROUND_TRUTH), radius_m=0.2),
            **options,
        )
