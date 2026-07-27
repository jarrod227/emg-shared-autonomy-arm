"""End-to-end synthetic-keypoint tests for the Objective 4.2 pipeline."""

import numpy as np
import pytest

from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointPair,
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


def make_pair(
    *,
    left_pixel=None,
    right_pixel=None,
    left_time=10.0,
    right_time=10.01,
    left_confidence=0.9,
    right_confidence=0.8,
):
    """Build one corresponding synthetic stereo keypoint pair."""
    if left_pixel is None:
        left_pixel = project_independently(GROUND_TRUTH)
    if right_pixel is None:
        right_pixel = project_independently(
            GROUND_TRUTH,
            right_camera=True,
        )
    return StereoKeypointPair(
        left_pixel=left_pixel,
        right_pixel=right_pixel,
        left_source_time_sec=left_time,
        right_source_time_sec=right_time,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
    )


def test_pipeline_recovers_point_and_becomes_valid_on_third_pair():
    pipeline = make_pipeline()

    results = []
    for index in range(3):
        left_time = 10.0 + index * 0.03
        pair = make_pair(
            left_time=left_time,
            right_time=left_time + 0.01,
        )
        results.append(pipeline.process(pair, now_sec=left_time + 0.02))

    assert not results[0].valid
    assert not results[1].valid
    assert results[2].valid
    np.testing.assert_allclose(results[2].point, GROUND_TRUTH)
    assert results[2].confidence == 0.8
    assert results[2].pair_skew_sec == pytest.approx(0.01)
    assert results[2].source_time_sec == pytest.approx(10.06)


def test_excessive_pair_skew_is_rejected_before_stability_gate():
    pipeline = make_pipeline(required_frames=1)
    pair = make_pair(left_time=10.0, right_time=10.03)

    result = pipeline.process(pair, now_sec=10.04)

    assert not result.valid
    assert result.reason == "excessive_pair_skew"
    assert result.pair_skew_sec == pytest.approx(0.03)


def test_missing_keypoint_publishes_explicit_invalid_result():
    pipeline = make_pipeline(required_frames=1)
    pair = make_pair()
    pair = StereoKeypointPair(
        left_pixel=pair.left_pixel,
        right_pixel=None,
        left_source_time_sec=pair.left_source_time_sec,
        right_source_time_sec=pair.right_source_time_sec,
        left_confidence=pair.left_confidence,
        right_confidence=pair.right_confidence,
    )

    result = pipeline.process(pair, now_sec=10.02)

    assert not result.valid
    assert result.point is None
    assert result.reason == "missing_keypoint"


def test_mismatched_keypoints_are_rejected_by_geometry():
    pipeline = make_pipeline(required_frames=1)
    right_pixel = np.asarray(
        project_independently(GROUND_TRUTH, right_camera=True)
    )
    right_pixel[1] += 8.0
    pair = make_pair(right_pixel=tuple(right_pixel))

    result = pipeline.process(pair, now_sec=10.02)

    assert not result.valid
    assert result.reason == "geometry_rejected"


def test_lowest_view_confidence_controls_output_and_validity():
    pipeline = make_pipeline(required_frames=1)
    pair = make_pair(left_confidence=0.95, right_confidence=0.6)

    result = pipeline.process(pair, now_sec=10.02)

    assert not result.valid
    assert result.reason == "low_confidence"
    assert result.confidence == 0.6
