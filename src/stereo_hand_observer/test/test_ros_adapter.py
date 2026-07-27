"""Tests for conversion into the existing HandObservation ROS contract."""

import pytest

from stereo_hand_observer.pipeline import PipelineResult
from stereo_hand_observer.ros_adapter import hand_observation_from_result


def make_result(**overrides):
    """Build one nominal stable pipeline result."""
    values = {
        "valid": True,
        "point": (0.4, 0.3, 1.0),
        "confidence": 0.9,
        "pair_skew_sec": 0.01,
        "reprojection_error_px": 0.4,
        "source_time_sec": 12.25,
        "stable_frames": 3,
        "reason": "stable",
    }
    values.update(overrides)
    return PipelineResult(**values)


def test_valid_result_populates_frozen_message_contract():
    message = hand_observation_from_result(make_result(), "world")

    assert message.valid
    assert message.header.frame_id == "world"
    assert message.header.stamp.sec == 12
    assert message.header.stamp.nanosec == 250_000_000
    assert (message.point.x, message.point.y, message.point.z) == (
        0.4,
        0.3,
        1.0,
    )
    assert message.confidence == pytest.approx(0.9)
    assert message.pair_skew_sec == pytest.approx(0.01)
    assert message.reprojection_error == pytest.approx(0.4)


def test_invalid_result_is_an_explicit_no_hand_message():
    message = hand_observation_from_result(
        make_result(valid=False, point=None, reason="missing_keypoint"),
        "world",
    )

    assert not message.valid
    assert (message.point.x, message.point.y, message.point.z) == (
        0.0,
        0.0,
        0.0,
    )


def test_integer_nanoseconds_preserve_large_epoch_stamp_exactly():
    source_nanoseconds = 1_785_174_214_909_537_099

    message = hand_observation_from_result(
        make_result(),
        "world",
        source_time_nanoseconds=source_nanoseconds,
    )

    assert message.header.stamp.sec == 1_785_174_214
    assert message.header.stamp.nanosec == 909_537_099


def test_invalid_confidence_is_clamped_to_message_contract():
    message = hand_observation_from_result(
        make_result(valid=False, point=None, confidence=1.2),
        "world",
    )

    assert message.confidence == 1.0


def test_valid_result_without_point_is_rejected():
    with pytest.raises(ValueError, match="must carry"):
        hand_observation_from_result(
            make_result(point=None),
            "world",
        )
