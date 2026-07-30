"""Tests for the markerless candidate ROS-to-pure adapter."""

from assistive_interfaces.msg import (
    ObjectCandidate,
    ObjectCandidateArray,
)
import pytest

from target_selector.candidate_adapter import candidate_frame_from_message


def candidate(track_id, class_label, position):
    """Build one ROS candidate with nominal confidence values."""
    message = ObjectCandidate()
    message.track_id = track_id
    message.class_label = class_label
    message.class_confidence = 0.9
    message.position.x, message.position.y, message.position.z = position
    message.localization_confidence = 0.8
    return message


def test_preserves_source_metadata_candidates_and_order():
    message = ObjectCandidateArray()
    message.header.stamp.sec = 1_785_174_214
    message.header.stamp.nanosec = 909_537_099
    message.header.frame_id = 'stereo_left_optical'
    message.valid = True
    message.pair_skew_sec = 0.006
    message.candidates = [
        candidate(9, 'cell_phone', (0.3, -0.1, 0.8)),
        candidate(2, 'cup', (0.2, 0.1, 0.7)),
    ]

    frame = candidate_frame_from_message(message)

    assert frame.source_time_sec == pytest.approx(
        1_785_174_214.909_537_099
    )
    assert frame.frame_id == 'stereo_left_optical'
    assert frame.valid
    assert frame.pair_skew_sec == pytest.approx(0.006)
    assert [item.track_id for item in frame.candidates] == [9, 2]
    assert frame.candidates[0].class_label == 'cell_phone'
    assert frame.candidates[0].position == pytest.approx((0.3, -0.1, 0.8))
    assert frame.candidates[0].class_confidence == pytest.approx(0.9)
    assert frame.candidates[0].localization_confidence == pytest.approx(0.8)


@pytest.mark.parametrize('valid', (True, False))
def test_empty_observation_preserves_validity(valid):
    message = ObjectCandidateArray()
    message.header.stamp.sec = 10
    message.header.frame_id = 'stereo_frame'
    message.valid = valid

    frame = candidate_frame_from_message(message)

    assert frame.valid is valid
    assert frame.candidates == ()


def test_invalid_ros_stamp_is_rejected():
    message = ObjectCandidateArray()
    message.header.stamp.nanosec = 1_000_000_000

    with pytest.raises(ValueError, match='non-negative ROS time'):
        candidate_frame_from_message(message)


def test_wrong_message_type_is_rejected():
    with pytest.raises(TypeError, match='ObjectCandidateArray'):
        candidate_frame_from_message(object())
