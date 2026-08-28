"""Tests for the source-preserving ObjectCandidateArray ROS adapter."""

from dataclasses import replace

from markerless_object_perception.candidate_builder import (
    CandidateBuildResult,
    ObjectCandidate,
)
from markerless_object_perception.ros_adapter import (
    object_candidate_array_from_result,
)
import pytest


def candidate(**overrides):
    """Build one nominal pure candidate."""
    values = {
        'class_label': 'bottle',
        'class_confidence': 0.91,
        'track_id': 7,
        'point': (0.25, -0.1, 0.7),
        'localization_confidence': 0.8,
        'localization_spread_m': 0.01,
        'valid_point_count': 40,
        'inlier_count': 32,
    }
    values.update(overrides)
    return ObjectCandidate(**values)


def build_result(*candidates):
    """Build one result without rejection diagnostics."""
    return CandidateBuildResult(
        candidates=tuple(candidates),
        rejections=(),
    )


def adapt(result, **overrides):
    """Convert with concise nominal source metadata."""
    values = {
        'source_time_nanoseconds': 1_785_174_214_909_537_099,
        'pair_skew_sec': 0.006,
    }
    values.update(overrides)
    return object_candidate_array_from_result(
        result,
        'stereo_left_optical',
        **values,
    )


def test_valid_result_preserves_source_metadata_and_candidate_fields():
    message = adapt(build_result(candidate()))

    assert message.valid
    assert message.header.frame_id == 'stereo_left_optical'
    assert message.header.stamp.sec == 1_785_174_214
    assert message.header.stamp.nanosec == 909_537_099
    assert message.pair_skew_sec == pytest.approx(0.006)
    assert len(message.candidates) == 1
    converted = message.candidates[0]
    assert converted.track_id == 7
    assert converted.class_label == 'bottle'
    assert converted.class_confidence == pytest.approx(0.91)
    assert (
        converted.position.x,
        converted.position.y,
        converted.position.z,
    ) == pytest.approx((0.25, -0.1, 0.7))
    assert converted.localization_confidence == pytest.approx(0.8)


def test_candidate_order_is_not_changed_by_adapter():
    message = adapt(
        build_result(
            candidate(track_id=9),
            candidate(track_id=2, class_label='cup'),
        )
    )

    assert [item.track_id for item in message.candidates] == [9, 2]


def test_fresh_negative_observation_is_valid_with_empty_candidates():
    message = adapt(build_result())

    assert message.valid
    assert message.candidates == []


def test_invalid_input_is_explicit_and_has_no_candidates():
    message = adapt(None, input_valid=False)

    assert not message.valid
    assert message.candidates == []


def test_invalid_input_rejects_ambiguous_build_result():
    with pytest.raises(ValueError, match='must not carry'):
        adapt(build_result(), input_valid=False)


@pytest.mark.parametrize(
    'field,value',
    [
        ('class_confidence', 1.1),
        ('localization_confidence', float('nan')),
        ('track_id', -1),
        ('point', (0.0, float('inf'), 0.5)),
    ],
)
def test_invalid_candidate_metadata_fails_closed(field, value):
    bad_candidate = replace(candidate(), **{field: value})

    with pytest.raises(ValueError):
        adapt(build_result(bad_candidate))


@pytest.mark.parametrize(
    'overrides',
    [
        {'source_time_nanoseconds': -1},
        {'source_time_nanoseconds': 1.5},
        {'pair_skew_sec': -0.1},
        {'pair_skew_sec': float('nan')},
    ],
)
def test_invalid_source_metadata_fails_closed(overrides):
    with pytest.raises(ValueError):
        adapt(build_result(), **overrides)
