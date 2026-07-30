"""Convert pure mask/XYZ results into the frozen ROS candidate contract."""

import math
import operator

from assistive_interfaces.msg import ObjectCandidate as ObjectCandidateMessage
from assistive_interfaces.msg import ObjectCandidateArray
from builtin_interfaces.msg import Time

from markerless_object_perception.candidate_builder import (
    CandidateBuildResult,
    ObjectCandidate,
)


_NANOSECONDS_PER_SECOND = 1_000_000_000
_UINT32_MAX = 4_294_967_295


def _stamp_from_nanoseconds(source_time_nanoseconds):
    try:
        total_nanoseconds = operator.index(source_time_nanoseconds)
    except TypeError as error:
        raise ValueError(
            'source_time_nanoseconds must be an integer'
        ) from error
    if isinstance(source_time_nanoseconds, bool) or total_nanoseconds < 0:
        raise ValueError(
            'source_time_nanoseconds must be a non-negative integer'
        )

    stamp = Time()
    stamp.sec = total_nanoseconds // _NANOSECONDS_PER_SECOND
    stamp.nanosec = total_nanoseconds % _NANOSECONDS_PER_SECOND
    return stamp


def object_candidate_array_from_result(
    result,
    frame_id,
    *,
    source_time_nanoseconds,
    pair_skew_sec,
    input_valid=True,
):
    """Build one observation without selecting or reordering candidates."""
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError('frame_id must be a non-empty string')
    if not isinstance(input_valid, bool):
        raise ValueError('input_valid must be a boolean')

    pair_skew_sec = _nonnegative_float(pair_skew_sec, 'pair_skew_sec')
    if input_valid:
        if not isinstance(result, CandidateBuildResult):
            raise TypeError(
                'a valid input must carry a CandidateBuildResult'
            )
    elif result is not None:
        raise ValueError('an invalid input must not carry a build result')

    message = ObjectCandidateArray()
    message.header.stamp = _stamp_from_nanoseconds(
        source_time_nanoseconds
    )
    message.header.frame_id = frame_id
    message.valid = input_valid
    message.pair_skew_sec = pair_skew_sec
    if input_valid:
        message.candidates = [
            _candidate_message(candidate)
            for candidate in result.candidates
        ]
    return message


def _candidate_message(candidate):
    if not isinstance(candidate, ObjectCandidate):
        raise TypeError('result candidates must be ObjectCandidate values')

    try:
        track_id = operator.index(candidate.track_id)
    except TypeError as error:
        raise ValueError('track_id must be an integer') from error
    if (
        isinstance(candidate.track_id, bool)
        or track_id < 0
        or track_id > _UINT32_MAX
    ):
        raise ValueError('track_id must fit a uint32')

    if (
        not isinstance(candidate.class_label, str)
        or not candidate.class_label.strip()
    ):
        raise ValueError('class_label must be a non-empty string')
    class_confidence = _confidence(
        candidate.class_confidence,
        'class_confidence',
    )
    localization_confidence = _confidence(
        candidate.localization_confidence,
        'localization_confidence',
    )

    if candidate.point is None or len(candidate.point) != 3:
        raise ValueError('candidate point must contain three coordinates')
    point = tuple(float(value) for value in candidate.point)
    if not all(math.isfinite(value) for value in point):
        raise ValueError('candidate point must be finite')

    message = ObjectCandidateMessage()
    message.track_id = track_id
    message.class_label = candidate.class_label.strip()
    message.class_confidence = class_confidence
    message.position.x, message.position.y, message.position.z = point
    message.localization_confidence = localization_confidence
    return message


def _nonnegative_float(value, name):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return converted


def _confidence(value, name):
    converted = _nonnegative_float(value, name)
    if converted > 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return converted
