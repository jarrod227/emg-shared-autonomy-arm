"""Adapt ROS object-candidate messages to the pure stability-gate model."""

from assistive_interfaces.msg import ObjectCandidateArray

from target_selector.candidate_stability import (
    CandidateFrame,
    CandidateMeasurement,
)


_NANOSECONDS_PER_SECOND = 1_000_000_000


def candidate_frame_from_message(message):
    """Preserve one ROS observation while converting it to pure values."""
    if not isinstance(message, ObjectCandidateArray):
        raise TypeError('message must be an ObjectCandidateArray')

    seconds = message.header.stamp.sec
    nanoseconds = message.header.stamp.nanosec
    if seconds < 0 or not 0 <= nanoseconds < _NANOSECONDS_PER_SECOND:
        raise ValueError('message stamp must be a non-negative ROS time')

    return CandidateFrame(
        source_time_sec=seconds + nanoseconds / _NANOSECONDS_PER_SECOND,
        frame_id=message.header.frame_id,
        valid=message.valid,
        pair_skew_sec=message.pair_skew_sec,
        candidates=tuple(
            CandidateMeasurement(
                track_id=candidate.track_id,
                class_label=candidate.class_label,
                class_confidence=candidate.class_confidence,
                position=(
                    candidate.position.x,
                    candidate.position.y,
                    candidate.position.z,
                ),
                localization_confidence=candidate.localization_confidence,
            )
            for candidate in message.candidates
        ),
    )
