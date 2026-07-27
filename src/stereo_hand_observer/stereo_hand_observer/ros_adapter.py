"""Convert the pure pipeline result into the frozen ROS message contract."""

import math

from assistive_interfaces.msg import HandObservation
from builtin_interfaces.msg import Time

from stereo_hand_observer.pipeline import PipelineResult


_NANOSECONDS_PER_SECOND = 1_000_000_000


def _stamp_from_seconds(source_time_sec):
    source_time_sec = float(source_time_sec)
    if not math.isfinite(source_time_sec) or source_time_sec < 0.0:
        raise ValueError("source_time_sec must be finite and non-negative")
    total_nanoseconds = round(
        source_time_sec * _NANOSECONDS_PER_SECOND
    )
    stamp = Time()
    stamp.sec = total_nanoseconds // _NANOSECONDS_PER_SECOND
    stamp.nanosec = total_nanoseconds % _NANOSECONDS_PER_SECOND
    return stamp


def hand_observation_from_result(result, frame_id):
    """Build one HandObservation without changing the pipeline decision."""
    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string")

    confidence = float(result.confidence)
    pair_skew_sec = float(result.pair_skew_sec)
    reprojection_error_px = float(result.reprojection_error_px)
    metrics = (confidence, pair_skew_sec, reprojection_error_px)
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("observation quality fields must be finite")
    if pair_skew_sec < 0.0 or reprojection_error_px < 0.0:
        raise ValueError("observation error fields must be non-negative")

    message = HandObservation()
    message.header.stamp = _stamp_from_seconds(result.source_time_sec)
    message.header.frame_id = frame_id
    message.valid = result.valid
    message.confidence = max(0.0, min(1.0, confidence))
    message.pair_skew_sec = pair_skew_sec
    message.reprojection_error = reprojection_error_px

    if result.valid:
        if result.point is None or len(result.point) != 3:
            raise ValueError("a valid result must carry one 3D point")
        point = tuple(float(value) for value in result.point)
        if not all(math.isfinite(value) for value in point):
            raise ValueError("a valid result point must be finite")
        message.point.x, message.point.y, message.point.z = point

    return message
