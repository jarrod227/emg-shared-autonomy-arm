"""ROS-independent processing of one synchronized live stereo frame pair."""

import math
import operator

from stereo_hand_observer.keypoint_detector import HandKeypointsDetection
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointSet,
)


def _source_timestamp_ms(source_time_sec, source_time_nanoseconds=None):
    if source_time_nanoseconds is not None:
        if isinstance(source_time_nanoseconds, bool):
            raise ValueError(
                "source time nanoseconds must be a non-negative integer"
            )
        try:
            source_time_nanoseconds = operator.index(
                source_time_nanoseconds
            )
        except TypeError as error:
            raise ValueError(
                "source time nanoseconds must be a non-negative integer"
            ) from error
        if source_time_nanoseconds < 0:
            raise ValueError(
                "source time nanoseconds must be a non-negative integer"
            )
        return source_time_nanoseconds // 1_000_000
    source_time_sec = float(source_time_sec)
    if not math.isfinite(source_time_sec) or source_time_sec < 0.0:
        raise ValueError("source time must be finite and non-negative")
    return math.floor(source_time_sec * 1000.0)


def _detect(
    detector,
    rgb_image,
    source_time_sec,
    source_time_nanoseconds,
):
    detect_at = getattr(detector, "detect_at", None)
    if callable(detect_at):
        return detect_at(
            rgb_image,
            _source_timestamp_ms(
                source_time_sec,
                source_time_nanoseconds,
            ),
        )
    return detector.detect(rgb_image)


class StereoFrameProcessor:
    """Run both 2D detectors and preserve fail-closed pipeline behavior."""

    def __init__(
        self,
        pipeline,
        left_detector,
        right_detector,
        *,
        require_handedness_match=True,
    ):
        if not isinstance(pipeline, StereoHandPipeline):
            raise TypeError("pipeline must be a StereoHandPipeline")
        for detector, name in (
            (left_detector, "left_detector"),
            (right_detector, "right_detector"),
        ):
            if not callable(getattr(detector, "detect", None)):
                raise TypeError(f"{name} must provide detect(rgb_image)")
        self._pipeline = pipeline
        self._left_detector = left_detector
        self._right_detector = right_detector
        self._require_handedness_match = bool(require_handedness_match)

    def process(
        self,
        left_rgb_image,
        right_rgb_image,
        *,
        left_source_time_sec,
        right_source_time_sec,
        now_sec,
        left_source_time_nanoseconds=None,
        right_source_time_nanoseconds=None,
    ):
        """Detect, associate, triangulate, and gate one image pair."""
        try:
            left_detection = _detect(
                self._left_detector,
                left_rgb_image,
                left_source_time_sec,
                left_source_time_nanoseconds,
            )
            right_detection = _detect(
                self._right_detector,
                right_rgb_image,
                right_source_time_sec,
                right_source_time_nanoseconds,
            )
        except Exception:
            return self._pipeline.invalidate(
                "detector_error",
                left_source_time_sec=left_source_time_sec,
                right_source_time_sec=right_source_time_sec,
            )

        for detection in (left_detection, right_detection):
            if (
                detection is not None
                and not isinstance(detection, HandKeypointsDetection)
            ):
                return self._pipeline.invalidate(
                    "detector_contract_error",
                    left_source_time_sec=left_source_time_sec,
                    right_source_time_sec=right_source_time_sec,
                )

        detections = (left_detection, right_detection)
        confidences = tuple(
            detection.confidence if detection is not None else 0.0
            for detection in detections
        )
        if left_detection is None or right_detection is None:
            keypoint_set = StereoKeypointSet(
                left_pixels=(
                    left_detection.pixels
                    if left_detection is not None
                    else None
                ),
                right_pixels=(
                    right_detection.pixels
                    if right_detection is not None
                    else None
                ),
                left_source_time_sec=left_source_time_sec,
                right_source_time_sec=right_source_time_sec,
                left_confidence=confidences[0],
                right_confidence=confidences[1],
            )
            return self._pipeline.process_set(keypoint_set, now_sec)

        handedness = (
            left_detection.handedness,
            right_detection.handedness,
        )
        if (
            self._require_handedness_match
            and all(value is not None for value in handedness)
            and handedness[0] != handedness[1]
        ):
            return self._pipeline.invalidate(
                "association_rejected",
                left_source_time_sec=left_source_time_sec,
                right_source_time_sec=right_source_time_sec,
                confidence=min(confidences),
            )

        keypoint_set = StereoKeypointSet(
            left_pixels=left_detection.pixels,
            right_pixels=right_detection.pixels,
            left_source_time_sec=left_source_time_sec,
            right_source_time_sec=right_source_time_sec,
            left_confidence=confidences[0],
            right_confidence=confidences[1],
        )
        return self._pipeline.process_set(keypoint_set, now_sec)
