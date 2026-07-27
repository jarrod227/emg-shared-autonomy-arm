"""ROS-independent processing of one synchronized live stereo frame pair."""

from stereo_hand_observer.keypoint_detector import HandKeypointDetection
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointPair,
)


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
    ):
        """Detect, associate, triangulate, and gate one image pair."""
        try:
            left_detection = self._left_detector.detect(left_rgb_image)
            right_detection = self._right_detector.detect(right_rgb_image)
        except Exception:
            return self._pipeline.invalidate(
                "detector_error",
                left_source_time_sec=left_source_time_sec,
                right_source_time_sec=right_source_time_sec,
            )

        for detection in (left_detection, right_detection):
            if (
                detection is not None
                and not isinstance(detection, HandKeypointDetection)
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
            pair = StereoKeypointPair(
                left_pixel=(
                    left_detection.pixel
                    if left_detection is not None
                    else None
                ),
                right_pixel=(
                    right_detection.pixel
                    if right_detection is not None
                    else None
                ),
                left_source_time_sec=left_source_time_sec,
                right_source_time_sec=right_source_time_sec,
                left_confidence=confidences[0],
                right_confidence=confidences[1],
            )
            return self._pipeline.process(pair, now_sec)

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

        pair = StereoKeypointPair(
            left_pixel=left_detection.pixel,
            right_pixel=right_detection.pixel,
            left_source_time_sec=left_source_time_sec,
            right_source_time_sec=right_source_time_sec,
            left_confidence=confidences[0],
            right_confidence=confidences[1],
        )
        return self._pipeline.process(pair, now_sec)
