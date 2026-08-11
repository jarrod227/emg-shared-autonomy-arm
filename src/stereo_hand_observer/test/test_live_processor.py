"""Tests for synchronized live-image processing without ROS transport."""

import numpy as np

from stereo_hand_observer.geometry import project_point
from stereo_hand_observer.keypoint_detector import HandKeypointsDetection
from stereo_hand_observer.live_processor import StereoFrameProcessor
from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import StereoHandPipeline
from stereo_hand_observer.synthetic_observer import rectified_stereo_model


GROUND_TRUTH = np.array([0.4, 0.3, 1.0])
LEFT_PROJECTION, RIGHT_PROJECTION, FUNDAMENTAL_MATRIX = (
    rectified_stereo_model(800.0, 800.0, 320.0, 240.0, 0.12)
)
RGB_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


class FakeDetector:
    """Return one configured result or raise a configured error."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def detect(self, _image):
        """Return the fake result for one view."""
        if self._error is not None:
            raise self._error
        return self._result


class TimestampDetector(FakeDetector):
    """Record timestamp-aware calls while reusing the fake result."""

    def __init__(self, result=None, error=None):
        super().__init__(result, error)
        self.timestamps = []

    def detect_at(self, image, timestamp_ms):
        """Record one source timestamp before returning the result."""
        self.timestamps.append(timestamp_ms)
        return self.detect(image)


class StrictTimestampDetector(TimestampDetector):
    """Mirror MediaPipe VIDEO's strictly increasing timestamp rule."""

    def detect_at(self, image, timestamp_ms):
        """Reject a timestamp that does not advance beyond the last call."""
        if self.timestamps and timestamp_ms <= self.timestamps[-1]:
            raise ValueError("timestamp_ms must increase strictly")
        return super().detect_at(image, timestamp_ms)


KNUCKLE_INDICES = (5, 9, 13, 17)


def detection(projection, *, confidence=0.9, handedness="right"):
    """Create one detector result projected from the known 3D point."""
    pixel = tuple(project_point(projection, GROUND_TRUTH))
    return HandKeypointsDetection(
        pixels={index: pixel for index in KNUCKLE_INDICES},
        confidence=confidence,
        handedness=handedness,
    )


def make_pipeline():
    """Create a one-frame stereo hand pipeline for processor tests."""
    return StereoHandPipeline(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        FUNDAMENTAL_MATRIX,
        DeliveryVolume(center=tuple(GROUND_TRUTH), radius_m=0.2),
        gate_config=StabilityGateConfig(
            required_frames=1,
            min_confidence=0.7,
            max_pair_skew_sec=0.02,
            max_reprojection_error_px=1.5,
            max_age_sec=0.2,
            max_point_step_m=0.05,
        ),
    )


def make_processor(left_detection, right_detection, *, left_error=None):
    """Create a one-frame gate around two fake view detectors."""
    return StereoFrameProcessor(
        make_pipeline(),
        FakeDetector(left_detection, left_error),
        FakeDetector(right_detection),
    )


def process(processor):
    """Process one fresh, synchronized fake RGB pair."""
    return processor.process(
        RGB_IMAGE,
        RGB_IMAGE,
        left_source_time_sec=10.0,
        right_source_time_sec=10.005,
        now_sec=10.01,
        left_source_time_nanoseconds=10_000_000_123,
        right_source_time_nanoseconds=10_005_000_456,
    )


def test_corresponding_live_detections_recover_stable_point():
    processor = make_processor(
        detection(LEFT_PROJECTION),
        detection(RIGHT_PROJECTION, confidence=0.8),
    )

    result = process(processor)

    assert result.valid
    np.testing.assert_allclose(result.point, GROUND_TRUTH)
    assert result.confidence == 0.8
    assert result.reason == "stable"


def test_timestamp_aware_detectors_receive_each_view_source_time():
    left_detector = TimestampDetector(detection(LEFT_PROJECTION))
    right_detector = TimestampDetector(detection(RIGHT_PROJECTION))
    processor = StereoFrameProcessor(
        make_pipeline(),
        left_detector,
        right_detector,
    )

    result = process(processor)

    assert result.valid
    assert left_detector.timestamps == [10000]
    assert right_detector.timestamps == [10005]


def test_same_millisecond_fails_closed_then_new_millisecond_recovers():
    left_detector = StrictTimestampDetector(detection(LEFT_PROJECTION))
    right_detector = StrictTimestampDetector(detection(RIGHT_PROJECTION))
    processor = StereoFrameProcessor(
        make_pipeline(),
        left_detector,
        right_detector,
    )

    def process_at(source_nanoseconds):
        source_sec = source_nanoseconds / 1e9
        return processor.process(
            RGB_IMAGE,
            RGB_IMAGE,
            left_source_time_sec=source_sec,
            right_source_time_sec=source_sec,
            now_sec=source_sec + 0.01,
            left_source_time_nanoseconds=source_nanoseconds,
            right_source_time_nanoseconds=source_nanoseconds,
        )

    first = process_at(10_000_100_000)
    same_millisecond = process_at(10_000_900_000)
    recovered = process_at(10_001_100_000)

    assert first.valid
    assert not same_millisecond.valid
    assert same_millisecond.reason == "detector_error"
    assert recovered.valid
    assert left_detector.timestamps == [10000, 10001]
    assert right_detector.timestamps == [10000, 10001]


def test_missing_detection_is_an_explicit_invalid_result():
    processor = make_processor(detection(LEFT_PROJECTION), None)

    result = process(processor)

    assert not result.valid
    assert result.point is None
    assert result.reason == "missing_keypoint"


def test_handedness_mismatch_rejects_cross_view_association():
    processor = make_processor(
        detection(LEFT_PROJECTION, handedness="left"),
        detection(RIGHT_PROJECTION, handedness="right"),
    )

    result = process(processor)

    assert not result.valid
    assert result.reason == "association_rejected"


def test_detector_exception_fails_closed_and_resets_stability():
    processor = make_processor(
        detection(LEFT_PROJECTION),
        detection(RIGHT_PROJECTION),
        left_error=RuntimeError("inference failed"),
    )

    result = process(processor)

    assert not result.valid
    assert result.reason == "detector_error"


def test_detector_contract_error_fails_closed():
    processor = make_processor(
        "not a HandKeypointsDetection",
        detection(RIGHT_PROJECTION),
    )

    result = process(processor)

    assert not result.valid
    assert result.reason == "detector_contract_error"
