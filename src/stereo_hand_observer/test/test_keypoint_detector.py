"""Contract tests for the optional MediaPipe detector adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from stereo_hand_observer.keypoint_detector import (
    MediaPipeHandKeypointDetector,
)


def fake_mediapipe(result):
    """Build the small MediaPipe Tasks surface used by the adapter."""

    class FakeLandmarker:
        options = None
        closed = False

        @classmethod
        def create_from_options(cls, options):
            cls.options = options
            return cls()

        def detect(self, _image):
            return result

        def close(self):
            type(self).closed = True

    class KeywordOptions:
        def __init__(self, **values):
            self.__dict__.update(values)

    module = SimpleNamespace(
        Image=lambda **values: SimpleNamespace(**values),
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        tasks=SimpleNamespace(
            BaseOptions=KeywordOptions,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE="image"),
                HandLandmarkerOptions=KeywordOptions,
                HandLandmarker=FakeLandmarker,
            ),
        ),
    )
    return module, FakeLandmarker


def result_with_hands(count):
    """Create a result with count identical 21-landmark hands."""
    hands = [
        [SimpleNamespace(x=0.25, y=0.5, z=-0.01) for _ in range(21)]
        for _ in range(count)
    ]
    handedness = [
        [SimpleNamespace(category_name="Right")]
        for _ in range(count)
    ]
    return SimpleNamespace(
        hand_landmarks=hands,
        handedness=handedness,
    )


def make_detector(tmp_path, result, **detector_options):
    """Create the adapter around a fake model file and fake task API."""
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"fake model")
    module, landmarker_class = fake_mediapipe(result)
    detector = MediaPipeHandKeypointDetector(
        model_path,
        min_detection_confidence=0.8,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.9,
        mediapipe_module=module,
        **detector_options,
    )
    return detector, landmarker_class


def test_unique_hand_returns_configured_pixel_and_quality_floor(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
    )

    detection = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert detection.pixel == (160.0, 240.0)
    assert detection.confidence == 0.7
    assert detection.handedness == "right"
    assert landmarker_class.options.num_hands == 2


def test_configured_landmark_is_selected_from_complete_hand(tmp_path):
    result = result_with_hands(1)
    result.hand_landmarks[0][5] = SimpleNamespace(
        x=0.75,
        y=0.25,
        z=-0.02,
    )
    detector, _ = make_detector(
        tmp_path,
        result,
        landmark_index=5,
    )

    detection = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert detection.pixel == (480.0, 120.0)


@pytest.mark.parametrize("hand_count", (0, 2))
def test_missing_or_ambiguous_hands_fail_closed(tmp_path, hand_count):
    detector, _ = make_detector(
        tmp_path,
        result_with_hands(hand_count),
    )

    assert detector.detect(
        np.zeros((48, 64, 3), dtype=np.uint8)
    ) is None


def test_closed_detector_rejects_further_inference(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
    )

    detector.close()

    assert landmarker_class.closed
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
