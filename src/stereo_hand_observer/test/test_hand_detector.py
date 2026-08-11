"""Tests for complete-hand detection before stereo point selection."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from stereo_hand_observer.hand_detector import (
    DetectedHand,
    MediaPipeHandDetector,
    NormalizedHandLandmark,
    draw_hand_landmarks,
)


def fake_mediapipe(result):
    """Build the MediaPipe Tasks surface used by the full-hand detector."""

    class FakeLandmarker:
        options = None
        closed = False
        create_count = 0
        detect_count = 0
        video_timestamps = []

        @classmethod
        def create_from_options(cls, options):
            cls.options = options
            cls.create_count += 1
            return cls()

        def detect(self, _image):
            type(self).detect_count += 1
            return result

        def detect_for_video(self, _image, timestamp_ms):
            type(self).video_timestamps.append(timestamp_ms)
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
                RunningMode=SimpleNamespace(IMAGE="image", VIDEO="video"),
                HandLandmarkerOptions=KeywordOptions,
                HandLandmarker=FakeLandmarker,
            ),
        ),
    )
    return module, FakeLandmarker


def result_with_hands(count, *, landmark_count=21):
    """Create count full MediaPipe-like hands with distinct landmarks."""
    hands = []
    for _ in range(count):
        hands.append(
            [
                SimpleNamespace(
                    x=0.20 + 0.02 * (index % 5),
                    y=0.20 + 0.03 * (index // 5),
                    z=-0.001 * index,
                )
                for index in range(landmark_count)
            ]
        )
    handedness = [
        [SimpleNamespace(category_name="Right", score=0.95)]
        for _ in range(count)
    ]
    return SimpleNamespace(
        hand_landmarks=hands,
        handedness=handedness,
    )


def make_detector(tmp_path, result, **detector_options):
    """Create a full-hand detector around a fake model and task API."""
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"fake model")
    module, landmarker_class = fake_mediapipe(result)
    detector = MediaPipeHandDetector(
        model_path,
        min_detection_confidence=0.8,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.9,
        mediapipe_module=module,
        **detector_options,
    )
    return detector, landmarker_class


def test_unique_hand_returns_complete_21_landmark_contract(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    first = detector.detect(image)
    second = detector.detect(image)

    assert isinstance(first, DetectedHand)
    assert len(first.landmarks) == 21
    landmark = first.landmarks[9]
    assert (landmark.x, landmark.y, landmark.z) == pytest.approx(
        (0.28, 0.23, -0.009)
    )
    assert first.confidence == 0.7
    assert first.handedness == "right"
    assert second == first
    assert landmarker_class.options.num_hands == 2
    assert landmarker_class.options.running_mode == "image"
    assert landmarker_class.create_count == 1
    assert landmarker_class.detect_count == 2


def test_video_mode_uses_timestamped_tracking_inference(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
        running_mode="video",
    )
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    first = detector.detect(image, timestamp_ms=1000)
    second = detector.detect(image, timestamp_ms=1200)

    assert first == second
    assert landmarker_class.options.running_mode == "video"
    assert landmarker_class.detect_count == 0
    assert landmarker_class.video_timestamps == [1000, 1200]


@pytest.mark.parametrize("timestamp_ms", (None, -1, 1.5, True))
def test_video_mode_rejects_invalid_timestamps(tmp_path, timestamp_ms):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
        running_mode="video",
    )

    with pytest.raises(ValueError, match="timestamp_ms"):
        detector.detect(
            np.zeros((48, 64, 3), dtype=np.uint8),
            timestamp_ms=timestamp_ms,
        )
    assert landmarker_class.video_timestamps == []


def test_video_mode_rejects_repeated_or_older_timestamps(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
        running_mode="video",
    )
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    detector.detect(image, timestamp_ms=1000)

    for timestamp_ms in (1000, 999):
        with pytest.raises(ValueError, match="increase strictly"):
            detector.detect(image, timestamp_ms=timestamp_ms)
    assert landmarker_class.video_timestamps == [1000]


@pytest.mark.parametrize("running_mode", (None, "stream", 1))
def test_unknown_running_mode_is_rejected(tmp_path, running_mode):
    with pytest.raises(ValueError, match="running_mode"):
        make_detector(
            tmp_path,
            result_with_hands(1),
            running_mode=running_mode,
        )


@pytest.mark.parametrize("hand_count", (0, 2))
def test_missing_or_ambiguous_hands_fail_closed(tmp_path, hand_count):
    detector, _ = make_detector(
        tmp_path,
        result_with_hands(hand_count),
    )

    assert detector.detect(
        np.zeros((48, 64, 3), dtype=np.uint8)
    ) is None


def test_incomplete_hand_result_is_rejected(tmp_path):
    detector, _ = make_detector(
        tmp_path,
        result_with_hands(1, landmark_count=20),
    )

    with pytest.raises(RuntimeError, match="exactly 21"):
        detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))


def test_non_finite_hand_landmark_is_rejected(tmp_path):
    result = result_with_hands(1)
    result.hand_landmarks[0][4].x = math.nan
    detector, _ = make_detector(tmp_path, result)

    with pytest.raises(RuntimeError, match="invalid hand landmarks"):
        detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    (
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
    ),
)
def test_invalid_rgb_images_are_rejected(tmp_path, image):
    detector, _ = make_detector(tmp_path, result_with_hands(1))

    with pytest.raises(ValueError, match="rgb_image"):
        detector.detect(image)


def test_closed_detector_rejects_further_inference(tmp_path):
    detector, landmarker_class = make_detector(
        tmp_path,
        result_with_hands(1),
    )

    detector.close()
    detector.close()

    assert landmarker_class.closed
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


def test_draw_hand_landmarks_renders_skeleton_and_representative_point():
    hand = DetectedHand(
        landmarks=tuple(
            NormalizedHandLandmark(
                x=0.20 + 0.02 * (index % 5),
                y=0.20 + 0.03 * (index // 5),
                z=-0.001 * index,
            )
            for index in range(21)
        ),
        confidence=0.8,
        handedness="right",
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    annotated = draw_hand_landmarks(image, hand)

    representative = hand.landmarks[9]
    x = int(round(representative.x * (image.shape[1] - 1)))
    y = int(round(representative.y * (image.shape[0] - 1)))
    assert np.count_nonzero(annotated) > 0
    np.testing.assert_array_equal(annotated[y, x], (0, 0, 255))
    assert not np.any(image)
