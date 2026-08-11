"""Replaceable 2D hand-keypoint detection backends for live stereo."""

from dataclasses import dataclass
import math

import numpy as np

from stereo_hand_observer.hand_detector import MediaPipeHandDetector


def _landmark_index(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("landmark indices must be integers in [0, 20]")
    if not 0 <= value <= 20:
        raise ValueError("landmark indices must be integers in [0, 20]")
    return value


@dataclass(frozen=True)
class HandKeypointsDetection:
    """In-frame pixels for several landmarks plus source-local metadata."""

    pixels: dict[int, tuple[float, float]]
    confidence: float
    handedness: str | None = None

    def __post_init__(self):
        try:
            items = dict(self.pixels).items()
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pixels must map landmark indices to pixel pairs"
            ) from error
        if not items:
            raise ValueError("pixels must contain at least one landmark")

        pixels = {}
        for index, value in items:
            index = _landmark_index(index)
            try:
                pixel = tuple(float(coordinate) for coordinate in value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "each pixel must contain two numeric values"
                ) from error
            if (
                len(pixel) != 2
                or not all(math.isfinite(value) for value in pixel)
            ):
                raise ValueError("each pixel must contain two finite values")
            pixels[index] = pixel

        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

        handedness = self.handedness
        if handedness is not None:
            if not isinstance(handedness, str) or not handedness.strip():
                raise ValueError("handedness must be a non-empty string or None")
            handedness = handedness.strip().lower()

        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "handedness", handedness)


class MediaPipeHandKeypointDetector:
    """Detect a complete hand, then expose palm landmarks to stereo geometry."""

    def __init__(
        self,
        model_path,
        *,
        landmark_indices=(5, 9, 13, 17),
        min_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode="image",
        mediapipe_module=None,
    ):
        try:
            indices = tuple(landmark_indices)
        except TypeError as error:
            raise ValueError(
                "landmark_indices must be an iterable of integers"
            ) from error
        if not indices:
            raise ValueError("landmark_indices must not be empty")
        for index in indices:
            _landmark_index(index)
        if len(set(indices)) != len(indices):
            raise ValueError("landmark_indices must not contain duplicates")

        self._landmark_indices = indices
        self._hand_detector = MediaPipeHandDetector(
            model_path,
            min_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=running_mode,
            mediapipe_module=mediapipe_module,
        )
        self._last_hand = None

    @property
    def last_hand(self):
        """Return the complete hand from the latest detect() call, if any."""
        return self._last_hand

    def detect(self, rgb_image):
        """Return in-frame configured pixels or None without any."""
        return self._detect(rgb_image)

    def detect_at(self, rgb_image, timestamp_ms):
        """Detect configured keypoints using a source timestamp."""
        return self._detect(rgb_image, timestamp_ms=timestamp_ms)

    def _detect(self, rgb_image, *, timestamp_ms=None):
        """Share image- and timestamp-driven complete-hand adaptation."""
        self._last_hand = None
        hand = self._hand_detector.detect(
            rgb_image,
            timestamp_ms=timestamp_ms,
        )
        if hand is None:
            return None
        self._last_hand = hand

        image = np.asarray(rgb_image)
        height, width = image.shape[:2]
        pixels = {}
        for index in self._landmark_indices:
            landmark = hand.landmarks[index]
            normalized = (landmark.x, landmark.y)
            if not all(0.0 <= value < 1.0 for value in normalized):
                continue
            pixels[index] = (
                normalized[0] * width,
                normalized[1] * height,
            )
        if not pixels:
            return None
        return HandKeypointsDetection(
            pixels=pixels,
            confidence=hand.confidence,
            handedness=hand.handedness,
        )

    def close(self):
        """Release MediaPipe resources; safe to call more than once."""
        self._last_hand = None
        self._hand_detector.close()
