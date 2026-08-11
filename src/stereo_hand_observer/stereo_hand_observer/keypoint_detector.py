"""Replaceable 2D hand-keypoint detection backends for live stereo."""

from dataclasses import dataclass
import math

import numpy as np

from stereo_hand_observer.hand_detector import MediaPipeHandDetector


@dataclass(frozen=True)
class HandKeypointDetection:
    """One image-space keypoint and source-local quality metadata."""

    pixel: tuple[float, float]
    confidence: float
    handedness: str | None = None

    def __post_init__(self):
        try:
            pixel = tuple(float(value) for value in self.pixel)
        except (TypeError, ValueError) as error:
            raise ValueError("pixel must contain two numeric values") from error
        if len(pixel) != 2 or not all(math.isfinite(value) for value in pixel):
            raise ValueError("pixel must contain two finite values")

        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

        handedness = self.handedness
        if handedness is not None:
            if not isinstance(handedness, str) or not handedness.strip():
                raise ValueError("handedness must be a non-empty string or None")
            handedness = handedness.strip().lower()

        object.__setattr__(self, "pixel", pixel)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "handedness", handedness)


class MediaPipeHandKeypointDetector:
    """Detect a complete hand, then expose one landmark to stereo geometry."""

    def __init__(
        self,
        model_path,
        *,
        landmark_index=9,
        min_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode="image",
        mediapipe_module=None,
    ):
        if (
            not isinstance(landmark_index, int)
            or isinstance(landmark_index, bool)
            or not 0 <= landmark_index <= 20
        ):
            raise ValueError("landmark_index must be an integer in [0, 20]")

        self._landmark_index = landmark_index
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
        """Return one bounded pixel or None when no hand is detected."""
        return self._detect(rgb_image)

    def detect_at(self, rgb_image, timestamp_ms):
        """Detect one keypoint using a source timestamp when supported."""
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
        landmark = hand.landmarks[self._landmark_index]
        normalized = (landmark.x, landmark.y)
        if not all(0.0 <= value < 1.0 for value in normalized):
            return None
        return HandKeypointDetection(
            pixel=(normalized[0] * width, normalized[1] * height),
            confidence=hand.confidence,
            handedness=hand.handedness,
        )

    def close(self):
        """Release MediaPipe resources; safe to call more than once."""
        self._last_hand = None
        self._hand_detector.close()
