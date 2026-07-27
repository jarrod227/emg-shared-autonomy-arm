"""Replaceable 2D hand-keypoint detection backends for live stereo."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


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


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


class MediaPipeHandKeypointDetector:
    """Detect one configurable MediaPipe hand landmark in an RGB image."""

    def __init__(
        self,
        model_path,
        *,
        landmark_index=9,
        min_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        mediapipe_module=None,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ValueError(
                f"MediaPipe hand-landmarker model does not exist: {model_path}"
            )
        if (
            not isinstance(landmark_index, int)
            or isinstance(landmark_index, bool)
            or not 0 <= landmark_index <= 20
        ):
            raise ValueError("landmark_index must be an integer in [0, 20]")

        detection_floor = _probability(
            min_detection_confidence,
            "min_detection_confidence",
        )
        presence_floor = _probability(
            min_hand_presence_confidence,
            "min_hand_presence_confidence",
        )
        tracking_floor = _probability(
            min_tracking_confidence,
            "min_tracking_confidence",
        )

        if mediapipe_module is None:
            try:
                import mediapipe as mediapipe_module
            except ImportError as error:
                raise RuntimeError(
                    "MediaPipe is not installed; install its Python package "
                    "before starting the live hand observer"
                ) from error

        self._mediapipe = mediapipe_module
        self._landmark_index = landmark_index
        self._confidence_floor = min(
            detection_floor,
            presence_floor,
            tracking_floor,
        )
        base_options = mediapipe_module.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )
        options = mediapipe_module.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mediapipe_module.tasks.vision.RunningMode.IMAGE
            ),
            num_hands=2,
            min_hand_detection_confidence=detection_floor,
            min_hand_presence_confidence=presence_floor,
            min_tracking_confidence=tracking_floor,
        )
        self._landmarker = (
            mediapipe_module.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        )

    def detect(self, rgb_image):
        """Return one bounded pixel or None when no hand is detected."""
        if self._landmarker is None:
            raise RuntimeError("detector is closed")

        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb_image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("rgb_image must use uint8 pixels")
        height, width = image.shape[:2]
        if height < 1 or width < 1:
            raise ValueError("rgb_image dimensions must be positive")

        media_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=np.ascontiguousarray(image),
        )
        result = self._landmarker.detect(media_image)
        if len(result.hand_landmarks) != 1:
            return None

        landmarks = result.hand_landmarks[0]
        if self._landmark_index >= len(landmarks):
            raise RuntimeError(
                "MediaPipe result does not contain the configured landmark"
            )
        landmark = landmarks[self._landmark_index]
        normalized = (float(landmark.x), float(landmark.y))
        if not all(math.isfinite(value) for value in normalized):
            raise RuntimeError("MediaPipe returned a non-finite landmark")
        if not all(0.0 <= value < 1.0 for value in normalized):
            return None

        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name or None
        return HandKeypointDetection(
            pixel=(normalized[0] * width, normalized[1] * height),
            confidence=self._confidence_floor,
            handedness=handedness,
        )

    def close(self):
        """Release MediaPipe resources; safe to call more than once."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
