"""Full-hand landmark detection and visualization for Objective 4.2."""

from dataclasses import dataclass
import math
import operator
from pathlib import Path

import numpy as np


HAND_LANDMARK_COUNT = 21
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _rgb_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("rgb_image must use uint8 pixels")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("rgb_image dimensions must be positive")
    return image


def _running_mode(value):
    if not isinstance(value, str):
        raise ValueError("running_mode must be 'image' or 'video'")
    value = value.strip().lower()
    if value not in {"image", "video"}:
        raise ValueError("running_mode must be 'image' or 'video'")
    return value


def _video_timestamp_ms(value):
    if isinstance(value, bool):
        raise ValueError(
            "timestamp_ms must be a non-negative integer in video mode"
        )
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(
            "timestamp_ms must be a non-negative integer in video mode"
        ) from error
    if value < 0:
        raise ValueError(
            "timestamp_ms must be a non-negative integer in video mode"
        )
    return value


@dataclass(frozen=True)
class NormalizedHandLandmark:
    """One finite MediaPipe landmark in normalized image coordinates."""

    x: float
    y: float
    z: float

    def __post_init__(self):
        values = tuple(float(value) for value in (self.x, self.y, self.z))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("hand landmark coordinates must be finite")
        object.__setattr__(self, "x", values[0])
        object.__setattr__(self, "y", values[1])
        object.__setattr__(self, "z", values[2])


@dataclass(frozen=True)
class DetectedHand:
    """Exactly one complete 21-landmark hand accepted by the detector."""

    landmarks: tuple[NormalizedHandLandmark, ...]
    confidence: float
    handedness: str | None = None

    def __post_init__(self):
        try:
            landmarks = tuple(self.landmarks)
        except TypeError as error:
            raise ValueError("landmarks must be an iterable") from error
        if len(landmarks) != HAND_LANDMARK_COUNT:
            raise ValueError(
                f"a detected hand must contain {HAND_LANDMARK_COUNT} landmarks"
            )
        if not all(
            isinstance(landmark, NormalizedHandLandmark)
            for landmark in landmarks
        ):
            raise ValueError(
                "landmarks must contain NormalizedHandLandmark values"
            )

        confidence = _probability(self.confidence, "confidence")
        handedness = self.handedness
        if handedness is not None:
            if not isinstance(handedness, str) or not handedness.strip():
                raise ValueError(
                    "handedness must be a non-empty string or None"
                )
            handedness = handedness.strip().lower()

        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "handedness", handedness)


class MediaPipeHandDetector:
    """Run one reusable MediaPipe Tasks model and return a complete hand."""

    def __init__(
        self,
        model_path,
        *,
        min_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode="image",
        mediapipe_module=None,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ValueError(
                f"MediaPipe hand-landmarker model does not exist: {model_path}"
            )

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
        self._confidence_floor = min(
            detection_floor,
            presence_floor,
            tracking_floor,
        )
        self._running_mode = _running_mode(running_mode)
        self._last_timestamp_ms = None

        if mediapipe_module is None:
            try:
                import mediapipe as mediapipe_module
            except ImportError as error:
                raise RuntimeError(
                    "MediaPipe is not installed; install its Python package "
                    "before starting the live hand observer"
                ) from error

        self._mediapipe = mediapipe_module
        base_options = mediapipe_module.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )
        mediapipe_running_mode = {
            "image": mediapipe_module.tasks.vision.RunningMode.IMAGE,
            "video": mediapipe_module.tasks.vision.RunningMode.VIDEO,
        }[self._running_mode]
        options = mediapipe_module.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mediapipe_running_mode,
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

    def detect(self, rgb_image, *, timestamp_ms=None):
        """Return one complete hand or None for missing/ambiguous hands."""
        if self._landmarker is None:
            raise RuntimeError("detector is closed")

        image = _rgb_image(rgb_image)
        media_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=np.ascontiguousarray(image),
        )
        if self._running_mode == "video":
            timestamp_ms = _video_timestamp_ms(timestamp_ms)
            if (
                self._last_timestamp_ms is not None
                and timestamp_ms <= self._last_timestamp_ms
            ):
                raise ValueError(
                    "timestamp_ms must increase strictly in video mode"
                )
            result = self._landmarker.detect_for_video(
                media_image,
                timestamp_ms,
            )
            self._last_timestamp_ms = timestamp_ms
        else:
            result = self._landmarker.detect(media_image)
        hands = tuple(getattr(result, "hand_landmarks", ()) or ())
        if len(hands) != 1:
            return None
        if len(hands[0]) != HAND_LANDMARK_COUNT:
            raise RuntimeError(
                "MediaPipe result does not contain exactly 21 landmarks"
            )

        try:
            landmarks = tuple(
                NormalizedHandLandmark(
                    landmark.x,
                    landmark.y,
                    landmark.z,
                )
                for landmark in hands[0]
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "MediaPipe returned invalid hand landmarks"
            ) from error

        handedness = None
        handedness_groups = getattr(result, "handedness", None)
        if handedness_groups and handedness_groups[0]:
            handedness = (
                getattr(handedness_groups[0][0], "category_name", None)
                or None
            )
        return DetectedHand(
            landmarks=landmarks,
            confidence=self._confidence_floor,
            handedness=handedness,
        )

    def close(self):
        """Release MediaPipe resources; safe to call more than once."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def draw_hand_landmarks(
    image_bgr,
    hand,
    *,
    representative_indices=(9,),
    cv2_module=None,
):
    """Return a BGR copy with the skeleton and chosen points drawn."""
    if not isinstance(hand, DetectedHand):
        raise TypeError("hand must be a DetectedHand")
    try:
        representative_indices = tuple(representative_indices)
    except TypeError as error:
        raise ValueError(
            "representative_indices must be an iterable of integers"
        ) from error
    for index in representative_indices:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < HAND_LANDMARK_COUNT
        ):
            raise ValueError(
                "representative_indices must contain integers in [0, 20]"
            )

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image_bgr must use uint8 pixels")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image_bgr dimensions must be positive")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to draw hand landmarks"
            ) from error

    height, width = image.shape[:2]
    pixels = []
    for landmark in hand.landmarks:
        if not (0.0 <= landmark.x < 1.0 and 0.0 <= landmark.y < 1.0):
            pixels.append(None)
            continue
        pixels.append(
            (
                int(round(landmark.x * (width - 1))),
                int(round(landmark.y * (height - 1))),
            )
        )

    output = np.array(image, copy=True)
    for start, end in HAND_CONNECTIONS:
        if pixels[start] is not None and pixels[end] is not None:
            cv2_module.line(
                output,
                pixels[start],
                pixels[end],
                (0, 255, 0),
                2,
            )
    for pixel in pixels:
        if pixel is not None:
            cv2_module.circle(output, pixel, 2, (255, 0, 0), -1)

    for index in representative_indices:
        representative = pixels[index]
        if representative is not None:
            cv2_module.circle(output, representative, 5, (0, 0, 255), -1)
    return output
