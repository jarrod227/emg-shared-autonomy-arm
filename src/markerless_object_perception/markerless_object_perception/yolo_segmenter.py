"""Adapt Ultralytics instance-segmentation tracks to project detections."""

from dataclasses import dataclass
import math
from pathlib import Path

from markerless_object_perception.candidate_builder import (
    InstanceMaskDetection,
)
import numpy as np


DEFAULT_OBJECT_CLASSES = ('bottle', 'cup', 'apple')
MODEL_CLASS_ALIASES = {
    'cell phone': 'cell_phone',
    'sports ball': 'sports_ball',
}


@dataclass(frozen=True)
class YoloSegmenterConfig:
    """Runtime options for the lightweight closed-set segmentation frontend."""

    model_path: str = 'yolo26n-seg.pt'
    min_confidence: float = 0.5
    iou_threshold: float = 0.7
    tracker: str = 'bytetrack.yaml'
    device: str = 'cpu'
    inference_size: int = 480
    allowed_classes: tuple[str, ...] = DEFAULT_OBJECT_CLASSES
    filter_classes: bool = True

    def __post_init__(self) -> None:
        model_path = str(self.model_path).strip()
        tracker = str(self.tracker).strip()
        device = str(self.device).strip()
        if not model_path:
            raise ValueError('model_path must not be empty')
        if not tracker:
            raise ValueError('tracker must not be empty')
        if not device:
            raise ValueError('device must not be empty')
        if (
            isinstance(self.inference_size, bool)
            or not isinstance(self.inference_size, int)
            or self.inference_size <= 0
        ):
            raise ValueError('inference_size must be a positive integer')
        if not isinstance(self.filter_classes, bool):
            raise ValueError('filter_classes must be a boolean')

        min_confidence = _unit_interval(
            self.min_confidence,
            'min_confidence',
        )
        iou_threshold = _unit_interval(
            self.iou_threshold,
            'iou_threshold',
        )
        allowed_classes = _class_names(self.allowed_classes)

        object.__setattr__(self, 'model_path', model_path)
        object.__setattr__(self, 'min_confidence', min_confidence)
        object.__setattr__(self, 'iou_threshold', iou_threshold)
        object.__setattr__(self, 'tracker', tracker)
        object.__setattr__(self, 'device', device)
        object.__setattr__(self, 'allowed_classes', allowed_classes)


class YoloInstanceSegmenter:
    """Run persistent YOLO tracking and return model-independent masks."""

    def __init__(
        self,
        config: YoloSegmenterConfig | None = None,
        *,
        model=None,
    ) -> None:
        self._config = config or YoloSegmenterConfig()
        if not isinstance(self._config, YoloSegmenterConfig):
            raise TypeError('config must be a YoloSegmenterConfig')
        self._model = model or _load_ultralytics_model(
            self._config.model_path
        )

    def track(self, frame) -> tuple[InstanceMaskDetection, ...]:
        """Segment one BGR frame while preserving tracker state between calls."""
        frame_array = np.asarray(frame)
        if (
            frame_array.ndim != 3
            or frame_array.shape[2] != 3
            or frame_array.size == 0
        ):
            raise ValueError('frame must have shape HxWx3')

        results = self._model.track(
            source=frame_array,
            persist=True,
            conf=self._config.min_confidence,
            iou=self._config.iou_threshold,
            tracker=self._config.tracker,
            device=self._config.device,
            imgsz=self._config.inference_size,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return ()

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return ()

        masks = getattr(result, 'masks', None)
        if masks is None:
            raise RuntimeError(
                'YOLO returned boxes without masks; use a -seg model'
            )

        class_ids = _to_numpy(boxes.cls).reshape(-1)
        confidences = _to_numpy(boxes.conf).reshape(-1)
        track_values = getattr(boxes, 'id', None)
        if track_values is None:
            return ()
        track_ids = _to_numpy(track_values).reshape(-1)
        mask_values = _to_numpy(masks.data)

        count = len(boxes)
        if (
            class_ids.size != count
            or confidences.size != count
            or track_ids.size != count
            or mask_values.ndim != 3
            or mask_values.shape[0] != count
        ):
            raise RuntimeError('YOLO boxes and masks have inconsistent sizes')

        frame_shape = frame_array.shape[:2]
        names = getattr(result, 'names', {})
        detections = []
        for index in range(count):
            mask = mask_values[index] > 0.5
            if mask.shape != frame_shape:
                raise RuntimeError(
                    'YOLO retina mask shape does not match the source frame'
                )
            class_label = _canonical_class_name(
                names,
                int(class_ids[index]),
            )
            if (
                self._config.filter_classes
                and class_label not in self._config.allowed_classes
            ):
                continue
            detections.append(
                InstanceMaskDetection(
                    class_label=class_label,
                    confidence=float(confidences[index]),
                    track_id=int(track_ids[index]),
                    mask=mask,
                )
            )

        return tuple(detections)


def _load_ultralytics_model(model_path: str):
    """Load Ultralytics lazily so core tests do not require PyTorch."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            'Ultralytics is not installed; install requirements-yolo.txt'
        ) from error
    return YOLO(str(Path(model_path).expanduser()))


def _to_numpy(value) -> np.ndarray:
    """Move a tensor-like value to CPU NumPy without assuming PyTorch."""
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names, class_id: int) -> str:
    """Resolve a class ID from the mapping/list returned by Ultralytics."""
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError(
            f'YOLO result has no name for class ID {class_id}'
        ) from error


def _canonical_class_name(names, class_id: int) -> str:
    """Map model-specific names onto the project's class contract."""
    class_name = _class_name(names, class_id).strip()
    return MODEL_CLASS_ALIASES.get(class_name, class_name)


def _class_names(values) -> tuple[str, ...]:
    """Validate and normalize the configured closed-set class names."""
    if isinstance(values, str):
        raise ValueError('allowed_classes must be a sequence of class names')
    try:
        class_names = tuple(values)
    except TypeError as error:
        raise ValueError(
            'allowed_classes must be a sequence of class names'
        ) from error
    if not class_names:
        raise ValueError('allowed_classes must not be empty')
    if any(
        not isinstance(name, str) or not name.strip()
        for name in class_names
    ):
        raise ValueError('allowed_classes must contain non-empty strings')
    class_names = tuple(name.strip() for name in class_names)
    if len(set(class_names)) != len(class_names):
        raise ValueError('allowed_classes must not contain duplicates')
    return class_names


def _unit_interval(value, name: str) -> float:
    """Convert a finite probability-like configuration value."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return converted
