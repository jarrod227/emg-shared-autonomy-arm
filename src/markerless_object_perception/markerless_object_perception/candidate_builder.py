"""Fuse instance masks with aligned stereo XYZ into object candidates."""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Iterable

from markerless_object_perception.masked_point_localizer import (
    MaskedPointLocalizer,
)
import numpy as np


@dataclass(frozen=True)
class InstanceMaskDetection:
    """One model detection after its mask has been resized to the XYZ image."""

    class_label: str
    confidence: float
    track_id: int
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.class_label, str) or not self.class_label.strip():
            raise ValueError('class_label must be a non-empty string')

        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as error:
            raise ValueError('confidence must be numeric') from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError('confidence must be finite and in [0, 1]')

        if (
            not isinstance(self.track_id, Integral)
            or isinstance(self.track_id, bool)
            or self.track_id < 0
        ):
            raise ValueError('track_id must be a non-negative integer')

        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError('mask must be a boolean HxW array')

        object.__setattr__(self, 'class_label', self.class_label.strip())
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(self, 'track_id', int(self.track_id))
        object.__setattr__(self, 'mask', mask)


@dataclass(frozen=True)
class ObjectCandidate:
    """One usable mask-derived 3D candidate in the stereo reference frame."""

    class_label: str
    class_confidence: float
    track_id: int
    point: tuple[float, float, float]
    localization_spread_m: float
    valid_point_count: int
    inlier_count: int


@dataclass(frozen=True)
class CandidateRejection:
    """Why one model detection could not become a usable 3D candidate."""

    class_label: str
    track_id: int
    reason: str


@dataclass(frozen=True)
class CandidateBuildResult:
    """Usable candidates and explicit per-detection rejection diagnostics."""

    candidates: tuple[ObjectCandidate, ...]
    rejections: tuple[CandidateRejection, ...]


@dataclass(frozen=True)
class CandidateBuilderConfig:
    """Quality threshold owned by the mask-to-candidate boundary."""

    min_detection_confidence: float = 0.5

    def __post_init__(self) -> None:
        try:
            value = float(self.min_detection_confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'min_detection_confidence must be numeric'
            ) from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                'min_detection_confidence must be finite and in [0, 1]'
            )
        object.__setattr__(self, 'min_detection_confidence', value)


class CandidateBuilder:
    """Convert a fresh set of 2D detections into localized 3D candidates."""

    def __init__(
        self,
        localizer: MaskedPointLocalizer | None = None,
        config: CandidateBuilderConfig | None = None,
    ) -> None:
        self._localizer = localizer or MaskedPointLocalizer()
        self._config = config or CandidateBuilderConfig()
        if not isinstance(self._localizer, MaskedPointLocalizer):
            raise TypeError('localizer must be a MaskedPointLocalizer')
        if not isinstance(self._config, CandidateBuilderConfig):
            raise TypeError('config must be a CandidateBuilderConfig')

    def build(
        self,
        detections: Iterable[InstanceMaskDetection],
        xyz_points,
    ) -> CandidateBuildResult:
        """Localize each accepted mask without choosing the final target."""
        candidates = []
        rejections = []
        seen_track_ids = set()

        for detection in detections:
            if not isinstance(detection, InstanceMaskDetection):
                raise TypeError(
                    'detections must contain InstanceMaskDetection values'
                )
            if detection.track_id in seen_track_ids:
                raise ValueError(
                    f'duplicate track_id in one frame: {detection.track_id}'
                )
            seen_track_ids.add(detection.track_id)

            if detection.confidence < self._config.min_detection_confidence:
                rejections.append(
                    CandidateRejection(
                        class_label=detection.class_label,
                        track_id=detection.track_id,
                        reason='low_detection_confidence',
                    )
                )
                continue

            localization = self._localizer.localize(
                detection.mask,
                xyz_points,
            )
            if not localization.valid:
                rejections.append(
                    CandidateRejection(
                        class_label=detection.class_label,
                        track_id=detection.track_id,
                        reason=localization.reason,
                    )
                )
                continue

            candidates.append(
                ObjectCandidate(
                    class_label=detection.class_label,
                    class_confidence=detection.confidence,
                    track_id=detection.track_id,
                    point=localization.point,
                    localization_spread_m=localization.spread_m,
                    valid_point_count=localization.valid_point_count,
                    inlier_count=localization.inlier_count,
                )
            )

        return CandidateBuildResult(
            candidates=tuple(candidates),
            rejections=tuple(rejections),
        )
