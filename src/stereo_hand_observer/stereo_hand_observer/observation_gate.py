"""Fail-closed quality and stability gating for stereo hand observations."""

from collections import deque
from dataclasses import dataclass
import math
import operator

import numpy as np


def _point_array(value, name):
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three numeric values") from error
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values")
    return point


def _finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative_float(value, name):
    value = _finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class DeliveryVolume:
    """A spherical 3D region matching the existing 4.1 delivery gate."""

    center: tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        center = tuple(_point_array(self.center, "center"))
        radius = _finite_float(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ValueError("radius_m must be greater than zero")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius_m", radius)

    def contains(self, point):
        """Return whether a finite 3D point is inside or on the boundary."""
        point = _point_array(point, "point")
        return bool(
            np.linalg.norm(point - np.asarray(self.center)) <= self.radius_m
        )


@dataclass(frozen=True)
class HandFrameCandidate:
    """One triangulated stereo result before temporal stability gating."""

    point: tuple[float, float, float]
    confidence: float
    pair_skew_sec: float
    reprojection_error_px: float
    source_time_sec: float


@dataclass(frozen=True)
class StabilityGateConfig:
    """Quality limits that every frame in a stable run must satisfy."""

    required_frames: int = 3
    min_confidence: float = 0.7
    max_pair_skew_sec: float = 0.05
    max_reprojection_error_px: float = 1.5
    max_age_sec: float = 0.25
    max_point_step_m: float = 0.05

    def __post_init__(self):
        try:
            required_frames = operator.index(self.required_frames)
        except TypeError as error:
            raise ValueError("required_frames must be an integer") from error
        if required_frames < 1:
            raise ValueError("required_frames must be at least one")

        min_confidence = _finite_float(
            self.min_confidence,
            "min_confidence",
        )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

        max_age_sec = _finite_float(self.max_age_sec, "max_age_sec")
        if max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be greater than zero")

        object.__setattr__(self, "required_frames", required_frames)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(
            self,
            "max_pair_skew_sec",
            _nonnegative_float(
                self.max_pair_skew_sec,
                "max_pair_skew_sec",
            ),
        )
        object.__setattr__(
            self,
            "max_reprojection_error_px",
            _nonnegative_float(
                self.max_reprojection_error_px,
                "max_reprojection_error_px",
            ),
        )
        object.__setattr__(self, "max_age_sec", max_age_sec)
        object.__setattr__(
            self,
            "max_point_step_m",
            _nonnegative_float(
                self.max_point_step_m,
                "max_point_step_m",
            ),
        )


@dataclass(frozen=True)
class GateDecision:
    """The fail-closed result to be converted into HandObservation.valid."""

    valid: bool
    stable_frames: int
    point: tuple[float, float, float] | None
    reason: str


class DeliveryStabilityGate:
    """Require N fresh, high-quality, nearby points in the delivery volume."""

    def __init__(self, volume, config=None):
        if not isinstance(volume, DeliveryVolume):
            raise TypeError("volume must be a DeliveryVolume")
        if config is None:
            config = StabilityGateConfig()
        if not isinstance(config, StabilityGateConfig):
            raise TypeError("config must be a StabilityGateConfig")
        self._volume = volume
        self._config = config
        self._points = deque(maxlen=config.required_frames)
        self._last_source_time_sec = None

    @property
    def stable_frames(self):
        """Return the current consecutive accepted-frame count."""
        return len(self._points)

    def reset(self):
        """Forget all temporal evidence after any invalid observation."""
        self._points.clear()
        self._last_source_time_sec = None

    def invalidate(self, reason):
        """Clear temporal evidence for a failure detected before gating."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        return self._reject(reason)

    def _reject(self, reason):
        self.reset()
        return GateDecision(
            valid=False,
            stable_frames=0,
            point=None,
            reason=reason,
        )

    def _accept(self, point, source_time_sec, pending_reason="warming_up"):
        self._points.append(tuple(point))
        self._last_source_time_sec = source_time_sec
        if len(self._points) < self._config.required_frames:
            return GateDecision(
                valid=False,
                stable_frames=len(self._points),
                point=None,
                reason=pending_reason,
            )

        robust_point = np.median(np.asarray(self._points), axis=0)
        if not self._volume.contains(robust_point):
            return self._reject("filtered_point_outside_volume")
        return GateDecision(
            valid=True,
            stable_frames=len(self._points),
            point=tuple(float(value) for value in robust_point),
            reason="stable",
        )

    def update(self, candidate, now_sec):
        """Consume one stereo frame and return the current gate decision."""
        now_sec = _finite_float(now_sec, "now_sec")
        if candidate is None:
            return self._reject("missing")
        if not isinstance(candidate, HandFrameCandidate):
            raise TypeError("candidate must be a HandFrameCandidate or None")

        try:
            point = _point_array(candidate.point, "candidate.point")
            confidence = _finite_float(
                candidate.confidence,
                "candidate.confidence",
            )
            pair_skew_sec = _nonnegative_float(
                candidate.pair_skew_sec,
                "candidate.pair_skew_sec",
            )
            reprojection_error_px = _nonnegative_float(
                candidate.reprojection_error_px,
                "candidate.reprojection_error_px",
            )
            source_time_sec = _finite_float(
                candidate.source_time_sec,
                "candidate.source_time_sec",
            )
        except (TypeError, ValueError):
            return self._reject("invalid_measurement")

        if not 0.0 <= confidence <= 1.0:
            return self._reject("invalid_confidence")
        if confidence < self._config.min_confidence:
            return self._reject("low_confidence")
        if pair_skew_sec > self._config.max_pair_skew_sec:
            return self._reject("excessive_pair_skew")
        if (
            reprojection_error_px
            > self._config.max_reprojection_error_px
        ):
            return self._reject("high_reprojection_error")

        age_sec = now_sec - source_time_sec
        if age_sec < 0.0:
            return self._reject("future_timestamp")
        if age_sec > self._config.max_age_sec:
            return self._reject("stale")
        if (
            self._last_source_time_sec is not None
            and source_time_sec <= self._last_source_time_sec
        ):
            return self._reject("non_increasing_timestamp")
        if not self._volume.contains(point):
            return self._reject("outside_delivery_volume")

        if self._points:
            previous = np.asarray(self._points[-1])
            if (
                np.linalg.norm(point - previous)
                > self._config.max_point_step_m
            ):
                self.reset()
                return self._accept(
                    point,
                    source_time_sec,
                    pending_reason="unstable",
                )

        return self._accept(point, source_time_sec)
