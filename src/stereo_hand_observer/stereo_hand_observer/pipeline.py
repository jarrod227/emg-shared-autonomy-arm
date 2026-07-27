"""Source-independent stereo hand-keypoint processing pipeline."""

from dataclasses import dataclass
import math

from stereo_hand_observer.geometry import (
    StereoGeometryError,
    triangulate_keypoint,
)
from stereo_hand_observer.observation_gate import (
    DeliveryStabilityGate,
    HandFrameCandidate,
    StabilityGateConfig,
)


@dataclass(frozen=True)
class StereoKeypointPair:
    """Corresponding hand pixels and metadata from one stereo image pair."""

    left_pixel: tuple[float, float] | None
    right_pixel: tuple[float, float] | None
    left_source_time_sec: float
    right_source_time_sec: float
    left_confidence: float
    right_confidence: float


@dataclass(frozen=True)
class PipelineResult:
    """ROS-independent data needed to populate one HandObservation message."""

    valid: bool
    point: tuple[float, float, float] | None
    confidence: float
    pair_skew_sec: float
    reprojection_error_px: float
    source_time_sec: float
    stable_frames: int
    reason: str


class StereoHandPipeline:
    """Triangulate one keypoint pair and apply fail-closed temporal gating."""

    def __init__(
        self,
        left_projection,
        right_projection,
        fundamental_matrix,
        delivery_volume,
        *,
        gate_config=None,
        max_epipolar_error_px=1.5,
        min_depth_m=1e-6,
    ):
        if gate_config is None:
            gate_config = StabilityGateConfig()
        if not isinstance(gate_config, StabilityGateConfig):
            raise TypeError("gate_config must be a StabilityGateConfig")
        max_epipolar_error_px = float(max_epipolar_error_px)
        min_depth_m = float(min_depth_m)
        if (
            not math.isfinite(max_epipolar_error_px)
            or max_epipolar_error_px < 0.0
        ):
            raise ValueError(
                "max_epipolar_error_px must be finite and non-negative"
            )
        if not math.isfinite(min_depth_m) or min_depth_m < 0.0:
            raise ValueError("min_depth_m must be finite and non-negative")

        self._left_projection = left_projection
        self._right_projection = right_projection
        self._fundamental_matrix = fundamental_matrix
        self._gate_config = gate_config
        self._max_epipolar_error_px = max_epipolar_error_px
        self._min_depth_m = min_depth_m
        self._gate = DeliveryStabilityGate(delivery_volume, gate_config)

    def _result(
        self,
        decision,
        *,
        confidence,
        pair_skew_sec,
        reprojection_error_px,
        source_time_sec,
    ):
        return PipelineResult(
            valid=decision.valid,
            point=decision.point,
            confidence=confidence,
            pair_skew_sec=pair_skew_sec,
            reprojection_error_px=reprojection_error_px,
            source_time_sec=source_time_sec,
            stable_frames=decision.stable_frames,
            reason=decision.reason,
        )

    def invalidate(
        self,
        reason,
        *,
        left_source_time_sec,
        right_source_time_sec,
        confidence=0.0,
    ):
        """Reset stability and return one explicit invalid observation."""
        try:
            left_time = float(left_source_time_sec)
            right_time = float(right_source_time_sec)
            confidence = float(confidence)
        except (TypeError, ValueError):
            left_time = right_time = 0.0
            confidence = 0.0
            reason = "invalid_pair_metadata"

        if not all(
            math.isfinite(value)
            for value in (left_time, right_time, confidence)
        ):
            left_time = right_time = 0.0
            confidence = 0.0
            reason = "invalid_pair_metadata"
        if not 0.0 <= confidence <= 1.0:
            confidence = 0.0
            reason = "invalid_pair_metadata"

        decision = self._gate.invalidate(reason)
        return self._result(
            decision,
            confidence=confidence,
            pair_skew_sec=abs(left_time - right_time),
            reprojection_error_px=0.0,
            source_time_sec=min(left_time, right_time),
        )

    def process(self, pair, now_sec):
        """Process one synchronized image pair into hand-observation data."""
        if not isinstance(pair, StereoKeypointPair):
            raise TypeError("pair must be a StereoKeypointPair")
        now_sec = float(now_sec)
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")

        times = (
            float(pair.left_source_time_sec),
            float(pair.right_source_time_sec),
        )
        confidences = (
            float(pair.left_confidence),
            float(pair.right_confidence),
        )
        if not all(math.isfinite(value) for value in times + confidences):
            decision = self._gate.invalidate("invalid_pair_metadata")
            return self._result(
                decision,
                confidence=0.0,
                pair_skew_sec=0.0,
                reprojection_error_px=0.0,
                source_time_sec=now_sec,
            )

        source_time_sec = min(times)
        pair_skew_sec = abs(times[0] - times[1])
        confidence = min(confidences)

        if pair_skew_sec > self._gate_config.max_pair_skew_sec:
            decision = self._gate.invalidate("excessive_pair_skew")
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
            )
        if pair.left_pixel is None or pair.right_pixel is None:
            decision = self._gate.invalidate("missing_keypoint")
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
            )

        try:
            triangulation = triangulate_keypoint(
                self._left_projection,
                self._right_projection,
                self._fundamental_matrix,
                pair.left_pixel,
                pair.right_pixel,
                max_epipolar_error_px=self._max_epipolar_error_px,
                max_reprojection_error_px=(
                    self._gate_config.max_reprojection_error_px
                ),
                min_depth_m=self._min_depth_m,
            )
        except (StereoGeometryError, TypeError, ValueError):
            decision = self._gate.invalidate("geometry_rejected")
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
            )

        reprojection_error_px = (
            triangulation.max_reprojection_error_px
        )
        candidate = HandFrameCandidate(
            point=tuple(triangulation.point),
            confidence=confidence,
            pair_skew_sec=pair_skew_sec,
            reprojection_error_px=reprojection_error_px,
            source_time_sec=source_time_sec,
        )
        decision = self._gate.update(candidate, now_sec)
        return self._result(
            decision,
            confidence=confidence,
            pair_skew_sec=pair_skew_sec,
            reprojection_error_px=reprojection_error_px,
            source_time_sec=source_time_sec,
        )
