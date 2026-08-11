"""Source-independent stereo hand-keypoint processing pipeline."""

from dataclasses import dataclass
import math
import operator

import numpy as np

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
class StereoKeypointSet:
    """Corresponding multi-landmark pixel sets from one stereo image pair."""

    left_pixels: dict[int, tuple[float, float]] | None
    right_pixels: dict[int, tuple[float, float]] | None
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
    diagnostic: str = ""


class StereoHandPipeline:
    """Triangulate stereo keypoints and apply fail-closed temporal gating."""

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
        min_consensus_points=3,
        max_palm_span_m=0.12,
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
        try:
            min_consensus_points = operator.index(min_consensus_points)
        except TypeError as error:
            raise ValueError(
                "min_consensus_points must be an integer"
            ) from error
        if min_consensus_points < 1:
            raise ValueError("min_consensus_points must be at least one")
        max_palm_span_m = float(max_palm_span_m)
        if not math.isfinite(max_palm_span_m) or max_palm_span_m <= 0.0:
            raise ValueError("max_palm_span_m must be finite and positive")

        self._min_consensus_points = min_consensus_points
        self._max_palm_span_m = max_palm_span_m
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
        diagnostic="",
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
            diagnostic=diagnostic,
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

    def process_set(self, keypoint_set, now_sec):
        """Triangulate consenting landmarks and gate their median point."""
        if not isinstance(keypoint_set, StereoKeypointSet):
            raise TypeError("keypoint_set must be a StereoKeypointSet")
        now_sec = float(now_sec)
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")

        times = (
            float(keypoint_set.left_source_time_sec),
            float(keypoint_set.right_source_time_sec),
        )
        confidences = (
            float(keypoint_set.left_confidence),
            float(keypoint_set.right_confidence),
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
        if not keypoint_set.left_pixels or not keypoint_set.right_pixels:
            decision = self._gate.invalidate("missing_keypoint")
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
            )

        shared_indices = sorted(
            set(keypoint_set.left_pixels) & set(keypoint_set.right_pixels)
        )
        survivors = {}
        failures = []
        for index in shared_indices:
            try:
                survivors[index] = triangulate_keypoint(
                    self._left_projection,
                    self._right_projection,
                    self._fundamental_matrix,
                    keypoint_set.left_pixels[index],
                    keypoint_set.right_pixels[index],
                    max_epipolar_error_px=self._max_epipolar_error_px,
                    max_reprojection_error_px=(
                        self._gate_config.max_reprojection_error_px
                    ),
                    min_depth_m=self._min_depth_m,
                )
            except (StereoGeometryError, TypeError, ValueError) as error:
                failures.append(f"{index}: {error}")

        if len(survivors) < self._min_consensus_points:
            decision = self._gate.invalidate("insufficient_consensus")
            diagnostic = (
                f"{len(survivors)}/{self._min_consensus_points} landmarks "
                "passed stereo geometry"
            )
            if failures:
                diagnostic += "; " + "; ".join(failures)
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
                diagnostic=diagnostic,
            )

        # Consistent-but-wrong horizontal matches pass the epipolar and
        # reprojection checks yet land far from the true palm in 3D, so
        # reject survivors that scatter beyond one physical palm span.
        cluster_median = np.median(
            np.asarray(
                [survivors[index].point for index in sorted(survivors)]
            ),
            axis=0,
        )
        kept = [
            index
            for index in sorted(survivors)
            if np.linalg.norm(survivors[index].point - cluster_median)
            <= self._max_palm_span_m
        ]
        if len(kept) < self._min_consensus_points:
            decision = self._gate.invalidate("palm_cluster_rejected")
            return self._result(
                decision,
                confidence=confidence,
                pair_skew_sec=pair_skew_sec,
                reprojection_error_px=0.0,
                source_time_sec=source_time_sec,
                diagnostic=(
                    f"only {len(kept)}/{len(survivors)} triangulated "
                    "landmarks lie within "
                    f"{self._max_palm_span_m:.3f} m of the palm median"
                ),
            )

        palm_point = tuple(
            float(value)
            for value in np.median(
                np.asarray([survivors[index].point for index in kept]),
                axis=0,
            )
        )
        reprojection_error_px = max(
            survivors[index].max_reprojection_error_px for index in kept
        )
        worst_epipolar_px = max(
            survivors[index].epipolar_error_px for index in kept
        )

        candidate = HandFrameCandidate(
            point=palm_point,
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
            diagnostic=(
                f"consensus={len(kept)}/{len(shared_indices)} "
                f"epi_max={worst_epipolar_px:.3f}px "
                f"reproj_max={reprojection_error_px:.3f}px "
                "xyz=("
                f"{palm_point[0]:.3f}, "
                f"{palm_point[1]:.3f}, "
                f"{palm_point[2]:.3f})m"
            ),
        )
