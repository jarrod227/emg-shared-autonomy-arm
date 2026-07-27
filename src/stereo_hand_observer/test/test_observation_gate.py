"""Tests for fail-closed delivery-volume and temporal stability gating."""

import numpy as np
import pytest

from stereo_hand_observer.observation_gate import (
    DeliveryStabilityGate,
    DeliveryVolume,
    HandFrameCandidate,
    StabilityGateConfig,
)


DELIVERY_CENTER = (0.4, 0.3, 1.0)
VOLUME = DeliveryVolume(center=DELIVERY_CENTER, radius_m=0.2)
CONFIG = StabilityGateConfig(
    required_frames=3,
    min_confidence=0.7,
    max_pair_skew_sec=0.02,
    max_reprojection_error_px=1.5,
    max_age_sec=0.2,
    max_point_step_m=0.05,
)


def candidate(
    *,
    point=DELIVERY_CENTER,
    confidence=0.9,
    pair_skew_sec=0.01,
    reprojection_error_px=0.4,
    source_time_sec=10.0,
):
    """Build one nominal synthetic stereo observation."""
    return HandFrameCandidate(
        point=point,
        confidence=confidence,
        pair_skew_sec=pair_skew_sec,
        reprojection_error_px=reprojection_error_px,
        source_time_sec=source_time_sec,
    )


def test_delivery_volume_is_spherical_and_includes_boundary():
    assert VOLUME.contains(DELIVERY_CENTER)
    assert VOLUME.contains((0.6, 0.3, 1.0))
    assert not VOLUME.contains((0.61, 0.3, 1.0))


def test_gate_becomes_valid_only_after_required_stable_frames():
    gate = DeliveryStabilityGate(VOLUME, CONFIG)
    points = (
        (0.40, 0.30, 1.00),
        (0.41, 0.29, 1.01),
        (0.39, 0.31, 0.99),
    )

    first = gate.update(
        candidate(point=points[0], source_time_sec=10.00),
        now_sec=10.01,
    )
    second = gate.update(
        candidate(point=points[1], source_time_sec=10.03),
        now_sec=10.04,
    )
    third = gate.update(
        candidate(point=points[2], source_time_sec=10.06),
        now_sec=10.07,
    )

    assert not first.valid
    assert first.stable_frames == 1
    assert not second.valid
    assert second.stable_frames == 2
    assert third.valid
    assert third.stable_frames == 3
    np.testing.assert_allclose(third.point, DELIVERY_CENTER)


def test_large_point_jump_starts_a_new_stability_run():
    gate = DeliveryStabilityGate(VOLUME, CONFIG)
    gate.update(candidate(source_time_sec=10.00), now_sec=10.01)
    jumped = gate.update(
        candidate(point=(0.48, 0.3, 1.0), source_time_sec=10.03),
        now_sec=10.04,
    )

    assert not jumped.valid
    assert jumped.reason == "unstable"
    assert jumped.stable_frames == 1

    gate.update(
        candidate(point=(0.49, 0.3, 1.0), source_time_sec=10.06),
        now_sec=10.07,
    )
    stable = gate.update(
        candidate(point=(0.47, 0.3, 1.0), source_time_sec=10.09),
        now_sec=10.10,
    )
    assert stable.valid
    np.testing.assert_allclose(stable.point, (0.48, 0.3, 1.0))


@pytest.mark.parametrize(
    ("bad_candidate", "now_sec", "expected_reason"),
    (
        (None, 10.01, "missing"),
        (
            candidate(point=(0.7, 0.3, 1.0)),
            10.01,
            "outside_delivery_volume",
        ),
        (
            candidate(confidence=0.69),
            10.01,
            "low_confidence",
        ),
        (
            candidate(pair_skew_sec=0.021),
            10.01,
            "excessive_pair_skew",
        ),
        (
            candidate(reprojection_error_px=1.51),
            10.01,
            "high_reprojection_error",
        ),
        (
            candidate(source_time_sec=9.79),
            10.01,
            "stale",
        ),
        (
            candidate(point=(np.nan, 0.3, 1.0)),
            10.01,
            "invalid_measurement",
        ),
    ),
)
def test_bad_frame_fails_closed_and_clears_stability(
    bad_candidate,
    now_sec,
    expected_reason,
):
    gate = DeliveryStabilityGate(VOLUME, CONFIG)
    gate.update(candidate(source_time_sec=9.94), now_sec=9.95)
    gate.update(candidate(source_time_sec=9.97), now_sec=9.98)
    assert gate.stable_frames == 2

    decision = gate.update(bad_candidate, now_sec=now_sec)

    assert not decision.valid
    assert decision.point is None
    assert decision.reason == expected_reason
    assert decision.stable_frames == 0
    assert gate.stable_frames == 0


def test_duplicate_source_timestamp_cannot_count_as_another_frame():
    gate = DeliveryStabilityGate(VOLUME, CONFIG)
    first = candidate(source_time_sec=10.0)
    gate.update(first, now_sec=10.01)

    duplicate = gate.update(first, now_sec=10.02)

    assert not duplicate.valid
    assert duplicate.reason == "non_increasing_timestamp"
    assert duplicate.stable_frames == 0


def test_invalid_frame_after_valid_run_immediately_revokes_validity():
    gate = DeliveryStabilityGate(VOLUME, CONFIG)
    for index in range(3):
        source_time = 10.0 + index * 0.03
        decision = gate.update(
            candidate(source_time_sec=source_time),
            now_sec=source_time + 0.01,
        )
    assert decision.valid

    invalid = gate.update(None, now_sec=10.10)

    assert not invalid.valid
    assert invalid.reason == "missing"
    assert gate.stable_frames == 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("required_frames", 0),
        ("min_confidence", 1.1),
        ("max_pair_skew_sec", -0.1),
        ("max_reprojection_error_px", np.inf),
        ("max_age_sec", 0.0),
        ("max_point_step_m", -0.1),
    ),
)
def test_bad_configuration_is_rejected(field, bad_value):
    values = {
        "required_frames": CONFIG.required_frames,
        "min_confidence": CONFIG.min_confidence,
        "max_pair_skew_sec": CONFIG.max_pair_skew_sec,
        "max_reprojection_error_px": CONFIG.max_reprojection_error_px,
        "max_age_sec": CONFIG.max_age_sec,
        "max_point_step_m": CONFIG.max_point_step_m,
    }
    values[field] = bad_value

    with pytest.raises(ValueError):
        StabilityGateConfig(**values)
