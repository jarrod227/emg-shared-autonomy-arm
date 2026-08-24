"""Behaviour of the proportional view output, before it reaches hardware.

The last test is the one that matters: it reads view.bin, produced by the
compiled C, and requires exact agreement. A host tool that predicts a
different activation than the board emits is worse than no prediction.
"""

import pathlib
import struct

import pytest

from emg_view_ref import (
    REFERENCE_DEN,
    REFERENCE_NUM,
    ABORT,
    CONFIRM,
    NEXT_TARGET,
    REST,
    ULNAR,
    view_activation,
    view_direction,
    view_reference,
)


def fallback(threshold):
    """What an uncalibrated board derives, for the tests written against it."""
    return threshold * REFERENCE_NUM // REFERENCE_DEN


def test_the_two_directions_are_opposite_and_nothing_else_steers():
    # A view channel that moved on CONFIRM or ABORT would steer during a
    # command meant to confirm or stop, in the one state where unintended
    # motion is hardest to notice.
    assert view_direction(NEXT_TARGET) == -1
    assert view_direction(ULNAR) == 1
    for other in (REST, CONFIRM, ABORT, "RADIAL", "SUPINATE"):
        assert view_direction(other) == 0


def test_activation_is_zero_at_and_below_the_threshold():
    # Not negative and not clamped from a negative: below the threshold there
    # is no intent to scale, and the controller reads zero as HOLD.
    assert view_activation(0, 100, fallback(100)) == 0
    assert view_activation(99, 100, fallback(100)) == 0
    assert view_activation(100, 100, fallback(100)) == 0
    assert view_activation(101, 100, fallback(100)) > 0


def test_activation_saturates_at_the_reference_rather_than_wrapping():
    threshold = 100
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    assert view_activation(reference, threshold, reference) == 65535
    # A wearer pushing well past their ceiling gets full deflection, not a
    # value that wrapped through a uint16.
    assert view_activation(reference * 10, threshold, reference) == 65535


def test_the_midpoint_of_the_span_is_about_half_deflection():
    threshold = 100
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    middle = threshold + (reference - threshold) // 2
    assert view_activation(middle, threshold, reference) == pytest.approx(32767, abs=400)


@pytest.mark.parametrize("threshold", [1, 2, 3, 7, 64, 1000])
def test_the_span_never_collapses_for_any_positive_threshold(threshold):
    # reference = floor(t * NUM / DEN) stays above t for every positive t as
    # long as the ratio exceeds one, so the span <= 0 guard in the C is
    # unreachable today. It is kept deliberately: the constants are interim,
    # and a ratio of one or less would divide by zero rather than fail closed.
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    assert reference > threshold
    assert view_activation(reference, threshold, reference) == 65535
    assert view_activation(threshold, threshold, reference) == 0


def test_multiplication_happens_before_division():
    # The other order truncates every value below the reference to zero, which
    # would look like a dead channel rather than an arithmetic mistake.
    threshold = 1000
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    just_above = threshold + (reference - threshold) // 100
    assert view_activation(just_above, threshold, reference) > 500


def test_activation_is_monotonic_in_effort():
    threshold = 80
    values = [view_activation(v, threshold, fallback(threshold)) for v in range(80, 300)]
    assert values == sorted(values)
    assert values[0] == 0 and values[-1] == 65535


FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "test" / "view.bin"


def test_the_reference_reproduces_the_compiled_c_exactly():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate view.bin")

    payload = FIXTURE.read_bytes()
    count, steps = struct.unpack_from("<ii", payload, 0)
    thresholds = struct.unpack_from(f"<{count}i", payload, 8)
    offset = 8 + 4 * count
    values = struct.unpack_from(f"<{count * steps}H", payload, offset)

    assert count > 0 and steps > 0
    mismatches = []
    for i, threshold in enumerate(thresholds):
        for step in range(steps):
            firmware = values[i * steps + step]
            expected = view_activation(step * 5, threshold,
                                       fallback(threshold))
            if firmware != expected:
                mismatches.append(
                    (threshold, step * 5, firmware, expected)
                )
    assert not mismatches, mismatches[:5]

    # The fixture has to exercise the range, or agreeing on zeros would pass.
    assert max(values) == 65535
    assert min(values) == 0
    assert len(set(values)) > 100


def test_the_reference_is_per_direction_and_falls_back_only_when_missing():
    """One number provably cannot serve both directions.

    Measured 2026-08-23 in a single capture on a single donning: wrist
    extension came to 4.19x the session threshold and ulnar deviation to
    5.51x. The compile-time constant that used to serve both is 3, below
    both, which saturated 58% of one session's LEFT commands.
    """
    assert view_reference(-1, 55, 231, 303) == 231
    assert view_reference(1, 55, 231, 303) == 303
    # Missing on one side only falls back on that side.
    assert view_reference(-1, 55, 0, 303) == fallback(55)
    assert view_reference(1, 55, 231, 0) == fallback(55)


def test_no_direction_means_no_activation_rather_than_a_default_reference():
    """Activation is a fraction of the span for the gesture being commanded.

    With no gesture there is no span, so publishing a number derived from
    whichever reference happened to be picked would make the units depend on
    an arbitrary choice.
    """
    assert view_reference(0, 55, 231, 303) == 0
    assert view_activation(400, 55, view_reference(0, 55, 231, 303)) == 0


def test_a_reference_inside_the_threshold_yields_no_deflection():
    # An empty span. Refused at the sender too, but the firmware must not
    # divide by it if one arrives anyway.
    assert view_activation(400, 100, 100) == 0
    assert view_activation(400, 100, 60) == 0


def test_a_larger_reference_makes_the_same_effort_read_lower():
    # The whole point of measuring it: the same muscle signal against a
    # correctly measured ceiling stops sitting at full deflection.
    # Real numbers: 2026-08-23 measured threshold 55 and a NEXT_TARGET
    # reference of 231, where the compile-time fallback would give 165.
    effort = 200
    assert view_activation(effort, 55, fallback(55)) == 65535, (
        "the fallback should saturate here; that is the defect"
    )
    measured = view_activation(effort, 55, 231)
    assert 0.7 < measured / 65535 < 0.9
