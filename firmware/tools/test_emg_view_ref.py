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
)


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
    assert view_activation(0, 100) == 0
    assert view_activation(99, 100) == 0
    assert view_activation(100, 100) == 0
    assert view_activation(101, 100) > 0


def test_activation_saturates_at_the_reference_rather_than_wrapping():
    threshold = 100
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    assert view_activation(reference, threshold) == 65535
    # A wearer pushing well past their ceiling gets full deflection, not a
    # value that wrapped through a uint16.
    assert view_activation(reference * 10, threshold) == 65535


def test_the_midpoint_of_the_span_is_about_half_deflection():
    threshold = 100
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    middle = threshold + (reference - threshold) // 2
    assert view_activation(middle, threshold) == pytest.approx(32767, abs=400)


@pytest.mark.parametrize("threshold", [1, 2, 3, 7, 64, 1000])
def test_the_span_never_collapses_for_any_positive_threshold(threshold):
    # reference = floor(t * NUM / DEN) stays above t for every positive t as
    # long as the ratio exceeds one, so the span <= 0 guard in the C is
    # unreachable today. It is kept deliberately: the constants are interim,
    # and a ratio of one or less would divide by zero rather than fail closed.
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    assert reference > threshold
    assert view_activation(reference, threshold) == 65535
    assert view_activation(threshold, threshold) == 0


def test_multiplication_happens_before_division():
    # The other order truncates every value below the reference to zero, which
    # would look like a dead channel rather than an arithmetic mistake.
    threshold = 1000
    reference = threshold * REFERENCE_NUM // REFERENCE_DEN
    just_above = threshold + (reference - threshold) // 100
    assert view_activation(just_above, threshold) > 500


def test_activation_is_monotonic_in_effort():
    threshold = 80
    values = [view_activation(v, threshold) for v in range(80, 300)]
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
            reference = view_activation(step * 5, threshold)
            if firmware != reference:
                mismatches.append((threshold, step * 5, firmware, reference))
    assert not mismatches, mismatches[:5]

    # The fixture has to exercise the range, or agreeing on zeros would pass.
    assert max(values) == 65535
    assert min(values) == 0
    assert len(set(values)) > 100
