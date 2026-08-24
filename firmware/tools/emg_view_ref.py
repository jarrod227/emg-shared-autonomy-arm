#!/usr/bin/env python3
"""Bit-exact mirror of firmware/src/emg_view.c.

Same reason as the other _ref modules: the host tools have to be able to say
what the board will emit, so a change can be tried and replayed offline before
it is flashed. Integer arithmetic throughout, in the same order as the C,
because the truncation is part of the answer.
"""

REFERENCE_NUM = 3
REFERENCE_DEN = 1

# Same values as emg_packet.h.
REST = "REST"
NEXT_TARGET = "NEXT_TARGET"
CONFIRM = "CONFIRM"
ABORT = "ABORT"

# Only the gestures assigned to a view direction map to anything. Everything
# else, REST included, is 0 and reads downstream as HOLD.
ULNAR = "ULNAR"

DIRECTIONS = {NEXT_TARGET: -1, ULNAR: 1}


def view_direction(decision):
    """-1, 0 or +1, from the post-activation decision."""
    return DIRECTIONS.get(decision, 0)


def view_reference(direction, threshold, reference_left, reference_right):
    """The calibrated ceiling for one direction, or the fallback.

    Zero for direction 0: activation is a fraction of the span for the
    gesture being commanded, so with no gesture there is nothing to take a
    fraction of.
    """
    direction = int(direction)
    if direction == 0:
        return 0
    measured = int(reference_left if direction < 0 else reference_right)
    if measured > 0:
        return measured
    threshold = int(threshold)
    if threshold <= 0:
        return 0
    return threshold * REFERENCE_NUM // REFERENCE_DEN


def view_activation(total_mav, threshold, reference):
    """0..65535 mapping to 0.0..1.0, zero at or below the threshold."""
    total_mav = int(total_mav)
    threshold = int(threshold)
    reference = int(reference)
    if threshold <= 0 or total_mav <= threshold:
        return 0
    span = reference - threshold
    if span <= 0:
        return 0
    above = total_mav - threshold
    if above >= span:
        return 65535
    return (above * 65535) // span


def activation_fraction(total_mav, threshold, reference):
    """The same value as a float, for readability in analysis scripts."""
    return view_activation(total_mav, threshold, reference) / 65535.0
