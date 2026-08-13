"""Tests for the sEMG feature extractor, host reference and firmware alike.

The last test is the golden vector: features.bin holds the sample sequence and
the feature sets the flashed C produced for it, and this recomputes them here.
Both sides are integer arithmetic, so the comparison is for **exact** equality
-- any difference at all is a real disagreement, not rounding.

    make -C firmware/test check
    python3 -m pytest firmware/tools/test_emg_features_ref.py
"""

import pathlib
import struct

import pytest

from emg_features_ref import (
    HOP,
    WINDOW,
    compute_features,
    mean_absolute_value,
    root_mean_square,
    sliding_features,
    waveform_length,
    zero_crossings,
)

FEATURES = pathlib.Path(__file__).resolve().parents[1] / "test" / "features.bin"


def test_constant_signal():
    samples = [-250] * WINDOW

    assert compute_features(samples, 10) == (250, 250, 0, 0)


def test_square_wave_is_exact():
    samples = [100 if index % 2 == 0 else -100 for index in range(WINDOW)]

    mav, rms, length, crossings = compute_features(samples, 10)

    assert (mav, rms) == (100, 100)
    assert length == 399 * 200
    assert crossings == 399


def test_zero_crossing_threshold_gates_noise():
    dither = [2 if index % 2 == 0 else -2 for index in range(WINDOW)]

    # A 4-count swing is noise against a 10-count gate and signal without one.
    assert zero_crossings(dither, 10) == 0
    assert zero_crossings(dither, 0) == 399


def test_exact_zeros_do_not_count_as_crossings():
    # +5, 0, +5 never changes sign, and 0 belongs to neither side.
    assert zero_crossings([5, 0, 5, 0, 5], 1) == 0
    assert zero_crossings([5, 0, -5], 1) == 0
    assert zero_crossings([5, -5], 1) == 1


def test_rms_truncates_like_the_firmware():
    # Squares 1,1,4,4 sum to 10; 10 // 4 is 2; floor(sqrt(2)) is 1. A float
    # implementation would give 1.58 and round to 2, so this pins the
    # truncation the C actually performs.
    assert root_mean_square([1, -1, 2, -2]) == 1
    # Squares 4,4,4,1 sum to 13; 13 // 4 is 3; floor(sqrt(3)) is 1.
    assert root_mean_square([2, 2, 2, 1]) == 1
    # An exact square must come back exact rather than one short.
    assert root_mean_square([9, -9, 9, -9]) == 9


def test_mav_and_wl_on_a_hand_worked_case():
    samples = [0, 3, -3, 6]

    assert mean_absolute_value(samples) == (0 + 3 + 3 + 6) // 4
    assert waveform_length(samples) == 3 + 6 + 9


def test_compute_rejects_a_partial_window():
    with pytest.raises(ValueError):
        compute_features([0] * (WINDOW - 1), 10)


def test_sliding_emits_on_window_then_every_hop():
    samples = [0] * (WINDOW + 3 * HOP)

    results = sliding_features(samples, 10)

    assert len(results) == 4


def test_firmware_features_match_the_reference_exactly():
    """Golden vector against the C that gets flashed.

    Integer in, integer out, on both sides -- so this asserts equality, not
    closeness. A one-count difference would mean the two implementations
    genuinely disagree about a definition.
    """
    if not FEATURES.exists():
        pytest.skip("run `make -C firmware/test check` to generate features.bin")

    payload = FEATURES.read_bytes()
    count, threshold, emitted = struct.unpack_from("<iii", payload, 0)
    assert count > WINDOW and emitted > 0

    offset = 12
    samples = list(struct.unpack_from(f"<{count}h", payload, offset))
    offset += 2 * count
    firmware = [
        struct.unpack_from("<4i", payload, offset + 16 * index)
        for index in range(emitted)
    ]

    reference = sliding_features(samples, threshold)

    assert len(reference) == emitted
    assert reference == firmware

    # The signal has to exercise the features, or agreeing on zeros would
    # pass this test.
    assert max(row[0] for row in firmware) > 50
    assert max(row[3] for row in firmware) > 10
    assert len({row[0] for row in firmware}) > 1
