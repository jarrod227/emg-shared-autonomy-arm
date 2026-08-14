"""Tests for the sEMG band-pass, host reference and firmware alike.

The last test is the one that matters: it reads golden.bin, produced by the
same C that gets flashed, and checks those outputs against a float reference
computed here. Fixed-point arithmetic is where the mistakes live, and this
catches them on a workstation rather than over a debug probe.

    make -C firmware/test check
    python3 -m pytest firmware/tools/test_emg_filter_ref.py
"""

import pathlib
import re
import struct

import numpy as np
import pytest

from emg_filter_ref import (
    COEFF_BITS,
    FixedFilter,
    MAX_SECTIONS,
    design_bandpass,
    design_emg_filter,
    filter_fixed,
    filter_float,
    format_c_initializer,
    to_fixed,
)

FIRMWARE = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = FIRMWARE / "test" / "golden.bin"
FILTER_SOURCE = FIRMWARE / "src" / "emg_filter.c"

# The quantization error measured across the design: max 0.50 counts is the
# rounding quantum itself, so 2 counts leaves room without hiding a real bug.
TOLERANCE_COUNTS = 2.0


def default_sections():
    """The cascade the firmware ships: band-pass plus mains notches."""
    return to_fixed(design_emg_filter())


def test_design_returns_two_sections_for_a_fourth_order_bandpass():
    sos = design_bandpass()

    assert sos.shape == (2, 6)
    assert np.allclose(sos[:, 3], 1.0)


@pytest.mark.parametrize(
    "low, high, rate",
    [(0.0, 450.0, 2000.0), (450.0, 20.0, 2000.0), (20.0, 1200.0, 2000.0)],
)
def test_design_rejects_impossible_bands(low, high, rate):
    # The last case is above Nyquist, which is the mistake most worth
    # refusing: it would silently alias rather than fail.
    with pytest.raises(ValueError):
        design_bandpass(low, high, rate)


def test_fixed_coefficients_fit_int32():
    for coeffs in default_sections():
        for value in coeffs:
            assert -(1 << 31) <= value < (1 << 31)


def test_to_fixed_rejects_too_many_sections():
    sos = design_bandpass(order=MAX_SECTIONS + 1)

    with pytest.raises(ValueError):
        to_fixed(sos)


def test_fixed_tracks_float_within_the_rounding_quantum():
    sos = design_emg_filter()
    steps = np.arange(4000) / 2000.0
    signal = (900 * np.sin(2 * np.pi * 80 * steps)
              + 400 * np.sin(2 * np.pi * 250 * steps) + 600)
    samples = np.clip(np.round(signal), -2048, 2047).astype(np.int16)

    error = filter_fixed(to_fixed(sos), samples) - filter_float(sos, samples)

    assert np.abs(error).max() < TOLERANCE_COUNTS


def test_the_band_pass_removes_dc():
    sections = default_sections()

    settled = filter_fixed(sections, np.full(3000, 2000, dtype=np.int16))

    assert np.all(settled[-200:] == 0)


def test_the_band_pass_keeps_in_band_content_and_drops_out_of_band():
    sections = default_sections()
    steps = np.arange(6000) / 2000.0

    def amplitude(frequency):
        wave = np.round(1000 * np.sin(2 * np.pi * frequency * steps))
        response = filter_fixed(sections, wave.astype(np.int16))
        return np.abs(response[3000:]).max()

    # 80 Hz sits in the pass band; 2 Hz and 900 Hz are well outside it.
    assert amplitude(80.0) > 900
    assert amplitude(2.0) < 100
    assert amplitude(900.0) < 100


def test_the_c_coefficient_table_matches_scipy():
    """Catch a hand-edited or stale table in emg_filter.c.

    The C file carries the numbers as literals so the MCU never needs float,
    which means nothing stops them drifting from the design they claim to
    come from except this test.
    """
    source = FILTER_SOURCE.read_text()
    match = re.search(
        r"emg_filter_20_450_notch50_at_2000\[[^\]]*\]\s*=\s*\{(.*?)\n\};",
        source, re.S,
    )
    assert match, "coefficient table not found in emg_filter.c"

    embedded = [
        tuple(int(value) for value in row.split(","))
        for row in re.findall(r"\{([^{}]*)\}", match.group(1))
    ]

    assert embedded == default_sections()


def test_generated_c_initializer_is_valid_c():
    text = format_c_initializer(default_sections())

    assert text.startswith("const emg_biquad_coeffs_t ")
    assert text.rstrip().endswith("};")
    assert text.count("{") == text.count("}")


def test_firmware_output_matches_the_float_reference():
    """Golden vector: the flashed C's output against scipy.

    golden.bin holds the input sequence and the outputs the C implementation
    produced for it. Re-filtering that input here and comparing is what shows
    the fixed-point firmware really implements the filter that was designed.
    """
    if not GOLDEN.exists():
        pytest.skip("run `make -C firmware/test check` to generate golden.bin")

    payload = GOLDEN.read_bytes()
    count = struct.unpack_from("<i", payload, 0)[0]
    assert count > 0
    samples = np.frombuffer(payload, dtype="<i2", count=count, offset=4)
    firmware = np.frombuffer(payload, dtype="<i4", count=count,
                             offset=4 + 2 * count).astype(np.int64)

    reference = filter_float(design_emg_filter(), samples)
    error = firmware - reference

    assert np.abs(error).max() < TOLERANCE_COUNTS
    # The signal has to actually exercise the filter, or agreeing on a flat
    # line would pass this test.
    assert np.abs(reference).max() > 100

    # The same bit-exact model must reproduce the firmware exactly, not just
    # within tolerance -- that is what makes the model usable for trying
    # numerical changes before writing them in C.
    assert np.array_equal(filter_fixed(default_sections(), samples), firmware)


def test_filtering_in_chunks_equals_filtering_all_at_once():
    """The property the live scope depends on.

    A filter reset between blocks puts a settling transient at the start of
    each one; for a 20 Hz high-pass that is a visible swing lasting about a
    tenth of a second, which on a scrolling plot looks like real signal.
    """
    sections = default_sections()
    steps = np.arange(3000) / 2000.0
    samples = np.clip(
        np.round(800 * np.sin(2 * np.pi * 90 * steps) + 600), -2048, 2047
    ).astype(np.int16)

    whole = FixedFilter(sections).process(samples)

    chunked = FixedFilter(sections)
    pieces = [chunked.process(samples[start:start + 96])
              for start in range(0, len(samples), 96)]

    assert np.array_equal(np.concatenate(pieces), whole)


def test_a_reset_filter_does_not_continue_the_previous_signal():
    # The complement of the test above: state really is state, so clearing it
    # has to change the result rather than being a no-op.
    sections = default_sections()
    samples = np.full(200, 1500, dtype=np.int16)

    stateful = FixedFilter(sections)
    first = stateful.process(samples)
    continued = stateful.process(samples)
    stateful.reset()
    restarted = stateful.process(samples)

    assert not np.array_equal(first, continued)
    assert np.array_equal(first, restarted)


def test_the_mains_notches_reject_50_and_150_hz():
    """The reason the notches exist, asserted after quantization.

    A narrow notch is the most quantization-sensitive filter shape here, so
    rejection is checked on the fixed-point cascade rather than on the float
    design it came from.
    """
    sections = default_sections()
    steps = np.arange(8000) / 2000.0

    def amplitude(frequency):
        wave = np.round(1000 * np.sin(2 * np.pi * frequency * steps))
        return np.abs(filter_fixed(sections, wave.astype(np.int16))[4000:]).max()

    assert amplitude(50.0) < 20        # mains fundamental, gone
    assert amplitude(150.0) < 20       # third harmonic, gone
    # And the notches must be narrow enough to leave the neighbourhood alone.
    assert amplitude(80.0) > 900
    assert amplitude(120.0) > 800
    assert amplitude(200.0) > 800


def test_every_quantized_pole_stays_inside_the_unit_circle():
    """Stability after quantization, which narrow notches can lose."""
    for index, (_, _, _, a1, a2) in enumerate(default_sections()):
        scale = float(1 << COEFF_BITS)
        radius = max(abs(root) for root in np.roots([1.0, a1 / scale, a2 / scale]))
        assert radius < 1.0, f"section {index} pole radius {radius}"
