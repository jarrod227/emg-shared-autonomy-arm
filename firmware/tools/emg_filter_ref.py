#!/usr/bin/env python3
"""Host reference for the fixed-point sEMG band-pass in ../src/emg_filter.c.

Three jobs:

  * design the coefficients with scipy and emit them as a C initializer,
  * provide a float reference the firmware's output is checked against,
  * provide a bit-exact model of the C, so a numerical idea can be tried out
    here before it is written in C and debugged over a debug probe.

Nothing here ships to the MCU. Coefficients are computed once on the host and
baked into the firmware as integers, because the Cortex-M3 has no FPU.

    python3 emg_filter_ref.py --emit-c
"""

import argparse

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

# Must match emg_filter.h.
COEFF_BITS = 29
STATE_BITS = 12
MAX_SECTIONS = 6

DEFAULT_ORDER = 2
DEFAULT_LOW_HZ = 20.0
DEFAULT_HIGH_HZ = 450.0
DEFAULT_RATE_HZ = 2000.0

# Mains hum sits inside the pass band, so the band-pass alone cannot touch it.
# Measured on a real session, notching the fundamental and third harmonic
# recovered gesture structure that was completely buried: channel 1 went from
# 96.6% mains and a 1.5x rest-to-contraction contrast to 7.3% and 6.4x. The
# second harmonic was measured too and contributed almost nothing, so it is
# left out and the cascade stays at four sections.
DEFAULT_MAINS_HZ = 50.0
DEFAULT_NOTCH_HARMONICS = (1, 3)
DEFAULT_NOTCH_Q = 30.0


def design_bandpass(low_hz=DEFAULT_LOW_HZ, high_hz=DEFAULT_HIGH_HZ,
                    rate_hz=DEFAULT_RATE_HZ, order=DEFAULT_ORDER):
    """Butterworth band-pass as scipy second-order sections.

    `order` is per edge, so the resulting filter is twice that: order 2 gives
    the conventional 4th-order band-pass in two biquads.
    """
    if not 0 < low_hz < high_hz < rate_hz / 2:
        raise ValueError("need 0 < low < high < rate/2")
    return butter(order, [low_hz, high_hz], btype="band", fs=rate_hz,
                  output="sos")


def design_notches(mains_hz=DEFAULT_MAINS_HZ,
                   harmonics=DEFAULT_NOTCH_HARMONICS,
                   rate_hz=DEFAULT_RATE_HZ, quality=DEFAULT_NOTCH_Q):
    """Narrow notches on the mains fundamental and chosen harmonics."""
    rows = []
    for harmonic in harmonics:
        centre = mains_hz * harmonic
        if not 0 < centre < rate_hz / 2:
            raise ValueError(f"notch at {centre} Hz is outside the Nyquist range")
        rows.append(tf2sos(*iirnotch(centre, quality, rate_hz)))
    return np.vstack(rows) if rows else np.empty((0, 6))


def design_emg_filter(low_hz=DEFAULT_LOW_HZ, high_hz=DEFAULT_HIGH_HZ,
                      rate_hz=DEFAULT_RATE_HZ, order=DEFAULT_ORDER,
                      mains_hz=DEFAULT_MAINS_HZ,
                      harmonics=DEFAULT_NOTCH_HARMONICS,
                      quality=DEFAULT_NOTCH_Q):
    """The cascade the firmware actually runs: band-pass then mains notches."""
    band = design_bandpass(low_hz, high_hz, rate_hz, order)
    notches = design_notches(mains_hz, harmonics, rate_hz, quality)
    return np.vstack([band, notches]) if len(notches) else band


def to_fixed(sos, coeff_bits=COEFF_BITS):
    """Quantize scipy sos rows to the (b0, b1, b2, a1, a2) integers C uses."""
    sections = np.asarray(sos, dtype=np.float64)
    if sections.ndim != 2 or sections.shape[1] != 6:
        raise ValueError("sos must be an N x 6 array")
    if len(sections) > MAX_SECTIONS:
        raise ValueError(f"at most {MAX_SECTIONS} sections fit in the C struct")
    if not np.allclose(sections[:, 3], 1.0):
        raise ValueError("every section must be normalized to a0 = 1")

    scale = 1 << coeff_bits
    fixed = []
    for row in sections:
        values = [int(round(value * scale))
                  for value in (row[0], row[1], row[2], row[4], row[5])]
        if any(abs(value) >= 1 << 31 for value in values):
            raise ValueError(f"coefficient overflows int32 at Q{coeff_bits}")
        fixed.append(tuple(values))
    return fixed


def filter_float(sos, samples):
    """Canonical float result. This is what the firmware is judged against."""
    return sosfilt(np.asarray(sos, dtype=np.float64),
                   np.asarray(samples, dtype=np.float64))


class FixedFilter:
    """Stateful bit-exact model of emg_filter_step, including its rounding.

    Stateful because the firmware is: a filter reset between blocks puts a
    settling transient at the start of every block, which for a 20 Hz
    high-pass is a visible swing lasting about a tenth of a second. Anything
    processing a live stream in chunks has to carry the state across them.

    Python's `>>` floors like C's does on negative values, so the explicit
    half-quantum add reproduces the C rather than merely approximating it.
    """

    def __init__(self, sections, coeff_bits=COEFF_BITS, state_bits=STATE_BITS):
        self._sections = [tuple(int(value) for value in row) for row in sections]
        self._coeff_bits = coeff_bits
        self._state_bits = state_bits
        self._coeff_half = 1 << (coeff_bits - 1)
        self._state_half = 1 << (state_bits - 1)
        self.reset()

    def reset(self):
        self._state = [[0, 0, 0, 0] for _ in self._sections]

    def process(self, samples):
        """Filter one block, carrying state into the next call."""
        output = np.empty(len(samples), dtype=np.int64)
        for index, sample in enumerate(samples):
            value = int(sample) << self._state_bits
            for coeffs, history in zip(self._sections, self._state):
                b0, b1, b2, a1, a2 = coeffs
                accumulator = (b0 * value + b1 * history[0] + b2 * history[1]
                               - a1 * history[2] - a2 * history[3])
                result = (accumulator + self._coeff_half) >> self._coeff_bits
                history[1], history[0] = history[0], value
                history[3], history[2] = history[2], result
                value = result
            output[index] = (value + self._state_half) >> self._state_bits
        return output


def filter_fixed(sections, samples, coeff_bits=COEFF_BITS,
                 state_bits=STATE_BITS):
    """Filter a complete signal from a cleared state."""
    return FixedFilter(sections, coeff_bits, state_bits).process(samples)


def format_c_initializer(sections, name="emg_filter_20_450_notch50_at_2000"):
    lines = [
        f"const emg_biquad_coeffs_t {name}[{len(sections)}] = {{",
    ]
    for coeffs in sections:
        body = ", ".join(str(value) for value in coeffs)
        lines.append(f"        {{{body}}},")
    lines.append("};")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=float, default=DEFAULT_LOW_HZ)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH_HZ)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER,
                        help="per edge; the band-pass is twice this")
    parser.add_argument("--mains", type=float, default=DEFAULT_MAINS_HZ,
                        help="mains frequency to notch (50 in most of the "
                             "world, 60 in the Americas)")
    parser.add_argument("--emit-c", action="store_true",
                        help="print the C initializer for emg_filter.c")
    arguments = parser.parse_args()

    sos = design_emg_filter(arguments.low, arguments.high, arguments.rate,
                            arguments.order, arguments.mains)
    sections = to_fixed(sos)

    if arguments.emit_c:
        print(format_c_initializer(sections))
        return 0

    print(f"Band-pass {arguments.low}-{arguments.high} Hz plus "
          f"{arguments.mains} Hz notches at {arguments.rate} Hz "
          f"-> {len(sections)} sections")
    for index, (row, coeffs) in enumerate(zip(sos, sections)):
        print(f"  section {index}")
        print(f"    float  b={row[0]:+.10f} {row[1]:+.10f} {row[2]:+.10f}"
              f"  a={row[4]:+.10f} {row[5]:+.10f}")
        print(f"    Q{COEFF_BITS}    {coeffs}")

    # Quantization cost, measured rather than asserted.
    rng = np.random.default_rng(7)
    steps = np.arange(4000) / arguments.rate
    signal = (900 * np.sin(2 * np.pi * 80 * steps)
              + 400 * np.sin(2 * np.pi * 250 * steps)
              + 600 + rng.normal(0, 25, steps.size))
    samples = np.clip(np.round(signal), -2048, 2047).astype(np.int16)
    error = filter_fixed(sections, samples) - filter_float(sos, samples)
    print(f"\n  fixed vs float over {samples.size} samples: "
          f"max {np.abs(error).max():.2f} counts, "
          f"RMS {np.sqrt((error ** 2).mean()):.3f} counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
