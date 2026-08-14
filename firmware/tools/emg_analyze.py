#!/usr/bin/env python3
"""Analyse a recorded sEMG session and judge electrode placement.

The question this answers is whether three channels carry three channels'
worth of information, or whether they are sitting on the same muscle and
carrying one. That is decided by correlating the **envelopes**, not the raw
signals.

Raw sEMG is a stochastic interference pattern: two electrodes over completely
different muscles still show near-zero sample-to-sample correlation, because
the motor units firing under each are independent. Redundant placement does
not show up there. It shows up in the amplitude envelope — if two channels
rise and fall together across gestures, they are reporting the same thing.

So the pipeline mirrors the firmware exactly: band-pass with the same
coefficients, then MAV over the same 200 ms / 50 ms windows, then correlate
those MAV series pairwise.

    python3 emg_analyze.py session.bin
"""

import argparse
import pathlib
import sys

import numpy as np

from emg_features_ref import HOP, WINDOW, mean_absolute_value
from emg_filter_ref import design_emg_filter, filter_fixed, to_fixed
from emg_protocol import TYPE_INFO, TYPE_RAW, PacketParser, decode_info, decode_raw

# Above this, two channels are telling the same story and one of them is
# not earning its place. Below 0.7 they are usefully distinct. The band
# between is a judgement call, which is why the script prints the number
# rather than only a verdict.
REDUNDANT_ABOVE = 0.90
DISTINCT_BELOW = 0.70
SEGMENTS = 10

# Any clipping at all makes the affected channel's amplitudes wrong, and a
# channel an order of magnitude quieter than the loudest is sitting at its
# noise floor. Correlation involving either is uninformative, so the verdict
# is withheld rather than reported against data that cannot support it.
MAX_CLIPPED_FRACTION = 0.001
MIN_AMPLITUDE_RATIO = 0.1

# The filter starts from a cleared state while the signal starts at the
# ~2048 mid-rail bias, so the first output is a large decaying transient
# from that step, not muscle. The firmware sees the same thing at boot; it
# is simply not information about placement. One second is comfortably past
# the 20 Hz section's settling.
WARMUP_SECONDS = 1.0

# Mains hum is measured on the *filtered* signal, because that is what the
# envelopes are computed from. The raw fraction is reported separately: a
# channel can be 97% mains before the notches and perfectly usable after,
# which is exactly what the first sessions turned out to be. A high raw
# fraction still means poor electrode contact and wasted ADC range, so it is
# worth saying, but it is not grounds for withholding a verdict.
MAX_MAINS_FRACTION = 0.30
MAINS_HARMONICS = 5
MAINS_HALF_WIDTH_HZ = 2.0


def load_session(path):
    """Return (info, channel sample arrays, parser stats, wear counts)."""
    parser = PacketParser()
    info = None
    columns = None
    wear_counts = {}
    pending = []

    def absorb(packet):
        block = decode_raw(packet.payload, info.channel_count)
        for channel in range(info.channel_count):
            if block.channel_attached(channel):
                wear_counts[channel] = wear_counts.get(channel, 0) + len(block.frames)
            columns[channel].extend(frame[channel] for frame in block.frames)

    for packet in parser.feed(pathlib.Path(path).read_bytes()):
        if packet.type == TYPE_INFO and info is None:
            info = decode_info(packet.payload)
            columns = [[] for _ in range(info.channel_count)]
            # A capture started mid-stream sees RAW before INFO. Those samples
            # are fine, they just could not be split into channels yet, and
            # dropping them would shift every segment time in the report.
            for held in pending:
                absorb(held)
            pending = []
        elif packet.type == TYPE_RAW:
            (absorb if info is not None else pending.append)(packet)

    if info is None:
        raise SystemExit("no INFO packet in the log; cannot interpret the samples")
    return info, [np.array(c, dtype=np.int64) for c in columns], parser.stats, wear_counts


def mav_series(samples, sections, rate_hz=2000.0):
    """Band-pass, then MAV per feature window — what the firmware computes.

    The filter's settling transient is dropped first. Left in, it dominates
    the opening windows: on a channel with only a couple of counts of real
    signal it produced a peak MAV of 77 against a raw standard deviation of
    1.8, which is enough to make a dead channel look alive.
    """
    filtered = filter_fixed(sections, np.clip(samples, -32768, 32767).astype(np.int16))
    warmup = min(int(WARMUP_SECONDS * rate_hz), max(0, len(filtered) - WINDOW))
    filtered = filtered[warmup:]
    return np.array([
        mean_absolute_value(filtered[end - WINDOW:end])
        for end in range(WINDOW, len(filtered) + 1, HOP)
    ], dtype=np.float64)


def mains_fraction(samples, rate_hz, mains_hz):
    """Share of in-band power sitting on a mains frequency and its harmonics."""
    values = np.asarray(samples, dtype=np.float64)
    values = (values - values.mean()) * np.hanning(len(values))
    freqs = np.fft.rfftfreq(len(values), 1.0 / rate_hz)
    power = np.abs(np.fft.rfft(values)) ** 2
    in_band = power[(freqs >= 20.0) & (freqs <= 450.0)].sum()
    if in_band <= 0.0:
        return 0.0
    hum = 0.0
    for harmonic in range(1, MAINS_HARMONICS + 1):
        centre = mains_hz * harmonic
        if centre > 450.0:
            break
        window = (freqs >= centre - MAINS_HALF_WIDTH_HZ) & (
            freqs <= centre + MAINS_HALF_WIDTH_HZ)
        hum += power[window].sum()
    return float(hum / in_band)


def channel_quality(columns, envelopes, rate_hz=2000.0, sections=None):
    """Flag channels whose correlations cannot mean anything.

    A dead channel correlates with nothing, which reads as "usefully
    distinct" — the exact way a placement verdict can come out green on data
    that does not support one.
    """
    strongest = max(float(row.max()) for row in envelopes) or 1.0
    problems = {}
    for channel, samples in enumerate(columns):
        clipped = np.count_nonzero((samples <= 0) | (samples >= 4095)) / len(samples)
        loudness = float(envelopes[channel].max()) / strongest
        measured = samples if sections is None else filter_fixed(
            sections, np.clip(samples, -32768, 32767).astype(np.int16))
        hum = max(mains_fraction(measured, rate_hz, 50.0),
                  mains_fraction(measured, rate_hz, 60.0))
        if clipped > MAX_CLIPPED_FRACTION:
            problems[channel] = f"clipped ({100 * clipped:.2f}% at the rails)"
        elif hum > MAX_MAINS_FRACTION:
            problems[channel] = (f"still mains-dominated after the notches "
                                 f"({100 * hum:.0f}% of in-band power)")
        elif loudness < MIN_AMPLITUDE_RATIO:
            problems[channel] = (f"near its noise floor (peak MAV is "
                                 f"{100 * loudness:.0f}% of the loudest channel)")
    return problems


def describe_channels(info, columns, wear_counts, total_frames):
    print("=" * 66)
    print("1. Per channel")
    print("=" * 66)
    for channel, samples in enumerate(columns):
        attached = wear_counts.get(channel, 0)
        share = 100.0 * attached / total_frames if total_frames else 0.0
        print(f"  ch{channel}: raw min={samples.min()} max={samples.max()} "
              f"mean={samples.mean():.0f} sd={samples.std():.1f}")
        print(f"       electrode attached for {attached}/{total_frames} frames "
              f"({share:.1f}%)")
        # Saturation is invisible downstream: a clipped peak still produces a
        # perfectly reasonable-looking MAV, just a wrong one. The analog gain
        # is fixed on the module, so the fix is a looser electrode or a
        # gentler contraction, not a software change.
        raw_hum = 100.0 * max(mains_fraction(samples, info.sample_rate_hz, 50.0),
                              mains_fraction(samples, info.sample_rate_hz, 60.0))
        note = "  (removed by the notches)" if raw_hum > 30 else ""
        print(f"       mains before filtering: {raw_hum:.0f}% of in-band power"
              f"{note}")
        clipped = int(np.count_nonzero((samples <= 0) | (samples >= 4095)))
        clipped_share = 100.0 * clipped / len(samples)
        if clipped_share > 0.1:
            print(f"       FAIL: {clipped} samples ({clipped_share:.2f}%) hit the "
                  "ADC rails; amplitudes are wrong wherever it clipped")
        if samples.std() < 1.0:
            print("       FAIL: essentially constant, this channel has no signal")
        elif share < 99.0:
            print("       WARN: contact was lost during the recording")


def report_correlation(envelopes, problems):
    print()
    print("=" * 66)
    print("2. Envelope correlation — the placement verdict")
    print("=" * 66)
    matrix = np.corrcoef(envelopes)
    count = len(envelopes)

    print("      " + "".join(f"    ch{index}" for index in range(count)))
    for row in range(count):
        cells = "".join(f"  {matrix[row][column]:+.2f}" for column in range(count))
        print(f"   ch{row}{cells}")

    worst = 0.0
    print()
    for row in range(count):
        for column in range(row + 1, count):
            value = abs(matrix[row][column])
            worst = max(worst, value)
            if value > REDUNDANT_ABOVE:
                mark = "REDUNDANT — same muscle group"
            elif value < DISTINCT_BELOW:
                mark = "OK — usefully distinct"
            else:
                mark = "marginal"
            print(f"   ch{row} vs ch{column}: |r| = {value:.2f}   {mark}")

    print()
    if problems:
        print("   VERDICT WITHHELD. These channels cannot support one:")
        for channel, reason in sorted(problems.items()):
            print(f"     ch{channel}: {reason}")
        print("   A clipped channel reports wrong amplitudes, a silent one")
        print("   correlates with nothing, and a mains-dominated one measures")
        print("   the room -- all three look identical to a good placement")
        print("   from amplitude alone. Fix the channels, then re-record.")
        return matrix
    if worst > REDUNDANT_ABOVE:
        print("   VERDICT: at least one pair is redundant. Move those two bands")
        print("   further apart around the forearm circumference, not along it.")
    elif worst < DISTINCT_BELOW:
        print("   VERDICT: all pairs are distinct. This placement carries three")
        print("   channels' worth of information.")
    else:
        print("   VERDICT: marginal. Usable, but separating the closest pair")
        print("   further would likely help class separation.")
    return matrix


def report_segments(envelopes, info):
    print()
    print("=" * 66)
    print(f"3. MAV over {SEGMENTS} segments — gestures should change the pattern")
    print("=" * 66)
    print("   Different gestures should give different *shapes* across channels,")
    print("   not just all three rising together.")
    print()
    windows = envelopes.shape[1]
    size = max(1, windows // SEGMENTS)
    window_sec = HOP / info.sample_rate_hz
    offset_sec = WARMUP_SECONDS + WINDOW / info.sample_rate_hz
    print("     time   " + "".join(f"    ch{i}" for i in range(len(envelopes)))
          + "   dominant")
    for start in range(0, windows, size):
        block = envelopes[:, start:start + size]
        if not block.size:
            continue
        means = block.mean(axis=1)
        dominant = "-" if means.max() < 5 else f"ch{int(means.argmax())}"
        cells = "".join(f"  {value:6.0f}" for value in means)
        print(f"   {offset_sec + start * window_sec:5.1f}s {cells}   {dominant}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="a .bin written by emg_record.py")
    arguments = parser.parse_args()

    info, columns, stats, wear_counts = load_session(arguments.session)
    total_frames = len(columns[0]) if columns else 0

    print(f"  {info.channel_count} channels, {info.sample_rate_hz} Hz, "
          f"{total_frames} frames ({total_frames / info.sample_rate_hz:.1f} s)")
    print(f"  packets accepted={stats.accepted} lost={stats.lost} "
          f"malformed={stats.malformed}")
    if stats.lost:
        print("  WARN: lost packets leave gaps; the channels are concatenated")
        print("        across them, which smears the correlation slightly.")
    if total_frames < WINDOW + HOP:
        raise SystemExit(f"need at least {WINDOW + HOP} frames to form windows")
    print()

    describe_channels(info, columns, wear_counts, total_frames)

    sections = to_fixed(design_emg_filter(rate_hz=float(info.sample_rate_hz)))
    envelopes = np.vstack([
        mav_series(samples, sections, float(info.sample_rate_hz))
        for samples in columns
    ])

    report_correlation(envelopes,
                       channel_quality(columns, envelopes,
                                       float(info.sample_rate_hz), sections))
    report_segments(envelopes, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
