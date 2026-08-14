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
from emg_filter_ref import design_bandpass, filter_fixed, to_fixed
from emg_protocol import TYPE_INFO, TYPE_RAW, PacketParser, decode_info, decode_raw

# Above this, two channels are telling the same story and one of them is
# not earning its place. Below 0.7 they are usefully distinct. The band
# between is a judgement call, which is why the script prints the number
# rather than only a verdict.
REDUNDANT_ABOVE = 0.90
DISTINCT_BELOW = 0.70
SEGMENTS = 10


def load_session(path):
    """Return (info, channel sample arrays, parser stats, wear counts)."""
    parser = PacketParser()
    info = None
    columns = None
    wear_counts = {}

    for packet in parser.feed(pathlib.Path(path).read_bytes()):
        if packet.type == TYPE_INFO and info is None:
            info = decode_info(packet.payload)
            columns = [[] for _ in range(info.channel_count)]
        elif packet.type == TYPE_RAW and info is not None:
            block = decode_raw(packet.payload, info.channel_count)
            for channel in range(info.channel_count):
                attached = block.channel_attached(channel)
                wear_counts[channel] = wear_counts.get(channel, 0) + (
                    len(block.frames) if attached else 0
                )
                columns[channel].extend(frame[channel] for frame in block.frames)

    if info is None:
        raise SystemExit("no INFO packet in the log; cannot interpret the samples")
    return info, [np.array(c, dtype=np.int64) for c in columns], parser.stats, wear_counts


def mav_series(samples, sections):
    """Band-pass, then MAV per feature window — what the firmware computes."""
    filtered = filter_fixed(sections, np.clip(samples, -32768, 32767).astype(np.int16))
    return np.array([
        mean_absolute_value(filtered[end - WINDOW:end])
        for end in range(WINDOW, len(filtered) + 1, HOP)
    ], dtype=np.float64)


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
        clipped = int(np.count_nonzero((samples <= 0) | (samples >= 4095)))
        clipped_share = 100.0 * clipped / len(samples)
        if clipped_share > 0.1:
            print(f"       FAIL: {clipped} samples ({clipped_share:.2f}%) hit the "
                  "ADC rails; amplitudes are wrong wherever it clipped")
        if samples.std() < 1.0:
            print("       FAIL: essentially constant, this channel has no signal")
        elif share < 99.0:
            print("       WARN: contact was lost during the recording")


def report_correlation(envelopes):
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
    print("     time   " + "".join(f"    ch{i}" for i in range(len(envelopes)))
          + "   dominant")
    for start in range(0, windows, size):
        block = envelopes[:, start:start + size]
        if not block.size:
            continue
        means = block.mean(axis=1)
        dominant = "-" if means.max() < 5 else f"ch{int(means.argmax())}"
        cells = "".join(f"  {value:6.0f}" for value in means)
        print(f"   {start * window_sec:5.1f}s {cells}   {dominant}")


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

    sections = to_fixed(design_bandpass(rate_hz=float(info.sample_rate_hz)))
    envelopes = np.vstack([mav_series(samples, sections) for samples in columns])

    report_correlation(envelopes)
    report_segments(envelopes, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
