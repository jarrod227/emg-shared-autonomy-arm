"""Tests for the session analyser.

The verdict this tool gives decides where electrodes go, so it is checked
against sessions built with a known answer: two channels sharing an envelope
must come back redundant, and independent ones must not.

    python3 -m pytest firmware/tools/test_emg_analyze.py
"""

import struct

import numpy as np
import pytest

from emg_analyze import DISTINCT_BELOW, REDUNDANT_ABOVE, load_session, mav_series
from emg_features_ref import HOP, WINDOW
from emg_filter_ref import design_bandpass, to_fixed
from emg_protocol import TYPE_INFO, TYPE_RAW
from test_emg_protocol import build

RATE = 2000
CHANNELS = 3
FRAMES_PER_PACKET = 32
SECONDS = 8


def envelope(centres, samples, width=0.5):
    times = np.arange(samples) / RATE
    shape = np.full(samples, 0.05)
    for centre in centres:
        shape += np.exp(-((times - centre) ** 2) / (2 * width ** 2))
    return shape


def synthetic_channel(shape, rng, gain=500.0):
    """Noise modulated by an envelope, biased to mid-rail like the real board.

    sEMG is an interference pattern, so the carrier is noise and only the
    envelope carries the information — which is exactly why the analyser
    correlates envelopes rather than samples.
    """
    values = 2048 + gain * shape * rng.normal(0.0, 1.0, shape.size)
    return np.clip(values, 0, 4095).astype(np.uint16)


def write_session(path, channels, wear=0b111):
    total = len(channels[0])
    stream = bytearray(build(
        TYPE_INFO, 0, 0,
        struct.pack("<HHBBBB", 0x0100, RATE, CHANNELS, 12, FRAMES_PER_PACKET, 0),
    ))
    for index in range(total // FRAMES_PER_PACKET):
        frames = []
        for offset in range(FRAMES_PER_PACKET):
            position = index * FRAMES_PER_PACKET + offset
            frames.extend(int(channel[position]) for channel in channels)
        payload = bytes([wear, 0]) + struct.pack(f"<{len(frames)}H", *frames)
        stream += build(TYPE_RAW, index & 0xFFFF, index * FRAMES_PER_PACKET * 500,
                        payload)
    path.write_bytes(bytes(stream))
    return path


def envelopes_of(path):
    info, columns, _, _ = load_session(path)
    sections = to_fixed(design_bandpass(rate_hz=float(info.sample_rate_hz)))
    return np.vstack([mav_series(samples, sections) for samples in columns])


def test_shared_envelopes_are_detected_as_redundant(tmp_path):
    rng = np.random.default_rng(3)
    samples = RATE * SECONDS
    shared = envelope([1, 3, 5, 7], samples)
    other = envelope([2, 4, 6], samples)
    channels = [synthetic_channel(shared, rng),
                synthetic_channel(shared, rng),
                synthetic_channel(other, rng)]

    matrix = np.corrcoef(envelopes_of(write_session(tmp_path / "s.bin", channels)))

    # The two sharing an envelope must be flagged; neither should be
    # confused with the independent one.
    assert abs(matrix[0][1]) > REDUNDANT_ABOVE
    assert abs(matrix[0][2]) < REDUNDANT_ABOVE
    assert abs(matrix[1][2]) < REDUNDANT_ABOVE


def test_independent_envelopes_are_distinct(tmp_path):
    rng = np.random.default_rng(11)
    samples = RATE * SECONDS
    channels = [
        synthetic_channel(envelope([1, 5], samples), rng),
        synthetic_channel(envelope([3, 7], samples), rng),
        synthetic_channel(envelope([2, 6], samples), rng),
    ]

    matrix = np.corrcoef(envelopes_of(write_session(tmp_path / "s.bin", channels)))

    for row in range(CHANNELS):
        for column in range(row + 1, CHANNELS):
            assert abs(matrix[row][column]) < REDUNDANT_ABOVE


def test_sample_level_correlation_would_have_missed_it(tmp_path):
    """Why the analyser correlates envelopes and not raw samples.

    Two channels driven by the *same* envelope but independent noise are
    completely redundant for classification, yet their raw sample streams are
    uncorrelated. Correlating samples would score this placement as fine.
    """
    rng = np.random.default_rng(5)
    samples = RATE * SECONDS
    shared = envelope([1, 3, 5, 7], samples)
    channels = [synthetic_channel(shared, rng), synthetic_channel(shared, rng)]
    path = write_session(tmp_path / "s.bin", channels + [channels[0]])

    _, columns, _, _ = load_session(path)
    raw = np.corrcoef(columns[0].astype(float), columns[1].astype(float))[0][1]
    envelope_r = np.corrcoef(envelopes_of(path))[0][1]

    assert abs(raw) < DISTINCT_BELOW      # looks fine, and is wrong
    assert abs(envelope_r) > REDUNDANT_ABOVE


def test_load_session_requires_an_info_packet(tmp_path):
    # Without INFO there is no channel count, and guessing one would silently
    # interleave the channels wrongly.
    path = tmp_path / "s.bin"
    path.write_bytes(build(TYPE_RAW, 0, 0, bytes([0b111, 0]) + b"\x00" * 12))

    with pytest.raises(SystemExit):
        load_session(path)


def test_windows_line_up_with_the_firmware(tmp_path):
    rng = np.random.default_rng(7)
    samples = RATE * SECONDS
    channels = [synthetic_channel(envelope([2], samples), rng)] * CHANNELS

    series = envelopes_of(write_session(tmp_path / "s.bin", channels))

    frames = (samples // FRAMES_PER_PACKET) * FRAMES_PER_PACKET
    assert series.shape == (CHANNELS, (frames - WINDOW) // HOP + 1)
