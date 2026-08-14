"""Tests for the sEMG recorder.

Drives everything from synthetic byte streams, so the whole tool is covered
without a board attached. Packets are built by the same helper the decoder
tests use, which is itself checked against the C encoder's fixture -- so these
streams are the shape real firmware will produce.

    python3 -m pytest firmware/tools/test_emg_record.py
"""

import json
import struct

import pytest

from emg_record import Recording, replay, write_sidecar
from emg_protocol import TYPE_INFO, TYPE_INTENT, TYPE_RAW
from test_emg_protocol import build

CHANNELS = 3
FRAMES_PER_PACKET = 32
SAMPLE_RATE = 2000
PACKET_PERIOD_US = round(1e6 * FRAMES_PER_PACKET / SAMPLE_RATE)  # 16000 us


def info_packet(sequence=0, timestamp_us=0):
    payload = struct.pack(
        "<HHBBBB", 0x0102, SAMPLE_RATE, CHANNELS, 12, FRAMES_PER_PACKET, 0
    )
    return build(TYPE_INFO, sequence, timestamp_us, payload)


ALL_ATTACHED = (1 << CHANNELS) - 1


def raw_packet(sequence, timestamp_us, value=2048, wear=ALL_ATTACHED,
               frames=FRAMES_PER_PACKET):
    samples = [value] * (frames * CHANNELS)
    payload = bytes([wear, 0]) + struct.pack(f"<{len(samples)}H", *samples)
    return build(TYPE_RAW, sequence, timestamp_us, payload)


def session(packet_count=10, *, start_sequence=0, skip=()):
    """A well-formed INFO packet followed by evenly spaced RAW packets."""
    stream = bytearray(info_packet())
    for index in range(packet_count):
        sequence = start_sequence + index
        if sequence in skip:
            continue
        stream += raw_packet(sequence, index * PACKET_PERIOD_US)
    return bytes(stream)


def test_counts_packets_by_type():
    recording = Recording()
    recording.feed(info_packet() + raw_packet(0, 0)
                   + build(TYPE_INTENT, 0, 0, b"\x00" * 8))

    summary = recording.summary(1.0)

    assert summary["packets"] == {"info": 1, "raw": 1, "intent": 1}
    assert summary["parser"]["accepted"] == 3
    assert summary["parser"]["malformed"] == 0


def test_reports_info_fields():
    recording = Recording()
    recording.feed(info_packet())

    assert recording.summary(1.0)["info"] == {
        "firmware_version": 0x0102,
        "sample_rate_hz": SAMPLE_RATE,
        "channel_count": CHANNELS,
        "adc_bits": 12,
        "frames_per_raw_packet": FRAMES_PER_PACKET,
    }


def test_counts_frames_using_the_declared_channel_count():
    recording = Recording()
    recording.feed(session(packet_count=5))

    assert recording.summary(1.0)["frames"] == 5 * FRAMES_PER_PACKET


def test_raw_before_info_is_not_counted_as_frames():
    # Without INFO the channel count is unknown, so frames cannot be derived.
    # Guessing one would silently corrupt every rate in the summary.
    recording = Recording()
    recording.feed(raw_packet(0, 0) + raw_packet(1, PACKET_PERIOD_US))

    summary = recording.summary(1.0)

    assert summary["packets"]["raw"] == 2
    assert "frames" not in summary


def test_derives_the_sample_rate_from_device_timestamps():
    recording = Recording()
    recording.feed(session(packet_count=11))

    # Ten inter-packet gaps of 16 ms each carry 10 * 32 frames.
    assert recording.device_sample_rate_hz() == pytest.approx(SAMPLE_RATE, rel=1e-6)


def test_device_rate_counts_real_frames_not_the_declared_batch_size():
    """A short final packet must not skew the rate.

    Deriving frames from `frames_per_raw_packet` would assume every packet is
    full, so any firmware that ships a partial batch would report a rate that
    is wrong by exactly the shortfall.
    """
    recording = Recording()
    stream = bytearray(info_packet())
    for index in range(3):
        stream += raw_packet(index, index * PACKET_PERIOD_US)
    # A fourth packet holding a single frame instead of 32.
    stream += raw_packet(3, 3 * PACKET_PERIOD_US, frames=1)

    recording.feed(bytes(stream))

    assert recording.raw_frames == 3 * FRAMES_PER_PACKET + 1
    # Span covers the first three full packets only: 96 frames over 48 ms.
    assert recording.device_sample_rate_hz() == pytest.approx(SAMPLE_RATE, rel=1e-6)


def test_wall_and_device_rates_are_reported_separately():
    recording = Recording()
    recording.feed(session(packet_count=11))

    # A host that took twice as long as the device timestamps say must show
    # the discrepancy rather than average it away.
    summary = recording.summary(elapsed_sec=0.32)

    assert summary["sample_rate_hz_device"] == pytest.approx(2000.0, abs=0.1)
    assert summary["sample_rate_hz_wall"] == pytest.approx(1100.0, abs=0.1)


def test_a_missing_packet_shows_up_as_loss():
    recording = Recording()
    recording.feed(session(packet_count=10, skip=(4,)))

    summary = recording.summary(1.0)

    assert summary["parser"]["lost"] == 1
    assert summary["packets"]["raw"] == 9


def test_leading_junk_is_discarded_without_being_called_malformed():
    recording = Recording()
    recording.feed(b"\x11\x22\x33" + info_packet())

    summary = recording.summary(1.0)

    assert summary["parser"]["accepted"] == 1
    assert summary["parser"]["malformed"] == 0
    assert summary["parser"]["discarded_bytes"] == 3


def test_a_wrong_sized_info_payload_is_counted_not_raised():
    recording = Recording()
    # Passes CRC but disagrees with the spec on payload size.
    recording.feed(build(TYPE_INFO, 0, 0, b"\x00" * 4))

    summary = recording.summary(1.0)

    assert "info" not in summary
    assert summary["parser"]["reasons"]["info_payload"] == 1


def test_replay_reproduces_the_recording(tmp_path):
    stream = session(packet_count=8, skip=(3,))
    log = tmp_path / "session.bin"
    log.write_bytes(stream)

    direct = Recording()
    direct.feed(stream)
    replayed = replay(log)

    assert replayed.summary(0.0) == direct.summary(0.0)
    assert replayed.summary(0.0)["parser"]["lost"] == 1


def test_replay_survives_a_log_split_across_read_chunks(tmp_path, monkeypatch):
    import emg_record

    # Force many tiny reads so packets straddle chunk boundaries.
    monkeypatch.setattr(emg_record, "_READ_CHUNK", 7)
    stream = session(packet_count=6)
    log = tmp_path / "session.bin"
    log.write_bytes(stream)

    assert replay(log).summary(0.0)["frames"] == 6 * FRAMES_PER_PACKET


def test_sidecar_is_written_next_to_the_log(tmp_path):
    recording = Recording()
    recording.feed(session(packet_count=3))
    summary = recording.summary(1.0)

    sidecar = write_sidecar(tmp_path / "session.bin", summary)

    assert sidecar.name == "session.json"
    assert json.loads(sidecar.read_text()) == summary


def test_counts_only_frames_with_every_electrode_attached():
    recording = Recording()
    stream = bytearray(info_packet())
    stream += raw_packet(0, 0)
    stream += raw_packet(1, PACKET_PERIOD_US, wear=0b101)   # channel 1 off
    stream += raw_packet(2, 2 * PACKET_PERIOD_US)

    recording.feed(bytes(stream))
    summary = recording.summary(1.0)

    assert summary["frames"] == 3 * FRAMES_PER_PACKET
    assert summary["frames_all_attached"] == 2 * FRAMES_PER_PACKET
    assert summary["usable_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    # Only channel 1 lost contact, and only for that packet's frames.
    assert summary["frames_detached_by_channel"] == {"1": FRAMES_PER_PACKET}


def test_a_fully_detached_recording_reports_nothing_usable():
    recording = Recording()
    stream = info_packet() + raw_packet(0, 0, wear=0b000)

    recording.feed(stream)
    summary = recording.summary(1.0)

    assert summary["frames_all_attached"] == 0
    assert summary["usable_fraction"] == 0.0
    assert set(summary["frames_detached_by_channel"]) == {"0", "1", "2"}
