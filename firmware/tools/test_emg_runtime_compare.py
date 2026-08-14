"""Focused tests for the absolute-grid MCU/host runtime comparator."""

import struct

import numpy as np

from emg_protocol import TYPE_INFO, TYPE_INTENT, TYPE_RAW
from emg_runtime_compare import (
    RuntimeEvent,
    aligned_frame_ends,
    analyze_runtime,
    compare_event_streams,
    load_deployed_model,
    load_runtime_recording,
    replay_host,
)
from test_emg_protocol import build


CHANNELS = 3
FRAMES_PER_PACKET = 32
SAMPLE_RATE = 2000
FRAME_PERIOD_US = 500
ALL_ATTACHED = 0b111


def _info_packet():
    payload = struct.pack(
        "<HHBBBB", 0x0100, SAMPLE_RATE, CHANNELS, 12, FRAMES_PER_PACKET, 0
    )
    return build(TYPE_INFO, 0, 0, payload)


def _raw_packet(sequence, start_frame, value=2048):
    values = [value] * (FRAMES_PER_PACKET * CHANNELS)
    payload = bytes((ALL_ATTACHED, 0)) + struct.pack(
        f"<{len(values)}H", *values
    )
    return build(
        TYPE_RAW,
        sequence,
        start_frame * FRAME_PERIOD_US,
        payload,
    )


def _intent_packet(sequence, end_frame, command=0):
    payload = struct.pack("<BBBbHH", command, 17, 255, 0, 0, 0)
    return build(
        TYPE_INTENT,
        sequence,
        end_frame * FRAME_PERIOD_US,
        payload,
    )


def _runtime_stream(*, first_frame=12_352, raw_packets=20,
                    event_command=None):
    stream = bytearray(_info_packet())
    for index in range(raw_packets):
        stream += _raw_packet(index, first_frame + index * FRAMES_PER_PACKET)
    ends = aligned_frame_ends(first_frame, raw_packets * FRAMES_PER_PACKET)
    for sequence, end in enumerate(ends):
        command = event_command if sequence == 0 and event_command is not None else 0
        stream += _intent_packet(sequence, int(end), command)
    return bytes(stream), ends


def test_absolute_grid_skips_local_400_when_capture_starts_at_offset_52():
    ends = aligned_frame_ends(12_352, 640)

    assert ends.tolist() == [12_800, 12_900]
    assert int(ends[0] - 12_352) == 448


def test_loads_generated_deployment_header_not_a_refitted_model():
    model = load_deployed_model()

    assert model.labels == ("REST", "NEXT_TARGET", "CONFIRM", "ABORT")
    assert model.fraction_bits == 18
    assert model.weights.shape == (4, 12)
    assert model.intercept.tolist() == [2216928, -6175264, -3904388, -5619152]


def test_zero_signal_runtime_replays_on_exact_absolute_timestamps(tmp_path):
    wire, expected_ends = _runtime_stream()
    path = tmp_path / "runtime.bin"
    path.write_bytes(wire)

    recording = load_runtime_recording(path)
    replay = replay_host(recording)

    assert recording.first_frame == 12_352
    assert replay.frame_ends.tolist() == expected_ends.tolist()
    assert replay.events == ()
    assert np.all(replay.valid)


def test_exact_event_match_passes():
    event = RuntimeEvent(1_500_000, "CONFIRM")

    comparison = compare_event_streams((event,), (event,))

    assert comparison.passed
    assert comparison.matched == (event,)


def test_differences_are_classified_without_misleading_order_pairing():
    firmware = (
        RuntimeEvent(1_000_000, "NEXT_TARGET"),
        RuntimeEvent(2_000_000, "CONFIRM"),
    )
    host = (
        RuntimeEvent(1_000_000, "ABORT"),
        RuntimeEvent(3_000_000, "CONFIRM"),
    )

    comparison = compare_event_streams(firmware, host)

    assert not comparison.passed
    assert comparison.command_mismatches == ((firmware[0], host[0]),)
    assert comparison.firmware_only == (firmware[1],)
    assert comparison.host_only == (host[1],)


def test_comparison_warmup_does_not_hide_ordinary_activity_event(tmp_path):
    wire, _ends = _runtime_stream(raw_packets=220, event_command=1)
    path = tmp_path / "ordinary.bin"
    path.write_bytes(wire)

    report = analyze_runtime(
        path,
        warmup_seconds=3.0,
        expect_no_events=True,
    )

    # The deliberately injected event is inside the host-history warm-up. It
    # is excluded from reproducibility comparison but still fails the safety
    # claim over the complete ordinary-activity recording.
    assert report.comparison.passed
    assert report.no_event_expectation_passed is False
    assert not report.passed


def test_no_event_mode_passes_a_clean_recording(tmp_path):
    wire, _ends = _runtime_stream(raw_packets=220)
    path = tmp_path / "ordinary.bin"
    path.write_bytes(wire)

    report = analyze_runtime(
        path,
        warmup_seconds=3.0,
        expect_no_events=True,
    )

    assert report.comparison.passed
    assert report.no_event_expectation_passed is True
    assert report.passed
