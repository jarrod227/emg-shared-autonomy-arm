#!/usr/bin/env python3
"""Compare deployed MCU EMG events with a host replay of the same RAW log.

The recorder stores RAW and INTENT packets in one wire log.  This tool rebuilds
the firmware feature grid from the RAW packet timestamps, replays the deployed
Q29 filter, integer features, generated Q18 model, activation stage, and frozen
event gate, then compares non-REST events at their exact device timestamps.

The absolute grid matters.  A capture usually begins after the MCU has already
processed many frames, so starting host windows at local frame 400 computes a
different set of windows whenever the first device frame is not a multiple of
the 100-frame feature hop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import re

import numpy as np

from emg_activation_ref import ActivationGate
from emg_event_gate_replay import EventGate, VALIDATED_GATE
from emg_features_ref import HOP, WINDOW, compute_features
from emg_filter_ref import design_emg_filter, filter_fixed, to_fixed
from emg_protocol import (
    COMMAND_NAMES,
    TYPE_INFO,
    TYPE_INTENT,
    TYPE_RAW,
    Info,
    PacketParser,
    decode_info,
    decode_intent,
    decode_raw,
)
from emg_train_lda import (
    CHANNEL_COUNT,
    FEATURE_NAMES,
    LABELS,
    QuantizedLDAModel,
    ZERO_CROSSING_THRESHOLD,
)


DEFAULT_MODEL_HEADER = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "emg_classifier_model.h"
)
DEFAULT_WARMUP_SECONDS = 3.0
UINT32_MODULO = 1 << 32


@dataclass(frozen=True)
class RuntimeEvent:
    """One discrete event at the feature window's device timestamp."""

    timestamp_us: int
    command: str


@dataclass(frozen=True)
class RuntimeRecording:
    """Validated RAW samples and MCU INTENT output from one wire log."""

    path: pathlib.Path
    info: Info
    samples: np.ndarray
    frame_valid: np.ndarray
    first_frame: int
    frame_period_us: int
    intent_timestamps_us: tuple[int, ...]
    firmware_events: tuple[RuntimeEvent, ...]
    parser_packets: int

    @property
    def duration_seconds(self):
        return len(self.samples) * self.frame_period_us / 1e6

    @property
    def first_timestamp_us(self):
        return self.first_frame * self.frame_period_us


@dataclass(frozen=True)
class HostReplay:
    """Host events plus the absolute feature grid used to produce them."""

    frame_ends: np.ndarray
    valid: np.ndarray
    events: tuple[RuntimeEvent, ...]


@dataclass(frozen=True)
class EventComparison:
    """Exact-timestamp comparison after the declared history warm-up."""

    cutoff_timestamp_us: int
    matched: tuple[RuntimeEvent, ...]
    command_mismatches: tuple[tuple[RuntimeEvent, RuntimeEvent], ...]
    firmware_only: tuple[RuntimeEvent, ...]
    host_only: tuple[RuntimeEvent, ...]

    @property
    def passed(self):
        return not (
            self.command_mismatches or self.firmware_only or self.host_only
        )


@dataclass(frozen=True)
class RuntimeReport:
    recording: RuntimeRecording
    replay: HostReplay
    comparison: EventComparison
    warmup_seconds: float
    expect_no_events: bool

    @property
    def no_event_expectation_passed(self):
        if not self.expect_no_events:
            return None
        # The warm-up exclusion belongs only to host/MCU reproducibility.
        # Ordinary-activity safety counts every MCU event in the whole log.
        return len(self.recording.firmware_events) == 0

    @property
    def passed(self):
        return self.comparison.passed and self.no_event_expectation_passed is not False


def _define_integer(text, name):
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+(-?\d+)[uUlL]*\b",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"model header is missing {name}")
    return int(match.group(1))


def _initializer_integers(text, name):
    match = re.search(
        rf"\b{re.escape(name)}\b.*?=\s*\{{(.*?)\}}\s*;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"model header is missing initializer {name}")
    return [int(value) for value in re.findall(r"-?\d+", match.group(1))]


def load_deployed_model(path=DEFAULT_MODEL_HEADER):
    """Load the generated C header, never refit a model during comparison."""
    header = pathlib.Path(path)
    try:
        text = header.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read deployed model header {header}: {error}") from error

    class_count = _define_integer(text, "EMG_CLASSIFIER_MODEL_CLASS_COUNT")
    feature_count = _define_integer(text, "EMG_CLASSIFIER_MODEL_FEATURE_COUNT")
    fraction_bits = _define_integer(text, "EMG_CLASSIFIER_MODEL_FRACTION_BITS")
    if class_count != len(LABELS) or feature_count != len(FEATURE_NAMES):
        raise ValueError(
            "deployed model geometry disagrees with the host feature contract"
        )

    command_ids = tuple(
        _define_integer(text, f"EMG_CLASSIFIER_MODEL_CLASS_{index}_COMMAND")
        for index in range(class_count)
    )
    try:
        labels = tuple(COMMAND_NAMES[command] for command in command_ids)
    except KeyError as error:
        raise ValueError(f"deployed model has unknown command {error.args[0]}") from error
    if labels != LABELS:
        raise ValueError(
            f"deployed model command order {labels} disagrees with host {LABELS}"
        )

    weights = _initializer_integers(text, "emg_classifier_model_weights")
    intercept = _initializer_integers(text, "emg_classifier_model_intercept")
    if len(weights) != class_count * feature_count:
        raise ValueError("deployed model weight count is malformed")
    if len(intercept) != class_count:
        raise ValueError("deployed model intercept count is malformed")
    return QuantizedLDAModel(
        labels=labels,
        feature_names=FEATURE_NAMES,
        fraction_bits=fraction_bits,
        weights=np.asarray(weights, dtype=np.int64).reshape(
            class_count, feature_count
        ),
        intercept=np.asarray(intercept, dtype=np.int64),
    )


def _parser_errors(stats):
    return {
        "lost": stats.lost,
        "malformed": stats.malformed,
        "duplicated": stats.duplicated,
        "time_reversed": stats.time_reversed,
        "discarded_bytes": stats.discarded_bytes,
    }


def load_runtime_recording(path):
    """Fail closed on any transport or timestamp gap before replaying."""
    source = pathlib.Path(path)
    try:
        wire = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read runtime log {source}: {error}") from error

    parser = PacketParser()
    packets = parser.feed(wire)
    errors = _parser_errors(parser.stats)
    if any(errors.values()):
        raise ValueError(f"runtime log has parser errors {errors}")

    infos = [decode_info(packet.payload) for packet in packets if packet.type == TYPE_INFO]
    if not infos:
        raise ValueError("runtime log has no INFO packet")
    info = infos[0]
    if any(item != info for item in infos[1:]):
        raise ValueError("runtime INFO packets disagree")
    if info.channel_count != CHANNEL_COUNT:
        raise ValueError(f"expected {CHANNEL_COUNT} channels, got {info.channel_count}")
    if 1_000_000 % info.sample_rate_hz:
        raise ValueError("sample period is not an integer number of microseconds")
    frame_period_us = 1_000_000 // info.sample_rate_hz

    raw_packets = [packet for packet in packets if packet.type == TYPE_RAW]
    if not raw_packets:
        raise ValueError("runtime log has no RAW packet")
    frames = []
    frame_valid = []
    previous_timestamp = None
    previous_frames = None
    for packet in raw_packets:
        if packet.timestamp_us % frame_period_us:
            raise ValueError("RAW timestamp is not on the device frame grid")
        if previous_timestamp is not None:
            if packet.timestamp_us < previous_timestamp:
                raise ValueError("timestamp wrap is unsupported; reconnect before capture")
            expected = previous_timestamp + previous_frames * frame_period_us
            if packet.timestamp_us != expected:
                raise ValueError(
                    f"RAW timestamp gap: expected {expected}, got {packet.timestamp_us}"
                )
        block = decode_raw(packet.payload, info.channel_count)
        if len(block.frames) != info.frames_per_raw_packet:
            raise ValueError("RAW packet frame count disagrees with INFO")
        frames.extend(block.frames)
        frame_valid.extend([block.all_attached] * len(block.frames))
        previous_timestamp = packet.timestamp_us
        previous_frames = len(block.frames)

    samples = np.asarray(frames, dtype=np.int64)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError("runtime RAW frame matrix is malformed")

    intent_packets = [packet for packet in packets if packet.type == TYPE_INTENT]
    if not intent_packets:
        raise ValueError("runtime log has no INTENT packet")
    intent_timestamps = []
    firmware_events = []
    previous_intent_timestamp = None
    expected_hop_us = HOP * frame_period_us
    for packet in intent_packets:
        if packet.timestamp_us % frame_period_us:
            raise ValueError("INTENT timestamp is not on the device frame grid")
        if previous_intent_timestamp is not None:
            if packet.timestamp_us < previous_intent_timestamp:
                raise ValueError("timestamp wrap is unsupported; reconnect before capture")
            if packet.timestamp_us - previous_intent_timestamp != expected_hop_us:
                raise ValueError("INTENT packets are not on a continuous feature-hop grid")
        intent = decode_intent(packet.payload)
        if intent.command not in COMMAND_NAMES:
            raise ValueError(f"unknown MCU command {intent.command}")
        intent_timestamps.append(packet.timestamp_us)
        if intent.command != 0:
            firmware_events.append(RuntimeEvent(
                timestamp_us=packet.timestamp_us,
                command=intent.command_name,
            ))
        previous_intent_timestamp = packet.timestamp_us

    first_timestamp_us = raw_packets[0].timestamp_us
    return RuntimeRecording(
        path=source,
        info=info,
        samples=samples,
        frame_valid=np.asarray(frame_valid, dtype=bool),
        first_frame=first_timestamp_us // frame_period_us,
        frame_period_us=frame_period_us,
        intent_timestamps_us=tuple(intent_timestamps),
        firmware_events=tuple(firmware_events),
        parser_packets=parser.stats.accepted,
    )


def aligned_frame_ends(first_frame, frame_count, *, window=WINDOW, hop=HOP):
    """Return absolute feature ends on the MCU's reset-anchored hop grid."""
    if first_frame < 0 or frame_count < 0:
        raise ValueError("frame positions must be non-negative")
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    earliest = first_frame + window
    first_end = ((earliest + hop - 1) // hop) * hop
    last_end = first_frame + frame_count
    if first_end > last_end:
        return np.empty(0, dtype=np.int64)
    return np.arange(first_end, last_end + 1, hop, dtype=np.int64)


def replay_host(recording, model=None):
    """Replay the deployed decision path on absolute device-frame windows."""
    model = load_deployed_model() if model is None else model
    sections = to_fixed(
        design_emg_filter(rate_hz=float(recording.info.sample_rate_hz))
    )
    filtered = [
        filter_fixed(
            sections,
            np.clip(recording.samples[:, channel], -32768, 32767).astype(np.int16),
        )
        for channel in range(CHANNEL_COUNT)
    ]
    frame_ends = aligned_frame_ends(recording.first_frame, len(recording.samples))
    if not len(frame_ends):
        raise ValueError("runtime log is shorter than one aligned feature window")

    rows = []
    local_ends = frame_ends - recording.first_frame
    for local_end in local_ends:
        begin = int(local_end) - WINDOW
        row = []
        for column in filtered:
            row.extend(compute_features(
                column[begin:int(local_end)], ZERO_CROSSING_THRESHOLD
            ))
        rows.append(row)
    features = np.asarray(rows, dtype=np.int64)
    predictions = model.predict(features)

    invalid_prefix = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(~recording.frame_valid, dtype=np.int64),
    ))
    valid = (
        invalid_prefix[local_ends] - invalid_prefix[local_ends - WINDOW]
    ) == 0

    activation = ActivationGate()
    gate = EventGate(VALIDATED_GATE)
    events = []
    mav_columns = (0, 4, 8)
    for index, prediction in enumerate(predictions):
        window_valid = bool(valid[index])
        total_mav = sum(int(features[index, column]) for column in mav_columns)
        decision = activation.apply(
            str(prediction), valid=window_valid, total_mav=total_mav
        )
        event = gate.push(decision, valid=window_valid)
        if event is not None:
            timestamp_us = int(frame_ends[index]) * recording.frame_period_us
            if timestamp_us >= UINT32_MODULO:
                raise ValueError("host replay timestamp crosses the uint32 wire wrap")
            events.append(RuntimeEvent(timestamp_us, event))
    return HostReplay(frame_ends=frame_ends, valid=valid, events=tuple(events))


def compare_event_streams(firmware_events, host_events, cutoff_timestamp_us=0):
    """Classify every post-cutoff event difference without order pairing."""
    firmware = {
        event.timestamp_us: event
        for event in firmware_events
        if event.timestamp_us >= cutoff_timestamp_us
    }
    host = {
        event.timestamp_us: event
        for event in host_events
        if event.timestamp_us >= cutoff_timestamp_us
    }
    if len(firmware) != sum(
        event.timestamp_us >= cutoff_timestamp_us for event in firmware_events
    ):
        raise ValueError("firmware emitted more than one event at one timestamp")
    if len(host) != sum(event.timestamp_us >= cutoff_timestamp_us for event in host_events):
        raise ValueError("host emitted more than one event at one timestamp")

    matched = []
    command_mismatches = []
    firmware_only = []
    host_only = []
    for timestamp in sorted(set(firmware) | set(host)):
        mcu = firmware.get(timestamp)
        replay = host.get(timestamp)
        if mcu is None:
            host_only.append(replay)
        elif replay is None:
            firmware_only.append(mcu)
        elif mcu.command != replay.command:
            command_mismatches.append((mcu, replay))
        else:
            matched.append(mcu)
    return EventComparison(
        cutoff_timestamp_us=int(cutoff_timestamp_us),
        matched=tuple(matched),
        command_mismatches=tuple(command_mismatches),
        firmware_only=tuple(firmware_only),
        host_only=tuple(host_only),
    )


def analyze_runtime(path, *, warmup_seconds=DEFAULT_WARMUP_SECONDS,
                    expect_no_events=False, model_header=DEFAULT_MODEL_HEADER):
    """Load, replay, compare, and attach the explicit no-event expectation."""
    warmup = float(warmup_seconds)
    if warmup < 0:
        raise ValueError("warm-up must be non-negative")
    recording = load_runtime_recording(path)
    if warmup >= recording.duration_seconds:
        raise ValueError("recording must be longer than the comparison warm-up")
    replay = replay_host(recording, load_deployed_model(model_header))
    cutoff = recording.first_timestamp_us + round(warmup * 1e6)
    comparison = compare_event_streams(
        recording.firmware_events, replay.events, cutoff
    )
    return RuntimeReport(
        recording=recording,
        replay=replay,
        comparison=comparison,
        warmup_seconds=warmup,
        expect_no_events=bool(expect_no_events),
    )


def _format_event(event):
    return f"{event.timestamp_us / 1e6:9.3f}s  {event.command}"


def print_report(report):
    recording = report.recording
    comparison = report.comparison
    intent_rate = len(recording.intent_timestamps_us) / recording.duration_seconds
    print("=" * 66)
    print(f"  log                       {recording.path}")
    print(f"  RAW frames                {len(recording.samples)}")
    print(f"  duration_sec              {recording.duration_seconds:.3f}")
    print(f"  parser_packets            {recording.parser_packets}")
    print(f"  INTENT packets/rate       {len(recording.intent_timestamps_us)} / {intent_rate:.2f} Hz")
    print(f"  first_device_frame        {recording.first_frame}")
    print(f"  first_host_feature_frame  {int(report.replay.frame_ends[0])}")
    print(f"  valid_host_hops           {int(np.count_nonzero(report.replay.valid))}/{len(report.replay.valid)}")
    print(f"  comparison_warmup_sec     {report.warmup_seconds:.3f}")
    print(f"  MCU events (whole log)    {len(recording.firmware_events)}")
    print(f"  host events (whole log)   {len(report.replay.events)}")
    print(f"  exact matches             {len(comparison.matched)}")
    print(f"  command mismatches        {len(comparison.command_mismatches)}")
    print(f"  MCU-only events           {len(comparison.firmware_only)}")
    print(f"  host-only events          {len(comparison.host_only)}")
    if report.expect_no_events:
        verdict = "PASS" if report.no_event_expectation_passed else "FAIL"
        print(f"  ordinary-activity events  {len(recording.firmware_events)} -> {verdict}")
    print("-" * 66)
    for event in comparison.firmware_only:
        print(f"  MCU only     {_format_event(event)}")
    for event in comparison.host_only:
        print(f"  host only    {_format_event(event)}")
    for mcu, host in comparison.command_mismatches:
        print(
            f"  mismatch     {mcu.timestamp_us / 1e6:9.3f}s  "
            f"MCU={mcu.command} host={host.command}"
        )
    print("=" * 66)
    print("  PASS" if report.passed else "  FAIL")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=pathlib.Path)
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_WARMUP_SECONDS,
        help="exclude this measured filter-history settling period only from "
             "MCU/host comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--expect-no-events",
        action="store_true",
        help="declare an ordinary-activity recording; any MCU event fails",
    )
    parser.add_argument(
        "--model-header",
        type=pathlib.Path,
        default=DEFAULT_MODEL_HEADER,
        help="generated deployed Q18 C model header",
    )
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    try:
        report = analyze_runtime(
            arguments.recording,
            warmup_seconds=arguments.warmup_seconds,
            expect_no_events=arguments.expect_no_events,
            model_header=arguments.model_header,
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        return 2
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
