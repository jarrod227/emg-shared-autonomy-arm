#!/usr/bin/env python3
"""Record the sEMG serial stream to a raw byte log, or replay a recording.

Records **bytes**, not decoded samples. If the decoder turns out to have a
bug, a recording of decoded values is unrecoverable, while a byte log can be
re-decoded as many times as needed. The decoded view is derived on the way
past, for the live summary and the metadata sidecar only.

    emg_record.py --seconds 60 --out session.bin
    emg_record.py --replay session.bin

Recording writes `session.bin` plus `session.json`, which holds the wall-clock
start, the INFO packet contents, the parser's error counts, and the
wall-clock sample rate. Replay re-parses
an existing log and prints the same summary, which is also how this tool is
tested without hardware attached.

RAW packets carry the per-channel wear mask alongside their samples, so the
summary can report how many frames were recorded with every electrode in
contact. Only those frames are training data — a detached electrode floats
rather than reading dead, so it looks like signal.
"""

import argparse
import datetime
import json
import pathlib
import sys
import time

from emg_protocol import (
    RAW_HEADER_SIZE,
    TYPE_INFO,
    TYPE_INTENT,
    TYPE_RAW,
    PacketParser,
    decode_info,
)

_TIMESTAMP_MODULO = 1 << 32
_TYPE_NAMES = {TYPE_INFO: "info", TYPE_RAW: "raw", TYPE_INTENT: "intent"}
_READ_CHUNK = 65536


class Recording:
    """Derives session metadata from a stream of bytes.

    Kept free of any I/O so the same object serves live recording and replay,
    and so the tests can drive it with a synthetic stream.
    """

    def __init__(self):
        self._parser = PacketParser()
        self.byte_count = 0
        self.counts = {TYPE_INFO: 0, TYPE_RAW: 0, TYPE_INTENT: 0}
        self.info = None
        self.raw_frames = 0
        self.first_raw_timestamp_us = None
        self.last_raw_timestamp_us = None
        self._last_raw_frames = 0
        self.raw_before_info = 0
        self.frames_all_attached = 0
        self.frames_detached = {}

    def feed(self, chunk):
        """Add bytes and fold every valid packet into the running summary."""
        self.byte_count += len(chunk)
        for packet in self._parser.feed(chunk):
            self.counts[packet.type] = self.counts.get(packet.type, 0) + 1
            if packet.type == TYPE_INFO:
                self._absorb_info(packet)
            elif packet.type == TYPE_RAW:
                self._absorb_raw(packet)

    def _absorb_info(self, packet):
        if self.info is not None:
            return
        try:
            self.info = decode_info(packet.payload)
        except ValueError:
            # A payload that passed CRC but is the wrong size means the
            # sender disagrees with the spec; count it rather than crash.
            self._parser.stats._note("info_payload")

    def _absorb_raw(self, packet):
        if self.info is None or self.info.channel_count <= 0:
            # Channel count is unknown until INFO arrives, so these frames
            # cannot be counted. Their timestamps must be excluded too:
            # spanning them while omitting their frames understates every
            # derived rate, which read 3% low until this was fixed.
            self.raw_before_info += 1
            return
        if self.first_raw_timestamp_us is None:
            self.first_raw_timestamp_us = packet.timestamp_us
        self.last_raw_timestamp_us = packet.timestamp_us
        channels = self.info.channel_count
        body = len(packet.payload) - RAW_HEADER_SIZE
        if body < 0:
            self._parser.stats._note("raw_payload")
            return
        frames = body // (2 * channels)
        self.raw_frames += frames
        self._last_raw_frames = frames

        # Wear state travels with the samples, so it can be attributed to
        # exactly these frames rather than to a nearby timestamp.
        wear_mask = packet.payload[0] if packet.payload else 0
        expected = (1 << channels) - 1
        if wear_mask & expected == expected:
            self.frames_all_attached += frames
        for channel in range(channels):
            if not wear_mask & (1 << channel):
                self.frames_detached[channel] = (
                    self.frames_detached.get(channel, 0) + frames
                )

    def frames_per_device_second(self):
        """Frames delivered per unit of the firmware's own clock.

        **This cannot validate the sample rate.** The firmware derives its
        timestamps from its own frame counter times the nominal period, so
        dividing frames by that span returns the nominal rate by
        construction. Only `sample_rate_hz_wall`, measured against the host
        clock, is an independent measurement of the rate.

        What this does detect is loss on the firmware side: a skipped half
        advances the timestamp without contributing frames, so the figure
        falls below nominal by exactly the fraction dropped.

        The span runs from the first packet's timestamp to the last one's, so
        it covers every frame except those in the last packet. Counting the
        frames actually present rather than assuming `frames_per_raw_packet`
        keeps this exact when a packet is short.
        """
        if self.first_raw_timestamp_us is None:
            return None
        frames = self.raw_frames - self._last_raw_frames
        if frames <= 0:
            return None
        span_us = (
            self.last_raw_timestamp_us - self.first_raw_timestamp_us
        ) % _TIMESTAMP_MODULO
        if span_us <= 0:
            return None
        return frames * 1e6 / span_us

    def summary(self, elapsed_sec):
        stats = self._parser.stats
        summary = {
            "elapsed_sec": round(elapsed_sec, 3),
            "bytes": self.byte_count,
            "byte_rate": round(self.byte_count / elapsed_sec, 1) if elapsed_sec else 0.0,
            "packets": {
                _TYPE_NAMES.get(key, str(key)): value
                for key, value in sorted(self.counts.items())
            },
            "parser": {
                "accepted": stats.accepted,
                "malformed": stats.malformed,
                "lost": stats.lost,
                "duplicated": stats.duplicated,
                "time_reversed": stats.time_reversed,
                "discarded_bytes": stats.discarded_bytes,
                "reasons": dict(stats.reasons),
            },
        }
        if self.info is not None:
            summary["info"] = {
                "firmware_version": self.info.firmware_version,
                "sample_rate_hz": self.info.sample_rate_hz,
                "channel_count": self.info.channel_count,
                "adc_bits": self.info.adc_bits,
                "frames_per_raw_packet": self.info.frames_per_raw_packet,
            }
        if self.raw_frames:
            summary["frames"] = self.raw_frames
            # Frames recorded with every electrode in contact. Anything else
            # is not training data: a detached electrode floats rather than
            # reading dead, so it looks like signal.
            summary["frames_all_attached"] = self.frames_all_attached
            summary["usable_fraction"] = round(
                self.frames_all_attached / self.raw_frames, 4
            )
            if self.frames_detached:
                summary["frames_detached_by_channel"] = {
                    str(channel): count
                    for channel, count in sorted(self.frames_detached.items())
                }
            if elapsed_sec:
                summary["sample_rate_hz_wall"] = round(self.raw_frames / elapsed_sec, 1)
        if self.info is not None:
            summary["sample_rate_hz_nominal"] = self.info.sample_rate_hz
        delivered = self.frames_per_device_second()
        if delivered is not None:
            summary["frames_per_device_second"] = round(delivered, 1)
        if self.raw_before_info:
            summary["raw_packets_before_info"] = self.raw_before_info
        return summary


def replay(path):
    """Re-parse an existing byte log and return the recording."""
    recording = Recording()
    with open(path, "rb") as log:
        while True:
            chunk = log.read(_READ_CHUNK)
            if not chunk:
                break
            recording.feed(chunk)
    return recording


def record(port, seconds, out_path):
    """Capture for a fixed window, writing bytes straight through to disk."""
    import serial  # Imported late so replay works without pyserial present.

    recording = Recording()
    started_wall = datetime.datetime.now().astimezone()
    connection = serial.Serial(port, 115200, timeout=0.1)
    connection.reset_input_buffer()
    start = time.perf_counter()
    with open(out_path, "wb") as log:
        while time.perf_counter() - start < seconds:
            chunk = connection.read(4096)
            if chunk:
                log.write(chunk)
                recording.feed(chunk)
    elapsed = time.perf_counter() - start
    connection.close()
    return recording, elapsed, started_wall


def write_sidecar(out_path, summary):
    """Write the metadata next to the byte log, same stem, .json suffix."""
    sidecar = pathlib.Path(out_path).with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return sidecar


def print_summary(summary):
    print("=" * 58)
    for key in ("elapsed_sec", "bytes", "byte_rate", "frames",
                "raw_packets_before_info", "frames_all_attached",
                "usable_fraction", "sample_rate_hz_nominal",
                "sample_rate_hz_wall", "frames_per_device_second"):
        if key in summary:
            print(f"  {key:24} {summary[key]}")
    if "frames_detached_by_channel" in summary:
        print("  frames_detached_by_channel")
        for channel, count in summary["frames_detached_by_channel"].items():
            print(f"    channel {channel:14} {count}")
    if "info" in summary:
        print("  info")
        for key, value in sorted(summary["info"].items()):
            print(f"    {key:22} {value}")
    print("  packets")
    for key, value in summary["packets"].items():
        print(f"    {key:22} {value}")
    print("  parser")
    for key, value in summary["parser"].items():
        print(f"    {key:22} {value}")
    print("=" * 58)

    parser_stats = summary["parser"]
    if parser_stats["accepted"] == 0:
        print("  FAIL: no valid packets. Wrong port, or firmware not sending v1.")
    elif parser_stats["lost"] or parser_stats["malformed"]:
        print("  WARN: the link dropped or corrupted data; see lost/malformed.")
    else:
        print("  OK: every packet accounted for.")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="session.bin")
    parser.add_argument("--replay", metavar="PATH",
                        help="re-parse an existing log instead of recording")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)

    if arguments.replay:
        recording = replay(arguments.replay)
        # Replay has no wall clock of its own, so rate-per-second figures come
        # from the device timestamps only; feeding 0 suppresses the wall ones.
        print_summary(recording.summary(0.0))
        return 0

    try:
        recording, elapsed, started_wall = record(
            arguments.port, arguments.seconds, arguments.out
        )
    except OSError as error:
        print(f"Could not record from {arguments.port}: {error}")
        print("Log in again so the dialout group applies, or run with sudo.")
        return 1

    summary = recording.summary(elapsed)
    summary["started"] = started_wall.isoformat()
    summary["port"] = arguments.port
    sidecar = write_sidecar(arguments.out, summary)
    print_summary(summary)
    print(f"  wrote {arguments.out} and {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
