#!/usr/bin/env python3
"""Capture a fixed window from the Cheez sEMG board and summarize it as text.

Answers three questions in one run:
  1. Is the analog chain alive at all, or is the value pinned?
  2. What sample rate is the firmware actually producing?
  3. Does the signal respond when the muscle contracts?

Runs for a fixed duration and exits on its own. Everything it learns is
printed, so the result survives without anyone having watched a plot.

Usage:  python3 emg_probe.py [seconds] [port]
"""

import re
import statistics
import sys
import time

import serial

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_SECONDS = 10.0
SEGMENTS = 10


def capture(port, seconds):
    """Read raw bytes for a fixed wall-clock window."""
    connection = serial.Serial(port, 115200, timeout=0.2)
    connection.reset_input_buffer()
    chunks = []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        data = connection.read(4096)
        if data:
            chunks.append(data)
    elapsed = time.perf_counter() - start
    connection.close()
    return b"".join(chunks), elapsed


def parse_samples(raw):
    """Pull integers out of the ASCII stream, one list per line."""
    text = raw.decode("ascii", errors="replace")
    rows = []
    # Drop the first and last line: both are almost certainly truncated by
    # where the capture window happened to open and close.
    for line in text.splitlines()[1:-1]:
        values = [int(value) for value in re.findall(r"-?\d+", line)]
        if values:
            rows.append(values)
    return rows


def describe(label, values):
    if not values:
        return f"  {label}: no data"
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (
        f"  {label}: min={min(values)} max={max(values)} "
        f"mean={statistics.mean(values):.1f} sd={spread:.1f}"
    )


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS
    port = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PORT

    print(f"Capturing {seconds:.0f} s on {port}.")
    print("Contract and relax the muscle several times during the window.\n")
    try:
        raw, elapsed = capture(port, seconds)
    except serial.SerialException as error:
        print(f"Could not open the serial port: {error}")
        print("Log in again so the dialout group applies, or run with sudo.")
        return 1

    print("=" * 58)
    print("1. Stream")
    print("=" * 58)
    print(f"  elapsed:     {elapsed:.2f} s")
    print(f"  bytes:       {len(raw)}")
    if not raw:
        print("\n  FAIL: nothing arrived. The firmware may be waiting for a")
        print("        start command, or the board is not transmitting.")
        return 0
    print(f"  byte rate:   {len(raw) / elapsed:.0f} B/s")

    rows = parse_samples(raw)
    if not rows:
        print("\n  FAIL: bytes arrived but no numbers parsed. First 120 bytes:")
        print(f"  {raw[:120]!r}")
        return 0

    widths = {len(row) for row in rows}
    print(f"\n  complete rows: {len(rows)}")
    print(f"  sample rate:   {len(rows) / elapsed:.1f} Hz")
    print(f"  values/row:    {sorted(widths)}  ->  ", end="")
    print("single channel" if widths == {1} else f"{max(widths)} channels")

    print("\n  first 3 rows:")
    for line in raw.decode("ascii", errors="replace").splitlines()[1:4]:
        print(f"    {line!r}")

    print()
    print("=" * 58)
    print("2. Value distribution")
    print("=" * 58)
    for channel in range(max(widths)):
        column = [row[channel] for row in rows if len(row) > channel]
        print(describe(f"channel {channel}", column))
        distinct = len(set(column))
        print(f"             distinct values: {distinct}")
        if distinct == 1:
            print(f"             FAIL: pinned at {column[0]}, no signal")
        elif distinct < 5:
            print("             WARN: almost no variation, likely just noise")
        else:
            print("             OK: signal is varying")

    print()
    print("=" * 58)
    print(f"3. Split into {SEGMENTS} segments (contractions should stand out)")
    print("=" * 58)
    column = [row[0] for row in rows if row]
    size = max(1, len(column) // SEGMENTS)
    for start in range(0, len(column), size):
        segment = column[start:start + size]
        if not segment:
            continue
        offset = start / len(column) * elapsed
        bar = "#" * min(40, (max(segment) - min(segment)) // 10)
        print(
            f"  t={offset:5.1f}s  min={min(segment):5d} max={max(segment):5d} "
            f"mean={statistics.mean(segment):7.1f}  {bar}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
