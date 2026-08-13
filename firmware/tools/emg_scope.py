#!/usr/bin/env python3
"""Live scope for the sEMG board's ASCII serial stream.

Reads the port in a background thread so redrawing never stalls the serial
read, and draws a scrolling window. Close the window to stop; a text summary
is printed on exit so the run still leaves evidence behind for anyone who was
not watching the screen.

Usage:  python3 emg_scope.py [--port P] [--seconds S] [--rate HZ] [--channels N]
"""

import argparse
import collections
import re
import statistics
import threading
import time

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation

PALETTE = ("#2b8cbe", "#e34a33", "#31a354")


def reader(connection, buffers, stop, widths):
    """Push one parsed sample per line into a per-channel ring buffer."""
    pending = b""
    while not stop.is_set():
        pending += connection.read(256)
        *lines, pending = pending.split(b"\n")
        for line in lines:
            values = [int(value) for value in re.findall(rb"-?\d+", line)]
            if not values:
                continue
            widths.append(len(values))
            for index, value in enumerate(values[: len(buffers)]):
                buffers[index].append(value)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="width of the scrolling window in seconds")
    parser.add_argument("--rate", type=float, default=500.0,
                        help="expected sample rate, used to size the window")
    parser.add_argument("--channels", type=int, default=1)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    span = max(64, int(arguments.seconds * arguments.rate))
    buffers = [collections.deque([0] * span, maxlen=span)
               for _ in range(arguments.channels)]
    widths = collections.deque(maxlen=2000)
    stop = threading.Event()

    try:
        connection = serial.Serial(arguments.port, 115200, timeout=0.05)
    except serial.SerialException as error:
        print(f"Could not open the serial port: {error}")
        print("Log in again so the dialout group applies, or run with sudo.")
        return 1
    connection.reset_input_buffer()

    thread = threading.Thread(
        target=reader, args=(connection, buffers, stop, widths), daemon=True
    )
    thread.start()
    started = time.perf_counter()

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.set_title(f"sEMG live - {arguments.port} - close the window to stop")
    axis.set_xlabel(f"most recent {arguments.seconds:g} s")
    axis.set_ylabel("ADC counts")
    axis.grid(alpha=0.3)
    samples = range(span)
    traces = [
        axis.plot(samples, list(buffer), color=PALETTE[index % len(PALETTE)],
                  linewidth=0.8, label=f"ch{index}")[0]
        for index, buffer in enumerate(buffers)
    ]
    if len(buffers) > 1:
        axis.legend(loc="upper right")

    def update(_):
        # Rescale to whatever is on screen. A fixed y-range either clips the
        # contraction bursts or flattens the resting trace: the measured
        # dynamic range on this board spans +/-2 to +/-1141 counts.
        low, high = 0, 0
        for trace, buffer in zip(traces, buffers):
            data = list(buffer)
            trace.set_ydata(data)
            low, high = min(low, min(data)), max(high, max(data))
        margin = max(32, int(0.1 * (high - low)))
        axis.set_ylim(low - margin, high + margin)
        return traces

    animation = FuncAnimation(figure, update, interval=40,
                              blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=1.0)
        connection.close()
    del animation

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 52)
    print(f"  elapsed:    {elapsed:.1f} s")
    print(f"  values/row: {sorted(set(widths)) or 'no data'}")
    for index, buffer in enumerate(buffers):
        data = list(buffer)
        if not any(data):
            print(f"  channel {index}: all zero, no signal")
            continue
        print(f"  channel {index}: min={min(data)} max={max(data)} "
              f"sd={statistics.pstdev(data):.1f} "
              f"(last {len(data)} points in the window)")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
