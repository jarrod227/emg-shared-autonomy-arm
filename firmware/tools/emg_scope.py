#!/usr/bin/env python3
"""Live three-channel scope for the sEMG firmware's binary stream.

Shows the band-passed signal per channel, because raw counts sit near 2048
and the interesting part is a few hundred counts of swing on top of that. The
number that actually matters for electrode placement is the MAV printed in
each subplot title: moving a band and watching which channel's MAV responds
to which gesture is far faster than record-then-analyse.

Close the window to stop; a text summary is printed on exit, so a run leaves
evidence behind for anyone who was not watching the screen.

    python3 emg_scope.py [--port P] [--seconds S]
"""

import argparse
import collections
import statistics
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from emg_features_ref import WINDOW, mean_absolute_value
from emg_filter_ref import FixedFilter, design_bandpass, to_fixed
from emg_protocol import TYPE_INFO, TYPE_RAW, PacketParser, decode_info, decode_raw

PALETTE = ("#2b8cbe", "#e34a33", "#31a354")


class ChannelStream:
    """Per-channel filtered ring buffer, filtered statefully as blocks arrive."""

    def __init__(self, sections, span):
        self.filter = FixedFilter(sections)
        self.samples = collections.deque([0] * span, maxlen=span)
        self.raw = collections.deque([2048] * span, maxlen=span)
        self.attached = False

    def extend(self, values):
        self.raw.extend(int(value) for value in values)
        self.samples.extend(self.filter.process(values))

    def rail_hits(self):
        """Raw samples pegged at the ADC limits in the visible window.

        Shown live because clipping is invisible in the filtered trace and in
        the MAV: a clipped peak still yields a plausible number, just a wrong
        one. Catching it while adjusting the band beats discovering it after
        the recording.
        """
        return sum(1 for value in self.raw if value <= 0 or value >= 4095)

    def mav(self):
        recent = list(self.samples)[-WINDOW:]
        return mean_absolute_value(recent) if recent else 0


def reader(connection, state, stop):
    """Parse packets into per-channel filtered streams until told to stop."""
    parser = PacketParser()
    while not stop.is_set():
        chunk = connection.read(4096)
        if not chunk:
            continue
        for packet in parser.feed(chunk):
            if packet.type == TYPE_INFO and state["info"] is None:
                state["info"] = decode_info(packet.payload)
            elif packet.type == TYPE_RAW and state["info"] is not None:
                block = decode_raw(packet.payload, state["info"].channel_count)
                if not state["channels"]:
                    continue
                frames = np.asarray(block.frames, dtype=np.int64)
                for index, stream in enumerate(state["channels"]):
                    stream.attached = block.channel_attached(index)
                    stream.extend(frames[:, index])
    state["stats"] = parser.stats


def wait_for_info(connection, state, stop, timeout=6.0):
    """Block until an INFO packet arrives; it carries the channel count."""
    start = time.perf_counter()
    while state["info"] is None:
        if time.perf_counter() - start > timeout:
            stop.set()
            return False
        time.sleep(0.05)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="width of the scrolling window")
    arguments = parser.parse_args()

    try:
        connection = serial.Serial(arguments.port, 115200, timeout=0.05)
    except serial.SerialException as error:
        print(f"Could not open the serial port: {error}")
        print("Log in again so the dialout group applies, or run with sudo.")
        return 1
    connection.reset_input_buffer()

    state = {"info": None, "channels": [], "stats": None}
    stop = threading.Event()
    thread = threading.Thread(target=reader, args=(connection, state, stop),
                              daemon=True)
    thread.start()

    print(f"Waiting for an INFO packet on {arguments.port} ...")
    if not wait_for_info(connection, state, stop):
        thread.join(timeout=1.0)
        connection.close()
        print("No INFO packet arrived. Is the firmware running, and is this")
        print("the binary protocol rather than the factory ASCII stream?")
        return 1

    info = state["info"]
    span = max(64, int(arguments.seconds * info.sample_rate_hz))
    sections = to_fixed(design_bandpass(rate_hz=float(info.sample_rate_hz)))
    state["channels"] = [ChannelStream(sections, span)
                         for _ in range(info.channel_count)]
    print(f"{info.channel_count} channels at {info.sample_rate_hz} Hz. "
          "Close the window to stop.")

    figure, axes = plt.subplots(info.channel_count, 1, sharex=True,
                               figsize=(10, 2.2 * info.channel_count))
    axes = np.atleast_1d(axes)
    positions = range(span)
    traces = []
    for index, axis in enumerate(axes):
        trace, = axis.plot(positions, [0] * span, linewidth=0.7,
                           color=PALETTE[index % len(PALETTE)])
        axis.grid(alpha=0.3)
        axis.set_ylabel("counts")
        traces.append(trace)
    axes[-1].set_xlabel(f"most recent {arguments.seconds:g} s (band-passed)")

    def update(_):
        for index, (axis, trace) in enumerate(zip(axes, traces)):
            stream = state["channels"][index]
            data = list(stream.samples)
            trace.set_ydata(data)
            # Autoscale per channel: a well-placed set has channels differing
            # by an order of magnitude between gestures, so one shared range
            # would flatten whichever is quiet.
            limit = max(64, int(1.2 * max(abs(min(data)), abs(max(data)))))
            axis.set_ylim(-limit, limit)
            contact = "attached" if stream.attached else "NO CONTACT"
            rails = stream.rail_hits()
            warning = f"   CLIPPING x{rails}" if rails else ""
            axis.set_title(
                f"ch{index}   MAV {stream.mav():5d}   [{contact}]{warning}",
                loc="left", fontsize=10,
                color="#b2182b" if (rails or not stream.attached) else "black")
        return traces

    animation = FuncAnimation(figure, update, interval=60, blit=False,
                              cache_frame_data=False)
    figure.tight_layout()
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=1.0)
        connection.close()
    del animation

    print("\n" + "=" * 58)
    for index, stream in enumerate(state["channels"]):
        data = list(stream.samples)
        spread = statistics.pstdev(data) if len(data) > 1 else 0.0
        contact = "attached" if stream.attached else "NO CONTACT"
        rails = stream.rail_hits()
        print(f"  ch{index}: last-window MAV {stream.mav()}  sd {spread:.1f}  "
              f"[{contact}]" + (f"  CLIPPED x{rails}" if rails else ""))
    stats = state.get("stats")
    if stats is not None:
        print(f"  packets accepted={stats.accepted} lost={stats.lost} "
              f"malformed={stats.malformed}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
