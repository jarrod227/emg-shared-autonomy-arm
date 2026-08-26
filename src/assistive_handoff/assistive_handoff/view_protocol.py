"""Prompted protocol for a closed-loop view-control session.

The calibration has always counted the wearer in -- "starting in 3... GO
(4 s)... done" -- and the closed-loop runs did not. They were a table the
wearer read between attempts while estimating six seconds by feel, and the
recordings show it: 87 s of session containing 16 s of commanding, separated
by gaps of 14 to 19 seconds that no analysis could distinguish from a signal
that had stopped working.

That ambiguity is the reason this exists, more than the ergonomics. Segmenting
a recording by the direction that was *observed* cannot tell a six-second hold
that produced three fragments from three fragments the wearer actually
performed. Publishing what was asked for, timestamped into the same bag,
turns "how much motion happened" into "how much of each intended hold produced
the direction it asked for", which is the question every one of these sessions
was trying to answer.

The marker stream is deliberately not an input to anything. Nothing subscribes
to it, the controller cannot see it, and it exists only to be recorded.

Every print flushes and every cue rings the terminal bell, because the first
session run with this tool produced prompts the wearer never saw. Python block-
buffers stdout when it is not a terminal, so a run whose output was redirected
delivered the whole script at once, after it had finished. The wearer performed
the protocol from memory and the markers labelled a timeline nobody was
following -- which is worse than having no markers, since the analysis then
scores gestures against the phase they happened to fall in. An alternating
protocol makes that failure look exactly like a classifier confusing the two
directions, and it did.

Run this where the wearer can see and hear it. It is the one part of a session
that cannot be launched somewhere its output will not be read.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String


# (action, effort, hold seconds). Alternating direction keeps the axis near
# the middle: a hold that reaches the band edge stops early, and its speed
# says more about the bound than about the effort that was asked for.
DEFAULT_PROTOCOL = (
    ("LEFT", "light", 6.0),
    ("RIGHT", "light", 6.0),
    ("LEFT", "medium", 6.0),
    ("RIGHT", "medium", 6.0),
    ("LEFT", "hard", 4.0),
    ("RIGHT", "hard", 4.0),
    ("LEFT", "light", 6.0),
    ("RIGHT", "medium", 6.0),
    ("RIGHT", "hard", 8.0),
)

GESTURES = {
    "LEFT": "lift your wrist upward",
    "RIGHT": "tilt your wrist toward the little-finger side",
}

EFFORTS = {
    "light": "gently -- just enough to move the arm",
    "medium": "moderately",
    "hard": "hard, but only as hard as you would hold for a whole session",
}

BELL = "\a"
REST_SECONDS = 3.0
COUNT_IN_SECONDS = 3


class ViewProtocol(Node):
    """Counts the wearer in and records what was asked for."""

    def __init__(self, protocol=DEFAULT_PROTOCOL):
        super().__init__("view_protocol")
        # Transient local so a late-started recorder still sees the phase in
        # progress rather than beginning mid-hold with no label.
        self._publisher = self.create_publisher(
            String,
            "/experiment_marker",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._protocol = tuple(protocol)

    def _mark(self, text):
        self._publisher.publish(String(data=text))
        rclpy.spin_once(self, timeout_sec=0.0)

    def _hold(self, label, seconds):
        """Publish one phase for its whole duration, spinning throughout."""
        self._mark(label)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        total = len(self._protocol)
        print(f"\n  {total} movements, alternating direction.", flush=True)
        print("  Hold steady once you start. If the arm jitters or reverses,", flush=True)
        print("  do NOT correct it -- keep the same gesture and effort.\n", flush=True)
        self._hold("rest", REST_SECONDS)

        for index, (direction, effort, seconds) in enumerate(self._protocol, 1):
            print(f"  {index}/{total}  {direction} {effort} -- "
                  f"{GESTURES[direction]}, {EFFORTS[effort]}")
            for remaining in range(COUNT_IN_SECONDS, 0, -1):
                print(f"    starting in {remaining}...", flush=True)
                self._hold(f"count_in {direction} {effort}", 1.0)
            print(f"{BELL}    GO ({seconds:.0f} s)", flush=True)
            self._hold(f"hold {direction} {effort}", seconds)
            print(f"{BELL}    RELAX", flush=True)
            self._hold("rest", REST_SECONDS)

        self._mark("done")
        print("\n  done -- stop the recording with Ctrl-C in its window.", flush=True)


def main(argv=None):
    rclpy.init(args=argv)
    node = ViewProtocol()
    try:
        node.run()
    except KeyboardInterrupt:
        node._mark("aborted")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
