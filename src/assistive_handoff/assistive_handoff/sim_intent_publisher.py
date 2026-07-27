"""Interactive Phase-0 /assistive_intent source for Objective 4.1.

Reads single-letter commands from stdin and publishes one AssistiveIntent
per line, with a per-session monotonically increasing sequence number as
the message contract requires:

    n  -> NEXT_TARGET
    c  -> CONFIRM
    a  -> ABORT
    q  -> quit

NEXT_TARGET is published on the source-independent intent contract for the
upstream target selector; the handoff controller itself does not cycle targets.

Also works non-interactively for scripted tests: `echo c | ros2 run ...`
publishes one CONFIRM and exits. QoS is the contract's reliable + volatile
(the default profile) — never TRANSIENT_LOCAL, so no intent is ever
retained or replayed.
"""

import sys
import threading
import time

import rclpy
from rclpy.node import Node

from assistive_interfaces.msg import AssistiveIntent

KEY_TO_COMMAND = {
    "n": ("NEXT_TARGET", AssistiveIntent.NEXT_TARGET),
    "c": ("CONFIRM", AssistiveIntent.CONFIRM),
    "a": ("ABORT", AssistiveIntent.ABORT),
}

SIMULATED_CONFIDENCE = 1.0


class SimIntentPublisher(Node):
    def __init__(self) -> None:
        super().__init__("sim_intent_publisher")
        self._pub = self.create_publisher(AssistiveIntent, "/assistive_intent", 10)
        self._sequence = 0
        self.get_logger().info(
            "keys: n=NEXT_TARGET  c=CONFIRM  a=ABORT  q=quit (then Enter)"
        )

    def publish_command(self, name: str, command: int) -> None:
        msg = AssistiveIntent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "sim_intent"
        msg.command = command
        msg.confidence = SIMULATED_CONFIDENCE
        msg.sequence = self._sequence
        self._sequence += 1
        self._pub.publish(msg)
        self.get_logger().info(f"published {name} (sequence {msg.sequence})")


def _stdin_loop(node: SimIntentPublisher) -> None:
    for line in sys.stdin:
        key = line.strip().lower()
        if key == "q":
            break
        if key in KEY_TO_COMMAND:
            node.publish_command(*KEY_TO_COMMAND[key])
        elif key:
            node.get_logger().warning(f"unknown key '{key}' (n/c/a/q)")
    # Give reliable QoS a moment to deliver the last message before exit
    # (matters for the piped one-shot use).
    time.sleep(0.5)
    rclpy.try_shutdown()


def main() -> None:
    rclpy.init()
    node = SimIntentPublisher()
    stdin_thread = threading.Thread(target=_stdin_loop, args=(node,), daemon=True)
    stdin_thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
