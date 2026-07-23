"""Target selector node.

Subscribes to /detected_markers (camera-frame poses), selects one marker,
applies the marker-to-grasp offset, transforms it into the planning frame,
and publishes the single /target_object_pose the reaching coordinator
consumes. This is the layer where the EMG intent signal will later choose
which marker to target.

M4 skeleton: starts and spins. Selection, offset, and TF transform land in
M4.5 and M5.
"""

import rclpy
from rclpy.node import Node


class SelectorNode(Node):
    """Turns detected markers into one /target_object_pose in the planning frame."""

    def __init__(self) -> None:
        super().__init__("target_selector")
        self.get_logger().info("target_selector started (M4 skeleton)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
