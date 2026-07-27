"""Phase-0 simulated /hand_observation source for Objective 4.1.

Publishes a valid, freshly stamped HandObservation at 10 Hz with a fixed
3D point, standing in for the Objective 4.2 stereo hand localizer. The
stereo-quality fields stay 0.0 as the message contract specifies for the
4.1 simulated publisher.

Stopping this node (Ctrl-C) makes the hand stream go stale and demonstrates
that the controller's parameterized freshness gate refuses release.
"""

import rclpy
from rclpy.node import Node

from assistive_interfaces.msg import HandObservation

PUBLISH_PERIOD_SEC = 0.1  # 10 Hz
FRAME_ID = "world"
# Fixed simulated palm position in the planning frame.
HAND_X, HAND_Y, HAND_Z = 0.4, 0.3, 1.0
SIMULATED_CONFIDENCE = 0.9


class SimHandPublisher(Node):
    def __init__(self) -> None:
        super().__init__("sim_hand_publisher")
        self._pub = self.create_publisher(HandObservation, "/hand_observation", 10)
        self._timer = self.create_timer(PUBLISH_PERIOD_SEC, self._publish)
        self.get_logger().info(
            f"publishing valid hand at ({HAND_X}, {HAND_Y}, {HAND_Z}) "
            f"in '{FRAME_ID}' every {PUBLISH_PERIOD_SEC}s"
        )

    def _publish(self) -> None:
        msg = HandObservation()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.valid = True
        msg.point.x = HAND_X
        msg.point.y = HAND_Y
        msg.point.z = HAND_Z
        msg.confidence = SIMULATED_CONFIDENCE
        # pair_skew_sec / reprojection_error stay 0.0 (contract: filled in 4.2)
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SimHandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
