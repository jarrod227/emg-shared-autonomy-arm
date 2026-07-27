"""Phase-0 simulated /target_object_pose source for Objective 4.1.

Re-stamps and republishes a fixed target pose at 2 Hz, using the same
retained (TRANSIENT_LOCAL) QoS as the real Objective 3.1 publishers
(target_selector, fixed_pose_publisher), so the handoff controller's
subscription QoS is exercised exactly as it is in integration.

The pose is the object1_demo fixed grasp pose — already verified reachable
by the Panda MoveIt pipeline — so motion-backend integration can consume it
unchanged.

Unlike the real publishers this one republishes continuously. Dedicated
controller tests cover the one-shot retained case: an old source stamp must
be rejected using parameterized age checking, regardless of when DDS delivers
the sample.
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

PUBLISH_PERIOD_SEC = 0.5  # 2 Hz
FRAME_ID = "world"
# object1_demo fixed_pose_publisher defaults (verified-reachable grasp pose).
TARGET_XYZ = (0.106982, 0.0, 1.121022)
TARGET_QUAT_XYZW = (0.653269, -0.270440, 0.653402, -0.270496)


class SimTargetPublisher(Node):
    def __init__(self) -> None:
        super().__init__("sim_target_publisher")
        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self._pub = self.create_publisher(
            PoseStamped, "/target_object_pose", latched_qos
        )
        self._timer = self.create_timer(PUBLISH_PERIOD_SEC, self._publish)
        self.get_logger().info(
            f"publishing fixed target at {TARGET_XYZ} in '{FRAME_ID}' "
            f"every {PUBLISH_PERIOD_SEC}s (retained QoS)"
        )

    def _publish(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = TARGET_XYZ
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = TARGET_QUAT_XYZW
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SimTargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
