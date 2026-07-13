"""Publish the fixed object1 pose for the Objective 2 target interface."""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class FixedPosePublisherNode(Node):
    """Publish one retained target pose when the node starts."""

    def __init__(self):
        super().__init__("fixed_pose_publisher")

        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("target_x", 0.106982)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 1.121022)
        self.declare_parameter("target_qx", 0.653269)
        self.declare_parameter("target_qy", -0.270440)
        self.declare_parameter("target_qz", 0.653402)
        self.declare_parameter("target_qw", -0.270496)

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.target_pose_publisher = self.create_publisher(
            PoseStamped,
            "target_object_pose",
            qos,
        )

        target_pose = self.build_target_pose()
        self.target_pose_publisher.publish(target_pose)
        self.get_logger().info(
            "Published fixed object1 pose on /target_object_pose "
            f"in frame {target_pose.header.frame_id}."
        )

    def build_target_pose(self):
        """Build a target pose from the node parameters."""
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.get_parameter("planning_frame").value
        target_pose.pose.position.x = self.get_parameter("target_x").value
        target_pose.pose.position.y = self.get_parameter("target_y").value
        target_pose.pose.position.z = self.get_parameter("target_z").value
        target_pose.pose.orientation.x = self.get_parameter("target_qx").value
        target_pose.pose.orientation.y = self.get_parameter("target_qy").value
        target_pose.pose.orientation.z = self.get_parameter("target_qz").value
        target_pose.pose.orientation.w = self.get_parameter("target_qw").value
        return target_pose


def main(args=None):
    rclpy.init(args=args)
    node = FixedPosePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
