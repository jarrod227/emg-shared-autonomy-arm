"""Move a simulated robot arm toward a fixed object pose."""

import rclpy
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node


class MoveToObjectNode(Node):
    """Starting point for the object1 fixed-pose demo."""

    def __init__(self) -> None:
        super().__init__("move_to_object")

        self.declare_parameter("planning_group", "panda_arm")
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("base_link", "panda_link0")
        self.declare_parameter("end_effector_frame", "panda_link8")
        self.declare_parameter("home_state", "ready")

        self.planning_group = self.get_parameter("planning_group").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.base_link = self.get_parameter("base_link").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.home_state = self.get_parameter("home_state").value

        self.move_group_client = ActionClient(self, MoveGroup, "move_action")

        # TODO: Define object1's fixed pose.
        # TODO: Connect to the MoveIt 2 planning interface.
        # TODO: Request motion to object1 and then return home.
        self.get_logger().info("object1_demo node started; MoveIt integration is the next step.")
        self.get_logger().info(
            "MoveIt target config: "
            f"planning_group={self.planning_group}, "
            f"planning_frame={self.planning_frame}, "
            f"base_link={self.base_link}, "
            f"end_effector_frame={self.end_effector_frame}, "
            f"home_state={self.home_state}"
        )
        self._log_move_group_server_status()

    def _log_move_group_server_status(self) -> None:
        """Log whether the MoveIt move_group action server is available."""
        if self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("Connected to MoveIt move_group action server.")
        else:
            self.get_logger().warn(
                "MoveIt move_group action server not available. "
                "Start the Panda MoveIt demo first."
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveToObjectNode()
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
