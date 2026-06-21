"""Move a simulated robot arm toward a fixed object pose."""

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from rclpy.node import Node


class MoveToObjectNode(Node):
    """Starting point for the object1 fixed-pose demo."""

    READY_JOINT_POSITIONS = (
        ("panda_joint1", 0.0),
        ("panda_joint2", -0.785),
        ("panda_joint3", 0.0),
        ("panda_joint4", -2.356),
        ("panda_joint5", 0.0),
        ("panda_joint6", 1.571),
        ("panda_joint7", 0.785),
    )

    def __init__(self) -> None:
        super().__init__("move_to_object")

        self.declare_parameter("planning_group", "panda_arm")
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("base_link", "panda_link0")
        self.declare_parameter("end_effector_frame", "panda_link8")
        self.declare_parameter("home_state", "ready")
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 1)

        self.planning_group = self.get_parameter("planning_group").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.base_link = self.get_parameter("base_link").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.home_state = self.get_parameter("home_state").value
        self.allowed_planning_time = self.get_parameter("allowed_planning_time").value
        self.num_planning_attempts = self.get_parameter("num_planning_attempts").value

        self.move_group_client = ActionClient(self, MoveGroup, "move_action")
        self.plan_only_goal = self._build_plan_only_goal()

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
        self._log_plan_only_goal()

    def _build_plan_only_goal(self) -> MoveGroup.Goal:
        """Create a local MoveIt goal skeleton without sending it."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.allowed_planning_time = float(self.allowed_planning_time)
        goal.request.num_planning_attempts = int(self.num_planning_attempts)
        goal.planning_options.plan_only = True

        ready_constraints = Constraints()
        for joint_name, position in self.READY_JOINT_POSITIONS:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            ready_constraints.joint_constraints.append(joint_constraint)
        goal.request.goal_constraints.append(ready_constraints)

        return goal

    def _log_plan_only_goal(self) -> None:
        """Log the local MoveIt goal skeleton fields."""
        self.get_logger().info(
            "MoveGroup goal skeleton: "
            f"group_name={self.plan_only_goal.request.group_name}, "
            f"allowed_planning_time={self.plan_only_goal.request.allowed_planning_time}, "
            f"num_planning_attempts={self.plan_only_goal.request.num_planning_attempts}, "
            f"plan_only={self.plan_only_goal.planning_options.plan_only}, "
            f"joint_constraints={len(self.plan_only_goal.request.goal_constraints[0].joint_constraints)}"
        )
        for constraint in self.plan_only_goal.request.goal_constraints[0].joint_constraints:
            self.get_logger().info(
                f"Ready constraint: {constraint.joint_name}={constraint.position} rad"
            )

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
