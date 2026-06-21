"""Move a simulated robot arm toward a fixed object pose."""

import time

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node


class MoveToObjectNode(Node):
    """Starting point for the object1 fixed-pose demo."""

    EXTENDED_JOINT_POSITIONS = (
        ("panda_joint1", 0.0),
        ("panda_joint2", 0.0),
        ("panda_joint3", 0.0),
        ("panda_joint4", 0.0),
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
        self.declare_parameter("execute", True)

        self.planning_group = self.get_parameter("planning_group").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.base_link = self.get_parameter("base_link").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.home_state = self.get_parameter("home_state").value
        self.allowed_planning_time = self.get_parameter("allowed_planning_time").value
        self.num_planning_attempts = self.get_parameter("num_planning_attempts").value
        self.execute = self.get_parameter("execute").value

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
        self._log_plan_only_goal()
        self._send_plan_only_goal()

    def _build_plan_only_goal(self) -> MoveGroup.Goal:
        """Create a MoveIt goal for Panda's extended state.

        When ``execute`` is True the goal plans and runs the trajectory;
        otherwise MoveIt only returns the plan without moving the arm.
        """
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.allowed_planning_time = float(self.allowed_planning_time)
        goal.request.num_planning_attempts = int(self.num_planning_attempts)
        goal.planning_options.plan_only = not self.execute

        extended_constraints = Constraints()
        for joint_name, position in self.EXTENDED_JOINT_POSITIONS:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            extended_constraints.joint_constraints.append(joint_constraint)
        goal.request.goal_constraints.append(extended_constraints)

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
                f"Extended constraint: {constraint.joint_name}={constraint.position} rad"
            )

    def _send_plan_only_goal(self) -> None:
        """Send the extended-state goal to MoveIt without executing a trajectory."""
        if not self._wait_for_move_group_server():
            return

        self._plan_request_start_time = time.monotonic()
        send_goal_future = self.move_group_client.send_goal_async(self.plan_only_goal)
        send_goal_future.add_done_callback(self._on_goal_response)
        mode = "plan-and-execute" if self.execute else "plan-only"
        self.get_logger().info(f"Sent {mode} goal to MoveIt.")

    def _wait_for_move_group_server(self) -> bool:
        """Return whether the MoveIt move_group action server is available."""
        if self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("Connected to MoveIt move_group action server.")
            return True

        self.get_logger().warn(
            "MoveIt move_group action server not available. "
            "Start the Panda MoveIt demo first."
        )
        return False

    def _on_goal_response(self, future) -> None:
        """Handle MoveIt's response to the plan-only goal request."""
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f"Failed to send MoveIt goal: {error}")
            return

        if not goal_handle.accepted:
            self.get_logger().error("MoveIt rejected the plan-only goal.")
            return

        self.get_logger().info("MoveIt accepted the plan-only goal.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_plan_result)

    def _log_planned_trajectory(self, result) -> None:
        """Log the joint trajectory returned by a successful plan-only request."""
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            self.get_logger().warn("Planning succeeded but returned no joint trajectory points.")
            return

        first_point = trajectory.points[0]
        final_point = trajectory.points[-1]
        self.get_logger().info(
            "Planned trajectory: "
            f"joint_names={list(trajectory.joint_names)}, "
            f"point_count={len(trajectory.points)}"
        )
        self.get_logger().info(
            f"First point positions={list(first_point.positions)}"
        )
        self.get_logger().info(
            f"Final point positions={list(final_point.positions)}"
        )

    def _on_plan_result(self, future) -> None:
        """Log the result of the plan-only request."""
        try:
            action_result = future.result()
        except Exception as error:
            self.get_logger().error(f"Failed to receive MoveIt result: {error}")
            return

        result = action_result.result
        wall_time = time.monotonic() - self._plan_request_start_time
        phase = "Motion" if self.execute else "Planning"
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(
                f"{phase} succeeded: "
                f"planning_time={result.planning_time:.3f}s, "
                f"wall_time={wall_time:.3f}s"
            )
            self._log_planned_trajectory(result)
            return

        # On failure MoveIt returns no executable trajectory, so the arm stays put.
        self.get_logger().error(
            f"{phase} failed: "
            f"error_code={result.error_code.val}, "
            f"message={result.error_code.message or 'no message'}, "
            f"wall_time={wall_time:.3f}s"
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
