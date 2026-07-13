"""Move a simulated robot arm to a received target pose and back home."""

import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker


class MoveToObjectNode(Node):
    """Reach a received Cartesian target pose, then return to the home pose.

    The reaching flow is a small state machine:

        IDLE -> REACHING (go to object1) -> RETURNING (go home) -> DONE
                      |                            |
                    failure --------------------> FAILED
    """

    # States of the reach/return-home sequence.
    IDLE = "IDLE"
    REACHING = "REACHING"
    RETURNING = "RETURNING"
    DONE = "DONE"
    FAILED = "FAILED"

    # Panda "ready" pose; the arm starts here, so it is the home pose.
    HOME_JOINT_POSITIONS = (
        ("panda_joint1", 0.0),
        ("panda_joint2", -0.785),
        ("panda_joint3", 0.0),
        ("panda_joint4", -2.356),
        ("panda_joint5", 0.0),
        ("panda_joint6", 1.571),
        ("panda_joint7", 0.785),
    )

    def __init__(self, **kwargs) -> None:
        super().__init__("move_to_object", **kwargs)

        self.declare_parameter("planning_group", "panda_arm")
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("base_link", "panda_link0")
        self.declare_parameter("end_effector_frame", "panda_link8")
        self.declare_parameter("home_state", "ready")
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 1)
        self.declare_parameter("execute", True)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("velocity_scaling", 0.2)
        self.declare_parameter("acceleration_scaling", 0.2)

        self.planning_group = self.get_parameter("planning_group").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.base_link = self.get_parameter("base_link").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.home_state = self.get_parameter("home_state").value
        self.allowed_planning_time = self.get_parameter("allowed_planning_time").value
        self.num_planning_attempts = self.get_parameter("num_planning_attempts").value
        self.execute = self.get_parameter("execute").value
        self.auto_start = self.get_parameter("auto_start").value
        self.velocity_scaling = float(
            self.get_parameter("velocity_scaling").value
        )
        self.acceleration_scaling = float(
            self.get_parameter("acceleration_scaling").value
        )
        self.object1_position = None
        self.object1_orientation = None
        self._target_received = False

        target_pose_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.target_pose_subscription = self.create_subscription(
            PoseStamped,
            "target_object_pose",
            self._on_target_pose,
            target_pose_qos,
        )

        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.object1_marker_publisher = self.create_publisher(
            Marker,
            "object1_target",
            marker_qos,
        )
        self.move_group_client = ActionClient(self, MoveGroup, "move_action")
        self.state = self.IDLE
        self._current_target = ""
        self._goal_start_time = 0.0

        self.get_logger().info("object1_demo node started.")
        self.get_logger().info(
            "MoveIt target config: "
            f"planning_group={self.planning_group}, "
            f"planning_frame={self.planning_frame}, "
            f"base_link={self.base_link}, "
            f"end_effector_frame={self.end_effector_frame}, "
            f"home_state={self.home_state}"
        )
        if self.auto_start:
            self.get_logger().info(
                "Waiting for a target pose on /target_object_pose."
            )

    def _on_target_pose(self, target_pose: PoseStamped) -> None:
        """Store the first valid target pose and start the reaching sequence."""
        if self._target_received:
            self.get_logger().warn("Ignoring an additional target pose.")
            return
        if not target_pose.header.frame_id:
            self.get_logger().error("Ignoring target pose with an empty frame_id.")
            return

        self.planning_frame = target_pose.header.frame_id
        self.object1_position = (
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )
        self.object1_orientation = (
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w,
        )
        self._target_received = True

        self.get_logger().info(
            "Received object1 target pose on /target_object_pose "
            f"in frame {self.planning_frame}."
        )
        self._publish_object1_marker()
        if self.auto_start:
            self._start_sequence()

    def _build_object1_marker(self) -> Marker:
        """Create the RViz marker for the received object1 target pose."""
        if self.object1_position is None or self.object1_orientation is None:
            raise RuntimeError("Cannot build marker before receiving a target pose.")

        marker = Marker()
        marker.header.frame_id = self.planning_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "object1"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        (
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ) = self.object1_position
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = self.object1_orientation
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.scale.z = 0.10
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.8
        return marker

    def _publish_object1_marker(self) -> None:
        """Publish the fixed target marker for RViz."""
        self.object1_marker_publisher.publish(self._build_object1_marker())
        self.get_logger().info("Published object1 target marker on /object1_target.")

    def _transition(self, new_state: str) -> None:
        """Record a state-machine transition."""
        self.get_logger().info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state

    def _start_sequence(self) -> None:
        """Begin the reach/return-home sequence if MoveIt is available."""
        if not self._wait_for_move_group_server():
            self._transition(self.FAILED)
            return

        self._transition(self.REACHING)
        self._send_pose_goal("object1")

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

    def _build_joint_goal(self, joint_positions) -> MoveGroup.Goal:
        """Create a MoveIt goal that drives the arm to a joint configuration.

        When ``execute`` is True the goal plans and runs the trajectory;
        otherwise MoveIt only returns the plan without moving the arm.
        """
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.allowed_planning_time = float(self.allowed_planning_time)
        goal.request.num_planning_attempts = int(self.num_planning_attempts)
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        goal.planning_options.plan_only = not self.execute

        constraints = Constraints()
        for joint_name, position in joint_positions:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        goal.request.goal_constraints.append(constraints)

        return goal

    def _build_pose_goal(self) -> MoveGroup.Goal:
        """Create a MoveIt goal for the received object1 end-effector pose."""
        if self.object1_position is None or self.object1_orientation is None:
            raise RuntimeError("Cannot build pose goal before receiving a target pose.")

        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.allowed_planning_time = float(self.allowed_planning_time)
        goal.request.num_planning_attempts = int(self.num_planning_attempts)
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        goal.planning_options.plan_only = not self.execute

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.planning_frame
        position_constraint.link_name = self.end_effector_frame
        position_constraint.weight = 1.0

        tolerance_region = SolidPrimitive()
        tolerance_region.type = SolidPrimitive.SPHERE
        tolerance_region.dimensions = [0.005]

        target_pose = Pose()
        (
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        ) = self.object1_position
        target_pose.orientation.w = 1.0
        position_constraint.constraint_region.primitives.append(tolerance_region)
        position_constraint.constraint_region.primitive_poses.append(target_pose)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.planning_frame
        orientation_constraint.link_name = self.end_effector_frame
        (
            orientation_constraint.orientation.x,
            orientation_constraint.orientation.y,
            orientation_constraint.orientation.z,
            orientation_constraint.orientation.w,
        ) = self.object1_orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints.append(constraints)
        return goal

    def _send_pose_goal(self, target_name: str) -> None:
        """Send the fixed Cartesian object1 goal."""
        self._current_target = target_name
        goal = self._build_pose_goal()
        self._goal_start_time = time.monotonic()
        send_goal_future = self.move_group_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._on_goal_response)

        mode = "plan-and-execute" if self.execute else "plan-only"
        self.get_logger().info(
            f"Sent {mode} pose goal toward {target_name} "
            f"in {self.planning_frame} for {self.end_effector_frame}."
        )

    def _send_joint_goal(self, target_name: str, joint_positions) -> None:
        """Send a joint-space goal toward the named target."""
        self._current_target = target_name
        goal = self._build_joint_goal(joint_positions)
        self._goal_start_time = time.monotonic()
        send_goal_future = self.move_group_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._on_goal_response)

        mode = "plan-and-execute" if self.execute else "plan-only"
        self.get_logger().info(
            f"Sent {mode} goal toward {target_name} "
            f"({len(joint_positions)} joint constraints)."
        )

    def _on_goal_response(self, future) -> None:
        """Handle MoveIt's accept/reject for the current goal."""
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f"Failed to send MoveIt goal: {error}")
            self._transition(self.FAILED)
            return

        if not goal_handle.accepted:
            self.get_logger().error(f"MoveIt rejected the {self._current_target} goal.")
            self._transition(self.FAILED)
            return

        self.get_logger().info(f"MoveIt accepted the {self._current_target} goal.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _log_planned_trajectory(self, result) -> None:
        """Log the joint trajectory returned for the current goal."""
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            self.get_logger().warn("Succeeded but returned no joint trajectory points.")
            return

        first_point = trajectory.points[0]
        final_point = trajectory.points[-1]
        self.get_logger().info(
            "Planned trajectory: "
            f"joint_names={list(trajectory.joint_names)}, "
            f"point_count={len(trajectory.points)}"
        )
        self.get_logger().info(f"First point positions={list(first_point.positions)}")
        self.get_logger().info(f"Final point positions={list(final_point.positions)}")

    def _on_result(self, future) -> None:
        """Handle the result for the current goal and advance the sequence."""
        try:
            action_result = future.result()
        except Exception as error:
            self.get_logger().error(f"Failed to receive MoveIt result: {error}")
            self._transition(self.FAILED)
            return

        result = action_result.result
        wall_time = time.monotonic() - self._goal_start_time
        phase = "Motion" if self.execute else "Planning"

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            # On failure MoveIt returns no executable trajectory, so the arm stays put.
            self.get_logger().error(
                f"{phase} to {self._current_target} failed: "
                f"error_code={result.error_code.val}, "
                f"message={result.error_code.message or 'no message'}, "
                f"wall_time={wall_time:.3f}s"
            )
            self._transition(self.FAILED)
            return

        self.get_logger().info(
            f"{phase} to {self._current_target} succeeded: "
            f"planning_time={result.planning_time:.3f}s, "
            f"wall_time={wall_time:.3f}s"
        )
        self._log_planned_trajectory(result)
        self._advance_sequence()

    def _advance_sequence(self) -> None:
        """Move to the next phase after a successful goal."""
        if self.state == self.REACHING:
            self._transition(self.RETURNING)
            self._send_joint_goal("home", self.HOME_JOINT_POSITIONS)
        elif self.state == self.RETURNING:
            self._transition(self.DONE)
            self.get_logger().info("Reach and return-home sequence complete.")


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
