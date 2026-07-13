"""Focused tests for the object1 reach/return-home logic.

These tests exercise the pure node logic (pose definitions, goal building,
state-machine transitions). They do not start MoveIt, so they need no
move_group server, no controllers, and no RViz. The node is constructed with
``auto_start=False`` so the reaching sequence does not fire during tests.
"""

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import Parameter

from moveit_msgs.msg import MoveItErrorCodes
from visualization_msgs.msg import Marker

from object1_demo.move_to_object import MoveToObjectNode


@pytest.fixture(scope="module", autouse=True)
def rclpy_context():
    """Initialize and tear down rclpy once for the whole test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def make_target_pose(frame_id="world"):
    """Create the fixed object1 pose carried by the Objective 2 interface."""
    target_pose = PoseStamped()
    target_pose.header.frame_id = frame_id
    target_pose.pose.position.x = 0.106982
    target_pose.pose.position.y = 0.0
    target_pose.pose.position.z = 1.121022
    target_pose.pose.orientation.x = 0.653269
    target_pose.pose.orientation.y = -0.270440
    target_pose.pose.orientation.z = 0.653402
    target_pose.pose.orientation.w = -0.270496
    return target_pose


def make_node(execute=True, auto_start=False, with_target=True):
    """Construct the node and optionally provide its target pose."""
    overrides = [
        Parameter("auto_start", Parameter.Type.BOOL, auto_start),
        Parameter("execute", Parameter.Type.BOOL, execute),
    ]
    node = MoveToObjectNode(parameter_overrides=overrides)
    if with_target:
        node._on_target_pose(make_target_pose())
    return node


def test_target_pose_starts_sequence_once():
    """The first target starts the flow; later targets are ignored."""
    node = make_node(auto_start=True, with_target=False)
    try:
        starts = []
        node._start_sequence = lambda: starts.append("started")

        node._on_target_pose(make_target_pose())
        node._on_target_pose(make_target_pose(frame_id="another_frame"))

        assert starts == ["started"]
        assert node.planning_frame == "world"
        assert node.object1_position == pytest.approx((0.106982, 0.0, 1.121022))
    finally:
        node.destroy_node()


def test_home_definition_has_seven_panda_joints():
    """Home remains a 7-joint Panda configuration."""
    names = [name for name, _ in MoveToObjectNode.HOME_JOINT_POSITIONS]
    assert names == [f"panda_joint{i}" for i in range(1, 8)]


def test_build_joint_goal_plan_and_execute():
    """With execute=True the goal plans and executes (plan_only False)."""
    node = make_node(execute=True)
    try:
        goal = node._build_joint_goal(node.HOME_JOINT_POSITIONS)

        assert goal.request.group_name == "panda_arm"
        assert goal.request.max_velocity_scaling_factor == pytest.approx(0.2)
        assert goal.request.max_acceleration_scaling_factor == pytest.approx(0.2)
        assert goal.planning_options.plan_only is False

        constraints = goal.request.goal_constraints[0].joint_constraints
        assert len(constraints) == 7
        assert constraints[0].joint_name == "panda_joint1"
        # The constraint positions match the requested configuration.
        sent = [(c.joint_name, c.position) for c in constraints]
        assert sent == list(node.HOME_JOINT_POSITIONS)
    finally:
        node.destroy_node()


def test_build_object1_pose_goal():
    """object1 is expressed as position and orientation constraints."""
    node = make_node(execute=False)
    try:
        goal = node._build_pose_goal()
        constraints = goal.request.goal_constraints[0]

        position = constraints.position_constraints[0]
        assert position.header.frame_id == "world"
        assert position.link_name == "panda_link8"
        target = position.constraint_region.primitive_poses[0].position
        assert (target.x, target.y, target.z) == pytest.approx(
            (0.106982, 0.0, 1.121022)
        )
        assert position.constraint_region.primitives[0].dimensions == pytest.approx(
            [0.005]
        )

        orientation = constraints.orientation_constraints[0]
        assert orientation.header.frame_id == "world"
        assert orientation.link_name == "panda_link8"
        quaternion = orientation.orientation
        actual_orientation = (
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
        assert actual_orientation == pytest.approx(
            (0.653269, -0.270440, 0.653402, -0.270496)
        )
        assert goal.planning_options.plan_only is True
    finally:
        node.destroy_node()


def test_build_object1_marker():
    """The RViz marker represents the same fixed object1 target."""
    node = make_node()
    try:
        marker = node._build_object1_marker()

        assert marker.header.frame_id == "world"
        assert marker.ns == "object1"
        assert marker.id == 0
        assert marker.type == Marker.SPHERE
        assert marker.action == Marker.ADD
        position = marker.pose.position
        assert (position.x, position.y, position.z) == pytest.approx(
            (0.106982, 0.0, 1.121022)
        )
        assert (marker.scale.x, marker.scale.y, marker.scale.z) == pytest.approx(
            (0.10, 0.10, 0.10)
        )
        assert (
            marker.color.r,
            marker.color.g,
            marker.color.b,
            marker.color.a,
        ) == pytest.approx((1.0, 0.1, 0.1, 0.8))
    finally:
        node.destroy_node()


def test_build_joint_goal_plan_only():
    """With execute=False the goal only plans (plan_only True)."""
    node = make_node(execute=False)
    try:
        goal = node._build_joint_goal(node.HOME_JOINT_POSITIONS)
        assert goal.planning_options.plan_only is True
    finally:
        node.destroy_node()


def test_state_machine_reaches_then_returns_home():
    """A successful reach advances to RETURNING and sends the home goal."""
    node = make_node()
    try:
        sent = []
        node._send_joint_goal = lambda name, positions: sent.append((name, positions))

        # Pretend we just succeeded reaching object1.
        node.state = MoveToObjectNode.REACHING
        node._advance_sequence()

        assert node.state == MoveToObjectNode.RETURNING
        assert sent == [("home", MoveToObjectNode.HOME_JOINT_POSITIONS)]

        # Pretend the return-home goal also succeeded.
        node._advance_sequence()
        assert node.state == MoveToObjectNode.DONE
        # No further goal is sent once the sequence is done.
        assert len(sent) == 1
    finally:
        node.destroy_node()


def test_failed_result_stops_the_sequence():
    """A non-success result transitions to FAILED and sends no further goal."""
    node = make_node()
    try:
        sent = []
        node._send_joint_goal = lambda name, positions: sent.append((name, positions))
        node._current_target = "object1"
        node.state = MoveToObjectNode.REACHING

        # Build a fake action result carrying a failure error code.
        class _FakeResult:
            class result:
                class error_code:
                    val = MoveItErrorCodes.PLANNING_FAILED
                    message = "fake failure"

                class planned_trajectory:
                    class joint_trajectory:
                        points = []

        class _FakeFuture:
            def result(self):
                return _FakeResult()

        node._on_result(_FakeFuture())

        assert node.state == MoveToObjectNode.FAILED
        assert sent == []
    finally:
        node.destroy_node()
