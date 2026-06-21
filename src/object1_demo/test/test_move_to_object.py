"""Focused tests for the object1 reach/return-home logic.

These tests exercise the pure node logic (pose definitions, goal building,
state-machine transitions). They do not start MoveIt, so they need no
move_group server, no controllers, and no RViz. The node is constructed with
``auto_start=False`` so the reaching sequence does not fire during tests.
"""

import pytest
import rclpy
from rclpy.parameter import Parameter

from moveit_msgs.msg import MoveItErrorCodes

from object1_demo.move_to_object import MoveToObjectNode


@pytest.fixture(scope="module", autouse=True)
def rclpy_context():
    """Initialize and tear down rclpy once for the whole test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def make_node(execute=True):
    """Construct the node without auto-starting the reaching sequence."""
    overrides = [
        Parameter("auto_start", Parameter.Type.BOOL, False),
        Parameter("execute", Parameter.Type.BOOL, execute),
    ]
    return MoveToObjectNode(parameter_overrides=overrides)


def test_pose_definitions_have_seven_panda_joints():
    """object1 and home are both 7-joint Panda configurations."""
    for config in (
        MoveToObjectNode.OBJECT1_JOINT_POSITIONS,
        MoveToObjectNode.HOME_JOINT_POSITIONS,
    ):
        names = [name for name, _ in config]
        assert names == [f"panda_joint{i}" for i in range(1, 8)]

    # Home is the Panda "ready" pose; object1 differs so the arm visibly moves.
    assert MoveToObjectNode.HOME_JOINT_POSITIONS != MoveToObjectNode.OBJECT1_JOINT_POSITIONS


def test_build_joint_goal_plan_and_execute():
    """With execute=True the goal plans and executes (plan_only False)."""
    node = make_node(execute=True)
    try:
        goal = node._build_joint_goal(node.OBJECT1_JOINT_POSITIONS)

        assert goal.request.group_name == "panda_arm"
        assert goal.planning_options.plan_only is False

        constraints = goal.request.goal_constraints[0].joint_constraints
        assert len(constraints) == 7
        assert constraints[0].joint_name == "panda_joint1"
        # The constraint positions match the requested configuration.
        sent = [(c.joint_name, c.position) for c in constraints]
        assert sent == list(node.OBJECT1_JOINT_POSITIONS)
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
