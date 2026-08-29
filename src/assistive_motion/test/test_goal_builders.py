"""What every MoveIt goal this project sends must contain.

These were node methods in the Objective 1 coordinator and were exercised
only through it. Pulling them out is what makes the fields checkable on their
own, which matters because a second caller now sends the same goals and a
third -- the Objective 5 real arm -- will need the same semantics.
"""

import pytest

from assistive_motion.goal_builders import (
    HOME_JOINT_POSITIONS,
    JOINT_TOLERANCE_RAD,
    ORIENTATION_TOLERANCE_RAD,
    POSITION_TOLERANCE_M,
    PlanningSettings,
    build_joint_goal,
    build_pose_goal,
)


def settings(**overrides):
    values = dict(
        planning_group="panda_arm",
        planning_frame="world",
        end_effector_frame="panda_link8",
        allowed_planning_time=5.0,
        num_planning_attempts=10,
        velocity_scaling=0.1,
        acceleration_scaling=0.1,
        execute=True,
    )
    values.update(overrides)
    return PlanningSettings(**values)


def test_execute_false_plans_without_moving_the_arm():
    """plan_only is the inverse of execute, and getting it backwards either
    plans a motion nobody sees or runs one nobody asked for."""
    assert build_joint_goal(settings(execute=True), HOME_JOINT_POSITIONS) \
        .planning_options.plan_only is False
    assert build_joint_goal(settings(execute=False), HOME_JOINT_POSITIONS) \
        .planning_options.plan_only is True


def test_a_joint_goal_carries_every_joint_with_a_two_sided_tolerance():
    goal = build_joint_goal(settings(), HOME_JOINT_POSITIONS)

    constraints = goal.request.goal_constraints[0].joint_constraints
    assert [c.joint_name for c in constraints] == [
        name for name, _ in HOME_JOINT_POSITIONS
    ]
    assert [c.position for c in constraints] == [
        position for _, position in HOME_JOINT_POSITIONS
    ]
    for constraint in constraints:
        assert constraint.tolerance_above == JOINT_TOLERANCE_RAD
        assert constraint.tolerance_below == JOINT_TOLERANCE_RAD


def test_a_pose_goal_constrains_position_and_orientation_of_one_link():
    goal = build_pose_goal(
        settings(), (0.3, -0.1, 0.6), (0.0, 0.0, 0.0, 1.0)
    )

    constraints = goal.request.goal_constraints[0]
    position = constraints.position_constraints[0]
    orientation = constraints.orientation_constraints[0]

    assert position.link_name == orientation.link_name == "panda_link8"
    assert position.header.frame_id == orientation.header.frame_id == "world"
    assert list(position.constraint_region.primitives[0].dimensions) == [
        POSITION_TOLERANCE_M
    ]
    placed = position.constraint_region.primitive_poses[0].position
    assert (placed.x, placed.y, placed.z) == (0.3, -0.1, 0.6)
    # The region carries no rotation of its own; the orientation constraint is
    # what fixes the end effector, and duplicating it would let the two
    # disagree.
    assert position.constraint_region.primitive_poses[0].orientation.w == 1.0
    assert orientation.absolute_x_axis_tolerance == ORIENTATION_TOLERANCE_RAD


def test_a_pose_goal_refuses_a_missing_target_rather_than_sending_zeros():
    with pytest.raises(ValueError, match="position and an orientation"):
        build_pose_goal(settings(), None, (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="position and an orientation"):
        build_pose_goal(settings(), (0.3, -0.1, 0.6), None)


def test_the_home_pose_is_the_panda_ready_pose():
    # Shared so that every backend returns to the same place; a per-package
    # copy is how two of them end up meaning different things by "home".
    assert len(HOME_JOINT_POSITIONS) == 7
    assert HOME_JOINT_POSITIONS[0] == ("panda_joint1", 0.0)
    assert HOME_JOINT_POSITIONS[3] == ("panda_joint4", -2.356)
