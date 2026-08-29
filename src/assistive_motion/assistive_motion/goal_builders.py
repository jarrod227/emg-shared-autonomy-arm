"""MoveIt goal construction, shared rather than reimplemented per caller.

Lifted unchanged from the Objective 1 reaching coordinator on 2026-08-29, where
it was verified against the Panda MoveIt path and then needed by a second
caller. The handoff controller has to issue the same goals, and the Objective 5
real-arm backend will need the same semantics, so the alternative was three
copies free to drift -- which this project has already been bitten by in a
different form, when a conclusion lived in a document and not in the tool.

Nothing here touches ROS state. These are pure functions over a small settings
record, so every field a goal carries can be checked without a node, a planning
scene, or a running MoveIt.
"""

from dataclasses import dataclass

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


# The Panda "ready" pose. The arm starts here, so it is also the home pose.
HOME_JOINT_POSITIONS = (
    ("panda_joint1", 0.0),
    ("panda_joint2", -0.785),
    ("panda_joint3", 0.0),
    ("panda_joint4", -2.356),
    ("panda_joint5", 0.0),
    ("panda_joint6", 1.571),
    ("panda_joint7", 0.785),
)

POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.01
JOINT_TOLERANCE_RAD = 0.001


@dataclass(frozen=True)
class PlanningSettings:
    """Everything a goal needs that is not the target itself.

    Kept explicit rather than read from parameters here: the caller owns its
    own parameter declarations, and a builder that reached for them would tie
    the goal's content to one node's parameter names.
    """

    planning_group: str
    planning_frame: str
    end_effector_frame: str
    allowed_planning_time: float
    num_planning_attempts: int
    velocity_scaling: float
    acceleration_scaling: float
    execute: bool


def _base_goal(settings: PlanningSettings) -> MoveGroup.Goal:
    goal = MoveGroup.Goal()
    goal.request.group_name = settings.planning_group
    goal.request.allowed_planning_time = float(settings.allowed_planning_time)
    goal.request.num_planning_attempts = int(settings.num_planning_attempts)
    goal.request.max_velocity_scaling_factor = settings.velocity_scaling
    goal.request.max_acceleration_scaling_factor = settings.acceleration_scaling
    # plan_only is the inverse of execute, which is the field that decides
    # whether the arm actually moves. Getting it backwards plans a motion
    # nobody sees, or runs one nobody asked for.
    goal.planning_options.plan_only = not settings.execute
    return goal


def build_joint_goal(settings: PlanningSettings, joint_positions):
    """A goal that drives the arm to a joint configuration."""
    goal = _base_goal(settings)
    constraints = Constraints()
    for joint_name, position in joint_positions:
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = joint_name
        joint_constraint.position = position
        joint_constraint.tolerance_above = JOINT_TOLERANCE_RAD
        joint_constraint.tolerance_below = JOINT_TOLERANCE_RAD
        joint_constraint.weight = 1.0
        constraints.joint_constraints.append(joint_constraint)
    goal.request.goal_constraints.append(constraints)
    return goal


def build_pose_goal(settings: PlanningSettings, position, orientation):
    """A goal for one end-effector pose, in the planning frame."""
    if position is None or orientation is None:
        raise ValueError("a pose goal needs both a position and an orientation")

    goal = _base_goal(settings)

    position_constraint = PositionConstraint()
    position_constraint.header.frame_id = settings.planning_frame
    position_constraint.link_name = settings.end_effector_frame
    position_constraint.weight = 1.0

    tolerance_region = SolidPrimitive()
    tolerance_region.type = SolidPrimitive.SPHERE
    tolerance_region.dimensions = [POSITION_TOLERANCE_M]

    target_pose = Pose()
    (
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z,
    ) = position
    # The region is placed at the target and the constraint carries no
    # orientation of its own; the orientation constraint below is what fixes
    # the end effector's rotation.
    target_pose.orientation.w = 1.0
    position_constraint.constraint_region.primitives.append(tolerance_region)
    position_constraint.constraint_region.primitive_poses.append(target_pose)

    orientation_constraint = OrientationConstraint()
    orientation_constraint.header.frame_id = settings.planning_frame
    orientation_constraint.link_name = settings.end_effector_frame
    (
        orientation_constraint.orientation.x,
        orientation_constraint.orientation.y,
        orientation_constraint.orientation.z,
        orientation_constraint.orientation.w,
    ) = orientation
    orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    orientation_constraint.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(position_constraint)
    constraints.orientation_constraints.append(orientation_constraint)
    goal.request.goal_constraints.append(constraints)
    return goal
