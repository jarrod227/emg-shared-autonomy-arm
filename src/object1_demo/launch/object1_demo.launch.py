"""Launch the fixed-pose provider and object1 reaching coordinator."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="object1_demo",
                executable="fixed_pose_publisher",
                name="fixed_pose_publisher",
                output="screen",
            ),
            Node(
                package="object1_demo",
                executable="move_to_object",
                name="move_to_object",
                output="screen",
            )
        ]
    )
