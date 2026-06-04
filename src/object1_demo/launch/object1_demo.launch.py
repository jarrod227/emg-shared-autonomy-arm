"""Launch the object1 fixed-pose demo node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="object1_demo",
                executable="move_to_object",
                name="move_to_object",
                output="screen",
            )
        ]
    )
