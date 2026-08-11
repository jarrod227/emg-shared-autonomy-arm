"""Launch DECXIN capture and atomic-composite hand observation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the single-callback DECXIN live hand-observation graph."""
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("stereo_hand_observer"),
            "config",
            "decxin_atomic_hand.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="DECXIN atomic hand-observer parameter file",
            ),
            DeclareLaunchArgument(
                "model_path",
                description="MediaPipe hand_landmarker.task file",
            ),
            Node(
                package="gscam",
                executable="gscam_node",
                namespace="stereo/composite",
                name="decxin_gscam",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="stereo_hand_observer",
                executable="live_observer",
                name="live_stereo_hand_observer",
                output="screen",
                parameters=[
                    config_file,
                    {"model_path": model_path},
                ],
            ),
        ]
    )
