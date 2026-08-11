"""Launch direct-capture DECXIN hand observation without a camera driver."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the single-node direct-capture hand-observation graph."""
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("stereo_hand_observer"),
            "config",
            "decxin_direct_hand.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="DECXIN direct-capture hand-observer parameters",
            ),
            DeclareLaunchArgument(
                "model_path",
                description="MediaPipe hand_landmarker.task file",
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
