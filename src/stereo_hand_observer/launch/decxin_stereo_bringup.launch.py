"""Launch DECXIN capture, stereo splitting, and image rectification."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the calibrated raw and rectified stereo camera graph."""
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("stereo_hand_observer"),
            "config",
            "decxin_stereo.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="DECXIN capture and splitter parameter file",
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
                executable="composite_stereo_splitter",
                name="composite_stereo_splitter",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="image_proc",
                executable="rectify_node",
                namespace="stereo/left",
                name="rectify",
                output="screen",
                remappings=[
                    ("image", "image_raw"),
                    ("camera_info", "camera_info"),
                    ("image_rect", "image_rect"),
                ],
            ),
            Node(
                package="image_proc",
                executable="rectify_node",
                namespace="stereo/right",
                name="rectify",
                output="screen",
                remappings=[
                    ("image", "image_raw"),
                    ("camera_info", "camera_info"),
                    ("image_rect", "image_rect"),
                ],
            ),
        ]
    )
