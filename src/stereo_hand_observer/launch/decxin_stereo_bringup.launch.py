"""Launch DECXIN capture, splitter-side rectification, and stereo depth."""

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
                package="stereo_image_proc",
                executable="disparity_node",
                namespace="stereo",
                name="disparity_node",
                output="screen",
                parameters=[
                    {
                        "approximate_sync": False,
                    }
                ],
            ),
            Node(
                package="stereo_image_proc",
                executable="point_cloud_node",
                namespace="stereo",
                name="point_cloud_node",
                output="screen",
                parameters=[
                    {
                        "approximate_sync": False,
                        "avoid_point_cloud_padding": True,
                        "use_color": False,
                    }
                ],
                remappings=[
                    ("left/image_rect_color", "left/image_rect"),
                ],
            ),
        ]
    )
