"""Launch the camera driver and the ArUco marker detection node together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    video_device = LaunchConfiguration("video_device")
    camera_info_path = os.path.join(
        get_package_share_directory("marker_pose_provider"),
        "config",
        "camera_info.yaml",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "video_device",
                default_value="/dev/video0",
                description="V4L2 camera device path",
            ),
            Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                name="camera",
                parameters=[
                    {
                        "video_device": video_device,
                        "camera_info_url": "file://" + camera_info_path,
                    }
                ],
            ),
            Node(
                package="marker_pose_provider",
                executable="marker_node",
                name="marker_pose_node",
            ),
        ]
    )
