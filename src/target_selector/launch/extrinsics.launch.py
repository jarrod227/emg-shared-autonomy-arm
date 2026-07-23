"""Publish the nominal camera-to-planning-frame extrinsic as a static TF.

Reads config/extrinsics.yaml (the single source of truth for the placeholder
values) and starts a static_transform_publisher so downstream nodes can look
up marker poses in the planning frame. See the yaml header for why these
values are invented in simulation.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_path = os.path.join(
        get_package_share_directory("target_selector"),
        "config",
        "extrinsics.yaml",
    )
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)

    translation = cfg["translation"]
    rotation = cfg["rotation_rpy"]
    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_extrinsic",
                arguments=[
                    "--x", str(translation["x"]),
                    "--y", str(translation["y"]),
                    "--z", str(translation["z"]),
                    "--roll", str(rotation["roll"]),
                    "--pitch", str(rotation["pitch"]),
                    "--yaw", str(rotation["yaw"]),
                    "--frame-id", cfg["parent_frame"],
                    "--child-frame-id", cfg["child_frame"],
                ],
            ),
        ]
    )
