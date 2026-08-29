"""Bring up the Phase-0 chain end to end, without hardware.

Every stage below has been validated on its own. This launch exists because
that is not the same claim: this project's last four integration failures were
each a pair of components that were individually correct and disagreed at the
seam -- a frozen clock offset, a reference level nobody sent, a fixture nobody
regenerated. Running the chain is how those are found.

    synthetic candidates -> markerless gate -> /target_object_pose
                                             \\
    simulated view commands ------------------> handoff controller
    simulated hand observations -------------/

`sim_intent_publisher` is deliberately absent: it reads single letters from
stdin (n / c / a / q), so it belongs in its own terminal where the operator can
drive the sequence. Start it after this launch is up.

MoveIt is also absent. The handoff controller still simulates its own motion
with timers, and `object1_demo`'s reaching coordinator is a one-shot
IDLE -> REACHING -> RETURNING -> DONE path rather than something that composes
with a state machine. Giving the controller a MoveIt backend that reuses that
package's verified goal construction is the next piece of work, and it should
be written against a chain that is known to talk to itself first.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    planning_frame = LaunchConfiguration("planning_frame")
    camera_frame = LaunchConfiguration("camera_frame")

    return LaunchDescription([
        DeclareLaunchArgument("planning_frame", default_value="world"),
        DeclareLaunchArgument(
            "camera_frame", default_value="stereo_left_optical"
        ),

        # A placeholder, and labelled as one wherever it appears. The real
        # camera-to-world extrinsic is an Objective 5 deliverable and does not
        # exist yet; Objective 4.2's connection to 4.1 is waiting on the same
        # number. Nothing measured may be derived from the poses this produces
        # -- it is here so the gate's TF lookup succeeds and the chain runs.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="placeholder_camera_extrinsic",
            arguments=[
                "--x", "0.4", "--y", "0.0", "--z", "0.6",
                "--roll", "0.0", "--pitch", "1.5708", "--yaw", "0.0",
                "--frame-id", planning_frame,
                "--child-frame-id", camera_frame,
            ],
        ),

        Node(
            package="markerless_object_perception",
            executable="synthetic_candidate_publisher",
            name="synthetic_candidates",
            parameters=[{"frame_id": camera_frame}],
            output="screen",
        ),

        Node(
            package="target_selector",
            executable="markerless_candidate_gate",
            name="markerless_candidate_gate",
            parameters=[{"planning_frame": planning_frame}],
            output="screen",
        ),

        Node(
            package="assistive_handoff",
            executable="sim_hand_publisher",
            name="sim_hand_publisher",
            output="screen",
        ),

        Node(
            package="assistive_handoff",
            executable="handoff_controller",
            name="handoff_controller",
            parameters=[{
                "proportional_search_available": True,
                "search_timeout_sec": 300.0,
            }],
            output="screen",
        ),
    ])
