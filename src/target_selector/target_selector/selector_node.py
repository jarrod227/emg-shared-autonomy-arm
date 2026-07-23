"""Target selector node.

Subscribes to /detected_markers (camera-frame poses), selects one marker,
applies the marker-to-grasp offset, transforms it into the planning frame,
and publishes the single /target_object_pose the reaching coordinator
consumes. This is the layer where the EMG intent signal will later choose
which marker to target.

M4.5 adds the marker-to-grasp offset math (pure functions, unit tested).
Selection and the TF transform to the planning frame land in M5.
"""

import math

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose
from rclpy.node import Node


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll/pitch/yaw (radians, XYZ order)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from a quaternion (x, y, z, w)."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(rot: np.ndarray):
    """Quaternion (x, y, z, w) from a 3x3 rotation matrix, trace method."""
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """4x4 homogeneous transform from a ROS Pose."""
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    """ROS Pose from a 4x4 homogeneous transform."""
    pose = Pose()
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    qx, qy, qz, qw = matrix_to_quaternion(matrix[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def offset_to_matrix(offset: dict) -> np.ndarray:
    """4x4 transform from a {translation, rotation_rpy} offset dict."""
    translation = offset["translation"]
    rotation = offset["rotation_rpy"]
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_to_matrix(
        rotation["roll"], rotation["pitch"], rotation["yaw"]
    )
    matrix[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return matrix


def apply_grasp_offset(marker_pose: Pose, offset: dict) -> Pose:
    """Compose a marker pose with a grasp offset expressed in the marker frame.

    The offset is right-multiplied (T_grasp = T_marker * T_offset) so it is
    interpreted in the marker's local frame and follows the marker's
    orientation, not the camera's.
    """
    grasp_matrix = pose_to_matrix(marker_pose) @ offset_to_matrix(offset)
    return matrix_to_pose(grasp_matrix)


def load_grasp_offsets(path: str) -> dict:
    """Load the grasp-offset config: {'default': offset, 'markers': {id: offset}}."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    markers = {int(k): v for k, v in (cfg.get("markers") or {}).items()}
    return {"default": cfg["default"], "markers": markers}


def offset_for_marker(offsets: dict, marker_id: int) -> dict:
    """Return the per-marker offset, falling back to 'default'."""
    return offsets["markers"].get(marker_id, offsets["default"])


class SelectorNode(Node):
    """Turns detected markers into one /target_object_pose in the planning frame."""

    def __init__(self) -> None:
        super().__init__("target_selector")
        self.get_logger().info("target_selector started (M4.5: offset math ready)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
