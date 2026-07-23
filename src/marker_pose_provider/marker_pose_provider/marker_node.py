"""ArUco marker localization node.

Stateless perception: subscribes to the camera driver's /image_raw and
/camera_info, detects ArUco markers, and publishes every detection to
/detected_markers. Target selection (picking the single /target_object_pose)
happens in a downstream selector node, not here.

M3: estimate each marker's 6-DOF pose in the camera frame with
solvePnPGeneric(SOLVEPNP_IPPE_SQUARE), resolve the planar z-flip ambiguity
with a per-marker temporal consistency check, and publish a PoseArray.
"""

import math

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# OpenCV 4.6 ships the legacy function-style aruco API; the ArucoDetector
# class only exists from 4.7. Detection is isolated here so an upgrade
# touches one function.
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_marker_ids(gray_image):
    """Return (corners, ids) for all markers found in a grayscale image."""
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _rejected = cv2.aruco.detectMarkers(
        gray_image, ARUCO_DICTIONARY, parameters=parameters
    )
    return corners, ids


def marker_object_points(marker_length: float) -> np.ndarray:
    """3D corner coordinates in the marker frame, in the order ArUco reports
    image corners (top-left, top-right, bottom-right, bottom-left), which is
    also the order SOLVEPNP_IPPE_SQUARE requires."""
    half = marker_length / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):
    """Solve planar PnP for one marker; return a list of (rvec, tvec, error).

    SOLVEPNP_IPPE_SQUARE returns up to two solutions (the planar ambiguity),
    ordered by reprojection error. The caller resolves which one to trust.
    """
    object_points = marker_object_points(marker_length)
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    _n, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return [
        (rvecs[i], tvecs[i], float(errors[i]))
        for i in range(len(rvecs))
    ]


def rotation_angle_between(rvec_a, rvec_b) -> float:
    """Angle in radians of the relative rotation between two rotation vectors."""
    rot_a, _ = cv2.Rodrigues(rvec_a)
    rot_b, _ = cv2.Rodrigues(rvec_b)
    relative = rot_a.T @ rot_b
    cos_angle = (np.trace(relative) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def pick_consistent_solution(solutions, previous_rvec):
    """Pick the PnP solution closest to the previous frame's orientation.

    Without history, fall back to the lowest reprojection error (the first
    entry, as solvePnPGeneric orders solutions by error). This suppresses the
    z-flip jumps a near-frontal planar marker otherwise produces.
    """
    if previous_rvec is None or len(solutions) == 1:
        return solutions[0]
    return min(
        solutions,
        key=lambda s: rotation_angle_between(s[0], previous_rvec),
    )


def rvec_tvec_to_pose(rvec, tvec) -> Pose:
    """Convert an OpenCV rotation vector + translation into a ROS Pose."""
    rot, _ = cv2.Rodrigues(rvec)
    # Rotation matrix -> quaternion (w, x, y, z), standard trace method.
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
    pose = Pose()
    pose.position.x = float(tvec[0])
    pose.position.y = float(tvec[1])
    pose.position.z = float(tvec[2])
    pose.orientation.w = qw
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    return pose


class MarkerPoseNode(Node):
    """Detects ArUco markers and publishes all detections to /detected_markers."""

    def __init__(self) -> None:
        super().__init__("marker_pose_node")
        self.declare_parameter("marker_length", 0.051)
        self._bridge = CvBridge()
        self._camera_matrix = None
        self._dist_coeffs = None
        self._previous_rvecs = {}
        self._image_sub = self.create_subscription(
            Image, "/image_raw", self._on_image, 10
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_info", self._on_camera_info, 10
        )
        self._pose_pub = self.create_publisher(PoseArray, "/detected_markers", 10)
        self.get_logger().info("marker_pose_node started, waiting for /image_raw")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._camera_matrix is None:
            self.get_logger().info("received camera intrinsics from /camera_info")
        self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _on_image(self, msg: Image) -> None:
        if self._camera_matrix is None:
            self.get_logger().info(
                "waiting for /camera_info before estimating poses",
                throttle_duration_sec=2.0,
            )
            return
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_marker_ids(gray)
        if ids is None:
            return
        marker_length = (
            self.get_parameter("marker_length").get_parameter_value().double_value
        )
        pose_array = PoseArray()
        pose_array.header = msg.header
        seen_ids = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            solutions = estimate_marker_pose(
                marker_corners, marker_length, self._camera_matrix, self._dist_coeffs
            )
            rvec, tvec, _error = pick_consistent_solution(
                solutions, self._previous_rvecs.get(marker_id)
            )
            self._previous_rvecs[marker_id] = rvec
            pose_array.poses.append(rvec_tvec_to_pose(rvec, tvec))
            seen_ids.append(marker_id)
        self._pose_pub.publish(pose_array)
        first = pose_array.poses[0].position
        self.get_logger().info(
            f"markers {seen_ids} first at "
            f"({first.x:.3f}, {first.y:.3f}, {first.z:.3f}) m",
            throttle_duration_sec=1.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MarkerPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
