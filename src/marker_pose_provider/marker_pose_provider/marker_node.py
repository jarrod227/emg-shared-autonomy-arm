"""ArUco marker localization node.

Stateless perception: subscribes to the camera driver's /image_raw and
/camera_info, detects ArUco markers, and publishes every detection to
/detected_markers. Target selection (picking the single /target_object_pose)
happens in a downstream selector node, not here.

M1: subscribe /image_raw, detect markers, log detected IDs. Pose estimation
and /detected_markers publishing arrive in later milestones.
"""

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

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


class MarkerPoseNode(Node):
    """Detects ArUco markers and publishes all detections to /detected_markers."""

    def __init__(self) -> None:
        super().__init__("marker_pose_node")
        self._bridge = CvBridge()
        self._image_sub = self.create_subscription(
            Image, "/image_raw", self._on_image, 10
        )
        self.get_logger().info("marker_pose_node started, waiting for /image_raw")

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _corners, ids = detect_marker_ids(gray)
        if ids is None:
            self.get_logger().info(
                "no markers detected", throttle_duration_sec=2.0
            )
            return
        id_list = sorted(int(marker_id) for marker_id in ids.flatten())
        self.get_logger().info(
            f"detected marker ids: {id_list}", throttle_duration_sec=1.0
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
