"""M6 accuracy probe: collect detection statistics at a known distance.

Run the camera + marker pipeline, place a marker at a tape-measured
distance from the lens, then run this node for a fixed duration:

    ros2 run marker_pose_provider accuracy_probe --ros-args \
        -p nominal_distance:=0.5 -p duration:=10.0

It reports, for the camera frame only (the honest part of the error
budget — the world-frame extrinsic is a simulation placeholder):
  - detection success rate  (PoseArray msgs / image frames)
  - z mean/std and error vs the nominal tape distance
  - x/y std (repeatability; no ground truth for lateral position)
  - rotation spread around the first sample (repeatability, not accuracy)
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image


def quaternion_angle(q1, q2) -> float:
    """Angle in radians between two quaternions given as (x, y, z, w)."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class AccuracyProbe(Node):
    """Collects /detected_markers statistics for a fixed duration."""

    def __init__(self) -> None:
        super().__init__("accuracy_probe")
        self.declare_parameter("nominal_distance", 0.0)
        self.declare_parameter("duration", 10.0)
        self._image_count = 0
        self._positions = []
        self._quaternions = []
        self.create_subscription(Image, "/image_raw", self._on_image, 10)
        self.create_subscription(
            PoseArray, "/detected_markers", self._on_markers, 10
        )
        duration = self.get_parameter("duration").value
        self._timer = self.create_timer(duration, self._finish)
        self.get_logger().info(
            f"collecting for {duration:.0f}s "
            f"(nominal distance "
            f"{self.get_parameter('nominal_distance').value:.3f} m)"
        )

    def _on_image(self, _msg: Image) -> None:
        self._image_count += 1

    def _on_markers(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        pose = msg.poses[0]
        self._positions.append(
            (pose.position.x, pose.position.y, pose.position.z)
        )
        self._quaternions.append(
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        )

    def _finish(self) -> None:
        self._timer.cancel()
        nominal = self.get_parameter("nominal_distance").value
        n_det = len(self._positions)
        n_img = self._image_count
        if n_img == 0:
            self.get_logger().error("no images received — is the camera running?")
            raise SystemExit(1)
        if n_det == 0:
            self.get_logger().error(
                f"{n_img} images but zero detections — is the marker in view?"
            )
            raise SystemExit(1)

        positions = np.array(self._positions)
        z = positions[:, 2]
        # The tape measures the straight line lens -> marker centre, which is
        # the Euclidean norm, not the optical-axis z component. Compare tape
        # against the norm; z alone under-reads whenever the marker is
        # off-centre in the image.
        norms = np.linalg.norm(positions, axis=1)
        rot_spread = [
            quaternion_angle(self._quaternions[0], q) for q in self._quaternions
        ]

        lines = [
            "==== accuracy probe result (camera frame) ====",
            f"nominal distance : {nominal:.3f} m",
            f"frames           : {n_img} images, {n_det} detections "
            f"({100.0 * n_det / n_img:.1f}% success)",
            f"position mean    : x {positions[:, 0].mean():+.4f}, "
            f"y {positions[:, 1].mean():+.4f}, z {z.mean():.4f} m",
            f"euclidean dist   : mean {norms.mean():.4f} m, "
            f"std {norms.std() * 1000:.2f} mm",
            f"dist error vs tape: {(norms.mean() - nominal) * 1000:+.1f} mm",
            f"x std / y std    : {positions[:, 0].std() * 1000:.2f} / "
            f"{positions[:, 1].std() * 1000:.2f} mm (repeatability)",
            f"rotation spread  : std {math.degrees(np.std(rot_spread)):.2f} deg "
            "vs first sample (repeatability, not accuracy)",
        ]
        for line in lines:
            self.get_logger().info(line)
        raise SystemExit(0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AccuracyProbe()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
