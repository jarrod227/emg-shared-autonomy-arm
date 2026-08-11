"""ROS 2 harness that feeds synthetic keypoints through the 4.2 pipeline.

This node is the software-first test source, not the final live-camera
observer.  It publishes the existing HandObservation contract so the
completed Objective 4.1 controller can consume 4.2 output unchanged.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node

from assistive_interfaces.msg import HandObservation
from stereo_hand_observer.geometry import project_point
from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointSet,
)
from stereo_hand_observer.ros_adapter import (
    hand_observation_from_result,
)


# Palm knuckles spread around the configured hand point so that each axis
# median returns that point exactly, matching the live multi-knuckle path.
SYNTHETIC_KNUCKLE_OFFSETS = {
    5: np.array([-0.03, 0.0, 0.0]),
    9: np.array([-0.01, 0.01, 0.0]),
    13: np.array([0.01, -0.01, 0.0]),
    17: np.array([0.03, 0.0, 0.0]),
}


def _finite_parameter(node, name):
    value = float(node.get_parameter(name).value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_parameter(node, name):
    value = _finite_parameter(node, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def rectified_stereo_model(fx, fy, cx, cy, baseline_m):
    """Build projection and fundamental matrices for horizontal stereo."""
    intrinsics = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )
    left_projection = intrinsics @ np.hstack(
        (np.eye(3), np.zeros((3, 1)))
    )
    right_projection = intrinsics @ np.hstack(
        (
            np.eye(3),
            np.array([[-baseline_m], [0.0], [0.0]]),
        )
    )
    fundamental_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return left_projection, right_projection, fundamental_matrix


class SyntheticStereoHandObserver(Node):
    """Publish quality-gated HandObservation from synthetic stereo pixels."""

    def __init__(self):
        super().__init__("synthetic_stereo_hand_observer")
        self._declare_parameters()

        frame_id = self.get_parameter("frame_id").value
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame_id must be a non-empty string")
        self._frame_id = frame_id

        publish_rate_hz = _positive_parameter(self, "publish_rate_hz")
        fx = _positive_parameter(self, "focal_length_x_px")
        fy = _positive_parameter(self, "focal_length_y_px")
        cx = _finite_parameter(self, "principal_point_x_px")
        cy = _finite_parameter(self, "principal_point_y_px")
        baseline_m = _positive_parameter(self, "baseline_m")
        model = rectified_stereo_model(fx, fy, cx, cy, baseline_m)
        self._left_projection, self._right_projection, fundamental = model

        self._hand_point = np.array(
            [
                _finite_parameter(self, "hand_point_x"),
                _finite_parameter(self, "hand_point_y"),
                _finite_parameter(self, "hand_point_z"),
            ]
        )
        if self._hand_point[2] <= 0.0:
            raise ValueError("hand_point_z must be greater than zero")

        delivery_volume = DeliveryVolume(
            center=(
                _finite_parameter(self, "delivery_center_x"),
                _finite_parameter(self, "delivery_center_y"),
                _finite_parameter(self, "delivery_center_z"),
            ),
            radius_m=_positive_parameter(self, "delivery_radius_m"),
        )
        gate_config = StabilityGateConfig(
            required_frames=self.get_parameter(
                "required_stable_frames"
            ).value,
            min_confidence=_finite_parameter(self, "min_confidence"),
            max_pair_skew_sec=_finite_parameter(
                self,
                "max_pair_skew_sec",
            ),
            max_reprojection_error_px=_finite_parameter(
                self,
                "max_reprojection_error_px",
            ),
            max_age_sec=_positive_parameter(self, "max_age_sec"),
            max_point_step_m=_finite_parameter(
                self,
                "max_point_step_m",
            ),
        )
        self._pipeline = StereoHandPipeline(
            self._left_projection,
            self._right_projection,
            fundamental,
            delivery_volume,
            gate_config=gate_config,
            max_epipolar_error_px=_finite_parameter(
                self,
                "max_epipolar_error_px",
            ),
        )

        self._simulated_pair_skew_sec = _finite_parameter(
            self,
            "simulated_pair_skew_sec",
        )
        if self._simulated_pair_skew_sec < 0.0:
            raise ValueError("simulated_pair_skew_sec must be non-negative")
        self._simulated_confidence = _finite_parameter(
            self,
            "simulated_confidence",
        )
        if not 0.0 <= self._simulated_confidence <= 1.0:
            raise ValueError("simulated_confidence must be in [0, 1]")
        self._right_vertical_error_px = _finite_parameter(
            self,
            "simulated_right_vertical_error_px",
        )
        self._simulate_missing_keypoint = self.get_parameter(
            "simulate_missing_keypoint"
        ).value

        self._publisher = self.create_publisher(
            HandObservation,
            "/hand_observation",
            10,
        )
        self._last_reason = None
        self._timer = self.create_timer(
            1.0 / publish_rate_hz,
            self._publish,
        )
        self.get_logger().info(
            "synthetic Objective 4.2 stereo observer started; "
            "no live cameras are in use"
        )

    def _declare_parameters(self):
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("focal_length_x_px", 600.0)
        self.declare_parameter("focal_length_y_px", 600.0)
        self.declare_parameter("principal_point_x_px", 320.0)
        self.declare_parameter("principal_point_y_px", 240.0)
        self.declare_parameter("baseline_m", 0.12)
        self.declare_parameter("hand_point_x", 0.4)
        self.declare_parameter("hand_point_y", 0.3)
        self.declare_parameter("hand_point_z", 1.0)
        self.declare_parameter("delivery_center_x", 0.4)
        self.declare_parameter("delivery_center_y", 0.3)
        self.declare_parameter("delivery_center_z", 1.0)
        self.declare_parameter("delivery_radius_m", 0.5)
        self.declare_parameter("required_stable_frames", 3)
        self.declare_parameter("min_confidence", 0.7)
        self.declare_parameter("max_pair_skew_sec", 0.02)
        self.declare_parameter("max_epipolar_error_px", 1.5)
        self.declare_parameter("max_reprojection_error_px", 1.5)
        self.declare_parameter("max_age_sec", 0.2)
        self.declare_parameter("max_point_step_m", 0.05)
        self.declare_parameter("simulated_pair_skew_sec", 0.01)
        self.declare_parameter("simulated_confidence", 0.9)
        self.declare_parameter("simulated_right_vertical_error_px", 0.0)
        self.declare_parameter("simulate_missing_keypoint", False)

    def _synthetic_keypoint_set(self, now_sec):
        left_pixels = {}
        right_pixels = {}
        for index, offset in SYNTHETIC_KNUCKLE_OFFSETS.items():
            point = self._hand_point + offset
            left_pixels[index] = tuple(
                project_point(self._left_projection, point)
            )
            right_pixel = project_point(self._right_projection, point)
            right_pixel[1] += self._right_vertical_error_px
            right_pixels[index] = tuple(right_pixel)
        if self._simulate_missing_keypoint:
            right_pixels = None

        left_time = max(0.0, now_sec - self._simulated_pair_skew_sec)
        return StereoKeypointSet(
            left_pixels=left_pixels,
            right_pixels=right_pixels,
            left_source_time_sec=left_time,
            right_source_time_sec=now_sec,
            left_confidence=self._simulated_confidence,
            right_confidence=self._simulated_confidence,
        )

    def _publish(self):
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9
        result = self._pipeline.process_set(
            self._synthetic_keypoint_set(now_sec),
            now_sec,
        )
        message = hand_observation_from_result(result, self._frame_id)
        self._publisher.publish(message)
        if result.reason != self._last_reason:
            self.get_logger().info(
                f"hand observation: valid={result.valid}, "
                f"stable_frames={result.stable_frames}, "
                f"reason={result.reason}"
            )
            self._last_reason = result.reason


def main(args=None):
    """Run the synthetic Objective 4.2 hand observer."""
    rclpy.init(args=args)
    node = SyntheticStereoHandObserver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
