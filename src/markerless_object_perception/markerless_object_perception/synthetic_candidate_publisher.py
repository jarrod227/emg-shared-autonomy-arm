"""
Publish synthetic mask/XYZ candidates through the Objective 3.2 contract.

This node verifies the ROS boundary without claiming live stereo input.  It
uses one static boolean mask and one aligned XYZ image, while every published
message receives a fresh ROS source timestamp.
"""

import math
import operator

from assistive_interfaces.msg import ObjectCandidateArray
from markerless_object_perception.candidate_builder import (
    CandidateBuilder,
    InstanceMaskDetection,
)
from markerless_object_perception.ros_adapter import (
    object_candidate_array_from_result,
)
import numpy as np
import rclpy
from rclpy.node import Node


_UINT32_MAX = 4_294_967_295


class SyntheticObjectCandidatePublisher(Node):
    """Publish one known localized object or a fresh empty observation."""

    def __init__(self, *, parameter_overrides=None):
        super().__init__(
            'synthetic_object_candidate_publisher',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()

        self._frame_id = _string_parameter(self, 'frame_id')
        candidate_topic = _string_parameter(self, 'candidate_topic')
        publish_rate_hz = _positive_parameter(self, 'publish_rate_hz')
        self._pair_skew_sec = _nonnegative_parameter(
            self,
            'pair_skew_sec',
        )
        self._simulate_no_detection = self.get_parameter(
            'simulate_no_detection'
        ).value
        if not isinstance(self._simulate_no_detection, bool):
            raise ValueError('simulate_no_detection must be a boolean')

        self._result = self._build_synthetic_result()
        self._publisher = self.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        self._timer = self.create_timer(
            1.0 / publish_rate_hz,
            self._publish,
        )
        mode = 'fresh empty observations' if self._simulate_no_detection else (
            'one synthetic localized candidate'
        )
        self.get_logger().info(
            f'synthetic Objective 3.2 publisher started: {mode}; '
            'no live stereo input is in use'
        )

    def _declare_parameters(self):
        self.declare_parameter('candidate_topic', '/object_candidates')
        self.declare_parameter('frame_id', 'stereo_left_optical')
        # 10 Hz, matching the live candidate path this node stands in for.
        # It defaulted to 5, which puts the frame gap exactly on the
        # markerless gate's max_frame_gap_sec of 0.2 s: measured 2026-08-29,
        # a chain driven at 5 Hz locked and expired a target ten times in
        # twelve seconds, and at 10 Hz locked once and held. A stand-in that
        # publishes at half the rate of the source exercises a condition the
        # real path does not produce, against a gate correctly sized for it.
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('pair_skew_sec', 0.005)
        self.declare_parameter('class_label', 'bottle')
        self.declare_parameter('class_confidence', 0.9)
        self.declare_parameter('track_id', 1)
        self.declare_parameter('object_point_x', 0.25)
        self.declare_parameter('object_point_y', 0.0)
        self.declare_parameter('object_point_z', 0.7)
        self.declare_parameter('simulate_no_detection', False)

    def _build_synthetic_result(self):
        xyz_points = np.full(
            (8, 8, 3),
            [
                _finite_parameter(self, 'object_point_x'),
                _finite_parameter(self, 'object_point_y'),
                _finite_parameter(self, 'object_point_z'),
            ],
            dtype=np.float64,
        )
        detections = ()
        if not self._simulate_no_detection:
            detections = (
                InstanceMaskDetection(
                    class_label=_string_parameter(self, 'class_label'),
                    confidence=_confidence_parameter(
                        self,
                        'class_confidence',
                    ),
                    track_id=_track_id_parameter(self, 'track_id'),
                    mask=np.ones((8, 8), dtype=bool),
                ),
            )

        result = CandidateBuilder().build(detections, xyz_points)
        if result.rejections:
            reasons = ', '.join(item.reason for item in result.rejections)
            raise ValueError(
                f'synthetic parameters produced rejected candidates: {reasons}'
            )
        return result

    def _publish(self):
        now = self.get_clock().now()
        message = object_candidate_array_from_result(
            self._result,
            self._frame_id,
            source_time_nanoseconds=now.nanoseconds,
            pair_skew_sec=self._pair_skew_sec,
        )
        self._publisher.publish(message)


def _string_parameter(node, name):
    value = node.get_parameter(name).value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _finite_parameter(node, name):
    try:
        value = float(node.get_parameter(name).value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive_parameter(node, name):
    value = _finite_parameter(node, name)
    if value <= 0.0:
        raise ValueError(f'{name} must be positive')
    return value


def _nonnegative_parameter(node, name):
    value = _finite_parameter(node, name)
    if value < 0.0:
        raise ValueError(f'{name} must be non-negative')
    return value


def _confidence_parameter(node, name):
    value = _nonnegative_parameter(node, name)
    if value > 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return value


def _track_id_parameter(node, name):
    raw_value = node.get_parameter(name).value
    try:
        value = operator.index(raw_value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if isinstance(raw_value, bool) or value < 0 or value > _UINT32_MAX:
        raise ValueError(f'{name} must fit a uint32')
    return value


def main(args=None):
    """Run the synthetic Objective 3.2 candidate publisher."""
    rclpy.init(args=args)
    node = SyntheticObjectCandidatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
