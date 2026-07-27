"""Live rectified-stereo ROS 2 adapter for Objective 4.2."""

import math
import operator
import time

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from assistive_interfaces.msg import HandObservation
from stereo_hand_observer.geometry import fundamental_from_projections
from stereo_hand_observer.keypoint_detector import (
    MediaPipeHandKeypointDetector,
)
from stereo_hand_observer.live_processor import StereoFrameProcessor
from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import PipelineResult, StereoHandPipeline
from stereo_hand_observer.ros_adapter import hand_observation_from_result


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


def _integer_parameter(node, name, minimum=1):
    value = node.get_parameter(name).value
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string_parameter(node, name):
    value = node.get_parameter(name).value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _stamp_seconds(message):
    return _stamp_nanoseconds(message) / 1e9


def _stamp_nanoseconds(message):
    return (
        message.header.stamp.sec * 1_000_000_000
        + message.header.stamp.nanosec
    )


def _projection_matrix(message, name):
    projection = np.asarray(message.p, dtype=np.float64)
    if projection.shape != (12,) or not np.all(np.isfinite(projection)):
        raise ValueError(f"{name}.p must contain 12 finite values")
    projection = projection.reshape(3, 4)
    if projection[0, 0] <= 0.0 or projection[1, 1] <= 0.0:
        raise ValueError(f"{name}.p must contain positive focal lengths")
    return projection


class LiveStereoHandObserver(Node):
    """Publish fail-closed hand observations from two rectified images."""

    def __init__(self, detector_factory=None, **node_kwargs):
        super().__init__("live_stereo_hand_observer", **node_kwargs)
        self._declare_parameters()
        self._bridge = CvBridge()

        self._fallback_frame_id = _string_parameter(
            self,
            "fallback_frame_id",
        )
        self._gate_config = StabilityGateConfig(
            required_frames=_integer_parameter(
                self,
                "required_stable_frames",
            ),
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
        self._delivery_volume = DeliveryVolume(
            center=(
                _finite_parameter(self, "delivery_center_x"),
                _finite_parameter(self, "delivery_center_y"),
                _finite_parameter(self, "delivery_center_z"),
            ),
            radius_m=_positive_parameter(self, "delivery_radius_m"),
        )
        self._max_epipolar_error_px = _finite_parameter(
            self,
            "max_epipolar_error_px",
        )
        if self._max_epipolar_error_px < 0.0:
            raise ValueError(
                "max_epipolar_error_px must be non-negative"
            )

        sync_slop_sec = _positive_parameter(self, "sync_slop_sec")
        if sync_slop_sec < self._gate_config.max_pair_skew_sec:
            raise ValueError(
                "sync_slop_sec must be at least max_pair_skew_sec"
            )
        sync_queue_size = _integer_parameter(self, "sync_queue_size")
        self._watchdog_timeout_sec = _positive_parameter(
            self,
            "watchdog_timeout_sec",
        )

        if detector_factory is None:
            detector_factory = self._mediapipe_detector_factory()
        self._detectors = (
            detector_factory("left"),
            detector_factory("right"),
        )
        for detector in self._detectors:
            if not callable(getattr(detector, "detect", None)):
                raise TypeError(
                    "detector_factory must return objects with detect()"
                )

        self._publisher = self.create_publisher(
            HandObservation,
            _string_parameter(self, "observation_topic"),
            10,
        )
        self._left_info = None
        self._right_info = None
        self._calibration_signature = None
        self._calibration_size = None
        self._pipeline = None
        self._processor = None
        self._output_frame_id = self._fallback_frame_id
        self._last_reason = None

        self._latest_source_nanoseconds = {"left": None, "right": None}
        self._latest_arrivals = {"left": None, "right": None}
        self._last_pair_arrival = None

        left_info_topic = _string_parameter(self, "left_camera_info_topic")
        right_info_topic = _string_parameter(
            self,
            "right_camera_info_topic",
        )
        self._left_info_subscription = self.create_subscription(
            CameraInfo,
            left_info_topic,
            self._on_left_camera_info,
            qos_profile_sensor_data,
        )
        self._right_info_subscription = self.create_subscription(
            CameraInfo,
            right_info_topic,
            self._on_right_camera_info,
            qos_profile_sensor_data,
        )

        self._left_image_subscriber = message_filters.Subscriber(
            self,
            Image,
            _string_parameter(self, "left_image_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        self._right_image_subscriber = message_filters.Subscriber(
            self,
            Image,
            _string_parameter(self, "right_image_topic"),
            qos_profile=qos_profile_sensor_data,
        )
        self._left_image_subscriber.registerCallback(
            self._record_left_image
        )
        self._right_image_subscriber.registerCallback(
            self._record_right_image
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._left_image_subscriber, self._right_image_subscriber],
            queue_size=sync_queue_size,
            slop=sync_slop_sec,
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._on_image_pair)

        watchdog_period = min(0.1, self._watchdog_timeout_sec / 2.0)
        self._watchdog = self.create_timer(
            watchdog_period,
            self._on_watchdog,
        )
        self.get_logger().info(
            "live stereo hand observer started; waiting for rectified "
            "images and CameraInfo"
        )

    def _declare_parameters(self):
        self.declare_parameter(
            "left_image_topic",
            "/stereo/left/image_rect",
        )
        self.declare_parameter(
            "right_image_topic",
            "/stereo/right/image_rect",
        )
        self.declare_parameter(
            "left_camera_info_topic",
            "/stereo/left/camera_info",
        )
        self.declare_parameter(
            "right_camera_info_topic",
            "/stereo/right/camera_info",
        )
        self.declare_parameter("observation_topic", "/hand_observation")
        self.declare_parameter(
            "fallback_frame_id",
            "stereo_left_optical_frame",
        )
        self.declare_parameter("model_path", "")
        self.declare_parameter("landmark_index", 9)
        self.declare_parameter("min_detection_confidence", 0.7)
        self.declare_parameter("min_hand_presence_confidence", 0.7)
        self.declare_parameter("min_tracking_confidence", 0.7)
        self.declare_parameter("require_handedness_match", True)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.1)
        self.declare_parameter("watchdog_timeout_sec", 0.25)
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

    def _mediapipe_detector_factory(self):
        model_path = self.get_parameter("model_path").value
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(
                "model_path must name a MediaPipe hand-landmarker model"
            )
        options = {
            "landmark_index": _integer_parameter(
                self,
                "landmark_index",
                minimum=0,
            ),
            "min_detection_confidence": _finite_parameter(
                self,
                "min_detection_confidence",
            ),
            "min_hand_presence_confidence": _finite_parameter(
                self,
                "min_hand_presence_confidence",
            ),
            "min_tracking_confidence": _finite_parameter(
                self,
                "min_tracking_confidence",
            ),
        }

        def create_detector(_side):
            return MediaPipeHandKeypointDetector(
                model_path.strip(),
                **options,
            )

        return create_detector

    def _on_left_camera_info(self, message):
        self._left_info = message
        self._refresh_calibration()

    def _on_right_camera_info(self, message):
        self._right_info = message
        self._refresh_calibration()

    def _refresh_calibration(self):
        if self._left_info is None or self._right_info is None:
            return
        signature = (
            tuple(self._left_info.p),
            self._left_info.width,
            self._left_info.height,
            self._left_info.header.frame_id,
            tuple(self._right_info.p),
            self._right_info.width,
            self._right_info.height,
            self._right_info.header.frame_id,
        )
        if signature == self._calibration_signature:
            return

        self._calibration_signature = signature
        self._pipeline = None
        self._processor = None
        try:
            left_projection = _projection_matrix(
                self._left_info,
                "left CameraInfo",
            )
            right_projection = _projection_matrix(
                self._right_info,
                "right CameraInfo",
            )
            left_size = (
                int(self._left_info.width),
                int(self._left_info.height),
            )
            right_size = (
                int(self._right_info.width),
                int(self._right_info.height),
            )
            if min(left_size + right_size) <= 0 or left_size != right_size:
                raise ValueError(
                    "left and right CameraInfo must have one positive size"
                )
            frame_id = self._left_info.header.frame_id
            if not frame_id:
                raise ValueError("left CameraInfo frame_id must not be empty")
            fundamental = fundamental_from_projections(
                left_projection,
                right_projection,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f"stereo calibration rejected: {error}"
            )
            return

        self._pipeline = StereoHandPipeline(
            left_projection,
            right_projection,
            fundamental,
            self._delivery_volume,
            gate_config=self._gate_config,
            max_epipolar_error_px=self._max_epipolar_error_px,
        )
        self._processor = StereoFrameProcessor(
            self._pipeline,
            self._detectors[0],
            self._detectors[1],
            require_handedness_match=self.get_parameter(
                "require_handedness_match"
            ).value,
        )
        self._calibration_size = left_size
        self._output_frame_id = frame_id
        self.get_logger().info(
            f"stereo calibration accepted in frame {frame_id}; "
            "stability history reset"
        )

    def _record_left_image(self, message):
        self._record_image("left", message)

    def _record_right_image(self, message):
        self._record_image("right", message)

    def _record_image(self, side, message):
        self._latest_source_nanoseconds[side] = _stamp_nanoseconds(message)
        self._latest_arrivals[side] = time.monotonic()

    def _on_image_pair(self, left_message, right_message):
        self._last_pair_arrival = time.monotonic()
        left_nanoseconds = _stamp_nanoseconds(left_message)
        right_nanoseconds = _stamp_nanoseconds(right_message)
        left_time = left_nanoseconds / 1e9
        right_time = right_nanoseconds / 1e9
        if self._processor is None:
            self._publish_invalid(
                "missing_or_invalid_camera_info",
                left_nanoseconds,
                right_nanoseconds,
            )
            return

        if (
            (left_message.width, left_message.height)
            != self._calibration_size
            or (right_message.width, right_message.height)
            != self._calibration_size
        ):
            self._publish_invalid(
                "image_size_mismatch",
                left_nanoseconds,
                right_nanoseconds,
            )
            return
        expected_frames = (
            self._left_info.header.frame_id,
            self._right_info.header.frame_id,
        )
        image_frames = (
            left_message.header.frame_id,
            right_message.header.frame_id,
        )
        if any(
            image_frame and image_frame != expected_frame
            for image_frame, expected_frame in zip(
                image_frames,
                expected_frames,
            )
        ):
            self._publish_invalid(
                "image_frame_mismatch",
                left_nanoseconds,
                right_nanoseconds,
            )
            return

        try:
            left_rgb = self._bridge.imgmsg_to_cv2(
                left_message,
                desired_encoding="rgb8",
            )
            right_rgb = self._bridge.imgmsg_to_cv2(
                right_message,
                desired_encoding="rgb8",
            )
        except Exception:
            self._publish_invalid(
                "image_conversion_error",
                left_nanoseconds,
                right_nanoseconds,
            )
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        try:
            result = self._processor.process(
                left_rgb,
                right_rgb,
                left_source_time_sec=left_time,
                right_source_time_sec=right_time,
                now_sec=now_sec,
            )
        except Exception:
            self._publish_invalid(
                "processing_error",
                left_nanoseconds,
                right_nanoseconds,
            )
            return
        self._publish_result(
            result,
            source_time_nanoseconds=min(
                left_nanoseconds,
                right_nanoseconds,
            ),
        )

    def _on_watchdog(self):
        if all(
            arrival is None for arrival in self._latest_arrivals.values()
        ):
            return
        now = time.monotonic()
        sides_recent = all(
            arrival is not None
            and now - arrival <= self._watchdog_timeout_sec
            for arrival in self._latest_arrivals.values()
        )
        pair_recent = (
            self._last_pair_arrival is not None
            and now - self._last_pair_arrival
            <= self._watchdog_timeout_sec
        )
        if sides_recent and pair_recent:
            return
        reason = "unpaired_images" if sides_recent else "input_timeout"
        left_nanoseconds = self._latest_source_nanoseconds["left"]
        right_nanoseconds = self._latest_source_nanoseconds["right"]
        if left_nanoseconds is None:
            left_nanoseconds = right_nanoseconds
        if right_nanoseconds is None:
            right_nanoseconds = left_nanoseconds
        self._publish_invalid(
            reason,
            left_nanoseconds,
            right_nanoseconds,
        )

    def _publish_invalid(
        self,
        reason,
        left_nanoseconds,
        right_nanoseconds,
    ):
        if left_nanoseconds is None or right_nanoseconds is None:
            source_nanoseconds = max(
                0,
                self.get_clock().now().nanoseconds,
            )
            left_nanoseconds = right_nanoseconds = source_nanoseconds
        left_time = left_nanoseconds / 1e9
        right_time = right_nanoseconds / 1e9
        if self._pipeline is not None:
            result = self._pipeline.invalidate(
                reason,
                left_source_time_sec=left_time,
                right_source_time_sec=right_time,
            )
        else:
            result = PipelineResult(
                valid=False,
                point=None,
                confidence=0.0,
                pair_skew_sec=abs(left_time - right_time),
                reprojection_error_px=0.0,
                source_time_sec=min(left_time, right_time),
                stable_frames=0,
                reason=reason,
            )
        self._publish_result(
            result,
            source_time_nanoseconds=min(
                left_nanoseconds,
                right_nanoseconds,
            ),
        )

    def _publish_result(self, result, *, source_time_nanoseconds=None):
        message = hand_observation_from_result(
            result,
            self._output_frame_id,
            source_time_nanoseconds=source_time_nanoseconds,
        )
        self._publisher.publish(message)
        if result.reason != self._last_reason:
            self.get_logger().info(
                f"hand observation: valid={result.valid}, "
                f"stable_frames={result.stable_frames}, "
                f"reason={result.reason}"
            )
            self._last_reason = result.reason

    def destroy_node(self):
        """Release detector resources before destroying the ROS node."""
        for detector in getattr(self, "_detectors", ()):
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        self._detectors = ()
        return super().destroy_node()


def main(args=None):
    """Run the live rectified-stereo hand observer."""
    rclpy.init(args=args)
    node = None
    try:
        node = LiveStereoHandObserver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
