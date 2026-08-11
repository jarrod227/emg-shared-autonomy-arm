"""Live topic-pair or atomic-composite ROS 2 adapter for Objective 4.2."""

from copy import deepcopy
import math
import operator
import time

from camera_info_manager import CameraInfoManager, CameraInfoMissingError
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from assistive_interfaces.msg import HandObservation
from stereo_hand_observer.composite_stereo_splitter import split_side_by_side
from stereo_hand_observer.geometry import fundamental_from_projections
from stereo_hand_observer.hand_detector import draw_hand_landmarks
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
from stereo_hand_observer.stereo_rectification import (
    build_rectification_maps,
    rectify_pair,
)


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


def _landmark_indices_parameter(node, name):
    values = node.get_parameter(name).value
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{name} must be a list of landmark indices")
    indices = []
    for value in values:
        try:
            index = operator.index(value)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain integers in [0, 20]"
            ) from error
        if not 0 <= index <= 20:
            raise ValueError(f"{name} must contain integers in [0, 20]")
        indices.append(index)
    if not indices:
        raise ValueError(f"{name} must contain at least one landmark")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(indices)


def _string_parameter(node, name):
    value = node.get_parameter(name).value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean_parameter(node, name):
    value = node.get_parameter(name).value
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


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
    """Publish fail-closed observations from topic or composite stereo."""

    def __init__(
        self,
        detector_factory=None,
        capture_factory=None,
        **node_kwargs,
    ):
        super().__init__("live_stereo_hand_observer", **node_kwargs)
        self._capture_factory = capture_factory
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

        self._input_mode = _string_parameter(self, "input_mode")
        if self._input_mode not in {"stereo_topics", "composite", "direct"}:
            raise ValueError(
                "input_mode must be 'stereo_topics', 'composite', or 'direct'"
            )
        self._watchdog_timeout_sec = _positive_parameter(
            self,
            "watchdog_timeout_sec",
        )
        self._palm_landmark_indices = _landmark_indices_parameter(
            self,
            "palm_landmark_indices",
        )
        self._min_consensus_points = _integer_parameter(
            self,
            "min_consensus_points",
        )
        if self._min_consensus_points > len(self._palm_landmark_indices):
            raise ValueError(
                "min_consensus_points cannot exceed the number of "
                "palm_landmark_indices; the gate could never pass"
            )
        self._max_palm_span_m = _positive_parameter(self, "max_palm_span_m")

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
        debug_image_topic = self.get_parameter("debug_image_topic").value
        if not isinstance(debug_image_topic, str):
            raise ValueError("debug_image_topic must be a string")
        self._debug_image_scale = _finite_parameter(
            self,
            "debug_image_scale",
        )
        if not 0.0 < self._debug_image_scale <= 1.0:
            raise ValueError(
                "debug_image_scale must be greater than zero and at most one"
            )
        self._debug_publisher = None
        if debug_image_topic.strip():
            self._debug_publisher = self.create_publisher(
                Image,
                debug_image_topic.strip(),
                qos_profile_sensor_data,
            )
        self._last_debug_error = None
        self._left_info = None
        self._right_info = None
        self._calibration_signature = None
        self._calibration_size = None
        self._pipeline = None
        self._processor = None
        self._output_frame_id = self._fallback_frame_id
        self._last_reason = None
        self._left_rectification_maps = None
        self._right_rectification_maps = None

        self._left_camera_info_manager = None
        self._right_camera_info_manager = None
        self._left_info_subscription = None
        self._right_info_subscription = None
        self._left_image_subscriber = None
        self._right_image_subscriber = None
        self._synchronizer = None
        self._composite_subscription = None
        self._capture = None
        self._capture_timer = None

        self._latest_source_nanoseconds = {"left": None, "right": None}
        self._latest_arrivals = {"left": None, "right": None}
        self._last_pair_arrival = None

        if self._input_mode == "stereo_topics":
            input_description = self._configure_stereo_topic_input()
        elif self._input_mode == "direct":
            input_description = self._configure_direct_input()
        else:
            input_description = self._configure_composite_input()

        watchdog_period = min(0.1, self._watchdog_timeout_sec / 2.0)
        self._watchdog = self.create_timer(
            watchdog_period,
            self._on_watchdog,
        )
        self.get_logger().info(
            "live stereo hand observer started; "
            f"{input_description}"
        )

    def _declare_parameters(self):
        self.declare_parameter("input_mode", "stereo_topics")
        self.declare_parameter(
            "composite_image_topic",
            "/stereo/composite/camera/image_raw",
        )
        self.declare_parameter("left_camera_info_url", "")
        self.declare_parameter("right_camera_info_url", "")
        self.declare_parameter(
            "left_camera_name",
            "decxin_stereo_left_1280x960",
        )
        self.declare_parameter(
            "right_camera_name",
            "decxin_stereo_right_1280x960",
        )
        self.declare_parameter("expected_composite_width", 2560)
        self.declare_parameter("expected_composite_height", 960)
        self.declare_parameter("expected_composite_encoding", "rgb8")
        self.declare_parameter(
            "left_frame_id",
            "stereo_left_optical_frame",
        )
        self.declare_parameter(
            "right_frame_id",
            "stereo_right_optical_frame",
        )
        self.declare_parameter("swap_halves", False)
        # Direct-capture mode: the observer opens the device itself, so the
        # 7.4 MB raw composite never crosses ROS transport.
        self.declare_parameter(
            "capture_device",
            "/dev/v4l/by-id/"
            "usb-DECXIN_DECXIN_Camera_01.00.00-video-index0",
        )
        self.declare_parameter("capture_fourcc", "MJPG")
        self.declare_parameter("capture_fps", 30.0)
        self.declare_parameter("capture_buffer_size", 1)
        self.declare_parameter("capture_period_sec", 0.005)
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
        self.declare_parameter("debug_image_topic", "")
        self.declare_parameter("debug_image_scale", 1.0)
        self.declare_parameter(
            "fallback_frame_id",
            "stereo_left_optical_frame",
        )
        self.declare_parameter("model_path", "")
        # Palm knuckles are the most cross-view consistent landmarks; a
        # measured 21-landmark sweep put them near 2 px of epipolar error
        # while fingertips reached tens of pixels.
        self.declare_parameter("palm_landmark_indices", [5, 9, 13, 17])
        self.declare_parameter("min_consensus_points", 3)
        self.declare_parameter("max_palm_span_m", 0.12)
        self.declare_parameter("min_detection_confidence", 0.7)
        self.declare_parameter("min_hand_presence_confidence", 0.7)
        self.declare_parameter("min_tracking_confidence", 0.7)
        self.declare_parameter("mediapipe_running_mode", "image")
        self.declare_parameter("require_handedness_match", True)
        self.declare_parameter("exact_sync", False)
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
            "landmark_indices": self._palm_landmark_indices,
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
            "running_mode": _string_parameter(
                self,
                "mediapipe_running_mode",
            ),
        }

        def create_detector(_side):
            return MediaPipeHandKeypointDetector(
                model_path.strip(),
                **options,
            )

        return create_detector

    def _configure_stereo_topic_input(self):
        sync_slop_sec = _positive_parameter(self, "sync_slop_sec")
        if sync_slop_sec < self._gate_config.max_pair_skew_sec:
            raise ValueError(
                "sync_slop_sec must be at least max_pair_skew_sec"
            )
        sync_queue_size = _integer_parameter(self, "sync_queue_size")
        left_info_topic = _string_parameter(
            self,
            "left_camera_info_topic",
        )
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
        image_subscribers = [
            self._left_image_subscriber,
            self._right_image_subscriber,
        ]
        if _boolean_parameter(self, "exact_sync"):
            self._synchronizer = message_filters.TimeSynchronizer(
                image_subscribers,
                sync_queue_size,
            )
            sync_description = "exact source timestamps"
        else:
            self._synchronizer = message_filters.ApproximateTimeSynchronizer(
                image_subscribers,
                queue_size=sync_queue_size,
                slop=sync_slop_sec,
                allow_headerless=False,
            )
            sync_description = (
                f"approximate timestamps ({sync_slop_sec:.3f}s)"
            )
        self._synchronizer.registerCallback(self._on_image_pair)
        return (
            "waiting for rectified image topics and CameraInfo; "
            f"pairing {sync_description}"
        )

    def _read_composite_shape_parameters(self):
        """Read the side-by-side frame geometry both split modes rely on."""
        self._expected_composite_width = _integer_parameter(
            self,
            "expected_composite_width",
        )
        if self._expected_composite_width % 2:
            raise ValueError("expected_composite_width must be even")
        self._expected_composite_height = _integer_parameter(
            self,
            "expected_composite_height",
        )
        self._expected_composite_encoding = _string_parameter(
            self,
            "expected_composite_encoding",
        )
        self._swap_halves = _boolean_parameter(self, "swap_halves")

    def _load_composite_calibration(self):
        """Load per-eye calibration and rectification maps from file URLs.

        Shared by the composite-topic and direct-capture modes: both split
        one side-by-side frame into two calibrated halves, and only the
        source of that frame differs.
        """
        left_frame_id = _string_parameter(self, "left_frame_id")
        right_frame_id = _string_parameter(self, "right_frame_id")
        try:
            self._left_camera_info_manager = CameraInfoManager(
                self,
                cname=_string_parameter(self, "left_camera_name"),
                url=_string_parameter(self, "left_camera_info_url"),
                namespace="atomic/left",
            )
            self._right_camera_info_manager = CameraInfoManager(
                self,
                cname=_string_parameter(self, "right_camera_name"),
                url=_string_parameter(self, "right_camera_info_url"),
                namespace="atomic/right",
            )
            self._left_camera_info_manager.loadCameraInfo()
            self._right_camera_info_manager.loadCameraInfo()
            self._left_info = deepcopy(
                self._left_camera_info_manager.getCameraInfo()
            )
            self._right_info = deepcopy(
                self._right_camera_info_manager.getCameraInfo()
            )
            self._left_info.header.frame_id = left_frame_id
            self._right_info.header.frame_id = right_frame_id
            expected_eye_size = (
                self._expected_composite_width // 2,
                self._expected_composite_height,
            )
            for side, camera_info in (
                ("left", self._left_info),
                ("right", self._right_info),
            ):
                actual_size = (camera_info.width, camera_info.height)
                if actual_size != expected_eye_size:
                    raise ValueError(
                        f"{side} calibration size must be "
                        f"{expected_eye_size[0]}x{expected_eye_size[1]}, "
                        f"got {actual_size[0]}x{actual_size[1]}"
                    )
            self._refresh_calibration()
            if self._processor is None:
                raise ValueError("stereo projection calibration was rejected")
            self._left_rectification_maps = build_rectification_maps(
                self._left_info
            )
            self._right_rectification_maps = build_rectification_maps(
                self._right_info
            )
        except (CameraInfoMissingError, TypeError, ValueError) as error:
            raise ValueError(
                f"failed to load composite stereo calibration: {error}"
            ) from error

    def _configure_composite_input(self):
        self._read_composite_shape_parameters()
        self._load_composite_calibration()

        composite_qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=1,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        self._composite_subscription = self.create_subscription(
            Image,
            _string_parameter(self, "composite_image_topic"),
            self._on_composite_image,
            composite_qos,
        )
        return (
            "waiting for one atomic composite image stream; "
            "left/right source timestamps are identical by transport"
        )

    def _default_capture_factory(self):
        """Open the composite device directly, keeping only newest frames."""
        import cv2

        device = _string_parameter(self, "capture_device")
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise ValueError(f"could not open capture device: {device}")
        fourcc = _string_parameter(self, "capture_fourcc")
        if len(fourcc) != 4:
            capture.release()
            raise ValueError("capture_fourcc must be exactly four characters")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self._expected_composite_width,
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self._expected_composite_height,
        )
        capture.set(cv2.CAP_PROP_FPS, _positive_parameter(self, "capture_fps"))
        # A one-frame driver buffer is what keeps read() returning the
        # newest frame instead of walking a backlog. Measured on the DECXIN
        # bench, read() stays near 25 ms, so stamping at read time is a fair
        # approximation of capture time.
        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            _integer_parameter(self, "capture_buffer_size"),
        )
        return capture

    def _configure_direct_input(self):
        self._read_composite_shape_parameters()
        self._load_composite_calibration()

        factory = self._capture_factory
        if factory is None:
            factory = self._default_capture_factory
        self._capture = factory()
        for name in ("read", "release"):
            if not callable(getattr(self._capture, name, None)):
                raise TypeError(
                    f"capture_factory must return an object with {name}()"
                )

        # A short period lets the blocking read/process cycle self-pace:
        # rclpy does not stack timer callbacks, so the loop simply runs
        # back to back while leaving room for the watchdog to fire.
        self._capture_timer = self.create_timer(
            _positive_parameter(self, "capture_period_sec"),
            self._on_capture_tick,
        )
        return (
            "capturing the composite stream directly from "
            f"{_string_parameter(self, 'capture_device')}; "
            "raw frames never cross ROS transport"
        )

    def _on_capture_tick(self):
        """Read one composite frame and run the shared atomic pipeline."""
        try:
            ok, frame = self._capture.read()
        except Exception:
            ok, frame = False, None
        arrival = time.monotonic()
        if not ok or frame is None:
            self._latest_arrivals["left"] = arrival
            self._latest_arrivals["right"] = arrival
            self._last_pair_arrival = arrival
            source_nanoseconds = self.get_clock().now().nanoseconds
            self._publish_invalid(
                "capture_read_failed",
                source_nanoseconds,
                source_nanoseconds,
            )
            return

        # Direct capture owns its own timestamp; read() has just returned,
        # so this is the closest honest estimate of exposure time.
        source_nanoseconds = self.get_clock().now().nanoseconds
        self._latest_source_nanoseconds["left"] = source_nanoseconds
        self._latest_source_nanoseconds["right"] = source_nanoseconds
        self._latest_arrivals["left"] = arrival
        self._latest_arrivals["right"] = arrival
        self._last_pair_arrival = arrival

        try:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("captured frame must be a colour image")
            if (
                array.shape[1] != self._expected_composite_width
                or array.shape[0] != self._expected_composite_height
            ):
                self._publish_invalid(
                    "composite_image_size_mismatch",
                    source_nanoseconds,
                    source_nanoseconds,
                )
                return
            import cv2

            composite_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            left_raw, right_raw = split_side_by_side(
                composite_rgb,
                self._expected_composite_width,
                self._expected_composite_height,
                swap_halves=self._swap_halves,
            )
        except Exception:
            self._publish_invalid(
                "image_conversion_error",
                source_nanoseconds,
                source_nanoseconds,
            )
            return

        try:
            left_rgb, right_rgb = rectify_pair(
                left_raw,
                right_raw,
                self._left_rectification_maps,
                self._right_rectification_maps,
            )
        except Exception:
            self._publish_invalid(
                "rectification_error",
                source_nanoseconds,
                source_nanoseconds,
            )
            return

        try:
            self._process_rgb_pair(
                left_rgb,
                right_rgb,
                left_nanoseconds=source_nanoseconds,
                right_nanoseconds=source_nanoseconds,
            )
        finally:
            completion = time.monotonic()
            self._latest_arrivals["left"] = completion
            self._latest_arrivals["right"] = completion
            self._last_pair_arrival = completion

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
            min_consensus_points=self._min_consensus_points,
            max_palm_span_m=self._max_palm_span_m,
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

    def _on_composite_image(self, message):
        source_nanoseconds = _stamp_nanoseconds(message)
        arrival = time.monotonic()
        self._latest_source_nanoseconds["left"] = source_nanoseconds
        self._latest_source_nanoseconds["right"] = source_nanoseconds
        self._latest_arrivals["left"] = arrival
        self._latest_arrivals["right"] = arrival
        self._last_pair_arrival = arrival

        try:
            if (
                message.width != self._expected_composite_width
                or message.height != self._expected_composite_height
            ):
                self._publish_invalid(
                    "composite_image_size_mismatch",
                    source_nanoseconds,
                    source_nanoseconds,
                )
                return
            if message.encoding != self._expected_composite_encoding:
                self._publish_invalid(
                    "composite_image_encoding_mismatch",
                    source_nanoseconds,
                    source_nanoseconds,
                )
                return

            try:
                composite_rgb = self._bridge.imgmsg_to_cv2(
                    message,
                    desired_encoding="rgb8",
                )
                left_raw, right_raw = split_side_by_side(
                    composite_rgb,
                    self._expected_composite_width,
                    self._expected_composite_height,
                    swap_halves=self._swap_halves,
                )
            except Exception:
                self._publish_invalid(
                    "image_conversion_error",
                    source_nanoseconds,
                    source_nanoseconds,
                )
                return

            try:
                left_rgb, right_rgb = rectify_pair(
                    left_raw,
                    right_raw,
                    self._left_rectification_maps,
                    self._right_rectification_maps,
                )
            except Exception:
                self._publish_invalid(
                    "rectification_error",
                    source_nanoseconds,
                    source_nanoseconds,
                )
                return

            self._process_rgb_pair(
                left_rgb,
                right_rgb,
                left_nanoseconds=source_nanoseconds,
                right_nanoseconds=source_nanoseconds,
            )
        finally:
            completion = time.monotonic()
            self._latest_arrivals["left"] = completion
            self._latest_arrivals["right"] = completion
            self._last_pair_arrival = completion

    def _on_image_pair(self, left_message, right_message):
        self._last_pair_arrival = time.monotonic()
        left_nanoseconds = _stamp_nanoseconds(left_message)
        right_nanoseconds = _stamp_nanoseconds(right_message)
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

        self._process_rgb_pair(
            left_rgb,
            right_rgb,
            left_nanoseconds=left_nanoseconds,
            right_nanoseconds=right_nanoseconds,
        )

    def _process_rgb_pair(
        self,
        left_rgb,
        right_rgb,
        *,
        left_nanoseconds,
        right_nanoseconds,
    ):
        if self._processor is None:
            self._publish_invalid(
                "missing_or_invalid_camera_info",
                left_nanoseconds,
                right_nanoseconds,
            )
            return
        left_time = left_nanoseconds / 1e9
        right_time = right_nanoseconds / 1e9
        try:
            result = self._processor.process(
                left_rgb,
                right_rgb,
                left_source_time_sec=left_time,
                right_source_time_sec=right_time,
                now_sec=self.get_clock().now().nanoseconds / 1e9,
                left_source_time_nanoseconds=left_nanoseconds,
                right_source_time_nanoseconds=right_nanoseconds,
            )
        except Exception:
            self._publish_invalid(
                "processing_error",
                left_nanoseconds,
                right_nanoseconds,
            )
            return
        publication_time_sec = self.get_clock().now().nanoseconds / 1e9
        source_time_sec = min(left_time, right_time)
        candidate_was_accepted = result.valid or result.stable_frames > 0
        if (
            candidate_was_accepted
            and publication_time_sec - source_time_sec
            > self._gate_config.max_age_sec
        ):
            result = self._pipeline.invalidate(
                "stale",
                left_source_time_sec=left_time,
                right_source_time_sec=right_time,
                confidence=result.confidence,
            )
        source_time_nanoseconds = min(
            left_nanoseconds,
            right_nanoseconds,
        )
        self._publish_result(
            result,
            source_time_nanoseconds=source_time_nanoseconds,
        )
        self._publish_debug_image(
            left_rgb,
            right_rgb,
            result,
            source_time_nanoseconds=source_time_nanoseconds,
        )

    def _publish_debug_image(
        self,
        left_rgb,
        right_rgb,
        result,
        *,
        source_time_nanoseconds,
    ):
        """Publish both rectified views with full-hand landmarks and status."""
        if self._debug_publisher is None:
            return
        try:
            import cv2

            annotated = []
            for label, rgb_image, detector in zip(
                ("LEFT", "RIGHT"),
                (left_rgb, right_rgb),
                self._detectors,
            ):
                view = cv2.cvtColor(
                    np.asarray(rgb_image),
                    cv2.COLOR_RGB2BGR,
                )
                hand = getattr(detector, "last_hand", None)
                if hand is not None:
                    view = draw_hand_landmarks(
                        view,
                        hand,
                        representative_indices=self._palm_landmark_indices,
                        cv2_module=cv2,
                    )
                cv2.putText(
                    view,
                    label,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                annotated.append(view)

            debug_image = np.hstack(annotated)
            status_color = (0, 255, 0) if result.valid else (0, 0, 255)
            status = (
                f"valid={str(result.valid).lower()} "
                f"stable_frames={result.stable_frames} "
                f"reason={result.reason}"
            )
            cv2.putText(
                debug_image,
                status,
                (12, debug_image.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )
            if result.diagnostic:
                cv2.putText(
                    debug_image,
                    result.diagnostic,
                    (12, debug_image.shape[0] - 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
            if self._debug_image_scale < 1.0:
                debug_image = cv2.resize(
                    debug_image,
                    (
                        max(
                            1,
                            round(
                                debug_image.shape[1]
                                * self._debug_image_scale
                            ),
                        ),
                        max(
                            1,
                            round(
                                debug_image.shape[0]
                                * self._debug_image_scale
                            ),
                        ),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            message = self._bridge.cv2_to_imgmsg(
                debug_image,
                encoding="bgr8",
            )
            source_time_nanoseconds = max(
                0,
                int(source_time_nanoseconds),
            )
            message.header.stamp.sec = (
                source_time_nanoseconds // 1_000_000_000
            )
            message.header.stamp.nanosec = (
                source_time_nanoseconds % 1_000_000_000
            )
            message.header.frame_id = self._output_frame_id
            self._debug_publisher.publish(message)
            self._last_debug_error = None
        except Exception as error:
            description = f"{type(error).__name__}: {error}"
            if description != self._last_debug_error:
                self.get_logger().warning(
                    f"debug image publication failed: {description}"
                )
                self._last_debug_error = description

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
        """Release detector and capture resources before destroying."""
        for detector in getattr(self, "_detectors", ()):
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        self._detectors = ()
        capture = getattr(self, "_capture", None)
        if capture is not None:
            release = getattr(capture, "release", None)
            if callable(release):
                release()
            self._capture = None
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
