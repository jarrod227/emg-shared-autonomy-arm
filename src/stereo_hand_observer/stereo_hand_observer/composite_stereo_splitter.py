"""Split a composite stereo image and publish matched calibration metadata."""

from copy import deepcopy
import math

from camera_info_manager import CameraInfoManager, CameraInfoMissingError
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from stereo_hand_observer.stereo_rectification import (
    build_rectification_maps,
    rectify_pair,
    resize_rectified_pair,
    scale_camera_info,
)


def split_side_by_side(
    image,
    expected_width,
    expected_height,
    swap_halves=False,
):
    """Return independent left/right arrays from one side-by-side image."""
    if expected_width <= 0 or expected_height <= 0:
        raise ValueError("expected dimensions must be positive")
    if expected_width % 2:
        raise ValueError("expected composite width must be even")

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must have shape HxW or HxWxC")
    if array.shape[0] != expected_height:
        raise ValueError(
            f"expected height {expected_height}, got {array.shape[0]}"
        )
    if array.shape[1] != expected_width:
        raise ValueError(
            f"expected width {expected_width}, got {array.shape[1]}"
        )

    midpoint = expected_width // 2
    first = array[:, :midpoint].copy(order="C")
    second = array[:, midpoint:].copy(order="C")
    if swap_halves:
        return second, first
    return first, second


class CompositeStereoSplitter(Node):
    """Convert one atomic composite frame into a matched stereo pair."""

    def __init__(self, *, capture_factory=None, **kwargs):
        super().__init__("composite_stereo_splitter", **kwargs)

        self._capture_factory = capture_factory
        self._input_mode = self.declare_parameter(
            "input_mode",
            "topic",
        ).value
        self._input_topic = self.declare_parameter(
            "input_topic",
            "/stereo/composite/camera/image_raw",
        ).value
        self._left_topic = self.declare_parameter(
            "left_topic",
            "/stereo/left/image_raw",
        ).value
        self._right_topic = self.declare_parameter(
            "right_topic",
            "/stereo/right/image_raw",
        ).value
        self._left_camera_info_topic = self.declare_parameter(
            "left_camera_info_topic",
            "/stereo/left/camera_info",
        ).value
        self._right_camera_info_topic = self.declare_parameter(
            "right_camera_info_topic",
            "/stereo/right/camera_info",
        ).value
        self._left_camera_info_url = self.declare_parameter(
            "left_camera_info_url",
            "",
        ).value
        self._right_camera_info_url = self.declare_parameter(
            "right_camera_info_url",
            "",
        ).value
        self._left_camera_name = self.declare_parameter(
            "left_camera_name",
            "decxin_stereo_left_1280x960",
        ).value
        self._right_camera_name = self.declare_parameter(
            "right_camera_name",
            "decxin_stereo_right_1280x960",
        ).value
        self._expected_width = self.declare_parameter(
            "expected_width",
            2560,
        ).value
        self._expected_height = self.declare_parameter(
            "expected_height",
            960,
        ).value
        self._expected_encoding = self.declare_parameter(
            "expected_encoding",
            "rgb8",
        ).value
        self._left_frame_id = self.declare_parameter(
            "left_frame_id",
            "stereo_left_optical_frame",
        ).value
        self._right_frame_id = self.declare_parameter(
            "right_frame_id",
            "stereo_right_optical_frame",
        ).value
        self._swap_halves = self.declare_parameter(
            "swap_halves",
            False,
        ).value
        self._rectify_images = self.declare_parameter(
            "rectify_images",
            False,
        ).value
        self._output_width = self.declare_parameter(
            "output_width",
            0,
        ).value
        self._output_height = self.declare_parameter(
            "output_height",
            0,
        ).value
        self._capture_device = self.declare_parameter(
            "capture_device",
            "/dev/video0",
        ).value
        self._capture_fourcc = self.declare_parameter(
            "capture_fourcc",
            "MJPG",
        ).value
        self._capture_fps = self.declare_parameter(
            "capture_fps",
            30.0,
        ).value
        self._capture_buffer_size = self.declare_parameter(
            "capture_buffer_size",
            1,
        ).value
        self._capture_period_sec = self.declare_parameter(
            "capture_period_sec",
            0.1,
        ).value

        self._validate_parameters()
        self._camera_info_enabled = bool(
            self._left_camera_info_url.strip()
        )
        self._bridge = CvBridge()
        self._last_rejection = None
        self._left_rectification_maps = None
        self._right_rectification_maps = None
        self._left_camera_info_template = None
        self._right_camera_info_template = None
        self._left_publisher = self.create_publisher(
            Image,
            self._left_topic,
            qos_profile_sensor_data,
        )
        self._right_publisher = self.create_publisher(
            Image,
            self._right_topic,
            qos_profile_sensor_data,
        )
        self._left_camera_info_publisher = None
        self._right_camera_info_publisher = None
        self._left_camera_info_manager = None
        self._right_camera_info_manager = None
        self._subscription = None
        self._capture = None
        self._capture_timer = None
        if self._camera_info_enabled:
            self._configure_camera_info()
        if self._input_mode == "topic":
            self._subscription = self.create_subscription(
                Image,
                self._input_topic,
                self._on_composite_image,
                qos_profile_sensor_data,
            )
            input_description = f"topic {self._input_topic}"
        else:
            self._configure_direct_input()
            input_description = f"device {self._capture_device}"
        output_kind = "rectified" if self._rectify_images else "raw"
        output_width = self._output_width or self._expected_width // 2
        output_height = self._output_height or self._expected_height
        self.get_logger().info(
            "Reading %s; splitting %dx%d composite images into "
            "%dx%d %s stereo pairs"
            % (
                input_description,
                self._expected_width,
                self._expected_height,
                output_width,
                output_height,
                output_kind,
            )
        )

    def _validate_parameters(self):
        string_parameters = {
            "input_mode": self._input_mode,
            "input_topic": self._input_topic,
            "left_topic": self._left_topic,
            "right_topic": self._right_topic,
            "left_camera_info_topic": self._left_camera_info_topic,
            "right_camera_info_topic": self._right_camera_info_topic,
            "expected_encoding": self._expected_encoding,
            "left_frame_id": self._left_frame_id,
            "right_frame_id": self._right_frame_id,
            "left_camera_name": self._left_camera_name,
            "right_camera_name": self._right_camera_name,
        }
        for name, value in string_parameters.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self._input_mode not in {"topic", "direct"}:
            raise ValueError("input_mode must be 'topic' or 'direct'")
        if self._expected_width <= 0 or self._expected_width % 2:
            raise ValueError("expected_width must be a positive even integer")
        if self._expected_height <= 0:
            raise ValueError("expected_height must be a positive integer")
        if not isinstance(self._swap_halves, bool):
            raise ValueError("swap_halves must be a boolean")
        if not isinstance(self._rectify_images, bool):
            raise ValueError("rectify_images must be a boolean")
        output_dimensions = (self._output_width, self._output_height)
        if any(isinstance(value, bool) for value in output_dimensions):
            raise ValueError("output dimensions must be integers")
        if not all(isinstance(value, int) for value in output_dimensions):
            raise ValueError("output dimensions must be integers")
        if (self._output_width == 0) != (self._output_height == 0):
            raise ValueError("output dimensions must both be zero or positive")
        if self._output_width < 0 or self._output_height < 0:
            raise ValueError("output dimensions must both be zero or positive")
        if self._output_width > 0 and not self._rectify_images:
            raise ValueError("resized output requires rectify_images")
        camera_info_urls = (
            self._left_camera_info_url,
            self._right_camera_info_url,
        )
        if not all(isinstance(url, str) for url in camera_info_urls):
            raise ValueError("camera-info URLs must be strings")
        if bool(camera_info_urls[0].strip()) != bool(
            camera_info_urls[1].strip()
        ):
            raise ValueError(
                "left and right camera-info URLs must both be set or both empty"
            )
        if self._rectify_images and not camera_info_urls[0].strip():
            raise ValueError(
                "rectify_images requires both camera-info URLs"
            )
        if self._input_mode == "direct":
            if (
                not isinstance(self._capture_device, str)
                or not self._capture_device.strip()
            ):
                raise ValueError("capture_device must be a non-empty string")
            if (
                not isinstance(self._capture_fourcc, str)
                or len(self._capture_fourcc) != 4
            ):
                raise ValueError(
                    "capture_fourcc must be exactly four characters"
                )
            if self._expected_encoding not in {"rgb8", "bgr8"}:
                raise ValueError(
                    "direct input requires expected_encoding rgb8 or bgr8"
                )
            self._validate_positive_number(
                self._capture_fps,
                "capture_fps",
            )
            self._validate_positive_number(
                self._capture_period_sec,
                "capture_period_sec",
            )
            if (
                isinstance(self._capture_buffer_size, bool)
                or not isinstance(self._capture_buffer_size, int)
                or self._capture_buffer_size <= 0
            ):
                raise ValueError(
                    "capture_buffer_size must be a positive integer"
                )

    @staticmethod
    def _validate_positive_number(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a positive finite number")

    def _default_capture_factory(self):
        """Open the composite device and keep only its newest frame."""
        capture = cv2.VideoCapture(self._capture_device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"could not open capture device: {self._capture_device}"
            )
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self._capture_fourcc),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._expected_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._expected_height)
        capture.set(cv2.CAP_PROP_FPS, float(self._capture_fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self._capture_buffer_size)
        return capture

    def _configure_direct_input(self):
        factory = self._capture_factory or self._default_capture_factory
        self._capture = factory()
        try:
            for name in ("read", "release"):
                if not callable(getattr(self._capture, name, None)):
                    raise TypeError(
                        "capture_factory must return an object with "
                        f"{name}()"
                    )
        except Exception:
            release = getattr(self._capture, "release", None)
            if callable(release):
                release()
            self._capture = None
            raise
        self._capture_timer = self.create_timer(
            float(self._capture_period_sec),
            self._on_capture_tick,
        )

    def _configure_camera_info(self):
        try:
            self._left_camera_info_manager = CameraInfoManager(
                self,
                cname=self._left_camera_name,
                url=self._left_camera_info_url,
                namespace="stereo/left",
            )
            self._right_camera_info_manager = CameraInfoManager(
                self,
                cname=self._right_camera_name,
                url=self._right_camera_info_url,
                namespace="stereo/right",
            )
            self._left_camera_info_manager.loadCameraInfo()
            self._right_camera_info_manager.loadCameraInfo()
            baseline = self._validated_camera_info_pair()
            left_info = self._left_camera_info_manager.getCameraInfo()
            right_info = self._right_camera_info_manager.getCameraInfo()
            if self._rectify_images:
                self._left_rectification_maps = build_rectification_maps(
                    left_info
                )
                self._right_rectification_maps = build_rectification_maps(
                    right_info
                )
            if self._output_width > 0:
                left_info = scale_camera_info(
                    left_info,
                    self._output_width,
                    self._output_height,
                )
                right_info = scale_camera_info(
                    right_info,
                    self._output_width,
                    self._output_height,
                )
            self._left_camera_info_template = deepcopy(left_info)
            self._right_camera_info_template = deepcopy(right_info)
        except (CameraInfoMissingError, TypeError, ValueError) as error:
            raise ValueError(
                f"failed to load stereo camera calibration: {error}"
            ) from error

        self._left_camera_info_publisher = self.create_publisher(
            CameraInfo,
            self._left_camera_info_topic,
            qos_profile_sensor_data,
        )
        self._right_camera_info_publisher = self.create_publisher(
            CameraInfo,
            self._right_camera_info_topic,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Loaded calibrated stereo CameraInfo (baseline %.2f mm)"
            % (baseline * 1000.0)
        )

    def _validate_camera_info(self, camera_info, side):
        expected_eye_width = self._expected_width // 2
        if camera_info.width != expected_eye_width:
            raise ValueError(
                f"{side} calibration width must be {expected_eye_width}, "
                f"got {camera_info.width}"
            )
        if camera_info.height != self._expected_height:
            raise ValueError(
                f"{side} calibration height must be {self._expected_height}, "
                f"got {camera_info.height}"
            )
        if not camera_info.distortion_model:
            raise ValueError(f"{side} distortion_model must not be empty")

        matrices = {
            "K": camera_info.k,
            "R": camera_info.r,
            "P": camera_info.p,
        }
        for name, values in matrices.items():
            array = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(array)) or not np.any(array):
                raise ValueError(
                    f"{side} calibration matrix {name} must be finite and nonzero"
                )
        if camera_info.k[0] <= 0.0 or camera_info.k[4] <= 0.0:
            raise ValueError(f"{side} calibration has invalid focal lengths")
        if camera_info.p[0] <= 0.0 or camera_info.p[5] <= 0.0:
            raise ValueError(
                f"{side} rectified projection has invalid focal lengths"
            )

    def _validated_camera_info_pair(self):
        left_info = self._left_camera_info_manager.getCameraInfo()
        right_info = self._right_camera_info_manager.getCameraInfo()
        self._validate_camera_info(left_info, "left")
        self._validate_camera_info(right_info, "right")
        baseline = -float(right_info.p[3]) / float(right_info.p[0])
        if not np.isfinite(baseline) or baseline <= 0.0:
            raise ValueError(
                "right projection matrix must encode a positive stereo baseline"
            )
        return baseline

    def _stamped_camera_info_pair(self, stamp):
        left_info = deepcopy(self._left_camera_info_template)
        right_info = deepcopy(self._right_camera_info_template)
        self._copy_stamp(stamp, left_info.header.stamp)
        self._copy_stamp(stamp, right_info.header.stamp)
        left_info.header.frame_id = self._left_frame_id
        right_info.header.frame_id = self._right_frame_id
        return left_info, right_info

    def _reject(self, reason):
        if reason != self._last_rejection:
            self.get_logger().warning(
                f"Rejecting composite frame: {reason}"
            )
            self._last_rejection = reason

    @staticmethod
    def _copy_stamp(source, destination):
        destination.sec = source.sec
        destination.nanosec = source.nanosec

    def _on_composite_image(self, message):
        if message.width != self._expected_width:
            self._reject(
                f"expected width {self._expected_width}, got {message.width}"
            )
            return
        if message.height != self._expected_height:
            self._reject(
                "expected height "
                f"{self._expected_height}, got {message.height}"
            )
            return
        if message.encoding != self._expected_encoding:
            self._reject(
                "expected encoding "
                f"{self._expected_encoding}, got {message.encoding}"
            )
            return

        try:
            composite = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding=self._expected_encoding,
            )
            self._publish_composite_array(
                composite,
                message.header.stamp,
            )
        except (CvBridgeError, cv2.error, TypeError, ValueError) as error:
            self._reject(str(error))

    def _on_capture_tick(self):
        """Read one newest device frame and publish a matched stereo pair."""
        try:
            ok, frame = self._capture.read()
        except Exception as error:
            self._reject(f"capture read failed: {error}")
            return
        if not ok or frame is None:
            self._reject("capture read failed")
            return

        try:
            composite = np.asarray(frame)
            if (
                composite.ndim != 3
                or composite.shape[2] != 3
                or composite.shape[0] != self._expected_height
                or composite.shape[1] != self._expected_width
            ):
                raise ValueError(
                    "captured frame must match the configured HxWx3 shape"
                )
            if self._expected_encoding == "rgb8":
                composite = cv2.cvtColor(
                    composite,
                    cv2.COLOR_BGR2RGB,
                )
            else:
                composite = np.ascontiguousarray(composite)
            self._publish_composite_array(
                composite,
                self.get_clock().now().to_msg(),
            )
        except (cv2.error, TypeError, ValueError) as error:
            self._reject(str(error))

    def _publish_composite_array(self, composite, stamp):
        """Split, optionally rectify/resize, and publish one atomic frame."""
        try:
            left, right = split_side_by_side(
                composite,
                self._expected_width,
                self._expected_height,
                self._swap_halves,
            )
            if self._rectify_images:
                left, right = rectify_pair(
                    left,
                    right,
                    self._left_rectification_maps,
                    self._right_rectification_maps,
                )
            if self._output_width > 0:
                left, right = resize_rectified_pair(
                    left,
                    right,
                    self._output_width,
                    self._output_height,
                )
            left_message = self._bridge.cv2_to_imgmsg(
                left,
                encoding=self._expected_encoding,
            )
            right_message = self._bridge.cv2_to_imgmsg(
                right,
                encoding=self._expected_encoding,
            )
            camera_info_pair = None
            if self._camera_info_enabled:
                camera_info_pair = self._stamped_camera_info_pair(
                    stamp
                )
        except (
            CameraInfoMissingError,
            CvBridgeError,
            cv2.error,
            TypeError,
            ValueError,
        ) as error:
            self._reject(str(error))
            return

        self._copy_stamp(stamp, left_message.header.stamp)
        self._copy_stamp(stamp, right_message.header.stamp)
        left_message.header.frame_id = self._left_frame_id
        right_message.header.frame_id = self._right_frame_id

        self._left_publisher.publish(left_message)
        self._right_publisher.publish(right_message)
        if camera_info_pair is not None:
            left_info, right_info = camera_info_pair
            self._left_camera_info_publisher.publish(left_info)
            self._right_camera_info_publisher.publish(right_info)
        self._last_rejection = None

    def destroy_node(self):
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()
            self._capture = None
        return super().destroy_node()


def main(args=None):
    """Run the composite stereo splitter node."""
    rclpy.init(args=args)
    node = None
    try:
        node = CompositeStereoSplitter()
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
