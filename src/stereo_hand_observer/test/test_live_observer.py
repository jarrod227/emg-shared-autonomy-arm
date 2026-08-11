"""ROS topic-level tests for the live Objective 4.2 image adapter."""

import json
import time

from builtin_interfaces.msg import Time
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from assistive_interfaces.msg import HandObservation
from stereo_hand_observer.geometry import project_point
from stereo_hand_observer.keypoint_detector import HandKeypointsDetection
from stereo_hand_observer.live_observer import LiveStereoHandObserver
from stereo_hand_observer.synthetic_observer import rectified_stereo_model


GROUND_TRUTH = np.array([0.1, 0.05, 1.0])
LEFT_PROJECTION, RIGHT_PROJECTION, _ = rectified_stereo_model(
    100.0,
    100.0,
    32.0,
    24.0,
    0.12,
)
WIDTH = 64
HEIGHT = 48


def stamp_from_nanoseconds(nanoseconds):
    """Create a builtin Time without depending on receipt time."""
    stamp = Time()
    stamp.sec = nanoseconds // 1_000_000_000
    stamp.nanosec = nanoseconds % 1_000_000_000
    return stamp


def spin_until(nodes, predicate, timeout=2.0):
    """Spin all nodes until a predicate succeeds or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)
        if predicate():
            return True
    return False


class MutableDetector:
    """Test detector whose output can change between image pairs."""

    def __init__(self, result):
        self.result = result
        self.closed = False
        self.calls = 0
        self.images = []
        self.delay_sec = 0.0
        self.timestamps = []

    def detect(self, image):
        """Return the current test result."""
        self.calls += 1
        self.images.append(np.asarray(image).copy())
        if self.delay_sec > 0.0:
            time.sleep(self.delay_sec)
        return self.result

    def detect_at(self, image, timestamp_ms):
        """Record the source time used by timestamp-aware live inference."""
        self.timestamps.append(timestamp_ms)
        return self.detect(image)

    def close(self):
        """Record resource cleanup."""
        self.closed = True


def keypoint(projection):
    """Project the known 3D hand into one detector result."""
    pixel = tuple(project_point(projection, GROUND_TRUTH))
    return HandKeypointsDetection(
        pixels={index: pixel for index in (5, 9, 13, 17)},
        confidence=0.9,
        handedness="right",
    )


class LiveObserverGraph:
    """One live observer plus its test publishers and subscriber."""

    def __init__(
        self,
        required_frames=3,
        watchdog_timeout=0.5,
        *,
        exact_sync=False,
        publish_debug=False,
        debug_image_scale=1.0,
    ):
        self.detectors = {
            "left": MutableDetector(keypoint(LEFT_PROJECTION)),
            "right": MutableDetector(keypoint(RIGHT_PROJECTION)),
        }

        def detector_factory(side):
            return self.detectors[side]

        values = {
            "left_image_topic": "/test_stereo/left/image_rect",
            "right_image_topic": "/test_stereo/right/image_rect",
            "left_camera_info_topic": "/test_stereo/left/camera_info",
            "right_camera_info_topic": "/test_stereo/right/camera_info",
            "observation_topic": "/test_hand_observation",
            "fallback_frame_id": "fallback_left_optical",
            "delivery_center_x": float(GROUND_TRUTH[0]),
            "delivery_center_y": float(GROUND_TRUTH[1]),
            "delivery_center_z": float(GROUND_TRUTH[2]),
            "delivery_radius_m": 0.2,
            "required_stable_frames": required_frames,
            "max_pair_skew_sec": 0.02,
            "sync_slop_sec": 0.05,
            "max_age_sec": 0.5,
            "watchdog_timeout_sec": watchdog_timeout,
            "exact_sync": exact_sync,
        }
        if publish_debug:
            values["debug_image_topic"] = "/test_stereo/hand_debug"
            values["debug_image_scale"] = debug_image_scale
        self.observer = LiveStereoHandObserver(
            detector_factory=detector_factory,
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in values.items()
            ],
        )
        self.helper = rclpy.create_node("live_observer_test_helper")
        self.left_image_publisher = self.helper.create_publisher(
            Image,
            values["left_image_topic"],
            qos_profile_sensor_data,
        )
        self.right_image_publisher = self.helper.create_publisher(
            Image,
            values["right_image_topic"],
            qos_profile_sensor_data,
        )
        self.left_info_publisher = self.helper.create_publisher(
            CameraInfo,
            values["left_camera_info_topic"],
            qos_profile_sensor_data,
        )
        self.right_info_publisher = self.helper.create_publisher(
            CameraInfo,
            values["right_camera_info_topic"],
            qos_profile_sensor_data,
        )
        self.messages = []
        self.debug_messages = []
        self.output_subscription = self.helper.create_subscription(
            HandObservation,
            values["observation_topic"],
            self.messages.append,
            10,
        )
        self.debug_subscription = None
        if publish_debug:
            self.debug_subscription = self.helper.create_subscription(
                Image,
                values["debug_image_topic"],
                self.debug_messages.append,
                qos_profile_sensor_data,
            )

    @property
    def nodes(self):
        """Return nodes that need executor progress."""
        return (self.observer, self.helper)

    def camera_info(self, projection, frame_id):
        """Build one rectified CameraInfo message."""
        message = CameraInfo()
        message.header.frame_id = frame_id
        message.width = WIDTH
        message.height = HEIGHT
        message.p = list(np.asarray(projection).reshape(-1))
        return message

    def publish_calibration(
        self,
        left_projection=LEFT_PROJECTION,
        right_projection=RIGHT_PROJECTION,
    ):
        """Publish both camera models and wait for acceptance."""
        left_info = self.camera_info(left_projection, "left_optical")
        right_info = self.camera_info(right_projection, "right_optical")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.left_info_publisher.publish(left_info)
            self.right_info_publisher.publish(right_info)
            if spin_until(
                self.nodes,
                lambda: self.observer._processor is not None,
                timeout=0.05,
            ):
                return True
        return False

    def image(self, stamp_ns, frame_id):
        """Build one valid RGB image with an exact source stamp."""
        message = Image()
        message.header.stamp = stamp_from_nanoseconds(stamp_ns)
        message.header.frame_id = frame_id
        message.width = WIDTH
        message.height = HEIGHT
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = WIDTH * 3
        message.data = bytes(HEIGHT * message.step)
        return message

    def publish_pair(self, skew_sec=0.005):
        """Publish one fresh pair and return its oldest source stamp."""
        before = len(self.messages)
        right_ns = self.helper.get_clock().now().nanoseconds
        left_ns = right_ns - round(skew_sec * 1e9)
        self.left_image_publisher.publish(
            self.image(left_ns, "left_optical")
        )
        self.right_image_publisher.publish(
            self.image(right_ns, "right_optical")
        )
        assert spin_until(
            self.nodes,
            lambda: len(self.messages) > before,
        ), "synchronized image pair produced no observation"
        return stamp_from_nanoseconds(left_ns)

    def destroy(self):
        """Destroy nodes and their short-lived ROS entities."""
        self.observer.destroy_node()
        self.helper.destroy_node()


def write_camera_info(path, camera_name, projection):
    """Write one temporary pinhole calibration for composite tests."""
    projection = np.asarray(projection, dtype=float).reshape(3, 4)
    intrinsic = projection[:, :3].reshape(-1).tolist()
    data = {
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "camera_name": camera_name,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": intrinsic,
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.0] * 5,
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": projection.reshape(-1).tolist(),
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.as_uri()


class CompositeObserverGraph:
    """One atomic-composite observer plus its ROS test endpoints."""

    input_topic = "/test_stereo/composite/image_raw"
    output_topic = "/test_composite_hand_observation"
    debug_topic = "/test_stereo/composite_hand_debug"

    def __init__(
        self,
        tmp_path,
        *,
        required_frames=3,
        watchdog_timeout=0.5,
        publish_debug=False,
        max_age_sec=0.5,
    ):
        self.detectors = {
            "left": MutableDetector(keypoint(LEFT_PROJECTION)),
            "right": MutableDetector(keypoint(RIGHT_PROJECTION)),
        }

        def detector_factory(side):
            return self.detectors[side]

        left_url = write_camera_info(
            tmp_path / "left.yaml",
            "test_atomic_left",
            LEFT_PROJECTION,
        )
        right_url = write_camera_info(
            tmp_path / "right.yaml",
            "test_atomic_right",
            RIGHT_PROJECTION,
        )
        values = {
            "input_mode": "composite",
            "composite_image_topic": self.input_topic,
            "left_camera_info_url": left_url,
            "right_camera_info_url": right_url,
            "left_camera_name": "test_atomic_left",
            "right_camera_name": "test_atomic_right",
            "expected_composite_width": WIDTH * 2,
            "expected_composite_height": HEIGHT,
            "expected_composite_encoding": "rgb8",
            "left_frame_id": "left_optical",
            "right_frame_id": "right_optical",
            "observation_topic": self.output_topic,
            "fallback_frame_id": "fallback_left_optical",
            "delivery_center_x": float(GROUND_TRUTH[0]),
            "delivery_center_y": float(GROUND_TRUTH[1]),
            "delivery_center_z": float(GROUND_TRUTH[2]),
            "delivery_radius_m": 0.2,
            "required_stable_frames": required_frames,
            "max_pair_skew_sec": 0.02,
            "max_age_sec": max_age_sec,
            "watchdog_timeout_sec": watchdog_timeout,
        }
        if publish_debug:
            values["debug_image_topic"] = self.debug_topic
        self.observer = LiveStereoHandObserver(
            detector_factory=detector_factory,
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in values.items()
            ],
        )
        self.helper = rclpy.create_node("composite_observer_test_helper")
        self.publisher = self.helper.create_publisher(
            Image,
            self.input_topic,
            qos_profile_sensor_data,
        )
        self.messages = []
        self.debug_messages = []
        self.output_subscription = self.helper.create_subscription(
            HandObservation,
            self.output_topic,
            self.messages.append,
            10,
        )
        self.debug_subscription = None
        if publish_debug:
            self.debug_subscription = self.helper.create_subscription(
                Image,
                self.debug_topic,
                self.debug_messages.append,
                qos_profile_sensor_data,
            )

    @property
    def nodes(self):
        """Return nodes that need executor progress."""
        return (self.observer, self.helper)

    def wait_for_connection(self):
        """Wait until the depth-one composite subscription is discovered."""
        return spin_until(
            self.nodes,
            lambda: self.publisher.get_subscription_count() > 0,
        )

    def image(self, left_value, right_value, *, stamp_ns, encoding="rgb8"):
        """Build one side-by-side RGB image with distinguishable halves."""
        composite = np.empty((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)
        composite[:, :WIDTH] = left_value
        composite[:, WIDTH:] = right_value
        message = Image()
        message.header.stamp = stamp_from_nanoseconds(stamp_ns)
        message.header.frame_id = "composite_optical"
        message.height = composite.shape[0]
        message.width = composite.shape[1]
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = composite.shape[1] * composite.shape[2]
        message.data = composite.tobytes()
        return message

    def publish(self, message):
        """Publish one composite and wait for its explicit observation."""
        before = len(self.messages)
        self.publisher.publish(message)
        assert spin_until(
            self.nodes,
            lambda: len(self.messages) > before,
        ), "composite image produced no hand observation"

    def destroy(self):
        """Destroy nodes and their short-lived ROS entities."""
        self.observer.destroy_node()
        self.helper.destroy_node()


class FakeCapture:
    """Stand in for cv2.VideoCapture so direct mode needs no hardware."""

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.reads = 0
        self.released = False
        self.ok = True
        self.frame_shape = None
        self.raise_on_read = None

    def read(self):
        """Return one BGR composite frame, mirroring OpenCV's contract."""
        self.reads += 1
        if self.raise_on_read is not None:
            raise self.raise_on_read
        if not self.ok:
            return False, None
        height, width = self.frame_shape or (self._height, self._width)
        return True, np.zeros((height, width, 3), dtype=np.uint8)

    def release(self):
        """Record that the device was handed back."""
        self.released = True


class DirectObserverGraph:
    """One direct-capture observer plus an injected fake device."""

    output_topic = "/test_direct_hand_observation"

    def __init__(self, tmp_path, *, required_frames=3, capture=None):
        self.detectors = {
            "left": MutableDetector(keypoint(LEFT_PROJECTION)),
            "right": MutableDetector(keypoint(RIGHT_PROJECTION)),
        }
        self.capture = capture or FakeCapture(WIDTH * 2, HEIGHT)
        values = {
            "input_mode": "direct",
            "capture_device": "/dev/null",
            "left_camera_info_url": write_camera_info(
                tmp_path / "left.yaml", "test_direct_left", LEFT_PROJECTION
            ),
            "right_camera_info_url": write_camera_info(
                tmp_path / "right.yaml", "test_direct_right", RIGHT_PROJECTION
            ),
            "left_camera_name": "test_direct_left",
            "right_camera_name": "test_direct_right",
            "expected_composite_width": WIDTH * 2,
            "expected_composite_height": HEIGHT,
            "left_frame_id": "left_optical",
            "right_frame_id": "right_optical",
            "observation_topic": self.output_topic,
            "fallback_frame_id": "fallback_left_optical",
            "delivery_center_x": float(GROUND_TRUTH[0]),
            "delivery_center_y": float(GROUND_TRUTH[1]),
            "delivery_center_z": float(GROUND_TRUTH[2]),
            "delivery_radius_m": 0.2,
            "required_stable_frames": required_frames,
            "max_pair_skew_sec": 0.02,
            "max_age_sec": 5.0,
            "watchdog_timeout_sec": 0.5,
            "capture_period_sec": 0.01,
        }
        self.observer = LiveStereoHandObserver(
            detector_factory=lambda side: self.detectors[side],
            capture_factory=lambda: self.capture,
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in values.items()
            ],
        )
        self.helper = rclpy.create_node("direct_observer_test_helper")
        self.messages = []
        self.helper.create_subscription(
            HandObservation,
            self.output_topic,
            self.messages.append,
            10,
        )

    @property
    def nodes(self):
        """Return nodes that need executor progress."""
        return (self.observer, self.helper)

    def wait_for(self, count):
        """Spin until the capture timer has produced count observations."""
        return spin_until(self.nodes, lambda: len(self.messages) >= count)

    def destroy(self):
        """Destroy nodes and their short-lived ROS entities."""
        self.observer.destroy_node()
        self.helper.destroy_node()


def test_direct_capture_drives_the_pipeline_without_ros_transport(tmp_path):
    rclpy.init()
    test_graph = DirectObserverGraph(tmp_path, required_frames=3)
    try:
        assert test_graph.wait_for(3), "capture timer produced no observations"
        assert test_graph.capture.reads >= 3
        assert test_graph.messages[-1].valid
        assert test_graph.messages[-1].header.frame_id == "left_optical"
        np.testing.assert_allclose(
            [
                test_graph.messages[-1].point.x,
                test_graph.messages[-1].point.y,
                test_graph.messages[-1].point.z,
            ],
            GROUND_TRUTH,
            atol=1e-6,
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_direct_capture_stamps_each_frame_with_node_time(tmp_path):
    rclpy.init()
    test_graph = DirectObserverGraph(tmp_path, required_frames=1)
    try:
        assert test_graph.wait_for(2)
        stamps = [
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
            for message in test_graph.messages[:2]
        ]
        # Direct capture owns its clock, so stamps must advance rather than
        # repeat the way a copied composite stamp could.
        assert stamps[1] > stamps[0]
        assert all(message.pair_skew_sec == 0.0 for message in
                   test_graph.messages[:2])
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    (
        ("not_ok", "capture_read_failed"),
        ("raises", "capture_read_failed"),
        ("wrong_size", "composite_image_size_mismatch"),
    ),
)
def test_direct_capture_failures_publish_explicit_invalid(
    tmp_path,
    failure,
    expected_reason,
):
    rclpy.init()
    capture = FakeCapture(WIDTH * 2, HEIGHT)
    if failure == "not_ok":
        capture.ok = False
    elif failure == "raises":
        capture.raise_on_read = RuntimeError("device disappeared")
    else:
        capture.frame_shape = (HEIGHT, WIDTH * 2 - 4)
    test_graph = DirectObserverGraph(
        tmp_path,
        required_frames=1,
        capture=capture,
    )
    try:
        assert test_graph.wait_for(1)
        assert not test_graph.messages[-1].valid
        assert test_graph.observer._last_reason == expected_reason
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_direct_capture_device_is_released_on_shutdown(tmp_path):
    rclpy.init()
    test_graph = DirectObserverGraph(tmp_path, required_frames=1)
    try:
        assert test_graph.wait_for(1)
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()
    assert test_graph.capture.released


def test_capture_factory_contract_is_enforced(tmp_path):
    rclpy.init()
    try:
        with pytest.raises(TypeError, match="read"):
            DirectObserverGraph(
                tmp_path,
                capture=object(),
            )
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture
def graph():
    """Provide one isolated ROS graph per test."""
    rclpy.init()
    test_graph = LiveObserverGraph()
    try:
        yield test_graph
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_three_image_pairs_become_valid_and_bad_pair_resets(graph):
    assert graph.publish_calibration()

    source_stamps = [graph.publish_pair() for _ in range(3)]

    assert [message.valid for message in graph.messages[-3:]] == [
        False,
        False,
        True,
    ]
    valid_message = graph.messages[-1]
    assert valid_message.header.frame_id == "left_optical"
    assert valid_message.header.stamp == source_stamps[-1]
    assert (
        valid_message.point.x,
        valid_message.point.y,
        valid_message.point.z,
    ) == pytest.approx(tuple(GROUND_TRUTH))

    graph.detectors["right"].result = None
    graph.publish_pair()
    assert not graph.messages[-1].valid
    graph.detectors["right"].result = keypoint(RIGHT_PROJECTION)
    graph.publish_pair()
    assert not graph.messages[-1].valid


def test_bad_camera_info_fails_closed_then_recovers():
    rclpy.init()
    test_graph = LiveObserverGraph(required_frames=1)
    try:
        left_info = test_graph.camera_info(
            LEFT_PROJECTION,
            "left_optical",
        )
        bad_right_info = test_graph.camera_info(
            LEFT_PROJECTION,
            "right_optical",
        )
        test_graph.left_info_publisher.publish(left_info)
        test_graph.right_info_publisher.publish(bad_right_info)
        spin_until(test_graph.nodes, lambda: False, timeout=0.1)
        test_graph.publish_pair()
        assert not test_graph.messages[-1].valid

        assert test_graph.publish_calibration()
        test_graph.publish_pair()
        assert test_graph.messages[-1].valid
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_pair_over_quality_skew_is_explicitly_invalid():
    rclpy.init()
    test_graph = LiveObserverGraph(required_frames=1)
    try:
        assert test_graph.publish_calibration()
        test_graph.publish_pair(skew_sec=0.03)
        assert not test_graph.messages[-1].valid
        assert test_graph.messages[-1].pair_skew_sec == pytest.approx(
            0.03,
            abs=1e-6,
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_exact_sync_does_not_send_different_stamps_to_detectors():
    rclpy.init()
    test_graph = LiveObserverGraph(
        required_frames=1,
        watchdog_timeout=1.0,
        exact_sync=True,
    )
    try:
        assert test_graph.publish_calibration()
        right_ns = test_graph.helper.get_clock().now().nanoseconds
        left_ns = right_ns - 5_000_000
        test_graph.left_image_publisher.publish(
            test_graph.image(left_ns, "left_optical")
        )
        test_graph.right_image_publisher.publish(
            test_graph.image(right_ns, "right_optical")
        )
        assert not spin_until(
            test_graph.nodes,
            lambda: any(
                detector.calls > 0
                for detector in test_graph.detectors.values()
            ),
            timeout=0.05,
        )

        test_graph.publish_pair(skew_sec=0.0)
        assert test_graph.messages[-1].valid
        assert all(
            detector.calls == 1
            for detector in test_graph.detectors.values()
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_debug_image_preserves_pair_stamp_and_combines_both_views():
    rclpy.init()
    test_graph = LiveObserverGraph(
        required_frames=1,
        publish_debug=True,
    )
    try:
        assert test_graph.publish_calibration()
        expected_stamp = test_graph.publish_pair()
        assert spin_until(
            test_graph.nodes,
            lambda: bool(test_graph.debug_messages),
        )

        message = test_graph.debug_messages[-1]
        assert message.header.stamp == expected_stamp
        assert message.header.frame_id == "left_optical"
        assert message.width == WIDTH * 2
        assert message.height == HEIGHT
        assert message.encoding == "bgr8"
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_debug_image_scale_reduces_only_published_view():
    rclpy.init()
    test_graph = LiveObserverGraph(
        required_frames=1,
        publish_debug=True,
        debug_image_scale=0.5,
    )
    try:
        assert test_graph.publish_calibration()
        test_graph.publish_pair()
        assert spin_until(
            test_graph.nodes,
            lambda: bool(test_graph.debug_messages),
        )

        message = test_graph.debug_messages[-1]
        assert message.width == WIDTH
        assert message.height == HEIGHT // 2
        assert message.encoding == "bgr8"
        assert all(
            detector.calls == 1
            for detector in test_graph.detectors.values()
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_watchdog_revokes_valid_result_after_one_stream_stops():
    rclpy.init()
    test_graph = LiveObserverGraph(
        required_frames=1,
        watchdog_timeout=0.08,
    )
    try:
        assert test_graph.publish_calibration()
        test_graph.publish_pair()
        assert test_graph.messages[-1].valid
        valid_count = len(test_graph.messages)

        left_ns = test_graph.helper.get_clock().now().nanoseconds
        test_graph.left_image_publisher.publish(
            test_graph.image(left_ns, "left_optical")
        )
        assert spin_until(
            test_graph.nodes,
            lambda: (
                len(test_graph.messages) > valid_count
                and not test_graph.messages[-1].valid
            ),
            timeout=0.5,
        ), "watchdog did not publish an invalid observation"
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_composite_mode_keeps_each_pair_atomic_and_preserves_stamp(
    tmp_path,
):
    rclpy.init()
    test_graph = CompositeObserverGraph(
        tmp_path,
        required_frames=3,
        publish_debug=True,
    )
    try:
        assert test_graph.wait_for_connection()
        expected_stamps = []
        base_stamp_ns = (
            test_graph.helper.get_clock().now().nanoseconds
            - 100_000_000
        )
        for index in range(3):
            stamp_ns = base_stamp_ns + index * 10_000_000
            message = test_graph.image(
                index + 1,
                index + 101,
                stamp_ns=stamp_ns,
            )
            expected_stamps.append(message.header.stamp)
            test_graph.publish(message)

        assert [message.valid for message in test_graph.messages[-3:]] == [
            False,
            False,
            True,
        ]
        assert all(
            message.pair_skew_sec == pytest.approx(0.0)
            for message in test_graph.messages[-3:]
        )
        assert [
            message.header.stamp for message in test_graph.messages[-3:]
        ] == expected_stamps
        for index, (left_image, right_image) in enumerate(
            zip(
                test_graph.detectors["left"].images,
                test_graph.detectors["right"].images,
            )
        ):
            assert np.all(left_image == index + 1)
            assert np.all(right_image == index + 101)
        expected_timestamp_ms = [
            stamp.sec * 1000 + stamp.nanosec // 1_000_000
            for stamp in expected_stamps
        ]
        assert test_graph.detectors["left"].timestamps == (
            expected_timestamp_ms
        )
        assert test_graph.detectors["right"].timestamps == (
            expected_timestamp_ms
        )
        assert all(
            later > earlier
            for earlier, later in zip(
                expected_timestamp_ms,
                expected_timestamp_ms[1:],
            )
        )

        assert spin_until(
            test_graph.nodes,
            lambda: len(test_graph.debug_messages) >= 3,
        )
        debug = test_graph.debug_messages[-1]
        assert debug.header.stamp == expected_stamps[-1]
        assert debug.header.frame_id == "left_optical"
        assert (debug.width, debug.height, debug.encoding) == (
            WIDTH * 2,
            HEIGHT,
            "bgr8",
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_composite_mode_rejects_bad_size_and_encoding(tmp_path):
    rclpy.init()
    test_graph = CompositeObserverGraph(
        tmp_path,
        required_frames=1,
        watchdog_timeout=1.0,
    )
    try:
        assert test_graph.wait_for_connection()
        wrong_size = test_graph.image(
            1,
            2,
            stamp_ns=test_graph.helper.get_clock().now().nanoseconds,
        )
        wrong_size.width -= 2
        test_graph.publish(wrong_size)
        assert not test_graph.messages[-1].valid
        assert (
            test_graph.observer._last_reason
            == "composite_image_size_mismatch"
        )

        wrong_encoding = test_graph.image(
            1,
            2,
            stamp_ns=test_graph.helper.get_clock().now().nanoseconds,
            encoding="bgr8",
        )
        test_graph.publish(wrong_encoding)
        assert not test_graph.messages[-1].valid
        assert (
            test_graph.observer._last_reason
            == "composite_image_encoding_mismatch"
        )
        assert all(
            detector.calls == 0
            for detector in test_graph.detectors.values()
        )
        assert all(
            message.pair_skew_sec == pytest.approx(0.0)
            for message in test_graph.messages
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_composite_watchdog_reports_input_timeout_not_unpaired(tmp_path):
    rclpy.init()
    test_graph = CompositeObserverGraph(
        tmp_path,
        required_frames=1,
        watchdog_timeout=0.08,
    )
    try:
        assert test_graph.wait_for_connection()
        source = test_graph.image(
            1,
            2,
            stamp_ns=test_graph.helper.get_clock().now().nanoseconds,
        )
        test_graph.publish(source)
        assert test_graph.messages[-1].valid
        valid_count = len(test_graph.messages)

        assert spin_until(
            test_graph.nodes,
            lambda: (
                len(test_graph.messages) > valid_count
                and test_graph.observer._last_reason == "input_timeout"
            ),
            timeout=0.5,
        )
        assert not test_graph.messages[-1].valid
        assert test_graph.messages[-1].pair_skew_sec == pytest.approx(0.0)
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_composite_result_that_ages_during_detection_is_stale(tmp_path):
    rclpy.init()
    test_graph = CompositeObserverGraph(
        tmp_path,
        required_frames=1,
        watchdog_timeout=1.0,
        max_age_sec=0.1,
    )
    try:
        assert test_graph.wait_for_connection()
        for detector in test_graph.detectors.values():
            detector.delay_sec = 0.08
        source = test_graph.image(
            1,
            2,
            stamp_ns=test_graph.helper.get_clock().now().nanoseconds,
        )

        test_graph.publish(source)

        assert not test_graph.messages[-1].valid
        assert test_graph.observer._last_reason == "stale"
        assert all(
            detector.calls == 1
            for detector in test_graph.detectors.values()
        )
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()
