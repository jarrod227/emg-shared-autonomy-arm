"""ROS topic-level tests for the live Objective 4.2 image adapter."""

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
from stereo_hand_observer.keypoint_detector import HandKeypointDetection
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

    def detect(self, _image):
        """Return the current test result."""
        return self.result

    def close(self):
        """Record resource cleanup."""
        self.closed = True


def keypoint(projection):
    """Project the known 3D hand into one detector result."""
    return HandKeypointDetection(
        pixel=tuple(project_point(projection, GROUND_TRUTH)),
        confidence=0.9,
        handedness="right",
    )


class LiveObserverGraph:
    """One live observer plus its test publishers and subscriber."""

    def __init__(self, required_frames=3, watchdog_timeout=0.5):
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
        }
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
        self.output_subscription = self.helper.create_subscription(
            HandObservation,
            values["observation_topic"],
            self.messages.append,
            10,
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
