"""Tests for the side-by-side stereo image adapter."""

import json
import time

from builtin_interfaces.msg import Time
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from stereo_hand_observer.composite_stereo_splitter import (
    CompositeStereoSplitter,
    split_side_by_side,
)


def stamp_from_nanoseconds(nanoseconds):
    """Create a builtin Time with an exact test timestamp."""
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


def spin_for(nodes, duration):
    """Give a ROS graph time to process messages for a fixed duration."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)


def test_split_side_by_side_preserves_first_half_as_left():
    composite = np.zeros((3, 8, 3), dtype=np.uint8)
    composite[:, :4] = (11, 22, 33)
    composite[:, 4:] = (44, 55, 66)

    left, right = split_side_by_side(composite, 8, 3)

    assert left.flags.c_contiguous
    assert right.flags.c_contiguous
    assert np.all(left == (11, 22, 33))
    assert np.all(right == (44, 55, 66))
    composite[:] = 0
    assert np.all(left == (11, 22, 33))
    assert np.all(right == (44, 55, 66))


def test_split_side_by_side_can_swap_halves():
    composite = np.hstack(
        (
            np.full((2, 3), 1, dtype=np.uint8),
            np.full((2, 3), 2, dtype=np.uint8),
        )
    )

    left, right = split_side_by_side(
        composite,
        6,
        2,
        swap_halves=True,
    )

    assert np.all(left == 2)
    assert np.all(right == 1)


@pytest.mark.parametrize(
    ("image", "width", "height", "message"),
    [
        (np.zeros((2, 8, 3)), 8, 3, "expected height"),
        (np.zeros((2, 8, 3)), 10, 2, "expected width"),
        (np.zeros((2, 7, 3)), 7, 2, "must be even"),
        (np.zeros((2, 4, 1, 1)), 4, 2, "shape HxW"),
    ],
)
def test_split_side_by_side_rejects_malformed_input(
    image,
    width,
    height,
    message,
):
    with pytest.raises(ValueError, match=message):
        split_side_by_side(image, width, height)


def write_camera_info(path, camera_name, width, height, tx):
    """Write a minimal calibrated CameraInfo file accepted by ROS."""
    focal_length = 100.0
    principal_x = width / 2.0
    principal_y = height / 2.0
    data = {
        "image_width": width,
        "image_height": height,
        "camera_name": camera_name,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                focal_length,
                0.0,
                principal_x,
                0.0,
                focal_length,
                principal_y,
                0.0,
                0.0,
                1.0,
            ],
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
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                focal_length,
                0.0,
                principal_x,
                tx,
                0.0,
                focal_length,
                principal_y,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.as_uri()


class SplitterGraph:
    """One splitter plus test ROS publishers and subscribers."""

    width = 8
    height = 4
    input_topic = "/test_composite/image_raw"
    left_topic = "/test_stereo/left/image_raw"
    right_topic = "/test_stereo/right/image_raw"
    left_info_topic = "/test_stereo/left/camera_info"
    right_info_topic = "/test_stereo/right/camera_info"

    def __init__(
        self,
        left_camera_info_url,
        right_camera_info_url,
        *,
        rectify_images=False,
        output_width=0,
        output_height=0,
    ):
        values = {
            "input_topic": self.input_topic,
            "left_topic": self.left_topic,
            "right_topic": self.right_topic,
            "left_camera_info_topic": self.left_info_topic,
            "right_camera_info_topic": self.right_info_topic,
            "left_camera_info_url": left_camera_info_url,
            "right_camera_info_url": right_camera_info_url,
            "left_camera_name": "test_left_camera",
            "right_camera_name": "test_right_camera",
            "expected_width": self.width,
            "expected_height": self.height,
            "expected_encoding": "rgb8",
            "left_frame_id": "test_left_optical_frame",
            "right_frame_id": "test_right_optical_frame",
            "swap_halves": False,
            "rectify_images": rectify_images,
            "output_width": output_width,
            "output_height": output_height,
        }
        self.splitter = CompositeStereoSplitter(
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in values.items()
            ],
        )
        self.helper = rclpy.create_node("composite_splitter_test_helper")
        self.publisher = self.helper.create_publisher(
            Image,
            self.input_topic,
            qos_profile_sensor_data,
        )
        self.left_messages = []
        self.right_messages = []
        self.left_info_messages = []
        self.right_info_messages = []
        self.left_subscription = self.helper.create_subscription(
            Image,
            self.left_topic,
            self.left_messages.append,
            qos_profile_sensor_data,
        )
        self.right_subscription = self.helper.create_subscription(
            Image,
            self.right_topic,
            self.right_messages.append,
            qos_profile_sensor_data,
        )
        self.left_info_subscription = self.helper.create_subscription(
            CameraInfo,
            self.left_info_topic,
            self.left_info_messages.append,
            qos_profile_sensor_data,
        )
        self.right_info_subscription = self.helper.create_subscription(
            CameraInfo,
            self.right_info_topic,
            self.right_info_messages.append,
            qos_profile_sensor_data,
        )

    @property
    def nodes(self):
        """Return nodes that need executor progress."""
        return (self.splitter, self.helper)

    def wait_for_connections(self):
        """Wait until the input and both output topics are discovered."""
        return spin_until(
            self.nodes,
            lambda: (
                self.publisher.get_subscription_count() > 0
                and self.helper.count_publishers(self.left_topic) > 0
                and self.helper.count_publishers(self.right_topic) > 0
                and self.helper.count_publishers(self.left_info_topic) > 0
                and self.helper.count_publishers(self.right_info_topic) > 0
            ),
        )

    def image_message(
        self,
        image,
        encoding="rgb8",
        stamp_ns=1_734_567_890_123_456_789,
    ):
        """Build an Image message without changing the source stamp."""
        message = Image()
        message.header.stamp = stamp_from_nanoseconds(stamp_ns)
        message.header.frame_id = "decxin_composite_optical_frame"
        message.height = image.shape[0]
        message.width = image.shape[1]
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = image.shape[1] * image.shape[2]
        message.data = image.tobytes()
        return message

    def destroy(self):
        """Destroy all short-lived ROS entities."""
        self.splitter.destroy_node()
        self.helper.destroy_node()


@pytest.fixture
def graph(tmp_path):
    """Provide one isolated splitter ROS graph per test."""
    left_url = write_camera_info(
        tmp_path / "left.yaml",
        "test_left_camera",
        SplitterGraph.width // 2,
        SplitterGraph.height,
        0.0,
    )
    right_url = write_camera_info(
        tmp_path / "right.yaml",
        "test_right_camera",
        SplitterGraph.width // 2,
        SplitterGraph.height,
        -6.4,
    )
    rclpy.init()
    test_graph = SplitterGraph(left_url, right_url)
    try:
        yield test_graph
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_ros_splitter_preserves_pixels_metadata_and_exact_stamp(graph):
    assert graph.wait_for_connections()
    composite = np.zeros(
        (graph.height, graph.width, 3),
        dtype=np.uint8,
    )
    composite[:, : graph.width // 2] = (10, 20, 30)
    composite[:, graph.width // 2 :] = (40, 50, 60)
    source = graph.image_message(composite)

    graph.publisher.publish(source)
    assert spin_until(
        graph.nodes,
        lambda: (
            graph.left_messages
            and graph.right_messages
            and graph.left_info_messages
            and graph.right_info_messages
        ),
    )

    left = graph.left_messages[-1]
    right = graph.right_messages[-1]
    left_info = graph.left_info_messages[-1]
    right_info = graph.right_info_messages[-1]
    expected_stamp = (source.header.stamp.sec, source.header.stamp.nanosec)
    assert (left.header.stamp.sec, left.header.stamp.nanosec) == expected_stamp
    assert (
        right.header.stamp.sec,
        right.header.stamp.nanosec,
    ) == expected_stamp
    assert (
        left_info.header.stamp.sec,
        left_info.header.stamp.nanosec,
    ) == expected_stamp
    assert (
        right_info.header.stamp.sec,
        right_info.header.stamp.nanosec,
    ) == expected_stamp
    assert left.header.frame_id == "test_left_optical_frame"
    assert right.header.frame_id == "test_right_optical_frame"
    assert left_info.header.frame_id == "test_left_optical_frame"
    assert right_info.header.frame_id == "test_right_optical_frame"
    assert (left_info.width, left_info.height) == (
        graph.width // 2,
        graph.height,
    )
    assert (right_info.width, right_info.height) == (
        graph.width // 2,
        graph.height,
    )
    assert tuple(left_info.k) == (
        100.0,
        0.0,
        2.0,
        0.0,
        100.0,
        2.0,
        0.0,
        0.0,
        1.0,
    )
    assert -right_info.p[3] / right_info.p[0] == pytest.approx(0.064)
    assert (left.width, left.height, left.encoding, left.step) == (
        graph.width // 2,
        graph.height,
        "rgb8",
        graph.width // 2 * 3,
    )
    assert (right.width, right.height, right.encoding, right.step) == (
        graph.width // 2,
        graph.height,
        "rgb8",
        graph.width // 2 * 3,
    )
    left_array = np.frombuffer(left.data, dtype=np.uint8).reshape(
        graph.height,
        graph.width // 2,
        3,
    )
    right_array = np.frombuffer(right.data, dtype=np.uint8).reshape(
        graph.height,
        graph.width // 2,
        3,
    )
    assert np.all(left_array == (10, 20, 30))
    assert np.all(right_array == (40, 50, 60))


def test_ros_splitter_can_rectify_before_publishing(tmp_path, monkeypatch):
    left_url = write_camera_info(
        tmp_path / "left_rectified.yaml",
        "test_left_camera",
        SplitterGraph.width // 2,
        SplitterGraph.height,
        0.0,
    )
    right_url = write_camera_info(
        tmp_path / "right_rectified.yaml",
        "test_right_camera",
        SplitterGraph.width // 2,
        SplitterGraph.height,
        -6.4,
    )
    calls = []

    def fake_rectify_pair(left, right, left_maps, right_maps):
        calls.append((left_maps, right_maps))
        return left + 1, right + 2

    monkeypatch.setattr(
        "stereo_hand_observer.composite_stereo_splitter.rectify_pair",
        fake_rectify_pair,
    )
    rclpy.init()
    test_graph = SplitterGraph(
        left_url,
        right_url,
        rectify_images=True,
        output_width=2,
        output_height=2,
    )
    try:
        assert test_graph.wait_for_connections()
        composite = np.zeros(
            (test_graph.height, test_graph.width, 3),
            dtype=np.uint8,
        )
        composite[:, : test_graph.width // 2] = 10
        composite[:, test_graph.width // 2 :] = 20
        source = test_graph.image_message(composite)

        test_graph.publisher.publish(source)
        assert spin_until(
            test_graph.nodes,
            lambda: (
                test_graph.left_messages
                and test_graph.right_messages
                and test_graph.left_info_messages
                and test_graph.right_info_messages
            ),
        )

        left = np.frombuffer(
            test_graph.left_messages[-1].data,
            dtype=np.uint8,
        ).reshape(2, 2, 3)
        right = np.frombuffer(
            test_graph.right_messages[-1].data,
            dtype=np.uint8,
        ).reshape(2, 2, 3)
        assert len(calls) == 1
        assert np.all(left == 11)
        assert np.all(right == 22)
        assert (
            test_graph.left_messages[-1].width,
            test_graph.left_messages[-1].height,
            test_graph.left_info_messages[-1].width,
            test_graph.left_info_messages[-1].height,
        ) == (2, 2, 2, 2)
        assert test_graph.left_info_messages[-1].k[0] == pytest.approx(50.0)
        assert test_graph.right_info_messages[-1].p[3] == pytest.approx(-3.2)
        assert (
            -test_graph.right_info_messages[-1].p[3]
            / test_graph.right_info_messages[-1].p[0]
        ) == pytest.approx(0.064)
        assert test_graph.left_messages[-1].header.stamp == source.header.stamp
        assert test_graph.right_messages[-1].header.stamp == source.header.stamp
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def test_ros_splitter_publishes_neither_side_for_bad_input(graph):
    assert graph.wait_for_connections()
    composite = np.zeros(
        (graph.height, graph.width, 3),
        dtype=np.uint8,
    )
    graph.publisher.publish(
        graph.image_message(composite, encoding="bgr8")
    )
    wrong_width = np.zeros(
        (graph.height, graph.width - 2, 3),
        dtype=np.uint8,
    )
    graph.publisher.publish(graph.image_message(wrong_width))
    corrupt_buffer = graph.image_message(composite)
    corrupt_buffer.data = b"too short"
    graph.publisher.publish(corrupt_buffer)

    spin_for(graph.nodes, 0.25)
    assert not graph.left_messages
    assert not graph.right_messages
    assert not graph.left_info_messages
    assert not graph.right_info_messages
