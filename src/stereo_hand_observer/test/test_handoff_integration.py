"""ROS topic-level integration tests from stereo geometry to handoff state."""

import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from assistive_handoff.handoff_controller import HandoffController, HandoffState
from assistive_interfaces.msg import AssistiveIntent, HandObservation
from stereo_hand_observer.geometry import project_point
from stereo_hand_observer.observation_gate import (
    DeliveryVolume,
    StabilityGateConfig,
)
from stereo_hand_observer.pipeline import (
    StereoHandPipeline,
    StereoKeypointSet,
)
from stereo_hand_observer.ros_adapter import hand_observation_from_result
from stereo_hand_observer.synthetic_observer import rectified_stereo_model


GROUND_TRUTH = np.array([0.4, 0.3, 1.0])
LEFT_PROJECTION, RIGHT_PROJECTION, FUNDAMENTAL_MATRIX = (
    rectified_stereo_model(800.0, 800.0, 320.0, 240.0, 0.12)
)


def spin_until(nodes, predicate, timeout=2.0):
    """Spin all test nodes until a condition becomes true or times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.01)
        if predicate():
            return True
    return False


class IntegrationGraph:
    """Completed 4.1 controller plus publishers owned by this 4.2 test."""

    def __init__(self):
        parameters = {
            "sim_motion_sec": 0.05,
            "motion_timeout_sec": 2.0,
            "ready_timeout_sec": 5.0,
            "simulate_stuck_motion": "release",
        }
        self.controller = HandoffController(
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in parameters.items()
            ]
        )
        self.helper = rclpy.create_node("stereo_handoff_integration_helper")
        self._intent_publisher = self.helper.create_publisher(
            AssistiveIntent,
            "/assistive_intent",
            10,
        )
        self._hand_publisher = self.helper.create_publisher(
            HandObservation,
            "/hand_observation",
            10,
        )
        self._target_publisher = self.helper.create_publisher(
            PoseStamped,
            "/target_object_pose",
            QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._intent_sequence = 0
        self.confirm_count = 0
        original_confirm = self.controller._on_confirm

        def counted_confirm():
            self.confirm_count += 1
            original_confirm()

        # Test-only instrumentation proves a refused CONFIRM was received;
        # otherwise remaining in READY could be a false pass after packet loss.
        self.controller._on_confirm = counted_confirm

    @property
    def nodes(self):
        """Return every node that must be spun for message delivery."""
        return (self.controller, self.helper)

    @property
    def state(self):
        """Expose the current controller state for focused assertions."""
        return self.controller._state

    def settle(self, seconds):
        """Spin the graph for a bounded interval."""
        spin_until(self.nodes, lambda: False, timeout=seconds)

    def send_target(self):
        """Publish one fresh retained target and prove controller receipt."""
        message = PoseStamped()
        message.header.stamp = self.helper.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.orientation.w = 1.0
        self._target_publisher.publish(message)
        assert spin_until(
            self.nodes,
            lambda: self.controller._last_target is not None,
        ), "target was not delivered"

    def send_confirm(self):
        """Publish CONFIRM and prove its callback was executed."""
        count_before = self.confirm_count
        message = AssistiveIntent()
        message.header.stamp = self.helper.get_clock().now().to_msg()
        message.header.frame_id = "integration_test"
        message.command = AssistiveIntent.CONFIRM
        message.confidence = 1.0
        message.sequence = self._intent_sequence
        self._intent_sequence += 1
        self._intent_publisher.publish(message)
        assert spin_until(
            self.nodes,
            lambda: self.confirm_count > count_before,
        ), "CONFIRM was not delivered"

    def publish_hand(self, message):
        """Publish one exact 4.2 output and prove controller receipt."""
        expected_stamp = (
            message.header.stamp.sec,
            message.header.stamp.nanosec,
        )
        self._hand_publisher.publish(message)

        def received_exact_message():
            latest = self.controller._last_hand
            if latest is None:
                return False
            return (
                latest.header.stamp.sec,
                latest.header.stamp.nanosec,
            ) == expected_stamp

        assert spin_until(self.nodes, received_exact_message), (
            "hand observation was not delivered"
        )

    def enter_ready(self):
        """Drive the controller through IDLE -> APPROACH -> READY."""
        self.settle(0.15)
        self.send_target()
        self.send_confirm()
        assert spin_until(
            self.nodes,
            lambda: self.state is HandoffState.READY,
        ), f"controller did not reach READY; state={self.state}"

    def destroy(self):
        """Destroy the short-lived integration graph."""
        self.controller.destroy_node()
        self.helper.destroy_node()


@pytest.fixture
def graph():
    """Provide one isolated ROS graph per integration-test case."""
    rclpy.init()
    test_graph = IntegrationGraph()
    try:
        yield test_graph
    finally:
        test_graph.destroy()
        if rclpy.ok():
            rclpy.shutdown()


def make_pipeline(required_frames=3, max_epipolar_error_px=1.5):
    """Build the exact 4.2 software gate used by these integration tests."""
    return StereoHandPipeline(
        LEFT_PROJECTION,
        RIGHT_PROJECTION,
        FUNDAMENTAL_MATRIX,
        DeliveryVolume(center=tuple(GROUND_TRUTH), radius_m=0.2),
        gate_config=StabilityGateConfig(
            required_frames=required_frames,
            min_confidence=0.7,
            max_pair_skew_sec=0.02,
            max_reprojection_error_px=1.5,
            max_age_sec=0.2,
            max_point_step_m=0.05,
        ),
        max_epipolar_error_px=max_epipolar_error_px,
    )


KNUCKLE_OFFSETS = {
    5: np.array([-0.03, 0.0, 0.0]),
    9: np.array([-0.01, 0.01, 0.0]),
    13: np.array([0.01, -0.01, 0.0]),
    17: np.array([0.03, 0.0, 0.0]),
}


def make_pair(
    now_sec,
    *,
    point=GROUND_TRUTH,
    confidence=0.9,
    pair_skew_sec=0.005,
    age_sec=0.0,
    missing_right=False,
    right_vertical_error_px=0.0,
):
    """Project one synthetic hand into a timestamped multi-knuckle set."""
    point = np.asarray(point, dtype=np.float64)
    left_pixels = {}
    right_pixels = {}
    for index, offset in KNUCKLE_OFFSETS.items():
        knuckle = point + offset
        left_pixels[index] = tuple(project_point(LEFT_PROJECTION, knuckle))
        right_pixel = project_point(RIGHT_PROJECTION, knuckle)
        right_pixel[1] += right_vertical_error_px
        right_pixels[index] = tuple(right_pixel)
    if missing_right:
        right_pixels = None

    right_source_time = now_sec - age_sec
    return StereoKeypointSet(
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        left_source_time_sec=right_source_time - pair_skew_sec,
        right_source_time_sec=right_source_time,
        left_confidence=confidence,
        right_confidence=confidence,
    )


def publish_pipeline_result(graph, result):
    """Cross the real Objective 4.2 -> Objective 4.1 message boundary."""
    message = hand_observation_from_result(result, "world")
    graph.publish_hand(message)


def test_release_requires_third_stable_stereo_observation(graph):
    graph.enter_ready()
    pipeline = make_pipeline(required_frames=3)
    reasons = []

    for index in range(3):
        now_sec = graph.helper.get_clock().now().nanoseconds / 1e9
        result = pipeline.process_set(make_pair(now_sec), now_sec)
        reasons.append(result.reason)
        publish_pipeline_result(graph, result)
        graph.send_confirm()
        if index < 2:
            graph.settle(0.08)
            assert graph.state is HandoffState.READY
        else:
            assert spin_until(
                graph.nodes,
                lambda: graph.state is HandoffState.RELEASE,
            ), "third stable frame did not permit release"

    assert reasons == ["warming_up", "warming_up", "stable"]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("outside", "outside_delivery_volume"),
        ("missing", "missing_keypoint"),
        ("stale", "stale"),
        ("unstable", "unstable"),
        ("low_confidence", "low_confidence"),
        ("excessive_skew", "excessive_pair_skew"),
        ("high_reprojection", "insufficient_consensus"),
    ),
)
def test_invalid_stereo_observation_blocks_release(
    graph,
    case,
    expected_reason,
):
    graph.enter_ready()
    pipeline = make_pipeline(
        max_epipolar_error_px=(
            10.0 if case == "high_reprojection" else 1.5
        )
    )
    now_sec = graph.helper.get_clock().now().nanoseconds / 1e9

    if case == "outside":
        pair = make_pair(now_sec, point=(0.7, 0.3, 1.0))
    elif case == "missing":
        pair = make_pair(now_sec, missing_right=True)
    elif case == "stale":
        pair = make_pair(now_sec, age_sec=1.0)
    elif case == "unstable":
        first_time = now_sec - 0.02
        pipeline.process_set(make_pair(first_time), first_time)
        pair = make_pair(now_sec, point=(0.48, 0.3, 1.0))
    elif case == "low_confidence":
        pair = make_pair(now_sec, confidence=0.6)
    elif case == "excessive_skew":
        pair = make_pair(now_sec, pair_skew_sec=0.03)
    else:
        pair = make_pair(now_sec, right_vertical_error_px=6.0)

    result = pipeline.process_set(pair, now_sec)
    assert not result.valid
    assert result.reason == expected_reason
    publish_pipeline_result(graph, result)

    graph.send_confirm()
    graph.settle(0.12)
    assert graph.state is HandoffState.READY, (
        f"{case} observation incorrectly permitted release"
    )
