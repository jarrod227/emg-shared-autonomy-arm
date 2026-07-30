"""ROS subscription test for the markerless candidate stability boundary."""

import time

from assistive_interfaces.msg import (
    ObjectCandidate,
    ObjectCandidateArray,
)
import pytest
import rclpy
from rclpy.parameter import Parameter

from target_selector.markerless_candidate_gate_node import (
    MarkerlessCandidateGateNode,
)


def spin_until(nodes, predicate, timeout=2.0):
    """Spin all nodes until the predicate succeeds or time expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)
        if predicate():
            return True
    return False


def candidate_message(source_time_nanoseconds, *, position=(0.3, 0.0, 0.7)):
    """Build one source-stamped candidate observation."""
    message = ObjectCandidateArray()
    message.header.stamp.sec = source_time_nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = source_time_nanoseconds % 1_000_000_000
    message.header.frame_id = 'test_stereo_frame'
    message.valid = True
    message.pair_skew_sec = 0.005
    detected = ObjectCandidate()
    detected.track_id = 7
    detected.class_label = 'bottle'
    detected.class_confidence = 0.9
    (
        detected.position.x,
        detected.position.y,
        detected.position.z,
    ) = position
    detected.localization_confidence = 0.85
    message.candidates = [detected]
    return message


def test_subscription_feeds_three_frames_into_stability_gate():
    topic = '/test/markerless_candidate_gate/input'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=topic),
                Parameter('required_frames', value=3),
                Parameter('max_age_sec', value=1.0),
                Parameter('max_frame_gap_sec', value=0.5),
            ]
        )
        helper = rclpy.create_node('markerless_candidate_gate_test_helper')
        publisher = helper.create_publisher(
            ObjectCandidateArray,
            topic,
            10,
        )
        base_time = helper.get_clock().now().nanoseconds - 100_000_000

        for index, position in enumerate(
            (
                (0.30, 0.00, 0.70),
                (0.31, 0.00, 0.70),
                (0.305, 0.00, 0.70),
            ),
            start=1,
        ):
            publisher.publish(
                candidate_message(
                    base_time + (index - 1) * 20_000_000,
                    position=position,
                )
            )
            assert spin_until(
                (gate_node, helper),
                lambda: gate_node.processed_message_count >= index,
            )

        decision = gate_node.last_decision
        assert decision.reason == 'stable'
        assert decision.frame_id == 'test_stereo_frame'
        assert decision.stable_counts == ((7, 3),)
        assert len(decision.stable_candidates) == 1
        assert decision.stable_candidates[0].position == pytest.approx(
            (0.305, 0.0, 0.7)
        )
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
