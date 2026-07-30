"""ROS topic tests for the synthetic Objective 3.2 candidate publisher."""

import time

from assistive_interfaces.msg import ObjectCandidateArray
from markerless_object_perception.synthetic_candidate_publisher import (
    SyntheticObjectCandidatePublisher,
)
import pytest
import rclpy
from rclpy.parameter import Parameter


def spin_until(nodes, predicate, timeout=2.0):
    """Spin all nodes until a predicate succeeds or time expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)
        if predicate():
            return True
    return False


def receive_one(topic, **overrides):
    """Run one isolated publisher/subscriber graph and return a message."""
    values = {
        'candidate_topic': topic,
        'publish_rate_hz': 50.0,
    }
    values.update(overrides)

    rclpy.init()
    publisher = None
    helper = None
    try:
        publisher = SyntheticObjectCandidatePublisher(
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in values.items()
            ]
        )
        helper = rclpy.create_node('synthetic_candidate_test_helper')
        messages = []
        subscription = helper.create_subscription(
            ObjectCandidateArray,
            topic,
            messages.append,
            10,
        )
        assert subscription is not None
        assert spin_until(
            (publisher, helper),
            lambda: bool(messages),
        ), 'synthetic publisher produced no ObjectCandidateArray'
        return messages[-1]
    finally:
        if publisher is not None:
            publisher.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_publishes_one_source_stamped_synthetic_candidate():
    message = receive_one(
        '/test/object_candidates/one',
        frame_id='test_stereo_frame',
        pair_skew_sec=0.007,
        class_label='cell_phone',
        class_confidence=0.88,
        track_id=12,
        object_point_x=0.3,
        object_point_y=-0.1,
        object_point_z=0.8,
    )

    assert message.valid
    assert message.header.frame_id == 'test_stereo_frame'
    assert message.header.stamp.sec > 0
    assert message.pair_skew_sec == pytest.approx(0.007)
    assert len(message.candidates) == 1
    candidate = message.candidates[0]
    assert candidate.track_id == 12
    assert candidate.class_label == 'cell_phone'
    assert candidate.class_confidence == pytest.approx(0.88)
    assert (
        candidate.position.x,
        candidate.position.y,
        candidate.position.z,
    ) == pytest.approx((0.3, -0.1, 0.8))
    assert candidate.localization_confidence == pytest.approx(1.0)


def test_no_detection_publishes_fresh_valid_empty_observation():
    message = receive_one(
        '/test/object_candidates/empty',
        simulate_no_detection=True,
    )

    assert message.valid
    assert message.header.stamp.sec > 0
    assert message.candidates == []


def test_rejected_synthetic_geometry_fails_during_node_startup():
    rclpy.init()
    try:
        with pytest.raises(ValueError, match='rejected candidates'):
            SyntheticObjectCandidatePublisher(
                parameter_overrides=[
                    Parameter('object_point_z', value=3.0),
                ]
            )
    finally:
        if rclpy.ok():
            rclpy.shutdown()
