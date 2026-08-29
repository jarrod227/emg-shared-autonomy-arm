"""ROS subscription test for the markerless candidate stability boundary."""

import time

from assistive_interfaces.msg import (
    AssistiveIntent,
    ObjectCandidate,
    ObjectCandidateArray,
)
from geometry_msgs.msg import PoseStamped, TransformStamped
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from target_selector.markerless_candidate_gate_node import (
    MarkerlessCandidateGateNode,
)
from tf2_ros import StaticTransformBroadcaster


def spin_until(nodes, predicate, timeout=2.0):
    """Spin all nodes until the predicate succeeds or time expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.005)
        if predicate():
            return True
    return False


def candidate_message(
    source_time_nanoseconds,
    *,
    position=(0.3, 0.0, 0.7),
    detections=None,
):
    """Build one source-stamped candidate observation."""
    message = ObjectCandidateArray()
    message.header.stamp.sec = source_time_nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = source_time_nanoseconds % 1_000_000_000
    message.header.frame_id = 'test_stereo_frame'
    message.valid = True
    message.pair_skew_sec = 0.005
    if detections is None:
        detections = ((7, 'bottle', position),)
    message.candidates = []
    for track_id, class_label, detected_position in detections:
        detected = ObjectCandidate()
        detected.track_id = track_id
        detected.class_label = class_label
        detected.class_confidence = 0.9
        (
            detected.position.x,
            detected.position.y,
            detected.position.z,
        ) = detected_position
        detected.localization_confidence = 0.85
        message.candidates.append(detected)
    return message


def intent_message(node, command, sequence):
    """Build one fresh simulated intent event."""
    message = AssistiveIntent()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = 'test_intent'
    message.command = command
    message.confidence = 1.0
    message.sequence = sequence
    return message


def broadcast_test_transform(node, translation=(0.0, 0.0, 0.0)):
    """Publish a static world-to-test-stereo transform."""
    broadcaster = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.stamp = node.get_clock().now().to_msg()
    transform.header.frame_id = 'world'
    transform.child_frame_id = 'test_stereo_frame'
    (
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ) = translation
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)
    return broadcaster


def retained_pose_qos():
    """Return the production target-pose QoS for late-joiner tests."""
    return QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )


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
        lock_decision = gate_node.last_lock_decision
        assert lock_decision.selected_candidate.track_id == 7
        assert lock_decision.selected_visible
        assert not lock_decision.confirmed
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_intent_cycles_and_confirms_current_stable_target():
    candidate_topic = '/test/markerless_candidate_gate/candidates'
    intent_topic = '/test/markerless_candidate_gate/intent'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node(
            'markerless_candidate_intent_test_helper'
        )
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent,
            intent_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
            ),
        )

        source_time = helper.get_clock().now().nanoseconds - 50_000_000
        candidate_publisher.publish(
            candidate_message(
                source_time,
                detections=(
                    (2, 'bottle', (0.3, 0.0, 0.7)),
                    (7, 'cup', (0.4, 0.0, 0.7)),
                ),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 1,
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 2

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.NEXT_TARGET, 0)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 1,
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 7
        assert not gate_node.last_lock_decision.confirmed

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 1)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                gate_node.processed_intent_count == 2
                and gate_node.last_lock_decision.ready
            ),
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 7
        assert gate_node.published_target_count == 0
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_selected_candidate_uses_exact_source_time_transform():
    candidate_topic = '/test/markerless_candidate_gate/source_time'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
                Parameter('planning_frame', value='world'),
            ]
        )
        helper = rclpy.create_node('markerless_source_time_test_helper')
        broadcaster = broadcast_test_transform(
            helper,
            translation=(1.0, -2.0, 0.5),
        )

        publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: publisher.get_subscription_count() == 1,
        )
        spin_until((gate_node, helper), lambda: False, timeout=0.1)

        source_time = helper.get_clock().now().nanoseconds - 20_000_000
        publisher.publish(
            candidate_message(
                source_time,
                position=(0.3, 0.1, 0.7),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_built_target_pose is not None,
        )

        target = gate_node.last_built_target_pose
        assert target.header.frame_id == 'world'
        target_stamp = (
            target.header.stamp.sec * 1_000_000_000
            + target.header.stamp.nanosec
        )
        assert target_stamp == source_time
        assert target.pose.position.x == pytest.approx(1.3)
        assert target.pose.position.y == pytest.approx(-1.9)
        assert target.pose.position.z == pytest.approx(1.2)
        assert broadcaster is not None
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_intent_sequence_comparison_handles_uint32_wrap():
    """Accept zero after uint32 max while rejecting duplicate/old values."""
    newer = MarkerlessCandidateGateNode._sequence_is_newer

    assert newer(0, 0xFFFFFFFF)
    assert not newer(7, 7)
    assert not newer(0xFFFFFFFF, 0)


def test_confirm_publishes_one_retained_pose_without_candidate_republish():
    candidate_topic = '/test/markerless_publish/candidates'
    intent_topic = '/test/markerless_publish/intent'
    target_topic = '/test/markerless_publish/target'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('target_topic', value=target_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_publish_test_helper')
        broadcaster = broadcast_test_transform(
            helper,
            translation=(1.0, -2.0, 0.5),
        )
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent,
            intent_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
            ),
        )
        spin_until((gate_node, helper), lambda: False, timeout=0.1)

        first_source_time = (
            helper.get_clock().now().nanoseconds - 20_000_000
        )
        candidate_publisher.publish(
            candidate_message(
                first_source_time,
                position=(0.3, 0.1, 0.7),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_built_target_pose is not None,
        )
        assert gate_node.published_target_count == 0

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 10)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.published_target_count == 1,
        )

        second_source_time = first_source_time + 10_000_000
        candidate_publisher.publish(
            candidate_message(
                second_source_time,
                position=(0.4, 0.1, 0.7),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 2,
        )
        assert gate_node.published_target_count == 1

        retained = []
        target_subscription = helper.create_subscription(
            PoseStamped,
            target_topic,
            retained.append,
            retained_pose_qos(),
        )
        assert spin_until(
            (gate_node, helper),
            lambda: len(retained) == 1,
        )
        target = retained[0]
        target_stamp = (
            target.header.stamp.sec * 1_000_000_000
            + target.header.stamp.nanosec
        )
        assert target_stamp == first_source_time
        assert target.pose.position.x == pytest.approx(1.3)
        assert target.pose.position.y == pytest.approx(-1.9)
        assert target.pose.position.z == pytest.approx(1.2)
        assert broadcaster is not None
        assert target_subscription is not None
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_abort_blocks_duplicate_and_delayed_confirm_sequences():
    candidate_topic = '/test/markerless_sequence/candidates'
    intent_topic = '/test/markerless_sequence/intent'
    target_topic = '/test/markerless_sequence/target'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('target_topic', value=target_topic),
                Parameter('required_frames', value=2),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_sequence_test_helper')
        broadcaster = broadcast_test_transform(helper)
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent,
            intent_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
            ),
        )
        spin_until((gate_node, helper), lambda: False, timeout=0.1)

        first_base_time = (
            helper.get_clock().now().nanoseconds - 30_000_000
        )
        for index in range(2):
            candidate_publisher.publish(
                candidate_message(
                    first_base_time + index * 10_000_000
                )
            )
            assert spin_until(
                (gate_node, helper),
                lambda: gate_node.processed_message_count >= index + 1,
            )
        assert gate_node.last_built_target_pose is not None

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 5)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.published_target_count == 1,
        )
        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 5)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 2,
        )
        assert gate_node.published_target_count == 1

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.ABORT, 7)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 3,
        )
        assert gate_node.last_lock_decision.selected_candidate is None

        second_base_time = (
            helper.get_clock().now().nanoseconds - 20_000_000
        )
        candidate_publisher.publish(
            candidate_message(second_base_time)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 3,
        )
        assert gate_node.last_decision.reason == 'warming_up'
        assert gate_node.last_built_target_pose is None

        candidate_publisher.publish(
            candidate_message(second_base_time + 10_000_000)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_built_target_pose is not None,
        )
        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 6)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 4,
        )
        assert not gate_node.last_lock_decision.confirmed
        assert gate_node.published_target_count == 1

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 8)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.published_target_count == 2,
        )
        assert broadcaster is not None
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_confirm_rejects_pose_older_than_candidate_max_age():
    candidate_topic = '/test/markerless_stale_confirm/candidates'
    intent_topic = '/test/markerless_stale_confirm/intent'
    target_topic = '/test/markerless_stale_confirm/target'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('target_topic', value=target_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=0.05),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_stale_confirm_test_helper')
        broadcaster = broadcast_test_transform(helper)
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent,
            intent_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
            ),
        )
        spin_until((gate_node, helper), lambda: False, timeout=0.1)

        candidate_publisher.publish(
            candidate_message(
                helper.get_clock().now().nanoseconds - 5_000_000
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_built_target_pose is not None,
        )
        spin_until((gate_node, helper), lambda: False, timeout=0.08)

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 1)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 1,
        )
        assert gate_node.last_lock_decision.ready
        assert gate_node.published_target_count == 0
        assert broadcaster is not None
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_tf_recovery_and_target_loss_require_new_confirm():
    candidate_topic = '/test/markerless_tf_recovery/candidates'
    intent_topic = '/test/markerless_tf_recovery/intent'
    target_topic = '/test/markerless_tf_recovery/target'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('target_topic', value=target_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_tf_recovery_test_helper')
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            10,
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent,
            intent_topic,
            10,
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
            ),
        )

        candidate_publisher.publish(
            candidate_message(
                helper.get_clock().now().nanoseconds - 5_000_000
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 1,
        )
        assert gate_node.last_built_target_pose is None

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 1)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 1,
        )
        assert gate_node.last_lock_decision.ready
        assert gate_node.published_target_count == 0

        broadcaster = broadcast_test_transform(helper)
        spin_until((gate_node, helper), lambda: False, timeout=0.1)
        candidate_publisher.publish(
            candidate_message(
                helper.get_clock().now().nanoseconds - 5_000_000
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_built_target_pose is not None,
        )
        assert gate_node.published_target_count == 0

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 2)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.published_target_count == 1,
        )

        candidate_publisher.publish(
            candidate_message(
                helper.get_clock().now().nanoseconds,
                detections=(),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 3,
        )
        assert gate_node.last_built_target_pose is None
        assert not gate_node.last_lock_decision.ready

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.CONFIRM, 3)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 3,
        )
        assert gate_node.published_target_count == 1
        assert broadcaster is not None
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def sweeping_qos():
    """Return the latched QoS the handoff controller publishes the flag with."""
    return QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def test_intent_is_suppressed_while_a_search_sweeps():
    # While a search sweeps, a gesture means view direction and nothing else.
    # This node had no idea the robot was searching, so it cycled candidates
    # underneath a wearer who was aiming the camera.
    candidate_topic = '/test/markerless_sweep_gate/candidates'
    intent_topic = '/test/markerless_sweep_gate/intent'
    sweeping_topic = '/test/markerless_sweep_gate/sweeping'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('search_sweeping_topic', value=sweeping_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_sweep_gate_test_helper')
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray, candidate_topic, 10
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent, intent_topic, 10
        )
        sweeping_publisher = helper.create_publisher(
            Bool, sweeping_topic, sweeping_qos()
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
                and sweeping_publisher.get_subscription_count() == 1
            ),
        )

        source_time = helper.get_clock().now().nanoseconds - 50_000_000
        candidate_publisher.publish(
            candidate_message(
                source_time,
                detections=(
                    (2, 'bottle', (0.3, 0.0, 0.7)),
                    (7, 'cup', (0.4, 0.0, 0.7)),
                ),
            )
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 1,
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 2

        sweeping_publisher.publish(Bool(data=True))
        assert spin_until(
            (gate_node, helper), lambda: gate_node.search_sweeping
        )

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.NEXT_TARGET, 0)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.suppressed_intent_count == 1,
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 2

        # Suppression is not rejection: the sequence watermark still advanced,
        # so the same command cannot be replayed once the sweep ends.
        sweeping_publisher.publish(Bool(data=False))
        assert spin_until(
            (gate_node, helper), lambda: not gate_node.search_sweeping
        )
        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.NEXT_TARGET, 0)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_intent_count == 2,
        )
        assert gate_node.last_lock_decision.selected_candidate.track_id == 2

        # A newer sequence acts normally again.
        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.NEXT_TARGET, 1)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                gate_node.last_lock_decision.selected_candidate.track_id == 7
            ),
        )
        assert gate_node.suppressed_intent_count == 1
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_abort_still_clears_the_lock_during_a_sweep():
    # ABORT keeps global priority. It is handled before the sweep gate for the
    # same reason it is handled before every other check in this callback.
    candidate_topic = '/test/markerless_sweep_abort/candidates'
    intent_topic = '/test/markerless_sweep_abort/intent'
    sweeping_topic = '/test/markerless_sweep_abort/sweeping'
    rclpy.init()
    gate_node = None
    helper = None
    try:
        gate_node = MarkerlessCandidateGateNode(
            parameter_overrides=[
                Parameter('candidate_topic', value=candidate_topic),
                Parameter('intent_topic', value=intent_topic),
                Parameter('search_sweeping_topic', value=sweeping_topic),
                Parameter('required_frames', value=1),
                Parameter('max_age_sec', value=1.0),
                Parameter('last_seen_timeout_sec', value=1.0),
            ]
        )
        helper = rclpy.create_node('markerless_sweep_abort_test_helper')
        candidate_publisher = helper.create_publisher(
            ObjectCandidateArray, candidate_topic, 10
        )
        intent_publisher = helper.create_publisher(
            AssistiveIntent, intent_topic, 10
        )
        sweeping_publisher = helper.create_publisher(
            Bool, sweeping_topic, sweeping_qos()
        )
        assert spin_until(
            (gate_node, helper),
            lambda: (
                candidate_publisher.get_subscription_count() == 1
                and intent_publisher.get_subscription_count() == 1
                and sweeping_publisher.get_subscription_count() == 1
            ),
        )

        source_time = helper.get_clock().now().nanoseconds - 50_000_000
        candidate_publisher.publish(candidate_message(source_time))
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.processed_message_count == 1,
        )
        sweeping_publisher.publish(Bool(data=True))
        assert spin_until(
            (gate_node, helper), lambda: gate_node.search_sweeping
        )

        intent_publisher.publish(
            intent_message(helper, AssistiveIntent.ABORT, 3)
        )
        assert spin_until(
            (gate_node, helper),
            lambda: gate_node.last_lock_decision.selected_candidate is None,
        )
        assert gate_node.suppressed_intent_count == 0
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if helper is not None:
            helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_the_flag_defaults_to_not_sweeping_before_any_controller_speaks():
    # The default has to be permissive, because the publisher is latched: a
    # node that starts after the controller gets the real value immediately,
    # and with no controller running there is no sweep to protect against.
    rclpy.init()
    gate_node = None
    try:
        gate_node = MarkerlessCandidateGateNode()
        assert not gate_node.search_sweeping
        assert gate_node.suppressed_intent_count == 0
    finally:
        if gate_node is not None:
            gate_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_the_synthetic_stand_in_publishes_fast_enough_for_this_gate():
    """The seam between two packages whose defaults were each defensible.

    The gate's 0.2 s maximum frame gap is sized for the live candidate path,
    measured delivering 150 of 150 valid frames at 10 Hz. The synthetic
    publisher stands in for that path and defaulted to 5 Hz, which puts the
    gap exactly on the limit: run together on 2026-08-29 the chain locked and
    expired a target ten times in twelve seconds, while the same chain at
    10 Hz locked once and held.

    Neither default was wrong alone, which is why every per-package test
    passed and nothing surfaced until the two ran in one graph. This pins the
    relationship so a change to either one is told which other file it has to
    look at.
    """
    from markerless_object_perception.synthetic_candidate_publisher import (
        SyntheticObjectCandidatePublisher,
    )

    rclpy.init()
    publisher = gate = None
    try:
        publisher = SyntheticObjectCandidatePublisher()
        gate = MarkerlessCandidateGateNode()
        rate = float(publisher.get_parameter('publish_rate_hz').value)
        max_gap = float(gate.get_parameter('max_frame_gap_sec').value)
    finally:
        for node in (publisher, gate):
            if node is not None:
                node.destroy_node()
        rclpy.shutdown()

    assert 1.0 / rate < max_gap, (
        f"the synthetic publisher's {rate} Hz gives a {1.0 / rate:.3f} s "
        f"frame gap and the gate allows {max_gap} s; a stand-in cannot be "
        f"slower than the source it stands in for"
    )
