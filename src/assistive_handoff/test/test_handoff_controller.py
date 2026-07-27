"""Node-level tests for HandoffController (Objective 4.1 M2 behaviors).

No simulator needed: the test process publishes intent / hand / target
itself (mirroring the sim publishers) and watches the controller's state.
Timing parameters are shrunk via parameter_overrides so every test runs in
well under a second of simulated-motion time.
"""

import time

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String

from assistive_handoff.handoff_controller import HandoffController, HandoffState
from assistive_interfaces.msg import AssistiveIntent, HandObservation

FAST_PARAMS = {
    "sim_motion_sec": 0.2,
    "motion_timeout_sec": 2.0,
    "ready_timeout_sec": 5.0,
}

HAND_POINT = (0.4, 0.3, 1.0)  # matches the default delivery center


def spin_until(nodes, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return True
    return False


class Graph:
    """Controller under test + a helper node faking every input stream."""

    def __init__(
        self,
        param_overrides,
        hand_point,
        hand_frame,
        hand_valid,
        publish_target,
        publish_hand,
    ):
        params = dict(FAST_PARAMS)
        params.update(param_overrides)
        self.controller = HandoffController(
            parameter_overrides=[
                Parameter(name, value=value) for name, value in params.items()
            ]
        )
        self.helper = rclpy.create_node("test_helper")

        self._intent_pub = self.helper.create_publisher(
            AssistiveIntent, "/assistive_intent", 10
        )
        self._hand_pub = self.helper.create_publisher(
            HandObservation, "/hand_observation", 10
        )
        self._target_pub = self.helper.create_publisher(
            PoseStamped,
            "/target_object_pose",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.states = []
        self.helper.create_subscription(
            String, "/handoff_state", lambda msg: self.states.append(msg.data), 10
        )

        self._sequence = 0
        self._hand_point = hand_point
        self._hand_frame = hand_frame
        self._hand_valid = hand_valid
        self._auto_target = publish_target
        self._auto_hand = publish_hand
        # Keep hand and target continuously fresh, like the sim publishers;
        # staleness tests disable the auto stream and publish aged stamps
        # through send_target()/send_hand() instead (an auto stream would
        # immediately overwrite an artificially aged message).
        self.helper.create_timer(0.05, lambda: self._publish_hand())
        self.helper.create_timer(0.05, lambda: self._publish_target())

    @property
    def nodes(self):
        return [self.controller, self.helper]

    @property
    def state(self):
        return self.controller._state

    def _publish_hand(self):
        if not self._auto_hand:
            return
        self._hand_pub.publish(self._make_hand(age_sec=0.0))

    def send_hand(self, age_sec=0.0):
        """Publish one hand observation stamped age_sec in the past."""
        self._hand_pub.publish(self._make_hand(age_sec))

    def _make_hand(self, age_sec):
        msg = HandObservation()
        stamp = self.helper.get_clock().now() - Duration(seconds=age_sec)
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self._hand_frame
        msg.valid = self._hand_valid
        msg.point.x, msg.point.y, msg.point.z = self._hand_point
        msg.confidence = 0.9
        return msg

    def wait_for_hand_receipt(self, timeout=2.0):
        """Same false-pass guard as wait_for_target_receipt, for the hand."""
        return spin_until(
            self.nodes, lambda: self.controller._last_hand is not None, timeout
        )

    def _publish_target(self):
        if not self._auto_target:
            return
        self._target_pub.publish(self._make_target(age_sec=0.0))

    def send_target(self, age_sec=0.0):
        """Publish one target whose source stamp is age_sec in the past."""
        self._target_pub.publish(self._make_target(age_sec))

    def _make_target(self, age_sec):
        msg = PoseStamped()
        stamp = self.helper.get_clock().now() - Duration(seconds=age_sec)
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = 0.106982
        msg.pose.position.z = 1.121022
        msg.pose.orientation.w = 1.0
        return msg

    def wait_for_target_receipt(self, timeout=2.0):
        """Guard against false passes: staying idle only proves a refusal if
        the controller demonstrably received the (stale) target first."""
        return spin_until(
            self.nodes, lambda: self.controller._last_target is not None, timeout
        )

    def send_intent(self, command):
        msg = AssistiveIntent()
        msg.header.stamp = self.helper.get_clock().now().to_msg()
        msg.header.frame_id = "test_intent"
        msg.command = command
        msg.confidence = 1.0
        msg.sequence = self._sequence
        self._sequence += 1
        self._intent_pub.publish(msg)

    def wait_for_state(self, state, timeout=3.0):
        return spin_until(self.nodes, lambda: self.state is state, timeout)

    def settle(self, seconds):
        spin_until(self.nodes, lambda: False, timeout=seconds)

    def confirm_to_ready(self):
        """Drive IDLE -> APPROACH -> READY (the shared test preamble)."""
        self.settle(0.3)  # let pubs/subs match and first inputs land
        self.send_intent(AssistiveIntent.CONFIRM)
        assert self.wait_for_state(HandoffState.READY), (
            f"never reached ready (state: {self.state})"
        )


@pytest.fixture
def make_graph():
    rclpy.init()
    graphs = []

    def factory(
        param_overrides={},
        hand_point=HAND_POINT,
        hand_frame="world",
        hand_valid=True,
        publish_target=True,
        publish_hand=True,
    ):
        graph = Graph(
            param_overrides,
            hand_point,
            hand_frame,
            hand_valid,
            publish_target,
            publish_hand,
        )
        graphs.append(graph)
        return graph

    yield factory

    for graph in graphs:
        graph.controller.destroy_node()
        graph.helper.destroy_node()
    rclpy.shutdown()


def test_happy_path_full_cycle(make_graph):
    g = make_graph()
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.IDLE), "never returned to idle"
    # Published sequence, deduplicated (the 1 Hz republish repeats states);
    # the cycle finishes before the first republish, so it starts at approach.
    dedup = [s for i, s in enumerate(g.states) if i == 0 or s != g.states[i - 1]]
    assert dedup[-5:] == ["approach", "ready", "release", "return_home", "idle"]


def test_confirm_without_target_stays_idle(make_graph):
    # No target stream at all: the gate must refuse CONFIRM.
    g = make_graph(publish_target=False)
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.IDLE


def test_abort_in_approach_returns_home(make_graph):
    # Freeze approach so the abort (not motion completion) causes the exit.
    g = make_graph({"simulate_stuck_motion": "approach"})
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.IDLE), "abort did not return home"
    assert "ready" not in g.states


def test_abort_in_release_cancels_and_returns_home(make_graph):
    # Freeze release: only ABORT can move the machine on from it.
    g = make_graph({"simulate_stuck_motion": "release"})
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.RELEASE)
    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.IDLE), "abort did not cancel release"


def test_ready_timeout_returns_home(make_graph):
    g = make_graph({"ready_timeout_sec": 0.4})
    g.confirm_to_ready()
    # No second CONFIRM: the dwell timeout alone must drive it home.
    assert g.wait_for_state(HandoffState.RETURN_HOME)
    assert g.wait_for_state(HandoffState.IDLE)


def test_release_refused_when_hand_far_from_delivery_center(make_graph):
    # Hand sits at the pickup-side point; delivery center moved 0.5 m away.
    g = make_graph(
        {
            "delivery_center_x": 0.0,
            "delivery_center_y": 0.0,
            "max_delivery_distance": 0.3,
        }
    )
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY, "release happened despite distance"


def test_release_refused_on_frame_mismatch(make_graph):
    g = make_graph(hand_frame="camera")
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY, "release happened despite frame mismatch"


def test_release_refused_on_no_hand_flag(make_graph):
    # valid=false is the *explicit* no-hand signal — a live stream saying
    # "I looked and there is no hand" (distinct from never-received below).
    g = make_graph(hand_valid=False)
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY, "release happened despite no-hand"


def test_release_refused_when_hand_never_received(make_graph):
    # No hand stream at all: _last_hand stays None and release must refuse.
    g = make_graph(publish_hand=False)
    g.confirm_to_ready()
    assert g.controller._last_hand is None
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY, "release happened with no hand stream"


def test_release_refused_when_hand_is_stale(make_graph):
    # A valid, in-range hand whose source stamp is simply too old.
    g = make_graph(publish_hand=False)
    g.confirm_to_ready()
    g.send_hand(age_sec=2.0)  # default hand_max_age_sec is 1.0
    assert g.wait_for_hand_receipt(), "aged hand never delivered"
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY, "stale hand accepted for release"


def test_half_second_old_hand_allowed_by_default(make_graph):
    g = make_graph(publish_hand=False)
    g.confirm_to_ready()
    g.send_hand(age_sec=0.5)
    assert g.wait_for_hand_receipt()
    g.send_intent(AssistiveIntent.CONFIRM)
    # 0.5s old is inside the default 1.0s window: release must start.
    assert spin_until(g.nodes, lambda: "release" in g.states), (
        "fresh-enough hand refused"
    )


def test_half_second_old_hand_refused_when_max_age_tightened(make_graph):
    # Same 0.5s-old observation as above, but the window is now 0.1s.
    g = make_graph({"hand_max_age_sec": 0.1}, publish_hand=False)
    g.confirm_to_ready()
    g.send_hand(age_sec=0.5)
    assert g.wait_for_hand_receipt()
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.READY
    assert "release" not in g.states, "stale hand accepted for release"


def test_hand_max_age_must_be_positive_and_finite():
    rclpy.init()
    try:
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                HandoffController(
                    parameter_overrides=[
                        Parameter("hand_max_age_sec", value=bad)
                    ]
                )
    finally:
        rclpy.shutdown()


def test_stuck_approach_watchdog_returns_home(make_graph):
    g = make_graph(
        {"simulate_stuck_motion": "approach", "motion_timeout_sec": 0.4}
    )
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    assert g.wait_for_state(HandoffState.IDLE), "watchdog never recovered"
    assert "ready" not in g.states


def test_return_home_failure_latches_fault(make_graph):
    g = make_graph(
        {"simulate_stuck_motion": "return_home", "motion_timeout_sec": 0.4}
    )
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.RETURN_HOME)
    # Watchdog fires; the fault must latch instead of transitioning to idle.
    assert spin_until(g.nodes, lambda: g.controller._fault, timeout=2.0)
    assert g.state is HandoffState.RETURN_HOME
    # A latched fault refuses new work: CONFIRM must change nothing.
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.RETURN_HOME
    assert "idle" not in g.states[1:], "fault state leaked back to idle"


def test_half_second_old_target_allowed_by_default(make_graph):
    g = make_graph(publish_target=False)
    g.settle(0.3)
    g.send_target(age_sec=0.5)
    assert g.wait_for_target_receipt()
    g.send_intent(AssistiveIntent.CONFIRM)
    # 0.5s old is well inside the default 2.0s window: approach must start.
    assert g.wait_for_state(HandoffState.APPROACH), "fresh-enough target refused"


def test_half_second_old_target_refused_when_max_age_tightened(make_graph):
    # Same 0.5s-old message as above, but the window is now 0.1s.
    g = make_graph({"target_max_age_sec": 0.1}, publish_target=False)
    g.settle(0.3)
    g.send_target(age_sec=0.5)
    assert g.wait_for_target_receipt()
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.IDLE, "stale target accepted"


def test_confirm_refused_when_retained_target_predates_subscriber(make_graph):
    # A latched publisher that outlives its single stale-stamped publish;
    # the controller is created afterwards, so the sample it receives comes
    # from DDS TRANSIENT_LOCAL retention — the exact late-joiner hazard.
    early = rclpy.create_node("early_target_publisher")
    early_pub = early.create_publisher(
        PoseStamped,
        "/target_object_pose",
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    )
    msg = PoseStamped()
    stamp = early.get_clock().now() - Duration(seconds=5.0)
    msg.header.stamp = stamp.to_msg()
    msg.header.frame_id = "world"
    msg.pose.orientation.w = 1.0
    early_pub.publish(msg)

    g = make_graph(publish_target=False)  # controller joins after the publish
    assert g.wait_for_target_receipt(), "retained sample never delivered"
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.5)
    assert g.state is HandoffState.IDLE, "retained stale target accepted"
    early.destroy_node()


def test_target_max_age_must_be_positive_and_finite():
    rclpy.init()
    try:
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                HandoffController(
                    parameter_overrides=[
                        Parameter("target_max_age_sec", value=bad)
                    ]
                )
    finally:
        rclpy.shutdown()


def test_speed_scale_out_of_range_rejected():
    rclpy.init()
    try:
        with pytest.raises(ValueError):
            HandoffController(
                parameter_overrides=[Parameter("speed_scale", value=1.5)]
            )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
