"""The 25 node-level tests for the completed Objective 4.1 Phase-0 controller.

No simulator needed: the test process publishes intent / hand / target
itself (mirroring the sim publishers) and watches the controller's state.
The suite covers the full simulated cycle, parameterized target/hand freshness
gates, timeouts, ABORT paths, callback races, and fault latching. Timing
parameters are shrunk via parameter_overrides so each test runs quickly.
"""

import time

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Float64, String

from assistive_handoff.handoff_controller import (
    HandoffController,
    HandoffState,
    SearchPhase,
)
from assistive_interfaces.msg import (
    AssistiveIntent,
    HandObservation,
    ViewControlCommand,
)

FAST_PARAMS = {
    "sim_motion_sec": 0.2,
    "motion_timeout_sec": 2.0,
    "ready_timeout_sec": 5.0,
}

SEARCH_PARAMS = {
    "view_update_period_sec": 0.01,
    "view_command_max_age_sec": 0.5,
    "view_watchdog_sec": 0.15,
    "search_timeout_sec": 2.0,
    "view_stable_command_count": 1,
    "view_target_update_min_rad": 0.0,
    "target_search_relative_limit": 0.6,
    "target_search_min_angle": -1.0,
    "target_search_max_angle": 1.0,
    "target_search_nominal_speed": 1.0,
    "target_search_acceleration": 4.0,
    "target_search_deceleration": 4.0,
    "handoff_search_relative_limit": 0.3,
    "handoff_search_min_angle": -1.0,
    "handoff_search_max_angle": 1.0,
    "handoff_search_nominal_speed": 0.5,
    "handoff_search_acceleration": 2.0,
    "handoff_search_deceleration": 4.0,
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
        self._view_pub = self.helper.create_publisher(
            ViewControlCommand,
            "/assistive_view_control",
            QoSProfile(
                depth=1, durability=QoSDurabilityPolicy.VOLATILE
            ),
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
        self.angles = []
        self.helper.create_subscription(
            Float64,
            "/simulated_view_angle",
            lambda msg: self.angles.append(msg.data),
            10,
        )

        self._sequence = 0
        self._view_sequence = 0
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

    def send_view(
        self,
        direction,
        *,
        activation=0.8,
        confidence=1.0,
        signal_quality=1.0,
        age_sec=0.0,
        sequence=None,
    ):
        msg = ViewControlCommand()
        stamp = self.helper.get_clock().now() - Duration(seconds=age_sec)
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "test_view"
        msg.direction = direction
        msg.activation = activation
        msg.confidence = confidence
        msg.signal_quality = signal_quality
        if sequence is None:
            sequence = self._view_sequence
            self._view_sequence += 1
        msg.sequence = sequence
        self._view_pub.publish(msg)

    def wait_for_state(self, state, timeout=3.0):
        return spin_until(
            self.nodes,
            lambda: (
                self.state is state
                and bool(self.states)
                and self.states[-1] == state.value
            ),
            timeout,
        )

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


def test_abort_in_ready_returns_home(make_graph):
    # READY's entry arms the dwell-timeout timer; ABORT must cancel it and
    # go home (a distinct entry path from the approach/release aborts).
    g = make_graph()
    g.confirm_to_ready()
    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.RETURN_HOME)
    assert g.wait_for_state(HandoffState.IDLE)
    assert "release" not in g.states


def test_abort_ignored_in_idle(make_graph):
    # A safety command arriving in the wrong state must cause no transition;
    # an unchanged epoch proves none happened (state value alone could hide
    # a spurious transition back to the same state).
    g = make_graph()
    g.settle(0.3)
    epoch_before = g.controller._epoch
    g.send_intent(AssistiveIntent.ABORT)
    g.settle(0.4)
    assert g.state is HandoffState.IDLE
    assert g.controller._epoch == epoch_before


def test_intents_ignored_in_return_home(make_graph):
    # Freeze return_home so it persists, then hammer it with CONFIRM and
    # ABORT: neither may transition, restart the motion, or latch a fault.
    g = make_graph({"simulate_stuck_motion": "return_home"})
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.RETURN_HOME)
    epoch_before = g.controller._epoch
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.3)
    g.send_intent(AssistiveIntent.ABORT)
    g.settle(0.3)
    assert g.state is HandoffState.RETURN_HOME
    assert g.controller._epoch == epoch_before
    assert not g.controller._fault


def test_confirm_ignored_in_approach(make_graph):
    # Operator mashing CONFIRM mid-motion must not skip ahead or restart
    # the motion.
    g = make_graph({"simulate_stuck_motion": "approach"})
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    epoch_before = g.controller._epoch
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.4)
    assert g.state is HandoffState.APPROACH
    assert g.controller._epoch == epoch_before
    assert "ready" not in g.states


def test_stale_epoch_callbacks_do_not_act(make_graph):
    # The epoch guard is the race-safety mechanism: a timer callback armed
    # before a transition must be a no-op after it. Call the callbacks
    # directly with an outdated epoch — deterministic, no timing games.
    g = make_graph({"simulate_stuck_motion": "approach"})
    g.settle(0.3)
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.APPROACH)
    stale_epoch = g.controller._epoch - 1
    g.controller._on_motion_result(stale_epoch, HandoffState.RELEASE)
    assert g.state is HandoffState.APPROACH, "stale result callback acted"
    g.controller._on_motion_timeout(stale_epoch)
    assert g.state is HandoffState.APPROACH, "stale timeout callback acted"
    assert not g.controller._fault


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


def make_search_graph(
    make_graph,
    overrides=None,
    *,
    publish_target=False,
    publish_hand=False,
):
    params = dict(SEARCH_PARAMS)
    if overrides:
        params.update(overrides)
    return make_graph(
        params,
        publish_target=publish_target,
        publish_hand=publish_hand,
    )


def test_view_search_not_started_by_hold_or_when_target_is_fresh(make_graph):
    g = make_search_graph(make_graph, publish_target=True)
    assert g.wait_for_target_receipt()

    g.send_view(ViewControlCommand.HOLD, activation=0.0)
    g.settle(0.1)
    assert g.state is HandoffState.IDLE

    g.send_view(ViewControlCommand.RIGHT)
    g.settle(0.1)
    assert g.state is HandoffState.IDLE


def test_target_search_requires_post_stop_target_and_confirm(make_graph):
    g = make_search_graph(make_graph)
    g.settle(0.2)

    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)

    # This target stops the camera, but cannot itself be used for approach.
    g.send_target()
    assert spin_until(
        g.nodes,
        lambda: (
            g.controller._search_phase
            is SearchPhase.WAITING_FOR_FRESH_OBSERVATION
        ),
    )
    g.send_intent(AssistiveIntent.CONFIRM)
    g.settle(0.1)
    assert g.state is HandoffState.TARGET_SEARCH

    # A second source-stamped pose acquired after the confirmed stop locks.
    g.send_target()
    assert spin_until(
        g.nodes,
        lambda: g.controller._search_phase is SearchPhase.LOCKED,
    )
    g.send_intent(AssistiveIntent.CONFIRM)
    assert g.wait_for_state(HandoffState.READY)
    assert g.controller._holding_object


def test_target_search_watchdog_holds_and_new_command_resumes(make_graph):
    g = make_search_graph(
        make_graph,
        {"view_watchdog_sec": 0.08},
    )
    g.settle(0.2)
    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)
    requested_target = g.controller._last_requested_view_target

    assert spin_until(
        g.nodes,
        lambda: not g.controller._view_motion.moving,
        timeout=1.0,
    )
    assert g.state is HandoffState.TARGET_SEARCH
    assert g.controller._view_motion.position < requested_target

    g.send_view(ViewControlCommand.LEFT)
    assert spin_until(
        g.nodes,
        lambda: g.controller._view_motion.velocity < 0.0,
    )


def test_new_view_command_serializes_preemption(make_graph):
    g = make_search_graph(
        make_graph,
        {"view_watchdog_sec": 1.0},
    )
    g.settle(0.2)
    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)

    g.send_view(ViewControlCommand.LEFT)
    g.settle(0.03)
    motion = g.controller._view_motion
    selected = (
        motion.pending_target
        if motion.pending_target is not None
        else motion.active_target
    )
    assert selected is not None and selected < 0.0
    assert motion.goal_count <= 1


def test_stale_low_quality_and_duplicate_view_commands_are_ignored(make_graph):
    g = make_search_graph(make_graph)
    g.settle(0.2)

    g.send_view(ViewControlCommand.RIGHT, age_sec=1.0)
    g.settle(0.1)
    assert g.state is HandoffState.IDLE

    g.send_view(ViewControlCommand.RIGHT, signal_quality=0.1)
    g.settle(0.1)
    assert g.state is HandoffState.IDLE

    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)
    accepted_sequence = g.controller._last_view_sequence

    g.send_view(ViewControlCommand.LEFT, sequence=accepted_sequence)
    g.settle(0.05)
    assert g.controller._last_view_sequence == accepted_sequence
    assert g.controller._view_candidate_direction == 1


def test_abort_in_target_search_emergency_stops_and_returns_home(make_graph):
    g = make_search_graph(make_graph)
    g.settle(0.2)
    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)

    g.send_intent(AssistiveIntent.ABORT)
    assert g.wait_for_state(HandoffState.RETURN_HOME)
    assert not g.controller._view_motion.moving
    assert g.wait_for_state(HandoffState.IDLE)


def test_handoff_search_refused_without_held_object(make_graph):
    g = make_search_graph(make_graph)
    g.controller._transition(HandoffState.HANDOFF_SEARCH)

    assert g.state is HandoffState.IDLE
    assert not g.controller._holding_object


def test_handoff_search_uses_loaded_limits_and_post_stop_hand(make_graph):
    g = make_search_graph(
        make_graph,
        publish_target=True,
        publish_hand=False,
    )
    g.confirm_to_ready()
    assert g.controller._holding_object

    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.HANDOFF_SEARCH)
    assert g.controller._active_search_profile.relative_limit == pytest.approx(
        SEARCH_PARAMS["handoff_search_relative_limit"]
    )
    assert g.controller._active_search_profile.nominal_speed == pytest.approx(
        SEARCH_PARAMS["handoff_search_nominal_speed"]
    )
    assert spin_until(g.nodes, lambda: g.controller._view_motion.velocity > 0.0)

    g.send_hand()
    assert spin_until(
        g.nodes,
        lambda: (
            g.controller._search_phase
            is SearchPhase.WAITING_FOR_FRESH_OBSERVATION
        ),
    )
    assert g.state is HandoffState.HANDOFF_SEARCH

    g.send_hand()
    assert g.wait_for_state(HandoffState.READY)
    assert g.controller._holding_object


def test_search_timeout_stops_view_before_return_home(make_graph):
    g = make_search_graph(
        make_graph,
        {
            "search_timeout_sec": 0.12,
            "view_watchdog_sec": 1.0,
        },
    )
    g.settle(0.2)
    g.send_view(ViewControlCommand.RIGHT)
    assert g.wait_for_state(HandoffState.TARGET_SEARCH)

    assert g.wait_for_state(HandoffState.RETURN_HOME)
    assert not g.controller._view_motion.moving
    assert g.wait_for_state(HandoffState.IDLE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
