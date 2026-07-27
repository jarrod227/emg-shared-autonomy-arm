"""Objective 4.1 M4: handoff state machine — timeouts, ABORT, failure
paths, and parameterized staleness rejection for both /target_object_pose
and /hand_observation (release fails closed).

States and transitions:

    IDLE        --CONFIRM + fresh target-->          APPROACH
    APPROACH    --motion done-->                     READY
    READY       --CONFIRM + release gates pass-->    RELEASE
    RELEASE     --release done-->                    RETURN_HOME
    RETURN_HOME --motion done-->                     IDLE

    APPROACH / READY / RELEASE  --ABORT-->           RETURN_HOME
        (M2's release is simulated, so aborting it just cancels a timer;
         the irreversible-gripper policy is a hardware-phase decision)
    READY       --no CONFIRM within ready_timeout--> RETURN_HOME
    APPROACH / RELEASE motion failure or timeout --> RETURN_HOME
    RETURN_HOME motion failure or timeout        --> stays RETURN_HOME with a
        latched fault: IDLE would falsely announce "safely home", so the node
        refuses all further intents until restarted.

Release gates (all must pass, checked on CONFIRM in READY):
- latest /hand_observation exists, valid, and fresh (source-stamp age);
- its frame_id matches delivery_frame (mismatch = refuse, no guessing);
- hand point is within max_delivery_distance of the configured delivery
  center (NOT of /target_object_pose — that is the pickup location).

Release gating is checked once, at the READY -> RELEASE decision (choice
made in M4): once a real gripper starts opening it may be past its
irreversible commit point, so losing the hand signal mid-release does not
automatically mean "interrupt". N-frame stability gating belongs to
Objective 4.2; the real-release cancellation policy is an Objective 5
(hardware-phase) decision.

Still out of scope (later milestones):
- M5: full failure-case test matrix.

Motion is simulated: _start_motion() arms a one-shot completion timer plus
a deadline (watchdog) timer; _on_motion_result() receives the "result".
These are the seam where a MoveIt action goal/result replaces the timers.
Every transition bumps a generation counter (_epoch); timer callbacks
carry the epoch they were armed in and do nothing if it has moved on, so a
stale callback can never mutate the state machine.

Freshness is always judged on header.stamp (source time), never on message
receipt time, per the assistive_interfaces contracts.
"""

import enum
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String

from assistive_interfaces.msg import AssistiveIntent, HandObservation

class HandoffState(enum.Enum):
    IDLE = "idle"
    APPROACH = "approach"
    READY = "ready"
    RELEASE = "release"
    RETURN_HOME = "return_home"


class HandoffController(Node):
    """Drives the handoff cycle from intent + hand + target observations."""

    def __init__(self, **node_kwargs) -> None:
        super().__init__("handoff_controller", **node_kwargs)

        # speed_scale is not consumed by the simulated motion; it is stored
        # for the future MoveIt backend (velocity scaling factor) and only
        # validated here so a bad config fails at startup, not mid-motion.
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("sim_motion_sec", 2.0)
        self.declare_parameter("motion_timeout_sec", 10.0)
        self.declare_parameter("ready_timeout_sec", 30.0)
        self.declare_parameter("max_delivery_distance", 0.5)
        self.declare_parameter("delivery_center_x", 0.4)
        self.declare_parameter("delivery_center_y", 0.3)
        self.declare_parameter("delivery_center_z", 1.0)
        self.declare_parameter("delivery_frame", "world")
        # Max source-stamp age for /target_object_pose. The topic is retained
        # (TRANSIENT_LOCAL), so a late-joining controller can be handed a
        # sample published long ago; this age gate is what makes accepting a
        # retained sample safe.
        self.declare_parameter("target_max_age_sec", 2.0)
        # Max source-stamp age for /hand_observation at the release gate.
        self.declare_parameter("hand_max_age_sec", 1.0)
        # Test hook: name a state ("approach"/"release"/"return_home") whose
        # simulated motion never completes, to exercise the watchdog. Without
        # this the deadline path would be untestable until real hardware.
        self.declare_parameter("simulate_stuck_motion", "")

        self._speed_scale = self.get_parameter("speed_scale").value
        if not 0.0 < self._speed_scale <= 1.0:
            raise ValueError(
                f"speed_scale must be in (0, 1], got {self._speed_scale}"
            )
        self._sim_motion_sec = self.get_parameter("sim_motion_sec").value
        self._motion_timeout_sec = self.get_parameter("motion_timeout_sec").value
        self._ready_timeout_sec = self.get_parameter("ready_timeout_sec").value
        self._max_delivery_distance = self.get_parameter(
            "max_delivery_distance"
        ).value
        self._delivery_center = (
            self.get_parameter("delivery_center_x").value,
            self.get_parameter("delivery_center_y").value,
            self.get_parameter("delivery_center_z").value,
        )
        self._delivery_frame = self.get_parameter("delivery_frame").value
        self._target_max_age_sec = self._positive_finite_param("target_max_age_sec")
        self._hand_max_age_sec = self._positive_finite_param("hand_max_age_sec")
        self._simulate_stuck_motion = self.get_parameter(
            "simulate_stuck_motion"
        ).value

        self._state = HandoffState.IDLE
        self._last_target: PoseStamped | None = None
        self._last_hand: HandObservation | None = None
        self._fault = False
        # Generation counter: bumped on every transition (and fault latch);
        # timer callbacks armed under an older epoch must not act.
        self._epoch = 0
        self._pending_timers = []

        self._intent_sub = self.create_subscription(
            AssistiveIntent, "/assistive_intent", self._on_intent, 10
        )
        self._hand_sub = self.create_subscription(
            HandObservation, "/hand_observation", self._on_hand, 10
        )
        # /target_object_pose is retained (TRANSIENT_LOCAL) by target_selector
        # and fixed_pose_publisher; subscribe with the same durability so a
        # late-starting controller still sees the target. Age checking is what
        # makes accepting a retained sample safe — hardcoded here, M3 hardens.
        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self._target_sub = self.create_subscription(
            PoseStamped, "/target_object_pose", self._on_target, latched_qos
        )

        self._state_pub = self.create_publisher(String, "/handoff_state", 10)
        # Republish the current state at 1 Hz so `ros2 topic echo` joined
        # mid-cycle still shows it; transitions also publish immediately.
        self._state_republish_timer = self.create_timer(1.0, self._publish_state)

        self.get_logger().info(
            "handoff_controller started in state 'idle' "
            f"(speed_scale={self._speed_scale}, "
            f"delivery_center={self._delivery_center} in "
            f"'{self._delivery_frame}')"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_target(self, msg: PoseStamped) -> None:
        self._last_target = msg

    def _on_hand(self, msg: HandObservation) -> None:
        self._last_hand = msg

    def _on_intent(self, msg: AssistiveIntent) -> None:
        if self._fault:
            self.get_logger().error(
                "fault latched (return_home failed): ignoring intent; "
                "restart the controller to recover"
            )
            return
        command = msg.command
        if command == AssistiveIntent.CONFIRM:
            self._on_confirm()
        elif command == AssistiveIntent.ABORT:
            self._on_abort()
        elif command == AssistiveIntent.NEXT_TARGET:
            # M2 still has a single simulated target; cycling arrives with
            # the multi-candidate selector integration.
            self.get_logger().info("NEXT_TARGET received: no-op in M2")
        else:
            self.get_logger().warning(f"unknown intent command {command}: ignoring")

    # ------------------------------------------------------------------
    # Intent handling
    # ------------------------------------------------------------------

    def _on_confirm(self) -> None:
        if self._state is HandoffState.IDLE:
            if self._target_is_fresh():
                self._transition(HandoffState.APPROACH)
            # else: _target_is_fresh already logged why the gate refused
        elif self._state is HandoffState.READY:
            if self._release_gates_pass():
                self._transition(HandoffState.RELEASE)
        else:
            self.get_logger().info(
                f"CONFIRM ignored in state '{self._state.value}'"
            )

    def _on_abort(self) -> None:
        if self._state in (
            HandoffState.APPROACH,
            HandoffState.READY,
            HandoffState.RELEASE,
        ):
            # Phase-0 release is a timer, so cancelling it is safe. The
            # real-gripper mid-RELEASE abort policy (commit point, whether
            # to interrupt at all) is an Objective 5 hardware-phase decision.
            self.get_logger().warning(
                f"ABORT in '{self._state.value}': cancelling, returning home"
            )
            self._transition(HandoffState.RETURN_HOME)
        else:
            self.get_logger().info(
                f"ABORT ignored in state '{self._state.value}'"
            )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _target_is_fresh(self) -> bool:
        if self._last_target is None:
            self.get_logger().warning("CONFIRM refused: no /target_object_pose yet")
            return False
        age = self._age_sec(self._last_target.header.stamp)
        if age > self._target_max_age_sec:
            self.get_logger().warning(
                f"CONFIRM refused: target is {age:.2f}s old "
                f"(max {self._target_max_age_sec}s)"
            )
            return False
        return True

    def _release_gates_pass(self) -> bool:
        hand = self._last_hand
        if hand is None:
            self.get_logger().warning("release refused: no /hand_observation yet")
            return False
        if not hand.valid:
            self.get_logger().warning("release refused: latest observation is no-hand")
            return False
        age = self._age_sec(hand.header.stamp)
        if age > self._hand_max_age_sec:
            self.get_logger().warning(
                f"release refused: hand observation is {age:.2f}s old "
                f"(max {self._hand_max_age_sec}s)"
            )
            return False
        if hand.header.frame_id != self._delivery_frame:
            # Refusing beats silently comparing points in different frames.
            self.get_logger().warning(
                f"release refused: hand frame '{hand.header.frame_id}' != "
                f"delivery frame '{self._delivery_frame}'"
            )
            return False
        distance = math.dist(
            (hand.point.x, hand.point.y, hand.point.z), self._delivery_center
        )
        if distance > self._max_delivery_distance:
            self.get_logger().warning(
                f"release refused: hand is {distance:.3f}m from delivery "
                f"center (max {self._max_delivery_distance}m)"
            )
            return False
        return True

    def _age_sec(self, stamp) -> float:
        return (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9

    def _positive_finite_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")
        return value

    # ------------------------------------------------------------------
    # State transitions, simulated motion, timeouts
    # ------------------------------------------------------------------

    def _transition(self, new_state: HandoffState) -> None:
        self._epoch += 1
        self._cancel_pending_timers()
        self.get_logger().info(
            f"state: {self._state.value} -> {new_state.value}"
        )
        self._state = new_state
        self._publish_state()

        # Entry actions: motion states start their (simulated) motion; READY
        # arms the dwell timeout so the arm never hovers indefinitely.
        if new_state is HandoffState.APPROACH:
            self._start_motion(HandoffState.READY)
        elif new_state is HandoffState.RELEASE:
            # Phase-0 release is a simulated transition: no gripper actuation.
            self._start_motion(HandoffState.RETURN_HOME)
        elif new_state is HandoffState.RETURN_HOME:
            self._start_motion(HandoffState.IDLE)
        elif new_state is HandoffState.READY:
            epoch = self._epoch
            self._pending_timers.append(
                self.create_timer(
                    self._ready_timeout_sec,
                    lambda: self._on_ready_timeout(epoch),
                )
            )

    def _start_motion(self, result_state: HandoffState) -> None:
        # M2: a one-shot timer stands in for sending a MoveIt action goal;
        # the deadline timer is the watchdog for a motion that never reports.
        # Later milestones replace the completion timer with the real goal
        # request and route the action result into _on_motion_result.
        epoch = self._epoch
        if self._simulate_stuck_motion == self._state.value:
            self.get_logger().warning(
                f"simulate_stuck_motion: motion in '{self._state.value}' "
                "will never complete"
            )
        else:
            self._pending_timers.append(
                self.create_timer(
                    self._sim_motion_sec,
                    lambda: self._on_motion_result(epoch, result_state),
                )
            )
        self._pending_timers.append(
            self.create_timer(
                self._motion_timeout_sec,
                lambda: self._on_motion_timeout(epoch),
            )
        )

    def _on_motion_result(
        self, epoch: int, result_state: HandoffState
    ) -> None:
        if epoch != self._epoch:
            return  # stale callback from a superseded motion
        self._transition(result_state)

    def _on_motion_timeout(self, epoch: int) -> None:
        if epoch != self._epoch:
            return
        self.get_logger().error(
            f"motion in '{self._state.value}' exceeded "
            f"{self._motion_timeout_sec}s"
        )
        self._on_motion_failure()

    def _on_motion_failure(self) -> None:
        if self._state is HandoffState.RETURN_HOME:
            # Nowhere safer to go, and claiming IDLE would falsely announce
            # "safely home". Stay put, latch the fault, refuse new work.
            self._fault = True
            self._epoch += 1
            self._cancel_pending_timers()
            self.get_logger().error(
                "fault latched: return_home failed; staying in 'return_home' "
                "and refusing intents until restart"
            )
        else:
            self.get_logger().warning(
                f"motion failure in '{self._state.value}': returning home"
            )
            self._transition(HandoffState.RETURN_HOME)

    def _on_ready_timeout(self, epoch: int) -> None:
        if epoch != self._epoch:
            return
        self.get_logger().warning(
            f"no CONFIRM within {self._ready_timeout_sec}s in 'ready': "
            "returning home"
        )
        self._transition(HandoffState.RETURN_HOME)

    def _cancel_pending_timers(self) -> None:
        for timer in self._pending_timers:
            self.destroy_timer(timer)
        self._pending_timers.clear()

    def _publish_state(self) -> None:
        self._state_pub.publish(String(data=self._state.value))


def main() -> None:
    rclpy.init()
    node = HandoffController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
