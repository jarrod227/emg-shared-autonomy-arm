"""Objective 4.1 M1: handoff state machine skeleton (happy path only).

States and M1 transitions:

    IDLE        --CONFIRM + fresh target-->        APPROACH
    APPROACH    --simulated motion done-->         READY
    READY       --CONFIRM + fresh valid hand-->    RELEASE
    RELEASE     --simulated release done-->        RETURN_HOME
    RETURN_HOME --simulated motion done-->         IDLE

Deliberately out of scope in M1 (later milestones):
- M2: timeouts, ABORT cancellation, failure transitions, configurable
  speed / delivery-distance parameters.
- M3: configurable max-age parameters and rejection of retained
  /target_object_pose samples (M1 uses hardcoded age constants below).
- M4: fail-closed release testing when /hand_observation is absent/stale.
- M5: failure-case tests.

Motion is simulated: _start_motion() arms a one-shot timer and
_on_motion_result() receives its "result". These two methods are the seam
where a MoveIt action goal / result callback replaces the timer in a later
milestone; the state logic must not need to change for that swap.

Freshness is always judged on header.stamp (source time), never on message
receipt time, per the assistive_interfaces contracts.
"""

import enum

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String

from assistive_interfaces.msg import AssistiveIntent, HandObservation

# M1 hardcoded freshness limits; promoted to declared parameters in M3.
# Sim publishers run at 2 Hz (target) and 10 Hz (hand), so a healthy stream
# stays far below these ages and a stopped publisher exceeds them quickly.
TARGET_MAX_AGE_SEC = 2.0
HAND_MAX_AGE_SEC = 1.0

# One-shot timer period standing in for real motion duration.
SIMULATED_MOTION_SEC = 2.0


class HandoffState(enum.Enum):
    IDLE = "idle"
    APPROACH = "approach"
    READY = "ready"
    RELEASE = "release"
    RETURN_HOME = "return_home"


class HandoffController(Node):
    """Drives the handoff cycle from intent + hand + target observations."""

    def __init__(self) -> None:
        super().__init__("handoff_controller")

        self._state = HandoffState.IDLE
        self._last_target: PoseStamped | None = None
        self._last_hand: HandObservation | None = None
        self._motion_timer = None

        # Intent and hand streams are reliable + volatile per the message
        # contracts (never retained); default depth-10 QoS matches that.
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

        self.get_logger().info("handoff_controller started in state 'idle'")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_target(self, msg: PoseStamped) -> None:
        self._last_target = msg

    def _on_hand(self, msg: HandObservation) -> None:
        self._last_hand = msg

    def _on_intent(self, msg: AssistiveIntent) -> None:
        command = msg.command
        if command == AssistiveIntent.CONFIRM:
            self._on_confirm()
        elif command == AssistiveIntent.NEXT_TARGET:
            # M1 has a single simulated target; cycling arrives with the
            # multi-candidate selector integration.
            self.get_logger().info("NEXT_TARGET received: no-op in M1")
        elif command == AssistiveIntent.ABORT:
            # Loud on purpose: silently ignoring a safety command would be
            # easy to miss. Cancellation transitions land in M2.
            self.get_logger().warning(
                "ABORT received but not handled until M2 — ignoring"
            )
        else:
            self.get_logger().warning(f"unknown intent command {command}: ignoring")

    # ------------------------------------------------------------------
    # CONFIRM gates
    # ------------------------------------------------------------------

    def _on_confirm(self) -> None:
        if self._state is HandoffState.IDLE:
            if self._target_is_fresh():
                self._transition(HandoffState.APPROACH)
            # else: _target_is_fresh already logged why the gate refused
        elif self._state is HandoffState.READY:
            if self._hand_is_fresh_and_valid():
                self._transition(HandoffState.RELEASE)
        else:
            self.get_logger().info(
                f"CONFIRM ignored in state '{self._state.value}'"
            )

    def _target_is_fresh(self) -> bool:
        if self._last_target is None:
            self.get_logger().warning("CONFIRM refused: no /target_object_pose yet")
            return False
        age = self._age_sec(self._last_target.header.stamp)
        if age > TARGET_MAX_AGE_SEC:
            self.get_logger().warning(
                f"CONFIRM refused: target is {age:.2f}s old "
                f"(max {TARGET_MAX_AGE_SEC}s)"
            )
            return False
        return True

    def _hand_is_fresh_and_valid(self) -> bool:
        if self._last_hand is None:
            self.get_logger().warning("release refused: no /hand_observation yet")
            return False
        if not self._last_hand.valid:
            self.get_logger().warning("release refused: latest observation is no-hand")
            return False
        age = self._age_sec(self._last_hand.header.stamp)
        if age > HAND_MAX_AGE_SEC:
            self.get_logger().warning(
                f"release refused: hand observation is {age:.2f}s old "
                f"(max {HAND_MAX_AGE_SEC}s)"
            )
            return False
        return True

    def _age_sec(self, stamp) -> float:
        return (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9

    # ------------------------------------------------------------------
    # State transitions and simulated motion
    # ------------------------------------------------------------------

    def _transition(self, new_state: HandoffState) -> None:
        self.get_logger().info(
            f"state: {self._state.value} -> {new_state.value}"
        )
        self._state = new_state
        self._publish_state()

        # Entry actions: states whose exit is a motion/action result rather
        # than an intent event start their (simulated) motion here.
        if new_state is HandoffState.APPROACH:
            self._start_motion(HandoffState.READY)
        elif new_state is HandoffState.RELEASE:
            # Phase-0 release is a simulated transition: no gripper actuation.
            self._start_motion(HandoffState.RETURN_HOME)
        elif new_state is HandoffState.RETURN_HOME:
            self._start_motion(HandoffState.IDLE)

    def _start_motion(self, result_state: HandoffState) -> None:
        # M1: a one-shot timer stands in for sending a MoveIt action goal.
        # Later milestones replace this body with the real goal request and
        # route the action result into _on_motion_result unchanged.
        self._motion_timer = self.create_timer(
            SIMULATED_MOTION_SEC,
            lambda: self._on_motion_result(result_state),
        )

    def _on_motion_result(self, result_state: HandoffState) -> None:
        # rclpy timers repeat, so a one-shot must destroy itself first.
        self.destroy_timer(self._motion_timer)
        self._motion_timer = None
        self._transition(result_state)

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
