"""Objective 4.1/4.3 Phase-0 simulated handoff and active-view controller.

The controller includes parameterized source-time freshness gates for both
/target_object_pose and /hand_observation, plus timeouts, ABORT preemption,
race-safe callbacks, and latched fault handling. Release fails closed.

States and transitions:

    IDLE        --CONFIRM + fresh target-->          APPROACH
    IDLE        --fresh view command + no target-->  TARGET_SEARCH
    TARGET_SEARCH --stop + fresh target + CONFIRM--> APPROACH
    APPROACH    --motion done-->                     READY
    READY       --view command + no hand + held-->   HANDOFF_SEARCH
    HANDOFF_SEARCH --stop + fresh hand-->            READY
    READY       --CONFIRM + release gates pass-->    RELEASE
    RELEASE     --release done-->                    RETURN_HOME
    RETURN_HOME --motion done-->                     IDLE

    SEARCH / APPROACH / READY / RELEASE --ABORT-->   RETURN_HOME
        (ABORT is ignored in IDLE and in RETURN_HOME itself: there is
        nothing to cancel, and returning home is already the safe action.
        An ABORT taken while holding releases the object in place first,
        so reaching IDLE always means the gripper is empty.)
        (Phase-0 release is simulated, so aborting it just cancels a timer;
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
for the Phase-0 controller): once a real gripper starts opening it may be
past its irreversible commit point, so losing the hand signal mid-release
does not automatically mean "interrupt". N-frame stability gating belongs
to Objective 4.2; the real-release cancellation policy is an Objective 5
(hardware-phase) decision.

Motion is simulated: _start_motion() arms a one-shot completion timer plus
a deadline (watchdog) timer; _on_motion_result() receives the "result".
These are the seam where a MoveIt action goal/result replaces the timers.
Every transition bumps a generation counter (_epoch); timer callbacks
carry the epoch they were armed in and do nothing if it has moved on, so a
stale callback can never mutate the state machine.

Freshness is always judged on header.stamp (source time), never on message
receipt time, per the assistive_interfaces contracts.

The source-independent behavior is covered by node-level controller tests.
"""

import enum
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Float64, String

from assistive_interfaces.msg import (
    AssistiveIntent,
    HandObservation,
    ViewControlCommand,
)

from assistive_handoff.view_search import (
    DiscreteViewSweep,
    SearchProfile,
    SimulatedViewMotion,
)


class HandoffState(enum.Enum):
    IDLE = "idle"
    TARGET_SEARCH = "target_search"
    APPROACH = "approach"
    HANDOFF_SEARCH = "handoff_search"
    READY = "ready"
    RELEASE = "release"
    RETURN_HOME = "return_home"


class SearchPhase(enum.Enum):
    """Internal stop-and-look phases; public states stay source-independent."""

    ACTIVE = "active"
    STOPPING_FOR_OBSERVATION = "stopping_for_observation"
    WAITING_FOR_FRESH_OBSERVATION = "waiting_for_fresh_observation"
    LOCKED = "locked"
    STOPPING_FOR_TIMEOUT = "stopping_for_timeout"


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

        # Objective 4.3 bounded active-view command and simulation settings.
        # All angles are radians. Relative travel is additionally hard-capped
        # by SearchProfile at pi/4 (45 degrees).
        self.declare_parameter("view_command_max_age_sec", 0.5)
        self.declare_parameter("view_command_future_tolerance_sec", 0.05)
        self.declare_parameter("view_watchdog_sec", 0.75)
        self.declare_parameter("search_timeout_sec", 20.0)
        self.declare_parameter("view_update_period_sec", 0.02)
        self.declare_parameter("view_confidence_min", 0.6)
        self.declare_parameter("view_signal_quality_min", 0.5)
        self.declare_parameter("view_activation_deadband", 0.05)
        self.declare_parameter("view_activation_smoothing_alpha", 0.4)
        self.declare_parameter("view_stable_command_count", 2)
        self.declare_parameter("view_target_update_min_rad", 0.02)
        self.declare_parameter("view_step_angle", math.radians(10.0))

        self.declare_parameter("target_search_center_angle", 0.0)
        self.declare_parameter("target_search_relative_limit", math.pi / 4.0)
        self.declare_parameter("target_search_min_angle", -math.pi / 3.0)
        self.declare_parameter("target_search_max_angle", math.pi / 3.0)
        self.declare_parameter("target_search_nominal_speed", 0.25)
        self.declare_parameter("target_search_acceleration", 0.5)
        self.declare_parameter("target_search_deceleration", 0.75)

        self.declare_parameter("handoff_search_center_angle", 0.0)
        self.declare_parameter("handoff_search_relative_limit", math.pi / 9.0)
        self.declare_parameter("handoff_search_min_angle", -math.pi / 3.0)
        self.declare_parameter("handoff_search_max_angle", math.pi / 3.0)
        self.declare_parameter("handoff_search_nominal_speed", 0.12)
        self.declare_parameter("handoff_search_acceleration", 0.3)
        self.declare_parameter("handoff_search_deceleration", 0.5)

        self._speed_scale = self.get_parameter("speed_scale").value
        if not 0.0 < self._speed_scale <= 1.0:
            raise ValueError(
                f"speed_scale must be in (0, 1], got {self._speed_scale}"
            )
        self._sim_motion_sec = self.get_parameter("sim_motion_sec").value
        self._motion_timeout_sec = self.get_parameter("motion_timeout_sec").value
        self._ready_timeout_sec = self.get_parameter("ready_timeout_sec").value
        self._max_delivery_distance = self._positive_finite_param(
            "max_delivery_distance"
        )
        self._delivery_center = (
            self._finite_param("delivery_center_x"),
            self._finite_param("delivery_center_y"),
            self._finite_param("delivery_center_z"),
        )
        self._delivery_frame = self._nonempty_string_param("delivery_frame")
        self._target_max_age_sec = self._positive_finite_param(
            "target_max_age_sec"
        )
        self._hand_max_age_sec = self._positive_finite_param("hand_max_age_sec")
        self._simulate_stuck_motion = self.get_parameter(
            "simulate_stuck_motion"
        ).value

        self._view_command_max_age_sec = self._positive_finite_param(
            "view_command_max_age_sec"
        )
        self._view_future_tolerance_sec = self._nonnegative_finite_param(
            "view_command_future_tolerance_sec"
        )
        self._view_watchdog_sec = self._positive_finite_param(
            "view_watchdog_sec"
        )
        self._search_timeout_sec = self._positive_finite_param(
            "search_timeout_sec"
        )
        self._view_update_period_sec = self._positive_finite_param(
            "view_update_period_sec"
        )
        self._view_confidence_min = self._unit_interval_param(
            "view_confidence_min"
        )
        self._view_signal_quality_min = self._unit_interval_param(
            "view_signal_quality_min"
        )
        self._view_activation_deadband = self._unit_interval_param(
            "view_activation_deadband", upper_inclusive=False
        )
        self._view_smoothing_alpha = self._unit_interval_param(
            "view_activation_smoothing_alpha", lower_inclusive=False
        )
        self._view_stable_command_count = self._positive_integer_param(
            "view_stable_command_count"
        )
        self._view_target_update_min_rad = self._nonnegative_finite_param(
            "view_target_update_min_rad"
        )
        self._view_step_angle = self._positive_finite_param("view_step_angle")
        self._target_search_profile = self._load_search_profile("target_search")
        self._handoff_search_profile = self._load_search_profile("handoff_search")
        if (
            self._handoff_search_profile.relative_limit
            > self._target_search_profile.relative_limit
        ):
            raise ValueError("handoff search angle limit must not exceed target search")
        if self._handoff_search_profile.nominal_speed > self._target_search_profile.nominal_speed:
            raise ValueError("handoff search speed must not exceed target search")

        self._state = HandoffState.IDLE
        self._last_target: PoseStamped | None = None
        self._last_hand: HandObservation | None = None
        self._fault = False
        # Generation counter: bumped on every transition (and fault latch);
        # timer callbacks armed under an older epoch must not act.
        self._epoch = 0
        self._pending_timers = []

        self._holding_object = False
        self._search_phase: SearchPhase | None = None
        self._search_started_at: Time | None = None
        self._search_stopped_at: Time | None = None
        self._last_view_source_time: Time | None = None
        self._last_view_sequence: int | None = None
        self._view_watchdog_holding = False
        self._view_candidate_direction: int | None = None
        self._view_candidate_count = 0
        self._view_smoothed_activation = 0.0
        self._last_requested_view_target: float | None = None
        self._search_input_mode: str | None = None
        self._discrete_sweep: DiscreteViewSweep | None = None
        self._last_discrete_sequence: int | None = None
        self._active_search_profile = self._target_search_profile
        self._view_motion = SimulatedViewMotion(self._target_search_profile)

        self._intent_sub = self.create_subscription(
            AssistiveIntent, "/assistive_intent", self._on_intent, 10
        )
        self._hand_sub = self.create_subscription(
            HandObservation, "/hand_observation", self._on_hand, 10
        )
        # /target_object_pose is retained (TRANSIENT_LOCAL) by target_selector
        # and fixed_pose_publisher; subscribe with the same durability so a
        # late-starting controller still sees the target. The parameterized
        # target_max_age_sec source-time gate makes retained acceptance safe.
        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self._target_sub = self.create_subscription(
            PoseStamped, "/target_object_pose", self._on_target, latched_qos
        )

        volatile_latest_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.VOLATILE
        )
        self._view_sub = self.create_subscription(
            ViewControlCommand,
            "/assistive_view_control",
            self._on_view_control,
            volatile_latest_qos,
        )

        self._state_pub = self.create_publisher(String, "/handoff_state", 10)
        self._view_angle_pub = self.create_publisher(
            Float64, "/simulated_view_angle", 10
        )
        # Republish the current state at 1 Hz so `ros2 topic echo` joined
        # mid-cycle still shows it; transitions also publish immediately.
        self._state_republish_timer = self.create_timer(1.0, self._publish_state)
        self._view_update_timer = self.create_timer(
            self._view_update_period_sec, self._on_view_tick
        )
        self._publish_view_angle()

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
        if self._state is HandoffState.TARGET_SEARCH:
            usable = self._target_observation_is_usable(msg)
            self._handle_search_observation(msg.header.stamp, usable)

    def _on_hand(self, msg: HandObservation) -> None:
        self._last_hand = msg
        if self._state is HandoffState.HANDOFF_SEARCH:
            usable = self._hand_is_release_ready(msg, log_reasons=False)
            self._handle_search_observation(msg.header.stamp, usable)

    def _on_view_control(self, msg: ViewControlCommand) -> None:
        if self._fault:
            return
        if self._state not in (
            HandoffState.IDLE,
            HandoffState.TARGET_SEARCH,
            HandoffState.READY,
            HandoffState.HANDOFF_SEARCH,
        ):
            return

        activation = self._validated_view_activation(msg)
        if activation is None:
            return
        requests_motion = (
            msg.direction in (
                ViewControlCommand.LEFT,
                ViewControlCommand.RIGHT,
            )
            and activation > self._view_activation_deadband
        )

        if self._state is HandoffState.IDLE:
            if self._target_is_available() or not requests_motion:
                return
            self._search_input_mode = "proportional"
            self._transition(HandoffState.TARGET_SEARCH)
        elif self._state is HandoffState.READY:
            if not self._holding_object:
                self.get_logger().warning(
                    "HANDOFF_SEARCH refused: holding_object is false"
                )
                return
            if (
                self._hand_is_release_ready(
                    self._last_hand, log_reasons=False
                )
                or not requests_motion
            ):
                return
            self._search_input_mode = "proportional"
            self._transition(HandoffState.HANDOFF_SEARCH)

        if self._state in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            if self._search_input_mode == "discrete":
                return
            self._apply_view_command(msg, activation)

    def _validated_view_activation(
        self, msg: ViewControlCommand
    ) -> float | None:
        if msg.direction not in (
            ViewControlCommand.HOLD,
            ViewControlCommand.LEFT,
            ViewControlCommand.RIGHT,
        ):
            self.get_logger().warning(
                f"view command ignored: unknown direction {msg.direction}"
            )
            return None

        age = self._age_sec(msg.header.stamp)
        if age < -self._view_future_tolerance_sec:
            self.get_logger().warning(
                f"view command ignored: source stamp is {-age:.3f}s future"
            )
            return None
        if age > self._view_command_max_age_sec:
            self.get_logger().warning(
                f"view command ignored: source stamp is {age:.3f}s old"
            )
            return None

        confidence = float(msg.confidence)
        signal_quality = float(msg.signal_quality)
        for name, value, minimum in (
            ("confidence", confidence, self._view_confidence_min),
            (
                "signal_quality",
                signal_quality,
                self._view_signal_quality_min,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                self.get_logger().warning(
                    f"view command ignored: invalid {name}={value}"
                )
                return None
            if value < minimum:
                self.get_logger().warning(
                    f"view command ignored: {name}={value:.3f} "
                    f"below {minimum:.3f}"
                )
                return None

        activation = float(msg.activation)
        if not math.isfinite(activation):
            self.get_logger().warning(
                "view command ignored: activation is not finite"
            )
            return None
        activation = min(1.0, max(0.0, activation))

        sequence = int(msg.sequence)
        if (
            self._last_view_sequence is not None
            and not self._sequence_is_newer(
                sequence, self._last_view_sequence
            )
        ):
            self.get_logger().warning(
                f"view command ignored: sequence {sequence} is not newer"
            )
            return None
        self._last_view_sequence = sequence
        return activation

    @staticmethod
    def _sequence_is_newer(sequence: int, previous: int) -> bool:
        delta = (sequence - previous) & 0xFFFFFFFF
        return 0 < delta < 0x80000000

    def _apply_view_command(
        self, msg: ViewControlCommand, activation: float
    ) -> None:
        if self._search_phase is not SearchPhase.ACTIVE:
            return

        self._last_view_source_time = Time.from_msg(msg.header.stamp)
        self._view_watchdog_holding = False
        if (
            msg.direction == ViewControlCommand.HOLD
            or activation <= self._view_activation_deadband
        ):
            self._view_motion.request_hold()
            self._reset_view_filter()
            return

        direction = (
            -1 if msg.direction == ViewControlCommand.LEFT else 1
        )
        if direction != self._view_candidate_direction:
            self._view_candidate_direction = direction
            self._view_candidate_count = 1
            self._view_smoothed_activation = activation
        else:
            self._view_candidate_count += 1
            alpha = self._view_smoothing_alpha
            self._view_smoothed_activation = (
                alpha * activation
                + (1.0 - alpha) * self._view_smoothed_activation
            )

        if self._view_candidate_count < self._view_stable_command_count:
            return
        target = self._active_search_profile.target_for(
            direction, self._view_smoothed_activation
        )
        if (
            self._last_requested_view_target is not None
            and abs(target - self._last_requested_view_target)
            < self._view_target_update_min_rad
        ):
            return
        self._last_requested_view_target = self._view_motion.request_target(
            target
        )

    def _reset_view_filter(self) -> None:
        self._view_candidate_direction = None
        self._view_candidate_count = 0
        self._view_smoothed_activation = 0.0
        self._last_requested_view_target = None

    def _target_observation_is_usable(self, msg: PoseStamped) -> bool:
        _, status = self._observation_age_status(
            msg.header.stamp, self._target_max_age_sec
        )
        return status is None

    def _handle_search_observation(self, stamp, usable: bool) -> None:
        if not usable or self._search_phase not in (
            SearchPhase.ACTIVE,
            SearchPhase.WAITING_FOR_FRESH_OBSERVATION,
        ):
            return
        reference = (
            self._search_started_at
            if self._search_phase is SearchPhase.ACTIVE
            else self._search_stopped_at
        )
        if reference is None:
            return
        if Time.from_msg(stamp).nanoseconds <= reference.nanoseconds:
            return

        if self._search_phase is SearchPhase.ACTIVE:
            self._view_motion.request_hold()
            self._reset_view_filter()
            self._search_phase = SearchPhase.STOPPING_FOR_OBSERVATION
            self.get_logger().info(
                f"{self._state.value}: observation acquired; stopping view"
            )
        elif self._state is HandoffState.TARGET_SEARCH:
            self._search_phase = SearchPhase.LOCKED
            self.get_logger().info(
                "target locked from a post-stop observation; waiting CONFIRM"
            )
        else:
            self.get_logger().info(
                "hand locked from a post-stop observation; entering ready"
            )
            self._transition(HandoffState.READY)

    def _on_view_tick(self) -> None:
        self._view_motion.step(self._view_update_period_sec)
        self._publish_view_angle()
        if self._state not in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            return

        if (
            self._search_phase is SearchPhase.ACTIVE
            and self._search_input_mode != "discrete"
        ):
            watchdog_expired = self._last_view_source_time is None
            if self._last_view_source_time is not None:
                age = (
                    self.get_clock().now() - self._last_view_source_time
                ).nanoseconds / 1e9
                watchdog_expired = age > self._view_watchdog_sec
            if watchdog_expired and not self._view_watchdog_holding:
                self._view_motion.request_hold()
                self._reset_view_filter()
                self._view_watchdog_holding = True
                self.get_logger().warning(
                    f"{self._state.value}: view watchdog expired; holding"
                )

        if (
            self._search_phase
            is SearchPhase.STOPPING_FOR_OBSERVATION
            and not self._view_motion.moving
        ):
            self._search_stopped_at = self.get_clock().now()
            self._search_phase = SearchPhase.WAITING_FOR_FRESH_OBSERVATION
            self.get_logger().info(
                f"{self._state.value}: view stopped; waiting fresh observation"
            )
        elif (
            self._search_phase is SearchPhase.STOPPING_FOR_TIMEOUT
            and not self._view_motion.moving
        ):
            self._transition(HandoffState.RETURN_HOME)

    def _publish_view_angle(self) -> None:
        self._view_angle_pub.publish(
            Float64(data=self._view_motion.position)
        )

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
            self._on_next_target(msg)
        else:
            self.get_logger().warning(f"unknown intent command {command}: ignoring")

    # ------------------------------------------------------------------
    # Intent handling
    # ------------------------------------------------------------------

    def _on_next_target(self, msg: AssistiveIntent) -> None:
        """Advance one bounded view step in a discrete search episode."""

        age = self._age_sec(msg.header.stamp)
        if age < -self._view_future_tolerance_sec:
            self.get_logger().warning(
                f"NEXT_TARGET view step ignored: source stamp is {-age:.3f}s future"
            )
            return
        if age > self._view_command_max_age_sec:
            self.get_logger().warning(
                f"NEXT_TARGET view step ignored: source stamp is {age:.3f}s old"
            )
            return
        sequence = int(msg.sequence)
        if (
            self._last_discrete_sequence is not None
            and not self._sequence_is_newer(
                sequence, self._last_discrete_sequence
            )
        ):
            self.get_logger().warning(
                f"NEXT_TARGET view step ignored: sequence {sequence} is not newer"
            )
            return
        self._last_discrete_sequence = sequence

        if self._state is HandoffState.IDLE:
            if self._target_is_available():
                return
            self._search_input_mode = "discrete"
            self._transition(HandoffState.TARGET_SEARCH)
        elif self._state is HandoffState.READY:
            if (
                not self._holding_object
                or self._hand_is_release_ready(
                    self._last_hand, log_reasons=False
                )
            ):
                return
            self._search_input_mode = "discrete"
            self._transition(HandoffState.HANDOFF_SEARCH)

        if self._state not in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            self.get_logger().info(
                "NEXT_TARGET handled by upstream selector outside search"
            )
            return
        if self._search_input_mode != "discrete":
            self.get_logger().info(
                "NEXT_TARGET view step ignored: proportional search owns episode"
            )
            return
        if (
            self._search_phase is not SearchPhase.ACTIVE
            or self._view_motion.moving
        ):
            self.get_logger().info(
                "NEXT_TARGET view step ignored: previous step still active"
            )
            return

        if self._discrete_sweep is None:
            raise RuntimeError("discrete search has no sweep policy")
        target = self._discrete_sweep.next_target(
            self._view_motion.position
        )
        self._last_requested_view_target = (
            self._view_motion.request_target(target)
        )

    def _on_confirm(self) -> None:
        if self._state is HandoffState.IDLE:
            if self._target_is_fresh():
                self._transition(HandoffState.APPROACH)
            # else: _target_is_fresh already logged why the gate refused
        elif self._state is HandoffState.TARGET_SEARCH:
            if self._search_phase is not SearchPhase.LOCKED:
                self.get_logger().info(
                    "CONFIRM ignored: target search has no post-stop lock"
                )
            elif self._target_is_fresh():
                self._transition(HandoffState.APPROACH)
            else:
                # The lock aged out while waiting for the user. Resume the
                # same bounded search; a new command is required by watchdog.
                self._search_phase = SearchPhase.ACTIVE
                self._search_started_at = self.get_clock().now()
                self._search_stopped_at = None
                self._last_view_source_time = None
                self._reset_view_filter()
        elif self._state is HandoffState.READY:
            if self._release_gates_pass():
                self._transition(HandoffState.RELEASE)
        else:
            self.get_logger().info(
                f"CONFIRM ignored in state '{self._state.value}'"
            )

    def _on_abort(self) -> None:
        if self._state in (
            HandoffState.TARGET_SEARCH,
            HandoffState.APPROACH,
            HandoffState.HANDOFF_SEARCH,
            HandoffState.READY,
            HandoffState.RELEASE,
        ):
            if self._state in (
                HandoffState.TARGET_SEARCH,
                HandoffState.HANDOFF_SEARCH,
            ):
                self._view_motion.emergency_stop()
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
        age, status = self._observation_age_status(
            self._last_target.header.stamp, self._target_max_age_sec
        )
        if status == "future":
            self.get_logger().warning(
                f"CONFIRM refused: target stamp is {-age:.2f}s in the future "
                f"(tolerance {self._view_future_tolerance_sec}s)"
            )
            return False
        if status == "stale":
            self.get_logger().warning(
                f"CONFIRM refused: target is {age:.2f}s old "
                f"(max {self._target_max_age_sec}s)"
            )
            return False
        return True

    def _target_is_available(self) -> bool:
        if self._last_target is None:
            return False
        _, status = self._observation_age_status(
            self._last_target.header.stamp, self._target_max_age_sec
        )
        return status is None

    def _release_gates_pass(self) -> bool:
        return self._hand_is_release_ready(
            self._last_hand, log_reasons=True
        )

    def _hand_is_release_ready(
        self,
        hand: HandObservation | None,
        *,
        log_reasons: bool,
    ) -> bool:
        def refuse(message: str) -> bool:
            if log_reasons:
                self.get_logger().warning(message)
            return False

        if hand is None:
            return refuse("release refused: no /hand_observation yet")
        if not hand.valid:
            return refuse(
                "release refused: latest observation is no-hand"
            )
        age, status = self._observation_age_status(
            hand.header.stamp, self._hand_max_age_sec
        )
        if status == "future":
            return refuse(
                f"release refused: hand stamp is {-age:.2f}s in the future "
                f"(tolerance {self._view_future_tolerance_sec}s)"
            )
        if status == "stale":
            return refuse(
                f"release refused: hand observation is {age:.2f}s old "
                f"(max {self._hand_max_age_sec}s)"
            )
        if hand.header.frame_id != self._delivery_frame:
            # Refusing beats silently comparing points in different frames.
            return refuse(
                f"release refused: hand frame '{hand.header.frame_id}' != "
                f"delivery frame '{self._delivery_frame}'"
            )
        point = (hand.point.x, hand.point.y, hand.point.z)
        if not all(math.isfinite(value) for value in point):
            return refuse("release refused: hand point contains non-finite values")
        distance = math.dist(point, self._delivery_center)
        if distance > self._max_delivery_distance:
            return refuse(
                f"release refused: hand is {distance:.3f}m from delivery "
                f"center (max {self._max_delivery_distance}m)"
            )
        return True

    def _age_sec(self, stamp) -> float:
        return (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9

    def _observation_age_status(
        self, stamp, max_age_sec: float
    ) -> tuple[float, str | None]:
        age = self._age_sec(stamp)
        if age < -self._view_future_tolerance_sec:
            return age, "future"
        if age > max_age_sec:
            return age, "stale"
        return age, None

    def _finite_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return float(value)

    def _nonempty_string_param(self, name: str) -> str:
        value = self.get_parameter(name).value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string, got {value!r}")
        return value

    def _positive_finite_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")
        return value

    def _nonnegative_finite_param(self, name: str) -> float:
        value = self.get_parameter(name).value
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(
                f"{name} must be finite and >= 0, got {value}"
            )
        return float(value)

    def _unit_interval_param(
        self,
        name: str,
        *,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
    ) -> float:
        value = self.get_parameter(name).value
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        lower_ok = value >= 0.0 if lower_inclusive else value > 0.0
        upper_ok = value <= 1.0 if upper_inclusive else value < 1.0
        if not lower_ok or not upper_ok:
            left = "[" if lower_inclusive else "("
            right = "]" if upper_inclusive else ")"
            raise ValueError(
                f"{name} must be in {left}0, 1{right}, got {value}"
            )
        return float(value)

    def _positive_integer_param(self, name: str) -> int:
        value = self.get_parameter(name).value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value}")
        return value

    def _load_search_profile(self, prefix: str) -> SearchProfile:
        return SearchProfile(
            center_angle=self.get_parameter(
                f"{prefix}_center_angle"
            ).value,
            relative_limit=self.get_parameter(
                f"{prefix}_relative_limit"
            ).value,
            min_angle=self.get_parameter(f"{prefix}_min_angle").value,
            max_angle=self.get_parameter(f"{prefix}_max_angle").value,
            nominal_speed=self.get_parameter(
                f"{prefix}_nominal_speed"
            ).value,
            acceleration=self.get_parameter(
                f"{prefix}_acceleration"
            ).value,
            deceleration=self.get_parameter(
                f"{prefix}_deceleration"
            ).value,
        )

    # ------------------------------------------------------------------
    # State transitions, simulated motion, timeouts
    # ------------------------------------------------------------------

    def _transition(self, new_state: HandoffState) -> None:
        if (
            new_state is HandoffState.HANDOFF_SEARCH
            and not self._holding_object
        ):
            self.get_logger().warning(
                "HANDOFF_SEARCH refused: holding_object is false"
            )
            return

        old_state = self._state
        if old_state in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ) and new_state not in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            self._view_motion.emergency_stop()
            self._search_phase = None
            self._search_started_at = None
            self._search_stopped_at = None
            self._search_input_mode = None
            self._discrete_sweep = None

        self._epoch += 1
        self._cancel_pending_timers()
        self.get_logger().info(
            f"state: {self._state.value} -> {new_state.value}"
        )
        self._state = new_state
        self._publish_state()

        # Entry actions: motion states start their (simulated) motion; READY
        # arms the dwell timeout so the arm never hovers indefinitely.
        if new_state in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            self._start_search(new_state)
        elif new_state is HandoffState.APPROACH:
            self._start_motion(HandoffState.READY)
        elif new_state is HandoffState.RELEASE:
            # Phase-0 release is a simulated transition: no gripper actuation.
            self._start_motion(HandoffState.RETURN_HOME)
        elif new_state is HandoffState.RETURN_HOME:
            # ABORT must end with an empty gripper, so an abort taken while
            # holding puts the object down before the arm leaves. The normal
            # handover path cannot double-release: _on_motion_result clears
            # the flag for RELEASE before it transitions here.
            if self._holding_object:
                self._release_in_place()
            self._start_motion(HandoffState.IDLE)
        elif new_state is HandoffState.READY:
            epoch = self._epoch
            self._pending_timers.append(
                self.create_timer(
                    self._ready_timeout_sec,
                    lambda: self._on_ready_timeout(epoch),
                )
            )
        elif new_state is HandoffState.IDLE:
            # Asserted, not assigned. This used to clear the flag, which meant
            # an abort taken while holding reached IDLE reporting an empty
            # gripper while the object was still in it -- the state model went
            # wrong silently and the next cycle would approach a second object
            # with a full gripper. Releasing is now the only way the flag
            # clears, so reaching here holding is a missing release, not a
            # condition to paper over.
            if self._holding_object:
                raise RuntimeError(
                    "reached IDLE while holding: some path to RETURN_HOME "
                    "skipped the release"
                )
            self._active_search_profile = self._target_search_profile
            self._view_motion = SimulatedViewMotion(
                self._target_search_profile
            )
            self._publish_view_angle()

    def _release_in_place(self) -> None:
        """Put the object down where the arm stopped, before returning home.

        Chosen over carrying the object home because carrying it would need an
        "idle but holding" condition, and that weakens the invariant this
        keeps: reaching IDLE means the gripper is empty. Dropping in place is
        acceptable for the tabletop scenario this system is scoped to; it is
        not a general policy, and a scenario where the arm can be over
        something it must not drop onto needs its own decision.

        ABORT stays immediate. What stops instantly is the arm, which is
        already halted at this point; the release happens at the pose it
        stopped in and adds no travel.

        Deliberately not routed through the RELEASE state, whose
        _release_gates_pass() encodes "a fresh, confident hand is ready to
        receive". An abort release must never require or imply that, and
        sharing the state would set a precedent for bypassing those gates.

        Phase-0 has no gripper actuation anywhere -- RELEASE is a timer too --
        so this is a flag change and a log line. Objective 5 owns the real
        actuation and the sequencing it needs: the release must complete
        before the homing motion starts.
        """
        self.get_logger().warning(
            "ABORT while holding an object: releasing in place before "
            "returning home"
        )
        self._holding_object = False

    def _start_search(self, state: HandoffState) -> None:
        if self._view_motion.moving:
            raise RuntimeError("cannot enter search while view motion is active")
        profile = (
            self._target_search_profile
            if state is HandoffState.TARGET_SEARCH
            else self._handoff_search_profile
        )
        self._view_motion.configure(profile)
        self._active_search_profile = profile
        if self._search_input_mode is None:
            self._search_input_mode = "proportional"
        self._discrete_sweep = (
            DiscreteViewSweep(profile, self._view_step_angle)
            if self._search_input_mode == "discrete"
            else None
        )
        self._search_phase = SearchPhase.ACTIVE
        self._search_started_at = self.get_clock().now()
        self._search_stopped_at = None
        self._last_view_source_time = None
        self._view_watchdog_holding = False
        self._reset_view_filter()

        epoch = self._epoch
        self._pending_timers.append(
            self.create_timer(
                self._search_timeout_sec,
                lambda: self._on_search_timeout(epoch),
            )
        )

    def _start_motion(self, result_state: HandoffState) -> None:
        # Phase-0 uses a one-shot timer in place of a MoveIt action goal; the
        # deadline timer is the watchdog for a motion that never reports.
        # Motion-backend integration replaces the completion timer with the
        # real goal request and routes its result into _on_motion_result.
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
        if self._state is HandoffState.APPROACH:
            self._holding_object = True
        elif self._state is HandoffState.RELEASE:
            self._holding_object = False
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

    def _on_search_timeout(self, epoch: int) -> None:
        if epoch != self._epoch or self._state not in (
            HandoffState.TARGET_SEARCH,
            HandoffState.HANDOFF_SEARCH,
        ):
            return
        self.get_logger().warning(
            f"{self._state.value}: search timeout; stopping before return"
        )
        self._view_motion.request_hold()
        self._reset_view_filter()
        self._search_phase = SearchPhase.STOPPING_FOR_TIMEOUT

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
