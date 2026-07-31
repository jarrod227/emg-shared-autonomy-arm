"""Subscribe to markerless candidates and feed the pure stability gate."""

import math
import os

from ament_index_python.packages import get_package_share_directory
from assistive_interfaces.msg import AssistiveIntent, ObjectCandidateArray
from geometry_msgs.msg import PointStamped, PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from target_selector.candidate_adapter import candidate_frame_from_message
from target_selector.candidate_stability import (
    CandidateFrame,
    CandidateStabilityConfig,
    CandidateStabilityGate,
    DEFAULT_OBJECT_CLASSES,
)
from target_selector.markerless_grasp import (
    build_markerless_grasp_pose,
    grasp_template_for_class,
    load_markerless_grasp_templates,
)
from target_selector.target_lock import TargetLockConfig, TargetLockManager
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener


class MarkerlessCandidateGateNode(Node):
    """Own the ROS subscription boundary around candidate stability logic."""

    def __init__(self, *, parameter_overrides=None):
        super().__init__(
            'markerless_candidate_gate',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()
        candidate_topic = self.get_parameter('candidate_topic').value
        if not isinstance(candidate_topic, str) or not candidate_topic.strip():
            raise ValueError('candidate_topic must be a non-empty string')
        intent_topic = self.get_parameter('intent_topic').value
        if not isinstance(intent_topic, str) or not intent_topic.strip():
            raise ValueError('intent_topic must be a non-empty string')
        target_topic = self.get_parameter('target_topic').value
        if not isinstance(target_topic, str) or not target_topic.strip():
            raise ValueError('target_topic must be a non-empty string')
        self._target_topic = target_topic.strip()
        planning_frame = self.get_parameter('planning_frame').value
        if not isinstance(planning_frame, str) or not planning_frame.strip():
            raise ValueError('planning_frame must be a non-empty string')
        self._planning_frame = planning_frame.strip()
        self._tf_timeout_sec = self._positive_finite_parameter(
            'tf_timeout_sec'
        )

        template_path = self.get_parameter('grasp_template_path').value
        if not isinstance(template_path, str):
            raise ValueError('grasp_template_path must be a string')
        if not template_path.strip():
            template_path = os.path.join(
                get_package_share_directory('target_selector'),
                'config',
                'markerless_grasp_templates.yaml',
            )
        self._grasp_templates = load_markerless_grasp_templates(
            template_path
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        config = CandidateStabilityConfig(
            required_frames=self.get_parameter('required_frames').value,
            min_class_confidence=self.get_parameter(
                'min_class_confidence'
            ).value,
            min_localization_confidence=self.get_parameter(
                'min_localization_confidence'
            ).value,
            max_pair_skew_sec=self.get_parameter(
                'max_pair_skew_sec'
            ).value,
            max_age_sec=self.get_parameter('max_age_sec').value,
            future_tolerance_sec=self.get_parameter(
                'future_tolerance_sec'
            ).value,
            max_frame_gap_sec=self.get_parameter(
                'max_frame_gap_sec'
            ).value,
            max_position_span_m=self.get_parameter(
                'max_position_span_m'
            ).value,
            allowed_classes=tuple(
                self.get_parameter('allowed_classes').value
            ),
        )
        self._candidate_max_age_sec = config.max_age_sec
        self._candidate_future_tolerance_sec = config.future_tolerance_sec
        self._gate = CandidateStabilityGate(config)
        self._lock = TargetLockManager(
            TargetLockConfig(
                last_seen_timeout_sec=self.get_parameter(
                    'last_seen_timeout_sec'
                ).value
            )
        )
        self._watchdog_period_sec = self._positive_finite_parameter(
            'watchdog_period_sec'
        )
        self._intent_max_age_sec = self._positive_finite_parameter(
            'intent_max_age_sec'
        )
        self._intent_future_tolerance_sec = (
            self._nonnegative_finite_parameter(
                'intent_future_tolerance_sec'
            )
        )
        self._intent_min_confidence = self._unit_interval_parameter(
            'intent_min_confidence'
        )
        self._last_decision = None
        self._last_lock_decision = None
        self._latest_stable_header = None
        self._last_built_target_pose = None
        self._last_published_target_pose = None
        self._last_intent_sequence = None
        self._processed_message_count = 0
        self._processed_intent_count = 0
        self._published_target_count = 0
        target_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._target_publisher = self.create_publisher(
            PoseStamped,
            self._target_topic,
            target_qos,
        )
        self._subscription = self.create_subscription(
            ObjectCandidateArray,
            candidate_topic.strip(),
            self._on_candidates,
            10,
        )
        self._intent_subscription = self.create_subscription(
            AssistiveIntent,
            intent_topic.strip(),
            self._on_intent,
            10,
        )
        self._watchdog_timer = self.create_timer(
            self._watchdog_period_sec,
            self._on_watchdog,
        )
        self.get_logger().info(
            f'waiting for markerless candidates on {candidate_topic.strip()} '
            f'and intent on {intent_topic.strip()}; confirmed targets publish '
            f'on {self._target_topic}'
        )

    @property
    def last_decision(self):
        """Return the latest pure gate result for diagnostics and tests."""
        return self._last_decision

    @property
    def processed_message_count(self):
        """Return how many candidate messages reached this subscription."""
        return self._processed_message_count

    @property
    def last_lock_decision(self):
        """Return current selected-track state for diagnostics and tests."""
        return self._last_lock_decision

    @property
    def processed_intent_count(self):
        """Return how many intent messages reached this subscription."""
        return self._processed_intent_count

    @property
    def last_built_target_pose(self):
        """Return the latest source-time-transformed preview pose."""
        return self._last_built_target_pose

    @property
    def last_published_target_pose(self):
        """Return the last target pose published after confirmation."""
        return self._last_published_target_pose

    @property
    def published_target_count(self):
        """Return how many confirmed target poses this node published."""
        return self._published_target_count

    def _declare_parameters(self):
        self.declare_parameter('candidate_topic', '/object_candidates')
        self.declare_parameter('intent_topic', '/assistive_intent')
        self.declare_parameter('target_topic', '/target_object_pose')
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('tf_timeout_sec', 0.1)
        self.declare_parameter('grasp_template_path', '')
        self.declare_parameter('required_frames', 3)
        self.declare_parameter('min_class_confidence', 0.5)
        self.declare_parameter('min_localization_confidence', 0.5)
        self.declare_parameter('max_pair_skew_sec', 0.05)
        self.declare_parameter('max_age_sec', 0.5)
        self.declare_parameter('future_tolerance_sec', 0.05)
        self.declare_parameter('max_frame_gap_sec', 0.2)
        self.declare_parameter('max_position_span_m', 0.03)
        self.declare_parameter(
            'allowed_classes',
            list(DEFAULT_OBJECT_CLASSES),
        )
        self.declare_parameter('last_seen_timeout_sec', 0.5)
        self.declare_parameter('watchdog_period_sec', 0.05)
        self.declare_parameter('intent_max_age_sec', 0.5)
        self.declare_parameter('intent_future_tolerance_sec', 0.05)
        self.declare_parameter('intent_min_confidence', 0.5)

    def _on_candidates(self, message):
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000
        try:
            frame = candidate_frame_from_message(message)
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f'rejected malformed ObjectCandidateArray: {error}'
            )
            frame = CandidateFrame(
                source_time_sec=float('nan'),
                frame_id='',
                valid=False,
                pair_skew_sec=0.0,
                candidates=(),
            )

        self._last_decision = self._gate.update(frame, now_sec)
        self._last_lock_decision = self._lock.update(
            self._last_decision,
            now_sec,
        )
        if self._last_decision.stable_candidates:
            self._latest_stable_header = message.header
        else:
            self._latest_stable_header = None
        self._refresh_built_target_pose()
        self._processed_message_count += 1
        if self._last_decision.stable_candidates:
            tracks = ', '.join(
                f'{candidate.track_id}:{candidate.class_label}'
                for candidate in self._last_decision.stable_candidates
            )
            self.get_logger().info(f'stable markerless candidates: {tracks}')
        if self._last_lock_decision.reason in (
            'target_locked',
            'target_relocked',
        ):
            selected = self._last_lock_decision.selected_candidate
            self.get_logger().info(
                f'locked markerless target '
                f'{selected.track_id}:{selected.class_label}'
            )

    def _on_intent(self, message):
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000
        self._processed_intent_count += 1
        if message.command == AssistiveIntent.ABORT:
            self._remember_abort_sequence(message)
            self._gate.reset()
            self._last_lock_decision = self._lock.abort()
            self._latest_stable_header = None
            self._last_built_target_pose = None
            self.get_logger().warning('ABORT cleared markerless target lock')
            return
        if message.command not in (
            AssistiveIntent.NEXT_TARGET,
            AssistiveIntent.CONFIRM,
        ):
            self.get_logger().warning(
                f'ignored unsupported intent command {message.command}'
            )
            return
        if not self._intent_is_usable(message, now_sec):
            return
        if not self._accept_intent_sequence(message):
            return

        if message.command == AssistiveIntent.NEXT_TARGET:
            self._last_lock_decision = self._lock.next_target(now_sec)
        else:
            self._last_lock_decision = self._lock.confirm(now_sec)
        self._refresh_built_target_pose()
        if message.command == AssistiveIntent.CONFIRM:
            self._publish_confirmed_target(message, now_sec)

        selected = self._last_lock_decision.selected_candidate
        selected_name = (
            'none'
            if selected is None
            else f'{selected.track_id}:{selected.class_label}'
        )
        self.get_logger().info(
            f'intent result={self._last_lock_decision.reason}, '
            f'selected={selected_name}'
        )

    @staticmethod
    def _sequence_is_newer(sequence, previous):
        delta = (sequence - previous) & 0xFFFFFFFF
        return 0 < delta < 0x80000000

    def _accept_intent_sequence(self, message):
        sequence = int(message.sequence)
        if (
            self._last_intent_sequence is not None
            and not self._sequence_is_newer(
                sequence,
                self._last_intent_sequence,
            )
        ):
            self.get_logger().warning(
                f'ignored duplicate or out-of-order intent sequence '
                f'{sequence}; last accepted={self._last_intent_sequence}'
            )
            return False
        self._last_intent_sequence = sequence
        return True

    def _remember_abort_sequence(self, message):
        sequence = int(message.sequence)
        if (
            self._last_intent_sequence is None
            or self._sequence_is_newer(
                sequence,
                self._last_intent_sequence,
            )
        ):
            self._last_intent_sequence = sequence

    def _publish_confirmed_target(self, message, now_sec):
        if (
            self._last_lock_decision is None
            or not self._last_lock_decision.ready
        ):
            return
        if self._last_built_target_pose is None:
            self.get_logger().warning(
                'CONFIRM accepted for the selected track, but no '
                'source-time-transformed pose is available; not publishing'
            )
            return

        stamp = self._last_built_target_pose.header.stamp
        source_time_sec = stamp.sec + stamp.nanosec / 1_000_000_000
        age_sec = now_sec - source_time_sec
        if (
            age_sec < -self._candidate_future_tolerance_sec
            or age_sec > self._candidate_max_age_sec
        ):
            self.get_logger().warning(
                'CONFIRM accepted for the selected track, but its source '
                'pose is stale or future-dated; not publishing'
            )
            return

        self._target_publisher.publish(self._last_built_target_pose)
        self._last_published_target_pose = self._last_built_target_pose
        self._published_target_count += 1
        selected = self._last_lock_decision.selected_candidate
        self.get_logger().info(
            f'published confirmed markerless target '
            f'{selected.track_id}:{selected.class_label} on '
            f'{self._target_topic} for intent sequence {message.sequence}'
        )

    def _intent_is_usable(self, message, now_sec):
        confidence = float(message.confidence)
        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self._intent_min_confidence
        ):
            self.get_logger().warning(
                'ignored low or invalid intent confidence'
            )
            return False

        seconds = message.header.stamp.sec
        nanoseconds = message.header.stamp.nanosec
        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            self.get_logger().warning('ignored invalid intent timestamp')
            return False
        source_time_sec = seconds + nanoseconds / 1_000_000_000
        age_sec = now_sec - source_time_sec
        if age_sec < -self._intent_future_tolerance_sec:
            self.get_logger().warning('ignored future-dated intent')
            return False
        if age_sec > self._intent_max_age_sec:
            self.get_logger().warning('ignored stale intent')
            return False
        return True

    def _on_watchdog(self):
        if self._last_lock_decision is None:
            return
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000
        previous = self._last_lock_decision.selected_candidate
        self._last_lock_decision = self._lock.tick(now_sec)
        if (
            previous is not None
            and self._last_lock_decision.reason == 'lock_expired'
        ):
            self._latest_stable_header = None
            self._last_built_target_pose = None
            self.get_logger().warning('markerless target lock expired')

    def _refresh_built_target_pose(self):
        self._last_built_target_pose = None
        selected = (
            None
            if self._last_lock_decision is None
            else self._last_lock_decision.selected_candidate
        )
        if (
            selected is None
            or not self._last_lock_decision.selected_visible
            or self._latest_stable_header is None
        ):
            return

        source_header = self._latest_stable_header
        source_point = PointStamped()
        source_point.header = source_header
        (
            source_point.point.x,
            source_point.point.y,
            source_point.point.z,
        ) = selected.position
        try:
            transform = self._tf_buffer.lookup_transform(
                self._planning_frame,
                source_header.frame_id,
                Time.from_msg(source_header.stamp),
                Duration(seconds=self._tf_timeout_sec),
            )
        except Exception as error:  # noqa: BLE001 - fail closed and retry
            self.get_logger().warning(
                f'tf lookup {self._planning_frame}<-'
                f'{source_header.frame_id} at candidate stamp failed: '
                f'{error}',
                throttle_duration_sec=1.0,
            )
            return

        planning_point = do_transform_point(source_point, transform)
        template = grasp_template_for_class(
            self._grasp_templates,
            selected.class_label,
        )
        target = PoseStamped()
        target.header.frame_id = self._planning_frame
        target.header.stamp.sec = source_header.stamp.sec
        target.header.stamp.nanosec = source_header.stamp.nanosec
        target.pose = build_markerless_grasp_pose(
            (
                planning_point.point.x,
                planning_point.point.y,
                planning_point.point.z,
            ),
            template,
        )
        self._last_built_target_pose = target

    def _positive_finite_parameter(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and greater than zero')
        return value

    def _nonnegative_finite_parameter(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')
        return value

    def _unit_interval_parameter(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')
        return value


def main(args=None):
    """Run the markerless candidate stability subscriber."""
    rclpy.init(args=args)
    node = MarkerlessCandidateGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
