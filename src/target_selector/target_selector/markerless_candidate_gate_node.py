"""Subscribe to markerless candidates and feed the pure stability gate."""

from assistive_interfaces.msg import ObjectCandidateArray
import rclpy
from rclpy.node import Node

from target_selector.candidate_adapter import candidate_frame_from_message
from target_selector.candidate_stability import (
    CandidateFrame,
    CandidateStabilityConfig,
    CandidateStabilityGate,
    DEFAULT_OBJECT_CLASSES,
)


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
        self._gate = CandidateStabilityGate(config)
        self._last_decision = None
        self._processed_message_count = 0
        self._subscription = self.create_subscription(
            ObjectCandidateArray,
            candidate_topic.strip(),
            self._on_candidates,
            10,
        )
        self.get_logger().info(
            f'waiting for markerless candidates on {candidate_topic.strip()}'
        )

    @property
    def last_decision(self):
        """Return the latest pure gate result for diagnostics and tests."""
        return self._last_decision

    @property
    def processed_message_count(self):
        """Return how many candidate messages reached this subscription."""
        return self._processed_message_count

    def _declare_parameters(self):
        self.declare_parameter('candidate_topic', '/object_candidates')
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
        self._processed_message_count += 1
        if self._last_decision.stable_candidates:
            tracks = ', '.join(
                f'{candidate.track_id}:{candidate.class_label}'
                for candidate in self._last_decision.stable_candidates
            )
            self.get_logger().info(f'stable markerless candidates: {tracks}')


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
