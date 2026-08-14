"""ROS 2 USB CDC bridge for Objective 3.5 discrete EMG intent events."""

import time

from assistive_interfaces.msg import AssistiveIntent
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time

from .confirmation import (
    ABORT,
    CONFIRM,
    NEXT_TARGET,
    IntentConfirmationGate,
)
from .runtime import DeviceClockMapper, SerialIntentReader


COMMAND_NAMES = {
    NEXT_TARGET: "NEXT_TARGET",
    CONFIRM: "CONFIRM",
    ABORT: "ABORT",
}


class EmgIntentBridge(Node):
    """Decode MCU packets, reject stale evidence, and publish confirmed intent."""

    def __init__(self, *, reader=None):
        super().__init__("emg_intent_bridge")
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout_sec", 0.05)
        self.declare_parameter("poll_period_sec", 0.01)
        self.declare_parameter("max_receipt_age_sec", 0.25)
        self.declare_parameter("confirmation_window_sec", 3.0)
        self.declare_parameter("intent_topic", "/assistive_intent")
        self.declare_parameter("frame_id", "stm32_emg")

        self._port = str(self.get_parameter("port").value)
        self._max_receipt_age_sec = float(
            self.get_parameter("max_receipt_age_sec").value
        )
        poll_period_sec = float(self.get_parameter("poll_period_sec").value)
        confirmation_window_sec = float(
            self.get_parameter("confirmation_window_sec").value
        )
        if self._max_receipt_age_sec <= 0.0 or poll_period_sec <= 0.0:
            raise ValueError("receipt age and poll period must be positive")

        intent_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._intent_publisher = self.create_publisher(
            AssistiveIntent,
            str(self.get_parameter("intent_topic").value),
            intent_qos,
        )
        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._gate = IntentConfirmationGate(confirmation_window_sec)
        self._clock_mapper = DeviceClockMapper()
        self._reader = reader or SerialIntentReader(
            self._port,
            baudrate=int(self.get_parameter("baudrate").value),
            timeout_sec=float(self.get_parameter("serial_timeout_sec").value),
        )
        self._output_sequence = 0
        self._published_count = 0
        self._stale_count = 0
        self._last_reanchors = 0
        self._last_queue_drops = 0
        self._last_event_margin = None
        self._last_event_quality = None
        self._last_receipt_monotonic_ns = None
        self._last_source_age_sec = None
        self._reported_reader_error = None
        self._reader.start()
        self._poll_timer = self.create_timer(poll_period_sec, self._poll_serial)
        self._diagnostic_timer = self.create_timer(1.0, self._publish_diagnostics)
        self.get_logger().info(
            f"reading {self._port}; NEXT_TARGET/CONFIRM require two events "
            f"within {confirmation_window_sec:.2f}s; ABORT is immediate"
        )

    def _poll_serial(self):
        if self._reader.queue_drops != self._last_queue_drops:
            self._gate.invalidate()
            self.get_logger().error(
                "serial-to-ROS queue overflow: discarded evidence and "
                "cleared pending confirmation"
            )
            self._last_queue_drops = self._reader.queue_drops

        if (
            self._reader.error is not None
            and self._reader.error != self._reported_reader_error
        ):
            self._reported_reader_error = self._reader.error
            self._gate.invalidate()
            self.get_logger().error(f"serial reader stopped: {self._reader.error}")

        while True:
            received = self._reader.pop_nowait()
            if received is None:
                return
            now_monotonic_ns = time.monotonic_ns()
            now_ros_ns = self.get_clock().now().nanoseconds
            receipt_age_sec = (
                now_monotonic_ns - received.received_monotonic_ns
            ) / 1e9
            self._last_receipt_monotonic_ns = received.received_monotonic_ns
            if receipt_age_sec > self._max_receipt_age_sec:
                self._stale_count += 1
                self._gate.invalidate()
                continue

            # Reconstruct the ROS time at serial receipt rather than stamping
            # at executor processing time. The mapper then preserves the MCU's
            # exact 50 ms source grid across variable host queue latency.
            receipt_ros_ns = now_ros_ns - max(
                0, now_monotonic_ns - received.received_monotonic_ns
            )
            source_ros_ns = self._clock_mapper.map(
                received.intent.timestamp_us,
                receipt_ros_ns,
            )
            if self._clock_mapper.reanchors != self._last_reanchors:
                # A clock discontinuity (ROS time set, or an MCU restart the
                # sequence check missed) makes any pending pair unpaired
                # evidence: its stamp and this one no longer share a mapping.
                self._last_reanchors = self._clock_mapper.reanchors
                self._gate.invalidate()
                self.get_logger().warning(
                    "device clock re-anchored: cleared pending confirmation"
                )
            self._last_source_age_sec = max(
                0.0, (now_ros_ns - source_ros_ns) / 1e9
            )
            if self._last_source_age_sec > self._max_receipt_age_sec:
                self._stale_count += 1
                self._gate.invalidate()
                continue

            confirmed = self._gate.push(received.intent)
            if confirmed is not None:
                self._publish_intent(confirmed, source_ros_ns)

    def _publish_intent(self, intent, source_ros_ns):
        message = AssistiveIntent()
        message.header.stamp = Time(nanoseconds=source_ros_ns).to_msg()
        message.header.frame_id = self._frame_id
        message.command = intent.command
        # The MCU byte is a compressed Q18 classifier margin, not a
        # probability; scaling it into [0, 1] made downstream >= 0.5 checks
        # reject half the events the whole pipeline had already accepted
        # (measured 2..162 over six correct live events). Publication itself
        # is the confidence statement here: activation threshold, five
        # agreeing windows, and the double-event policy have all passed. The
        # raw margin stays observable in /diagnostics.
        message.confidence = 1.0
        message.sequence = self._output_sequence
        self._output_sequence = (self._output_sequence + 1) & 0xFFFF_FFFF
        self._intent_publisher.publish(message)
        self._published_count += 1
        self._last_event_margin = intent.confidence
        self._last_event_quality = intent.signal_quality
        self.get_logger().info(
            f"published {COMMAND_NAMES[intent.command]} "
            f"sequence={message.sequence} margin={intent.confidence} "
            f"quality={intent.signal_quality}"
        )

    def _diagnostic_values(self):
        parser = self._reader.decoder.parser.stats
        last_age = "never"
        if self._last_receipt_monotonic_ns is not None:
            last_age = (
                f"{(time.monotonic_ns() - self._last_receipt_monotonic_ns) / 1e9:.3f}"
            )
        values = {
            "connected": self._reader.connected,
            "bytes_received": self._reader.bytes_received,
            "packets_accepted": parser.accepted,
            "packets_lost": parser.lost,
            "packets_malformed": parser.malformed,
            "packets_duplicated": parser.duplicated,
            "packets_time_reversed": parser.time_reversed,
            "discarded_bytes": parser.discarded_bytes,
            "intent_sequence_gaps": self._reader.decoder.intent_sequence_gaps,
            "intent_payload_errors": self._reader.decoder.payload_errors,
            "queue_drops": self._reader.queue_drops,
            "stale_packets": self._stale_count,
            "clock_reanchors": self._clock_mapper.reanchors,
            "published_intents": self._published_count,
            "last_event_margin": self._last_event_margin,
            "last_event_quality": self._last_event_quality,
            "pending_command": self._gate.pending_command,
            "last_receipt_age_sec": last_age,
            "last_source_age_sec": self._last_source_age_sec,
        }
        return [KeyValue(key=key, value=str(value)) for key, value in values.items()]

    def _publish_diagnostics(self):
        parser = self._reader.decoder.parser.stats
        anomalies = (
            parser.lost
            + parser.malformed
            + parser.duplicated
            + parser.time_reversed
            + self._reader.queue_drops
            + self._stale_count
        )
        if self._reader.error is not None or not self._reader.connected:
            level = DiagnosticStatus.ERROR
            summary = self._reader.error or "serial port not connected"
        elif anomalies:
            level = DiagnosticStatus.WARN
            summary = "stream anomalies detected; affected evidence discarded"
        else:
            level = DiagnosticStatus.OK
            summary = "serial stream healthy"
        status = DiagnosticStatus(
            level=level,
            name=f"{self.get_name()}: USB CDC intent stream",
            message=summary,
            hardware_id=self._port,
            values=self._diagnostic_values(),
        )
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self._diagnostic_publisher.publish(message)

    def destroy_node(self):
        self._reader.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EmgIntentBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
