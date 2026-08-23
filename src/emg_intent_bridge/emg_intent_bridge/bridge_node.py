"""ROS 2 USB CDC bridge for Objective 3.5 discrete EMG intent events."""

import json
import pathlib
import time

from assistive_interfaces.msg import AssistiveIntent, ViewControlCommand
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
    REST,
    IntentConfirmationGate,
)
from .protocol_loader import (
    SET_MODE_APPLY,
    SET_MODE_DEFAULTS,
    SET_RESULT_ACCEPTED,
    encode_set_activation,
)
from .runtime import DeviceClockMapper, SerialIntentReader


# The handshake is confirmed by the board *being in* the wanted state, not
# by matching an acknowledgement to a send: a board already holding the
# right configuration from an identical previous session is exactly as
# calibrated as one that just applied it. Resends therefore reuse one
# sequence value; the firmware's application is idempotent.
# The wire uses -1/0/+1; ROS uses named constants. Kept as a table so an
# unexpected value raises a KeyError instead of silently steering one way:
# DeviceIntent already rejects anything outside this domain on construction.
_VIEW_DIRECTIONS = {
    -1: ViewControlCommand.LEFT,
    0: ViewControlCommand.HOLD,
    1: ViewControlCommand.RIGHT,
}

_HANDSHAKE_SEQUENCE = 1
_HANDSHAKE_RESEND_SEC = 3.0


COMMAND_NAMES = {
    NEXT_TARGET: "NEXT_TARGET",
    CONFIRM: "CONFIRM",
    ABORT: "ABORT",
}


class EmgIntentBridge(Node):
    """Decode MCU packets, reject stale evidence, and publish confirmed intent."""

    def __init__(self, *, reader=None, parameter_overrides=None):
        super().__init__("emg_intent_bridge",
                         parameter_overrides=parameter_overrides)
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout_sec", 0.05)
        self.declare_parameter("poll_period_sec", 0.01)
        self.declare_parameter("max_receipt_age_sec", 0.25)
        self.declare_parameter("confirmation_window_sec", 5.5)
        self.declare_parameter("confirm_abort_override_sec", 0.25)
        self.declare_parameter("intent_topic", "/assistive_intent")
        self.declare_parameter(
            "view_control_topic", "/assistive_view_control"
        )
        self.declare_parameter("frame_id", "stm32_emg")
        # Path to a calibration JSON from emg_calibrate.py, or "" for none.
        # Both cases send an explicit request on startup: with a file, the
        # calibrated values; without one, a return to compile-time defaults.
        # "Send nothing" is deliberately not a state — an un-reset board
        # would silently keep a previous wearer's RAM configuration.
        self.declare_parameter("calibration_file", "")
        # How many packets to weigh before freezing the device-clock anchor.
        # 20 is one second at the MCU's 20 Hz feature hop -- long enough that
        # a single badly buffered receipt cannot become a permanent forward
        # bias on every stamp, short enough not to be felt at startup.
        self.declare_parameter("clock_anchor_window", 20)

        self._port = str(self.get_parameter("port").value)
        self._max_receipt_age_sec = float(
            self.get_parameter("max_receipt_age_sec").value
        )
        poll_period_sec = float(self.get_parameter("poll_period_sec").value)
        confirmation_window_sec = float(
            self.get_parameter("confirmation_window_sec").value
        )
        confirm_abort_override_sec = float(
            self.get_parameter("confirm_abort_override_sec").value
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
        # Matches the controller's subscription exactly: KEEP_LAST(1) and
        # VOLATILE, per the ViewControlCommand contract. A mismatch here is not
        # an error anyone sees, it is silence.
        view_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._view_publisher = self.create_publisher(
            ViewControlCommand,
            str(self.get_parameter("view_control_topic").value),
            view_qos,
        )
        self._view_sequence = 0
        self._view_published_count = 0
        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._gate = IntentConfirmationGate(
            confirmation_window_sec,
            abort_override_window_sec=confirm_abort_override_sec,
        )
        self._clock_mapper = DeviceClockMapper(
            anchor_window=int(
                self.get_parameter("clock_anchor_window").value
            )
        )
        self._reader = reader or SerialIntentReader(
            self._port,
            baudrate=int(self.get_parameter("baudrate").value),
            timeout_sec=float(self.get_parameter("serial_timeout_sec").value),
        )
        self._output_sequence = 0
        self._published_count = 0
        self._stale_count = 0
        self._warmup_skipped = 0
        self._last_anchor_bias_sec = None
        self._future_biased_packets = 0
        # 0.05 s is what the handoff controller allows; anything past it will
        # be refused there, so it is the right place to start complaining.
        # One second of it is a frozen anchor, not a scheduling hiccup.
        self._future_bias_warn_sec = 0.05
        self._future_bias_run = 20
        self._last_reanchors = 0
        self._last_queue_drops = 0
        self._last_event_margin = None
        self._last_event_quality = None
        self._last_mcu_event_us = None
        self._last_receipt_monotonic_ns = None
        self._last_source_age_sec = None
        self._reported_reader_error = None
        self._handshake = self._load_handshake(
            str(self.get_parameter("calibration_file").value)
        )
        self._handshake_confirmed = False
        self._handshake_sends = 0
        self._handshake_last_send_monotonic = None
        self._held_for_handshake = 0
        self._reader.start()
        self._poll_timer = self.create_timer(poll_period_sec, self._poll_serial)
        self._diagnostic_timer = self.create_timer(1.0, self._publish_diagnostics)
        self._handshake_timer = self.create_timer(0.5, self._drive_handshake)
        self.get_logger().info(
            f"reading {self._port}; NEXT_TARGET/CONFIRM require two events "
            f"within {confirmation_window_sec:.2f}s; CONFIRM waits "
            f"{confirm_abort_override_sec:.2f}s for an ABORT override; "
            f"ABORT is immediate; activation handshake: "
            f"{self._handshake['description']}"
        )

    @staticmethod
    def _load_handshake(calibration_file):
        """Build the startup request the board must be brought to."""
        if not calibration_file:
            return {
                "mode": SET_MODE_DEFAULTS,
                # Values are ignored for mode=0 but the packet carries them;
                # send the compile-time defaults so a captured trace reads
                # sensibly.
                "factor": 3,
                "baseline_shift": 4,
                "threshold_floor": 110,
                "description": "no calibration file; restoring defaults",
            }
        path = pathlib.Path(calibration_file)
        summary = json.loads(path.read_text())
        verdict = summary.get("verdict")
        if verdict == "fail":
            # A failing calibration means the donning could not separate
            # preparation from gesture. Refusing to start is the honest
            # response; silently falling back to defaults would look like a
            # calibrated system.
            raise ValueError(
                f"{path} records a failed calibration; re-place the "
                f"electrodes and calibrate again"
            )
        request = {
            "mode": SET_MODE_APPLY,
            "factor": int(summary["factor"]),
            "baseline_shift": int(summary["baseline_shift"]),
            "threshold_floor": int(summary["threshold_floor"]),
            "description": (
                f"{path.name} (floor {summary['threshold_floor']}, "
                f"verdict {verdict})"
            ),
        }
        # Encode once now so an out-of-range file fails at startup with a
        # clear message instead of as an endless rejected handshake.
        encode_set_activation(
            _HANDSHAKE_SEQUENCE, mode=request["mode"],
            factor=request["factor"],
            baseline_shift=request["baseline_shift"],
            threshold_floor=request["threshold_floor"],
        )
        return request

    def _handshake_state_matches(self, state):
        """Is the board in the state the handshake is driving it to?"""
        if state is None:
            return False
        if self._handshake["mode"] == SET_MODE_DEFAULTS:
            # Defaults are the wanted state however the board got there --
            # including power-on, before any request arrives.
            return not state.from_host
        return (
            state.from_host
            and state.last_result == SET_RESULT_ACCEPTED
            and (state.factor, state.baseline_shift, state.threshold_floor)
            == (self._handshake["factor"], self._handshake["baseline_shift"],
                self._handshake["threshold_floor"])
        )

    def _drive_handshake(self):
        if self._handshake_confirmed:
            return
        state = self._reader.decoder.last_activation_state
        if self._handshake_state_matches(state):
            self._handshake_confirmed = True
            self.get_logger().info(
                f"activation handshake confirmed: K={state.factor} "
                f"shift={state.baseline_shift} floor={state.threshold_floor} "
                f"source={'host' if state.from_host else 'defaults'}"
            )
            return
        now = time.monotonic()
        overdue = (
            self._handshake_last_send_monotonic is None
            or now - self._handshake_last_send_monotonic
            >= _HANDSHAKE_RESEND_SEC
        )
        if not overdue:
            return
        sent = self._reader.write(encode_set_activation(
            _HANDSHAKE_SEQUENCE, mode=self._handshake["mode"],
            factor=self._handshake["factor"],
            baseline_shift=self._handshake["baseline_shift"],
            threshold_floor=self._handshake["threshold_floor"],
        ))
        if sent:
            self._handshake_last_send_monotonic = now
            self._handshake_sends += 1
            if self._handshake_sends > 1:
                self.get_logger().warning(
                    f"activation handshake unconfirmed after "
                    f"{self._handshake_sends - 1} attempt(s); resending"
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
            if source_ros_ns is None:
                # Still choosing an anchor. Publishing here would mean
                # stamping with an offset the mapper has already decided not
                # to trust, which is the bias this warm-up exists to avoid.
                self._warmup_skipped += 1
                if self._clock_mapper.reanchors != self._last_reanchors:
                    self._last_reanchors = self._clock_mapper.reanchors
                    self._gate.invalidate()
                    self.get_logger().warning(
                        "device clock re-anchored: cleared pending "
                        "confirmation and restarted the anchor warm-up"
                    )
                continue
            if self._clock_mapper.reanchors != self._last_reanchors:
                # A clock discontinuity (ROS time set, or an MCU restart the
                # sequence check missed) makes any pending pair unpaired
                # evidence: its stamp and this one no longer share a mapping.
                self._last_reanchors = self._clock_mapper.reanchors
                self._gate.invalidate()
                self.get_logger().warning(
                    "device clock re-anchored: cleared pending confirmation"
                )
            # Signed, and kept separately: last_source_age_sec is clamped at
            # zero, so a stamp in the future reads as a perfectly fresh 0.0.
            # That clamp is why a session where every command was refused for
            # being 0.13 s in the future showed nothing but healthy numbers
            # here. Negative bias means the anchor is running ahead of the
            # host clock, and a consumer's future tolerance will refuse it.
            signed_age_sec = (now_ros_ns - source_ros_ns) / 1e9
            self._last_anchor_bias_sec = signed_age_sec
            if signed_age_sec < -self._future_bias_warn_sec:
                self._future_biased_packets += 1
                if self._future_biased_packets == self._future_bias_run:
                    self.get_logger().error(
                        f"source stamps run {-signed_age_sec:.3f}s ahead of "
                        "the host clock; a consumer that allows less future "
                        "than that will refuse every command. The anchor is "
                        "frozen for the life of this process -- restart the "
                        "bridge to re-run the warm-up."
                    )
            else:
                self._future_biased_packets = 0
            self._last_source_age_sec = max(0.0, signed_age_sec)
            if self._last_source_age_sec > self._max_receipt_age_sec:
                self._stale_count += 1
                self._gate.invalidate()
                continue

            self._publish_view_command(received.intent, source_ros_ns)

            if received.intent.command != REST:
                # Every MCU event is logged with its device time, published or
                # not. Counters alone cannot say whether a pair missed the
                # confirmation window or the second gesture never arrived.
                gap = "first"
                if self._last_mcu_event_us is not None:
                    gap = (
                        f"{(received.intent.timestamp_us - self._last_mcu_event_us) / 1e6:.2f}s"
                        " since previous"
                    )
                self._last_mcu_event_us = received.intent.timestamp_us
                self.get_logger().info(
                    f"MCU event {COMMAND_NAMES.get(received.intent.command)} "
                    f"at {received.intent.timestamp_us / 1e6:.2f}s ({gap}), "
                    f"margin={received.intent.confidence}, "
                    f"quality={received.intent.signal_quality}"
                )

            confirmed = self._gate.push(received.intent)
            if confirmed is None:
                continue
            if not self._handshake_confirmed and confirmed.command != ABORT:
                # The board has not confirmed it is judging with the wanted
                # activation configuration, so ordinary commands may have
                # passed the wrong threshold. ABORT still goes through:
                # refusing a stop because of an unfinished handshake trades
                # a certain hazard for a configuration formality.
                self._held_for_handshake += 1
                self.get_logger().warning(
                    f"held {COMMAND_NAMES[confirmed.command]}: activation "
                    f"handshake not confirmed"
                )
                continue
            # Deferred CONFIRM is released by a later REST liveness packet.
            # Preserve the paired MCU event's source stamp rather than using
            # release time; the mapper preserves device-time intervals.
            confirmed_source_ros_ns = source_ros_ns - max(
                0,
                received.intent.timestamp_us - confirmed.timestamp_us,
            ) * 1000
            self._publish_intent(confirmed, confirmed_source_ros_ns)

    def _publish_view_command(self, intent, source_ros_ns):
        """Publish the proportional half of one INTENT packet.

        Every packet carries both halves. The event gate emits at most one
        discrete command per gesture; direction and activation are continuous,
        one per 50 ms hop, and go to a different consumer. They are published
        here without waiting on the confirmation gate, which exists to reject
        single spurious *events* and has no meaning for a stream the wearer is
        steering visually.

        REST packets are published too, as HOLD. The protocol states the
        absence of intent rather than implying it, and a silent view channel is
        indistinguishable from a dead one; the controller's watchdog would then
        have to guess which it was.
        """
        message = ViewControlCommand()
        message.header.stamp = Time(nanoseconds=source_ros_ns).to_msg()
        message.header.frame_id = self._frame_id
        if not self._handshake_confirmed:
            # The board has not confirmed which activation configuration it is
            # judging with, and activation is normalized against exactly that.
            # HOLD rather than silence: it keeps the controller's watchdog fed
            # with a true statement instead of making a dead link and a
            # deliberately still one look identical.
            direction = ViewControlCommand.HOLD
            activation = 0.0
        else:
            direction = _VIEW_DIRECTIONS[intent.direction]
            activation = intent.activation / 65535.0
        message.direction = direction
        message.activation = activation
        # Same reasoning as the discrete path, and the same measured numbers:
        # the MCU byte is a Q18 classifier margin, and live events ran 2..162.
        # Scaled into [0, 1] that is 0.008..0.635 against a view_confidence_min
        # of 0.6, so almost every view command would be dropped by a gate the
        # margin was never validated against. Publication is the confidence
        # statement; the raw margin stays observable in /diagnostics.
        message.confidence = 1.0
        # signal_quality is mapped faithfully, unlike confidence, because it
        # does measure something physical: the firmware derives it from ADC
        # saturations in the window, and a clipping window is a real reason to
        # refuse to steer.
        message.signal_quality = intent.signal_quality / 255.0
        message.sequence = self._view_sequence
        self._view_sequence = (self._view_sequence + 1) & 0xFFFF_FFFF
        self._view_publisher.publish(message)
        self._view_published_count += 1

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
            "mcu_events": {
                COMMAND_NAMES[command]: count
                for command, count in sorted(
                    self._reader.decoder.command_counts.items()
                )
                if command in COMMAND_NAMES
            },
            "mcu_rest_windows": self._reader.decoder.command_counts[REST],
            "mcu_last_signal_quality":
                self._reader.decoder.last_signal_quality,
            "handshake_confirmed": self._handshake_confirmed,
            "handshake_request": self._handshake["description"],
            "handshake_sends": self._handshake_sends,
            "held_for_handshake": self._held_for_handshake,
            "mcu_activation_state": self._reader.decoder.last_activation_state,
            "queue_drops": self._reader.queue_drops,
            "stale_packets": self._stale_count,
            "clock_warmup_skipped": self._warmup_skipped,
            "clock_anchor_bias_sec": self._last_anchor_bias_sec,
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
        elif not self._handshake_confirmed and self._handshake_sends > 1:
            # More than one send means the first confirmation window
            # elapsed. NEXT_TARGET/CONFIRM are being held; ABORT passes.
            level = DiagnosticStatus.ERROR
            summary = (
                "activation handshake unconfirmed; ordinary commands held"
            )
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
