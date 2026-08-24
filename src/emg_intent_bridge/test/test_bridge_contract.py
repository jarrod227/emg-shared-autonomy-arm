"""Node-level checks that what the bridge publishes is what consumers accept.

The pure-logic tests prove the confirmation policy. They cannot catch a
mismatch in the field the policy hands downstream: the MCU margin byte was
being scaled into AssistiveIntent.confidence, and target_selector rejects
anything below intent_min_confidence (0.5 by default). Six correct live
events carried margins of 2, 55, 108, 157, 158 and 162, so three of them
would have been discarded after the whole pipeline had already accepted
them. This test runs a real node against a fake reader and pins the contract.
"""

import json
import queue

from assistive_interfaces.msg import AssistiveIntent, ViewControlCommand
import pytest
import rclpy
from rclpy.parameter import Parameter

from emg_intent_bridge.bridge_node import EmgIntentBridge
from emg_intent_bridge.confirmation import (
    ABORT,
    CONFIRM,
    NEXT_TARGET,
    REST,
    DeviceIntent,
)
from emg_intent_bridge.protocol_loader import (
    ACTIVATION_SOURCE_DEFAULTS,
    ACTIVATION_SOURCE_HOST,
    SET_RESULT_ACCEPTED,
    SET_RESULT_NONE,
    ActivationState,
)
from emg_intent_bridge.runtime import ReceivedIntent


SELECTOR_DEFAULT_MIN_CONFIDENCE = 0.5
# handoff_controller's view_confidence_min / view_signal_quality_min defaults.
CONTROLLER_VIEW_MIN_CONFIDENCE = 0.6
CONTROLLER_VIEW_MIN_SIGNAL_QUALITY = 0.5


def activation_state(*, source=ACTIVATION_SOURCE_DEFAULTS, factor=3,
                     baseline_shift=4, threshold_floor=110,
                     last_result=SET_RESULT_NONE, applied_sequence=0,
                     reference_left=0, reference_right=0):
    return ActivationState(source, factor, baseline_shift, last_result,
                           threshold_floor, applied_sequence,
                           reference_left, reference_right)


class _FakeStats:
    def __init__(self):
        self.accepted = 0
        self.lost = 0
        self.malformed = 0
        self.duplicated = 0
        self.time_reversed = 0
        self.discarded_bytes = 0


class _FakeParser:
    def __init__(self):
        self.stats = _FakeStats()


class _FakeDecoder:
    def __init__(self):
        self.parser = _FakeParser()
        self.intent_sequence_gaps = 0
        self.payload_errors = 0
        self.command_counts = {0: 0}
        self.last_signal_quality = None
        self.last_activation_state = None


class FakeReader:
    """Stand in for SerialIntentReader without touching a serial port."""

    def __init__(self):
        self.decoder = _FakeDecoder()
        self.connected = True
        self.error = None
        self.queue_drops = 0
        self.bytes_received = 0
        self.started = False
        self.stopped = False
        self.written = []
        self._queue = queue.Queue()

    def offer(self, intent, received_monotonic_ns):
        self._queue.put(ReceivedIntent(intent, received_monotonic_ns))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def write(self, data):
        self.written.append(bytes(data))
        return True

    def pop_nowait(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def bridge():
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        # Anchor on the first packet: these tests feed a handful of
        # packets with synthetic receipt times, not a warm-up's worth.
        Parameter("clock_anchor_window", value=1),
    ])
    # The board reports the power-on default state, which satisfies the
    # no-calibration-file handshake; these tests are about the intent path.
    reader.decoder.last_activation_state = activation_state()
    node._drive_handshake()
    assert node._handshake_confirmed
    published = []
    node.create_subscription(
        AssistiveIntent, "/assistive_intent", published.append, 10
    )
    yield node, reader, published
    node.destroy_node()


def device_intent(command, timestamp_us, *, confidence=200, quality=255,
                  direction=0, activation=0):
    return DeviceIntent(
        sequence=timestamp_us // 50_000,
        timestamp_us=timestamp_us,
        command=command,
        confidence=confidence,
        signal_quality=quality,
        direction=direction,
        activation=activation,
    )


def view_sink(node):
    """Collect everything the node publishes on the view-control topic."""
    received = []
    node.create_subscription(
        ViewControlCommand, "/assistive_view_control", received.append, 10
    )
    return received


def drain(node, reader, intents, *, expect):
    import time

    for intent in intents:
        reader.offer(intent, time.monotonic_ns())
    node._poll_serial()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if len(expect[0]) >= expect[1]:
            return


def test_published_view_confidence_clears_the_controller_gate(bridge):
    # The same defect the discrete path had, on a different gate. The MCU byte
    # is a classifier margin; live events measured 2..162, which is
    # 0.008..0.635 once scaled, against a view_confidence_min of 0.6. Mapping
    # it would drop almost every view command silently.
    node, reader, _published = bridge
    views = view_sink(node)

    drain(node, reader, [
        device_intent(REST, 1_000_000, confidence=2, direction=1,
                      activation=40_000),
    ], expect=(views, 1))

    assert len(views) == 1
    assert views[0].confidence >= CONTROLLER_VIEW_MIN_CONFIDENCE
    assert 0.0 <= views[0].confidence <= 1.0


def test_signal_quality_is_mapped_faithfully_unlike_confidence(bridge):
    # signal_quality measures ADC saturation in the window, which is a real
    # reason to refuse to steer, so it is not overridden the way confidence is.
    node, reader, _published = bridge
    views = view_sink(node)

    drain(node, reader, [
        device_intent(REST, 1_000_000, quality=255),
        device_intent(REST, 1_050_000, quality=0),
    ], expect=(views, 2))

    assert views[0].signal_quality == pytest.approx(1.0)
    assert views[1].signal_quality == pytest.approx(0.0)
    assert views[1].signal_quality < CONTROLLER_VIEW_MIN_SIGNAL_QUALITY


def test_every_packet_publishes_a_view_command_including_rest(bridge):
    # A silent view channel is indistinguishable from a dead one. REST is
    # published as HOLD so the controller's watchdog is never left guessing.
    node, reader, _published = bridge
    views = view_sink(node)

    drain(node, reader, [
        device_intent(REST, 1_000_000),
        device_intent(REST, 1_050_000, direction=-1, activation=65_535),
        device_intent(REST, 1_100_000, direction=1, activation=0),
    ], expect=(views, 3))

    assert [view.direction for view in views] == [
        ViewControlCommand.HOLD,
        ViewControlCommand.LEFT,
        ViewControlCommand.RIGHT,
    ]
    assert [view.activation for view in views] == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    assert [view.sequence for view in views] == [0, 1, 2]


def test_the_view_stream_does_not_wait_for_the_confirmation_gate(bridge):
    # The gate rejects a single spurious *event*; it has no meaning for a
    # stream the wearer is steering by watching the arm. One packet is enough.
    node, reader, published = bridge
    views = view_sink(node)

    drain(node, reader, [
        device_intent(NEXT_TARGET, 1_000_000, direction=1, activation=32_768),
    ], expect=(views, 1))

    assert published == []
    assert len(views) == 1
    assert views[0].direction == ViewControlCommand.RIGHT
    assert views[0].activation == pytest.approx(0.5, abs=1e-4)


def test_an_unconfirmed_handshake_publishes_hold_not_silence():
    # activation is normalized against the configuration the board confirms it
    # is judging with. Without that confirmation the number means nothing, but
    # going silent would make a deliberately still source and a dead link look
    # identical to the watchdog.
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        # Anchor on the first packet: these tests feed a handful of
        # packets with synthetic receipt times, not a warm-up's worth.
        Parameter("clock_anchor_window", value=1),
    ])
    views = view_sink(node)
    try:
        assert not node._handshake_confirmed

        drain(node, reader, [
            device_intent(REST, 1_000_000, direction=-1, activation=65_535),
        ], expect=(views, 1))

        assert len(views) == 1
        assert views[0].direction == ViewControlCommand.HOLD
        assert views[0].activation == pytest.approx(0.0)
    finally:
        node.destroy_node()


def test_published_confidence_clears_the_selector_default(bridge):
    node, reader, published = bridge

    # The weakest margin ever measured on a correct event.
    drain(node, reader, [
        device_intent(CONFIRM, 1_000_000, confidence=2),
        device_intent(CONFIRM, 1_500_000, confidence=2),
        device_intent(REST, 1_750_000),
    ], expect=(published, 1))

    assert len(published) == 1
    assert published[0].command == AssistiveIntent.CONFIRM
    assert published[0].confidence >= SELECTOR_DEFAULT_MIN_CONFIDENCE
    assert 0.0 <= published[0].confidence <= 1.0


def test_abort_reaches_ros_even_with_a_dead_quality_byte(bridge):
    node, reader, published = bridge

    drain(node, reader, [
        device_intent(ABORT, 1_000_000, quality=0),
    ], expect=(published, 1))

    assert len(published) == 1
    assert published[0].command == AssistiveIntent.ABORT


def test_abort_cancels_confirm_before_it_reaches_ros(bridge):
    node, reader, published = bridge

    drain(node, reader, [
        device_intent(CONFIRM, 1_000_000),
        device_intent(CONFIRM, 1_500_000),
        device_intent(REST, 1_600_000),
        device_intent(ABORT, 1_750_000),
    ], expect=(published, 1))

    assert [message.command for message in published] == [
        AssistiveIntent.ABORT
    ]


def test_stamps_advance_with_the_device_grid(bridge):
    node, reader, published = bridge

    drain(node, reader, [
        device_intent(ABORT, 1_000_000),
        device_intent(ABORT, 1_500_000),
    ], expect=(published, 2))

    assert len(published) == 2
    first, second = (
        message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        for message in published
    )
    assert second - first == 500_000_000


def test_unconfirmed_handshake_holds_ordinary_commands_but_not_abort():
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        # Anchor on the first packet: these tests feed a handful of
        # packets with synthetic receipt times, not a warm-up's worth.
        Parameter("clock_anchor_window", value=1),
    ])
    published = []
    node.create_subscription(
        AssistiveIntent, "/assistive_intent", published.append, 10
    )
    try:
        # No ACTIVATION_STATE has arrived at all: the board's configuration
        # is unknown, so ordinary commands must wait. The handshake request
        # itself must have gone out.
        node._drive_handshake()
        assert not node._handshake_confirmed
        assert len(reader.written) == 1

        drain(node, reader, [
            device_intent(NEXT_TARGET, 1_000_000),
            device_intent(NEXT_TARGET, 1_500_000),
            device_intent(ABORT, 1_750_000),
        ], expect=(published, 1))

        assert [message.command for message in published] == [
            AssistiveIntent.ABORT
        ]
        assert node._held_for_handshake == 1

        # The board comes back with the default state: the held pair is
        # gone (fail-closed), but new pairs flow.
        reader.decoder.last_activation_state = activation_state()
        node._drive_handshake()
        assert node._handshake_confirmed
        # This test advances one second of device time in almost no host time.
        # Reset the mapper between its two logical phases; clock-discontinuity
        # behavior has dedicated tests in test_runtime.py.
        node._clock_mapper.reset()
        drain(node, reader, [
            device_intent(CONFIRM, 2_000_000),
            device_intent(CONFIRM, 2_500_000),
            device_intent(REST, 2_750_000),
        ], expect=(published, 2))
        assert published[-1].command == AssistiveIntent.CONFIRM
    finally:
        node.destroy_node()


def test_calibration_file_drives_the_handshake_to_the_stored_values(tmp_path):

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 158,
        "verdict": "pass",
    }))
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        Parameter("clock_anchor_window", value=1),
        Parameter("calibration_file", value=str(calibration)),
    ])
    try:
        node._drive_handshake()
        assert len(reader.written) == 1

        # The power-on default state is NOT the wanted state when a
        # calibration was configured: the board must actually hold it.
        reader.decoder.last_activation_state = activation_state()
        node._drive_handshake()
        assert not node._handshake_confirmed

        # Host-sourced but with a different floor is a different
        # calibration, not this one.
        reader.decoder.last_activation_state = activation_state(
            source=ACTIVATION_SOURCE_HOST, threshold_floor=110,
            last_result=SET_RESULT_ACCEPTED,
        )
        node._drive_handshake()
        assert not node._handshake_confirmed

        reader.decoder.last_activation_state = activation_state(
            source=ACTIVATION_SOURCE_HOST, threshold_floor=158,
            last_result=SET_RESULT_ACCEPTED,
        )
        node._drive_handshake()
        assert node._handshake_confirmed
    finally:
        node.destroy_node()


def test_a_failed_calibration_file_refuses_to_start(tmp_path):

    calibration = tmp_path / "failed.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 103,
        "verdict": "fail",
    }))

    # Refusing beats silently falling back to defaults, which would look
    # like a calibrated system to everything downstream.
    with pytest.raises(ValueError, match="failed calibration"):
        EmgIntentBridge(reader=FakeReader(), parameter_overrides=[
            Parameter("clock_anchor_window", value=1),
            Parameter("calibration_file", value=str(calibration)),
        ])


def test_measured_references_reach_the_board_and_are_confirmed(tmp_path):
    """The references are the gain of the proportional channel.

    A board that accepted the threshold but not them would steer at the
    wrong scale while every log line said the handshake was confirmed, which
    is the failure mode this whole change exists to remove.
    """
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 55,
        "verdict": "pass",
        "reference_levels": {"NEXT_TARGET": 230.6, "ULNAR": 302.8},
    }))
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        Parameter("clock_anchor_window", value=1),
        Parameter("calibration_file", value=str(calibration)),
    ])
    try:
        assert node._handshake["reference_left"] == 231
        assert node._handshake["reference_right"] == 303

        # Right threshold, no references: not this calibration.
        reader.decoder.last_activation_state = activation_state(
            source=ACTIVATION_SOURCE_HOST, threshold_floor=55,
            last_result=SET_RESULT_ACCEPTED,
        )
        node._drive_handshake()
        assert not node._handshake_confirmed

        # One of the two applied is still not this calibration.
        reader.decoder.last_activation_state = activation_state(
            source=ACTIVATION_SOURCE_HOST, threshold_floor=55,
            last_result=SET_RESULT_ACCEPTED, reference_left=231,
        )
        node._drive_handshake()
        assert not node._handshake_confirmed

        reader.decoder.last_activation_state = activation_state(
            source=ACTIVATION_SOURCE_HOST, threshold_floor=55,
            last_result=SET_RESULT_ACCEPTED, reference_left=231,
            reference_right=303,
        )
        node._drive_handshake()
        assert node._handshake_confirmed
    finally:
        node.destroy_node()


def test_a_calibration_recorded_before_references_existed_still_starts(
    tmp_path,
):
    # Zero means "use the firmware fallback", so older files stay usable at
    # the gain they were collected with rather than refusing to load.
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 158,
        "verdict": "pass",
    }))
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
        Parameter("clock_anchor_window", value=1),
        Parameter("calibration_file", value=str(calibration)),
    ])
    try:
        assert node._handshake["reference_left"] == 0
        assert node._handshake["reference_right"] == 0
        assert "fallback" in node._handshake["description"]
    finally:
        node.destroy_node()
