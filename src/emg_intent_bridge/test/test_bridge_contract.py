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

from assistive_interfaces.msg import AssistiveIntent
import pytest
import rclpy

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


def activation_state(*, source=ACTIVATION_SOURCE_DEFAULTS, factor=3,
                     baseline_shift=4, threshold_floor=110,
                     last_result=SET_RESULT_NONE, applied_sequence=0):
    return ActivationState(source, factor, baseline_shift, last_result,
                           threshold_floor, applied_sequence)


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
    node = EmgIntentBridge(reader=reader)
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


def device_intent(command, timestamp_us, *, confidence=200, quality=255):
    return DeviceIntent(
        sequence=timestamp_us // 50_000,
        timestamp_us=timestamp_us,
        command=command,
        confidence=confidence,
        signal_quality=quality,
    )


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
    node = EmgIntentBridge(reader=reader)
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
    from rclpy.parameter import Parameter

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 158,
        "verdict": "pass",
    }))
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader, parameter_overrides=[
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
    from rclpy.parameter import Parameter

    calibration = tmp_path / "failed.json"
    calibration.write_text(json.dumps({
        "factor": 3, "baseline_shift": 4, "threshold_floor": 103,
        "verdict": "fail",
    }))

    # Refusing beats silently falling back to defaults, which would look
    # like a calibrated system to everything downstream.
    with pytest.raises(ValueError, match="failed calibration"):
        EmgIntentBridge(reader=FakeReader(), parameter_overrides=[
            Parameter("calibration_file", value=str(calibration)),
        ])
