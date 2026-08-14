"""Node-level checks that what the bridge publishes is what consumers accept.

The pure-logic tests prove the confirmation policy. They cannot catch a
mismatch in the field the policy hands downstream: the MCU margin byte was
being scaled into AssistiveIntent.confidence, and target_selector rejects
anything below intent_min_confidence (0.5 by default). Six correct live
events carried margins of 2, 55, 108, 157, 158 and 162, so three of them
would have been discarded after the whole pipeline had already accepted
them. This test runs a real node against a fake reader and pins the contract.
"""

import queue

from assistive_interfaces.msg import AssistiveIntent
import pytest
import rclpy

from emg_intent_bridge.bridge_node import EmgIntentBridge
from emg_intent_bridge.confirmation import ABORT, CONFIRM, DeviceIntent
from emg_intent_bridge.runtime import ReceivedIntent


SELECTOR_DEFAULT_MIN_CONFIDENCE = 0.5


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
        self._queue = queue.Queue()

    def offer(self, intent, received_monotonic_ns):
        self._queue.put(ReceivedIntent(intent, received_monotonic_ns))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def pop_nowait(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class _FakeStats:
    accepted = 0
    lost = 0
    malformed = 0
    duplicated = 0
    time_reversed = 0
    discarded_bytes = 0


class _FakeParser:
    stats = _FakeStats()


class _FakeDecoder:
    parser = _FakeParser()
    intent_sequence_gaps = 0
    payload_errors = 0


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def bridge():
    reader = FakeReader()
    node = EmgIntentBridge(reader=reader)
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
        device_intent(CONFIRM, 2_000_000, confidence=2),
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
