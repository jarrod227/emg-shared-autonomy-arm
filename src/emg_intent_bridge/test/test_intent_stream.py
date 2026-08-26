import struct

from emg_intent_bridge.confirmation import CONFIRM, REST
from emg_intent_bridge.intent_stream import IntentStreamDecoder
from emg_intent_bridge.protocol_loader import MAGIC, TYPE_INTENT, TYPE_RAW, crc16


def packet(packet_type, sequence, timestamp_us, payload=b""):
    header = MAGIC + bytes((1, packet_type)) + struct.pack(
        "<HHI", len(payload), sequence, timestamp_us
    )
    body = header[2:] + payload
    return header + payload + struct.pack("<H", crc16(body))


def intent_packet(sequence, timestamp_us, command=REST, *, confidence=100,
                  quality=255):
    payload = struct.pack(
        "<BBBbHH", command, confidence, quality, 0, 0, 0
    )
    return packet(TYPE_INTENT, sequence, timestamp_us, payload)


def test_incremental_stream_ignores_raw_and_decodes_intent():
    decoder = IntentStreamDecoder()
    wire = (
        packet(TYPE_RAW, 0, 0, b"\x07\x00")
        + intent_packet(7, 400_000, CONFIRM, confidence=222)
    )

    decoded = []
    for byte in wire:
        decoded.extend(decoder.feed(bytes((byte,))))

    assert len(decoded) == 1
    assert decoded[0].sequence == 7
    assert decoded[0].timestamp_us == 400_000
    assert decoded[0].command == CONFIRM
    assert decoded[0].confidence == 222


def test_sequence_gap_is_attached_to_the_first_packet_after_the_gap():
    decoder = IntentStreamDecoder()

    decoded = decoder.feed(
        intent_packet(10, 100_000)
        + intent_packet(13, 250_000, CONFIRM)
    )

    assert [item.stream_discontinuity for item in decoded] == [False, True]
    assert decoder.intent_sequence_gaps == 2


def test_duplicate_and_reversed_intents_never_reach_the_gate():
    decoder = IntentStreamDecoder()

    decoded = decoder.feed(
        intent_packet(20, 1_000_000)
        + intent_packet(20, 1_000_000)
        + intent_packet(19, 950_000)
        + intent_packet(21, 1_050_000)
    )

    assert [item.sequence for item in decoded] == [20, 21]
    assert decoder.duplicate_intents == 1
    assert decoder.reversed_intents == 1


def test_sequence_and_timestamp_wrap_are_unwrapped_forward():
    decoder = IntentStreamDecoder()

    decoded = decoder.feed(
        intent_packet(65535, 0xFFFF_F000)
        + intent_packet(0, 0x0000_B350)
    )

    assert len(decoded) == 2
    assert decoded[1].timestamp_us - decoded[0].timestamp_us == 50_000
    assert not decoded[1].stream_discontinuity


def test_wrong_sized_intent_payload_is_counted_and_dropped():
    decoder = IntentStreamDecoder()

    assert decoder.feed(packet(TYPE_INTENT, 0, 0, b"\x00")) == []
    assert decoder.payload_errors == 1


def test_activation_state_is_captured_without_entering_the_intent_stream():
    from emg_intent_bridge.protocol_loader import TYPE_ACTIVATION_STATE

    decoder = IntentStreamDecoder()
    state_payload = struct.pack("<BBBBiHHii", 1, 3, 4, 1, 158, 9, 21, 231, 303)
    wire = (
        packet(TYPE_ACTIVATION_STATE, 0, 0, state_payload)
        + intent_packet(1, 400_000, CONFIRM)
    )

    decoded = decoder.feed(wire)

    # The state is a side channel: it must not appear as an intent, and its
    # sequence numbering must not disturb the intent continuity tracking.
    assert len(decoded) == 1
    assert decoded[0].command == CONFIRM
    assert decoded[0].stream_discontinuity is False
    state = decoder.last_activation_state
    assert state.from_host
    assert (state.factor, state.baseline_shift) == (3, 4)
    assert state.threshold_floor == 158
    assert state.applied_sequence == 9
    assert (state.reference_left, state.reference_right) == (231, 303)
    assert decoder.activation_state_count == 1
