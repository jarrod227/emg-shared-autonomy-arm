"""Tests for the host-side sEMG protocol decoder.

Packets are built here byte by byte straight from PROTOCOL.md rather than by
calling a shared encoder, so a mistake in the decoder cannot be cancelled out
by the same mistake in its own encoder. The last test closes the loop the
other way, against bytes the independent C implementation produced.

    python3 -m pytest firmware/tools/test_emg_protocol.py
"""

import pathlib
import struct

import pytest

from emg_protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD,
    SET_MODE_APPLY,
    SET_RESULT_ACCEPTED,
    TYPE_ACTIVATION_STATE,
    TYPE_INFO,
    TYPE_INTENT,
    TYPE_RAW,
    TYPE_SET_ACTIVATION,
    PacketParser,
    crc16,
    decode_activation_state,
    decode_info,
    decode_intent,
    decode_raw,
    encode_packet,
    encode_set_activation,
)

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "test" / "fixture.bin"


def build(packet_type, sequence, timestamp_us, payload=b"", *,
          version=1, length=None, corrupt_crc=False):
    """Assemble one packet, with hooks for producing invalid ones."""
    declared = len(payload) if length is None else length
    header = MAGIC + bytes([version, packet_type]) + struct.pack(
        "<HHI", declared, sequence, timestamp_us
    )
    body = header[2:] + payload
    checksum = crc16(body) ^ (0xFFFF if corrupt_crc else 0)
    return header + payload + struct.pack("<H", checksum)


def test_crc_matches_the_published_check_value():
    assert crc16(b"123456789") == 0x29B1


def test_parses_a_single_packet():
    parser = PacketParser()
    packets = parser.feed(build(TYPE_INTENT, 7, 12345, b"\x01\x02"))

    assert len(packets) == 1
    assert packets[0].type == TYPE_INTENT
    assert packets[0].sequence == 7
    assert packets[0].timestamp_us == 12345
    assert packets[0].payload == b"\x01\x02"
    assert parser.stats.accepted == 1
    assert parser.stats.malformed == 0


def test_parses_several_packets_in_one_chunk():
    parser = PacketParser()
    stream = b"".join(build(TYPE_RAW, index, index * 100, b"\xaa\xbb")
                      for index in range(4))

    packets = parser.feed(stream)

    assert [packet.sequence for packet in packets] == [0, 1, 2, 3]
    assert parser.stats.lost == 0


def test_reassembles_a_packet_split_one_byte_at_a_time():
    parser = PacketParser()
    stream = build(TYPE_INTENT, 3, 999, b"\x05\x06\x07")

    collected = []
    for index in range(len(stream)):
        collected.extend(parser.feed(stream[index:index + 1]))

    assert len(collected) == 1
    assert collected[0].payload == b"\x05\x06\x07"


def test_skips_leading_junk_and_still_finds_the_packet():
    parser = PacketParser()
    junk = b"\x00\xa5\xff\x5a\x11"

    packets = parser.feed(junk + build(TYPE_INFO, 0, 1, b"\x01"))

    assert len(packets) == 1
    assert parser.stats.discarded_bytes == len(junk)
    assert parser.stats.malformed == 0


def test_a_corrupt_crc_is_counted_and_does_not_hide_the_next_packet():
    parser = PacketParser()
    stream = (build(TYPE_INTENT, 0, 0, b"\x01\x02", corrupt_crc=True)
              + build(TYPE_INTENT, 1, 10, b"\x03\x04"))

    packets = parser.feed(stream)

    assert [packet.sequence for packet in packets] == [1]
    assert parser.stats.reasons["crc"] == 1


def test_an_oversized_length_is_rejected_without_trusting_it():
    parser = PacketParser()
    # Declaring more than MAX_PAYLOAD must fail the bound check rather than
    # make the parser wait forever for bytes that will never arrive.
    stream = (build(TYPE_RAW, 0, 0, b"\x01", length=MAX_PAYLOAD + 1)
              + build(TYPE_RAW, 1, 0, b"\x02"))

    packets = parser.feed(stream)

    assert [packet.sequence for packet in packets] == [1]
    assert parser.stats.reasons["length"] == 1


@pytest.mark.parametrize(
    "kwargs, reason",
    [({"version": 2}, "version"), ({"packet_type": 0x7F}, "type")],
)
def test_unknown_version_or_type_is_malformed(kwargs, reason):
    parser = PacketParser()
    defaults = {"packet_type": TYPE_RAW, "sequence": 0, "timestamp_us": 0}
    defaults.update(kwargs)

    packets = parser.feed(build(payload=b"\x01", **defaults))

    assert packets == []
    assert parser.stats.reasons[reason] == 1


def test_counts_lost_packets_from_the_sequence_gap():
    parser = PacketParser()
    stream = (build(TYPE_RAW, 0, 0, b"\x01")
              + build(TYPE_RAW, 4, 100, b"\x02"))

    parser.feed(stream)

    assert parser.stats.lost == 3
    assert parser.stats.duplicated == 0


def test_counts_a_repeated_sequence_as_duplicated():
    parser = PacketParser()
    stream = (build(TYPE_RAW, 9, 0, b"\x01")
              + build(TYPE_RAW, 9, 50, b"\x01"))

    parser.feed(stream)

    assert parser.stats.duplicated == 1
    assert parser.stats.lost == 0


def test_sequence_wrap_is_not_reported_as_loss():
    parser = PacketParser()
    stream = (build(TYPE_RAW, 65535, 0, b"\x01")
              + build(TYPE_RAW, 0, 100, b"\x02"))

    parser.feed(stream)

    assert parser.stats.lost == 0


def test_sequences_are_tracked_per_type():
    parser = PacketParser()
    # Interleaving two types, each perfectly in order, must produce no loss.
    stream = b"".join(
        build(TYPE_RAW, index, index * 10, b"\x01")
        + build(TYPE_INTENT, index, index * 10, b"\x02")
        for index in range(5)
    )

    parser.feed(stream)

    assert parser.stats.accepted == 10
    assert parser.stats.lost == 0
    assert parser.stats.duplicated == 0


def test_a_backwards_timestamp_is_counted_but_a_wrap_is_not():
    backwards = PacketParser()
    backwards.feed(build(TYPE_RAW, 0, 5000, b"\x01")
                   + build(TYPE_RAW, 1, 4000, b"\x02"))
    assert backwards.stats.time_reversed == 1

    wrapped = PacketParser()
    wrapped.feed(build(TYPE_RAW, 0, 0xFFFFFF00, b"\x01")
                 + build(TYPE_RAW, 1, 0x00000100, b"\x02"))
    assert wrapped.stats.time_reversed == 0


def test_payload_decoders():
    info = decode_info(struct.pack("<HHBBBB", 0x0102, 2000, 3, 12, 32, 0))
    assert (info.sample_rate_hz, info.channel_count, info.adc_bits) == (2000, 3, 12)

    intent = decode_intent(struct.pack("<BBBbHH", 2, 200, 150, -1, 40000, 0))
    assert intent.command_name == "CONFIRM"
    assert (intent.direction, intent.activation) == (-1, 40000)

    block = decode_raw(b"\x05\x00" + struct.pack("<6H", 0, 1, 2048, 4095, 1234, 777), 3)
    assert block.frames == [(0, 1, 2048), (4095, 1234, 777)]
    assert block.wear_mask == 0x05
    assert block.channel_attached(0) and not block.channel_attached(1)
    assert not block.all_attached


@pytest.mark.parametrize(
    "decoder, payload", [(decode_info, b"\x00" * 7), (decode_intent, b"\x00" * 9)]
)
def test_payload_decoders_reject_wrong_sizes(decoder, payload):
    with pytest.raises(ValueError):
        decoder(payload)


def test_raw_decoder_rejects_a_partial_frame():
    with pytest.raises(ValueError):
        decode_raw(b"\x00" * 12, 3)


def test_raw_decoder_rejects_a_payload_too_short_for_its_header():
    with pytest.raises(ValueError):
        decode_raw(b"\x07", 3)


def test_activation_state_decoder():
    state = decode_activation_state(
        struct.pack("<BBBBiHHii", 1, 3, 4, 1, 110, 0x0102, 0, 231, 303)
    )

    assert state.from_host
    assert (state.factor, state.baseline_shift) == (3, 4)
    assert state.last_result == SET_RESULT_ACCEPTED
    assert state.threshold_floor == 110
    assert state.applied_sequence == 0x0102
    # Reported, not merely accepted: a host confirms a calibration by
    # watching this reflect what it sent, so a field it cannot read back is
    # a field it cannot verify was applied.
    assert (state.reference_left, state.reference_right) == (231, 303)
    with pytest.raises(ValueError):
        decode_activation_state(b"\x00" * 19)


def test_a_board_with_no_measured_reference_reports_zero():
    state = decode_activation_state(
        struct.pack("<BBBBiHHii", 1, 3, 4, 1, 110, 0x0102, 0, 0, 0)
    )

    assert (state.reference_left, state.reference_right) == (0, 0)


def test_own_encoder_round_trips_through_own_parser():
    parser = PacketParser()
    wire = encode_set_activation(5, mode=SET_MODE_APPLY, factor=3,
                                 baseline_shift=4, threshold_floor=110)

    packets = parser.feed(wire)

    assert len(packets) == 1
    assert packets[0].type == TYPE_SET_ACTIVATION
    assert packets[0].sequence == 5
    # timestamp is 0 by contract: the host does not own the device clock.
    assert packets[0].timestamp_us == 0
    mode, factor, shift, _reserved, floor, left, right = struct.unpack(
        "<BBBBiii", packets[0].payload
    )
    assert (mode, factor, shift, floor) == (1, 3, 4, 110)
    # Omitted means "no measurement, use the fallback", not "zero span".
    assert (left, right) == (0, 0)


def test_measured_references_travel_on_the_wire():
    parser = PacketParser()
    wire = encode_set_activation(5, mode=SET_MODE_APPLY, factor=3,
                                 baseline_shift=4, threshold_floor=55,
                                 reference_left=231, reference_right=303)

    payload = parser.feed(wire)[0].payload

    assert struct.unpack("<BBBBiii", payload)[5:] == (231, 303)


def test_a_reference_inside_the_threshold_is_refused_at_the_sender():
    # The span activation maps is the one above the threshold, so a
    # reference at or below it is an empty span. Sending it and letting the
    # board silently produce no deflection is the worse failure.
    with pytest.raises(ValueError, match="reference_left"):
        encode_set_activation(5, mode=SET_MODE_APPLY, factor=3,
                              baseline_shift=4, threshold_floor=55,
                              reference_left=55, reference_right=303)
    with pytest.raises(ValueError, match="reference_right"):
        encode_set_activation(5, mode=SET_MODE_APPLY, factor=3,
                              baseline_shift=4, threshold_floor=55,
                              reference_left=231, reference_right=20)


def test_set_encoder_rejects_what_the_firmware_would():
    # A host bug should fail loudly at the sender, not travel to the board
    # and come back as a silent last_result=2.
    good = dict(mode=SET_MODE_APPLY, factor=3, baseline_shift=4,
                threshold_floor=110)
    for bad in (
        dict(good, mode=7),
        dict(good, factor=0),
        dict(good, factor=256),
        dict(good, baseline_shift=0),
        dict(good, baseline_shift=9),
        dict(good, threshold_floor=0),
        dict(good, threshold_floor=3 * 32767),
    ):
        with pytest.raises(ValueError):
            encode_set_activation(0, **bad)


def test_generic_encoder_rejects_oversized_payloads():
    with pytest.raises(ValueError):
        encode_packet(TYPE_RAW, 0, 0, b"\x00" * (MAX_PAYLOAD + 1))


def test_decodes_the_fixture_produced_by_the_c_encoder():
    """Cross-implementation check: C wrote these bytes, Python reads them.

    The two sides were written from PROTOCOL.md without reading each other,
    so agreement here is evidence the spec is unambiguous.
    """
    if not FIXTURE.exists():
        pytest.skip(f"run `make check` in firmware/test to generate {FIXTURE.name}")

    parser = PacketParser()
    fixture_bytes = FIXTURE.read_bytes()
    packets = parser.feed(fixture_bytes)

    assert [packet.type for packet in packets] == [
        TYPE_INFO, TYPE_RAW, TYPE_RAW, TYPE_RAW, TYPE_INTENT, TYPE_INTENT,
        TYPE_ACTIVATION_STATE, TYPE_SET_ACTIVATION,
    ]

    info = decode_info(packets[0].payload)
    assert info == type(info)(0x0102, 2000, 3, 12, 32)
    assert packets[0].timestamp_us == 1000

    first = decode_raw(packets[1].payload, info.channel_count)
    assert first.frames == [(0, 1, 2048), (4095, 1234, 777)]
    assert first.all_attached
    # The fixture drops channel 1 on the third RAW packet.
    assert decode_raw(packets[3].payload, info.channel_count).wear_mask == 0x05

    intent = decode_intent(packets[4].payload)
    assert intent.command_name == "CONFIRM"
    assert (intent.confidence, intent.signal_quality) == (200, 150)
    assert (intent.direction, intent.activation) == (-1, 40000)

    state = decode_activation_state(packets[6].payload)
    assert state.from_host
    assert (state.factor, state.baseline_shift) == (3, 4)
    assert state.threshold_floor == 110
    assert state.applied_sequence == 0x0102
    assert (state.reference_left, state.reference_right) == (231, 303)

    # The C-encoded SET_ACTIVATION must be byte-identical to this module's
    # own encoding of the same request: the Python sender and the firmware
    # receiver were written independently from PROTOCOL.md, and this is
    # where a disagreement would surface.
    python_wire = encode_set_activation(5, mode=SET_MODE_APPLY, factor=3,
                                        baseline_shift=4,
                                        threshold_floor=110,
                                        reference_left=231,
                                        reference_right=303)
    assert python_wire in fixture_bytes
    assert packets[7].sequence == 5

    # The fixture deliberately embeds leading junk, a RAW sequence jump of
    # 1 -> 3, and a repeated INTENT sequence.
    assert parser.stats.accepted == 8
    assert parser.stats.malformed == 0
    assert parser.stats.lost == 1
    assert parser.stats.duplicated == 1
    assert parser.stats.discarded_bytes == 5
    assert HEADER_SIZE == 12
