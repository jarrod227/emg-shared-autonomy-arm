"""Host side of the Cheez sEMG serial protocol v1.

Decodes every device-to-host packet and encodes the host-to-device
SET_ACTIVATION request. Written independently from `../src/emg_packet.c`
against `../PROTOCOL.md`; `test_emg_protocol.py` checks the two agree on
bytes the C encoder produced — in both directions, since the C fixture also
contains a SET_ACTIVATION this encoder must reproduce byte for byte.

The parser is fail-closed. A packet reaches the caller only after its magic,
version, type, length bound, and CRC all check out, and every rejection is
counted rather than silently swallowed, so a degrading link shows up as
numbers instead of as missing data.
"""

from dataclasses import dataclass, field
import struct

MAGIC = b"\xa5\x5a"
PROTOCOL_VERSION = 1
HEADER_SIZE = 12
CRC_SIZE = 2
MAX_PAYLOAD = 512
# RAW payloads open with a wear bitmask and one reserved byte.
RAW_HEADER_SIZE = 2
MAX_PACKET = HEADER_SIZE + MAX_PAYLOAD + CRC_SIZE

TYPE_INFO = 0x00
TYPE_RAW = 0x01
TYPE_INTENT = 0x02
TYPE_ACTIVATION_STATE = 0x03
# 0x80-0xFF is the host-to-device range.
TYPE_SET_ACTIVATION = 0x80
KNOWN_TYPES = (TYPE_INFO, TYPE_RAW, TYPE_INTENT, TYPE_ACTIVATION_STATE,
               TYPE_SET_ACTIVATION)

ACTIVATION_SOURCE_DEFAULTS = 0
ACTIVATION_SOURCE_HOST = 1
SET_RESULT_NONE = 0
SET_RESULT_ACCEPTED = 1
SET_RESULT_REJECTED = 2
SET_MODE_DEFAULTS = 0
SET_MODE_APPLY = 1

COMMAND_NAMES = {0: "REST", 1: "NEXT_TARGET", 2: "CONFIRM", 3: "ABORT"}

_SEQUENCE_MODULO = 1 << 16
_TIMESTAMP_MODULO = 1 << 32


def crc16(data):
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no XOR."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class Packet:
    """One packet that passed every validity check."""

    type: int
    sequence: int
    timestamp_us: int
    payload: bytes


@dataclass(frozen=True)
class Info:
    firmware_version: int
    sample_rate_hz: int
    channel_count: int
    adc_bits: int
    frames_per_raw_packet: int


@dataclass(frozen=True)
class Intent:
    command: int
    confidence: int
    signal_quality: int
    direction: int
    activation: int

    @property
    def command_name(self):
        return COMMAND_NAMES.get(self.command, f"UNKNOWN({self.command})")


@dataclass
class ParserStats:
    """Counts of every failure class Objective 3.5 requires detecting."""

    accepted: int = 0
    malformed: int = 0
    lost: int = 0
    duplicated: int = 0
    time_reversed: int = 0
    discarded_bytes: int = 0
    reasons: dict = field(default_factory=dict)

    def _note(self, reason):
        self.malformed += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def decode_info(payload):
    if len(payload) != 8:
        raise ValueError(f"INFO payload must be 8 bytes, got {len(payload)}")
    firmware, rate, channels, bits, frames, _ = struct.unpack("<HHBBBB", payload)
    return Info(firmware, rate, channels, bits, frames)


def decode_intent(payload):
    if len(payload) != 8:
        raise ValueError(f"INTENT payload must be 8 bytes, got {len(payload)}")
    command, confidence, quality, direction, activation, _ = struct.unpack(
        "<BBBbHH", payload
    )
    return Intent(command, confidence, quality, direction, activation)


@dataclass(frozen=True)
class ActivationState:
    """What the firmware is judging with right now, and how it got there.

    A state report rather than an ACK: the host sends SET_ACTIVATION and
    watches this until it reflects the request, resending if it does not.
    """

    source: int          # ACTIVATION_SOURCE_*
    factor: int
    baseline_shift: int
    last_result: int     # SET_RESULT_*
    threshold_floor: int
    applied_sequence: int

    @property
    def from_host(self):
        return self.source == ACTIVATION_SOURCE_HOST


def decode_activation_state(payload):
    if len(payload) != 12:
        raise ValueError(
            f"ACTIVATION_STATE payload must be 12 bytes, got {len(payload)}"
        )
    source, factor, shift, result, floor, applied, _ = struct.unpack(
        "<BBBBiHH", payload
    )
    return ActivationState(source, factor, shift, result, floor, applied)


def encode_packet(packet_type, sequence, timestamp_us, payload=b""):
    """Frame one packet exactly as the C encoder does."""
    if not 0 <= len(payload) <= MAX_PAYLOAD:
        raise ValueError(f"payload must be 0..{MAX_PAYLOAD} bytes")
    header = MAGIC + bytes([PROTOCOL_VERSION, packet_type]) + struct.pack(
        "<HHI", len(payload), sequence % _SEQUENCE_MODULO,
        timestamp_us % _TIMESTAMP_MODULO
    )
    body = header[2:] + payload
    return header + payload + struct.pack("<H", crc16(body))


def encode_set_activation(sequence, *, mode, factor, baseline_shift,
                          threshold_floor):
    """Build the host-to-device calibration packet.

    Range checking here mirrors what the firmware will accept, so a host
    bug fails loudly at the sender instead of as a silent last_result=2.
    timestamp_us is 0 by contract: the host does not own the device clock.
    """
    if mode not in (SET_MODE_DEFAULTS, SET_MODE_APPLY):
        raise ValueError(f"unknown SET_ACTIVATION mode {mode}")
    if not 1 <= factor <= 255:
        raise ValueError("factor must be in 1..255")
    if not 1 <= baseline_shift <= 8:
        raise ValueError("baseline_shift must be in 1..8")
    if not 1 <= threshold_floor < 3 * 32767:
        raise ValueError("threshold_floor must be in 1..3*32767-1")
    payload = struct.pack("<BBBBi", mode, factor, baseline_shift, 0,
                          threshold_floor)
    return encode_packet(TYPE_SET_ACTIVATION, sequence, 0, payload)


@dataclass(frozen=True)
class RawBlock:
    """One RAW packet's samples plus the wear state they were taken under."""

    wear_mask: int
    frames: list

    def channel_attached(self, channel):
        return bool(self.wear_mask & (1 << channel))

    @property
    def all_attached(self):
        return all(self.channel_attached(index)
                   for index in range(len(self.frames[0]) if self.frames else 0))


def decode_raw(payload, channel_count):
    """Split a RAW payload into its wear mask and per-channel frames."""
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    if len(payload) < RAW_HEADER_SIZE:
        raise ValueError(f"RAW payload must be at least {RAW_HEADER_SIZE} bytes")
    wear_mask = payload[0]
    body = payload[RAW_HEADER_SIZE:]
    if len(body) % (2 * channel_count):
        raise ValueError(
            f"RAW sample area {len(body)} B is not whole frames of "
            f"{channel_count} channels"
        )
    values = struct.unpack(f"<{len(body) // 2}H", body)
    frames = [
        tuple(values[start:start + channel_count])
        for start in range(0, len(values), channel_count)
    ]
    return RawBlock(wear_mask=wear_mask, frames=frames)


class PacketParser:
    """Incremental byte-stream parser with resynchronization."""

    def __init__(self):
        self._buffer = bytearray()
        self._last_sequence = {}
        self._last_timestamp = {}
        self.stats = ParserStats()

    def feed(self, data):
        """Add bytes and return every packet that became complete and valid."""
        self._buffer.extend(data)
        packets = []
        while True:
            packet = self._take_one(packets)
            if packet is None:
                break
        return packets

    def _resync(self, reason):
        """Drop one byte and rescan.

        Never skip `length` bytes on a rejection: a corrupted length is
        exactly the case being recovered from, and trusting it turns one bad
        packet into an unbounded desync.
        """
        self.stats._note(reason)
        del self._buffer[0]
        self.stats.discarded_bytes += 1

    def _take_one(self, packets):
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                # Keep one byte in case the magic straddles this chunk and
                # the next; everything before it can never start a packet.
                keep = len(self._buffer) - (len(MAGIC) - 1)
                if keep > 0:
                    del self._buffer[:keep]
                    self.stats.discarded_bytes += keep
                return None
            if start > 0:
                del self._buffer[:start]
                self.stats.discarded_bytes += start
            if len(self._buffer) < HEADER_SIZE:
                return None

            version = self._buffer[2]
            packet_type = self._buffer[3]
            length = int.from_bytes(self._buffer[4:6], "little")
            if version != PROTOCOL_VERSION:
                self._resync("version")
                continue
            if packet_type not in KNOWN_TYPES:
                self._resync("type")
                continue
            if length > MAX_PAYLOAD:
                self._resync("length")
                continue

            total = HEADER_SIZE + length + CRC_SIZE
            if len(self._buffer) < total:
                return None

            body = bytes(self._buffer[2:HEADER_SIZE + length])
            expected = int.from_bytes(
                self._buffer[HEADER_SIZE + length:total], "little"
            )
            if crc16(body) != expected:
                self._resync("crc")
                continue

            packet = Packet(
                type=packet_type,
                sequence=int.from_bytes(self._buffer[6:8], "little"),
                timestamp_us=int.from_bytes(self._buffer[8:12], "little"),
                payload=bytes(self._buffer[HEADER_SIZE:HEADER_SIZE + length]),
            )
            del self._buffer[:total]
            self._account(packet)
            packets.append(packet)
            return packet

    def _account(self, packet):
        """Update per-type sequence and timestamp continuity counters."""
        self.stats.accepted += 1
        previous = self._last_sequence.get(packet.type)
        if previous is not None:
            gap = (packet.sequence - previous) % _SEQUENCE_MODULO
            if gap == 0:
                self.stats.duplicated += 1
            elif gap > 1:
                self.stats.lost += gap - 1
        self._last_sequence[packet.type] = packet.sequence

        # A legal 32-bit microsecond wrap happens every ~71.6 minutes and
        # looks like a huge forward jump, so only small backward steps are
        # treated as time going the wrong way.
        last_time = self._last_timestamp.get(packet.type)
        if last_time is not None:
            delta = (packet.timestamp_us - last_time) % _TIMESTAMP_MODULO
            if delta > _TIMESTAMP_MODULO // 2:
                self.stats.time_reversed += 1
        self._last_timestamp[packet.type] = packet.timestamp_us
