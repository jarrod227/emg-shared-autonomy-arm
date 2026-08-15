"""Incremental adapter from protocol-v1 bytes to safe device intent records."""

from collections import Counter

from .protocol_loader import (
    COMMAND_NAMES,
    TYPE_ACTIVATION_STATE,
    TYPE_INTENT,
    PacketParser,
    decode_activation_state,
    decode_intent,
)

from .confirmation import DeviceIntent


_SEQUENCE_MODULO = 1 << 16
_TIMESTAMP_MODULO = 1 << 32


class IntentStreamDecoder:
    """Decode INTENT packets and reject duplicate or reversed source state."""

    def __init__(self):
        self.parser = PacketParser()
        # Counted per command so a silent bridge can be told apart from a
        # silent MCU. Without this the published count alone cannot say
        # whether the gate dropped events or none ever arrived.
        self.command_counts = Counter()
        self.last_command = None
        self.last_signal_quality = None
        self.payload_errors = 0
        self.unknown_commands = 0
        # ACTIVATION_STATE is a side channel, not part of the intent stream:
        # the startup handshake and diagnostics read these directly rather
        # than draining feed()'s return list, so a caller that only wants
        # intents is unaffected by their arrival.
        self.activation_state_errors = 0
        self.activation_state_count = 0
        self.last_activation_state = None
        self.duplicate_intents = 0
        self.reversed_intents = 0
        self.intent_sequence_gaps = 0
        self._last_sequence = None
        self._last_wire_timestamp = None
        self._unwrapped_timestamp = None

    def _sequence_delta(self, sequence):
        if self._last_sequence is None:
            return None
        return (sequence - self._last_sequence) % _SEQUENCE_MODULO

    def _unwrap_timestamp(self, timestamp_us):
        if self._last_wire_timestamp is None:
            return int(timestamp_us)
        delta = (timestamp_us - self._last_wire_timestamp) % _TIMESTAMP_MODULO
        if delta > _TIMESTAMP_MODULO // 2:
            return None
        return self._unwrapped_timestamp + delta

    def feed(self, chunk):
        """Return every new, forward INTENT contained in ``chunk``."""
        decoded = []
        for packet in self.parser.feed(chunk):
            if packet.type == TYPE_ACTIVATION_STATE:
                try:
                    self.last_activation_state = decode_activation_state(
                        packet.payload
                    )
                    self.activation_state_count += 1
                except ValueError:
                    self.activation_state_errors += 1
                continue
            if packet.type != TYPE_INTENT:
                continue
            try:
                intent = decode_intent(packet.payload)
            except ValueError:
                self.payload_errors += 1
                continue
            if intent.command not in COMMAND_NAMES:
                self.unknown_commands += 1
                continue

            sequence_delta = self._sequence_delta(packet.sequence)
            if sequence_delta == 0:
                self.duplicate_intents += 1
                continue
            if (
                sequence_delta is not None
                and sequence_delta > _SEQUENCE_MODULO // 2
            ):
                self.reversed_intents += 1
                continue

            timestamp = self._unwrap_timestamp(packet.timestamp_us)
            if timestamp is None:
                self.reversed_intents += 1
                continue

            gap = 0 if sequence_delta is None else max(0, sequence_delta - 1)
            self.intent_sequence_gaps += gap
            self.command_counts[intent.command] += 1
            self.last_command = intent.command
            self.last_signal_quality = intent.signal_quality
            self._last_sequence = packet.sequence
            self._last_wire_timestamp = packet.timestamp_us
            self._unwrapped_timestamp = timestamp
            decoded.append(DeviceIntent(
                sequence=packet.sequence,
                timestamp_us=timestamp,
                command=intent.command,
                confidence=intent.confidence,
                signal_quality=intent.signal_quality,
                stream_discontinuity=gap > 0,
            ))
        return decoded
