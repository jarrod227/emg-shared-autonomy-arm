"""Pure safety policy between MCU candidate events and ROS intent events.

The ten-minute ordinary-activity run showed that a natural action can have the
same EMG shape and strength as a command. No scalar threshold can recover the
missing information: K=4 retained all six deliberate gestures but still let
eight ordinary-activity events through, while K=5 lost half the deliberate
gestures. The MVP therefore asks for a second occurrence of the same ordinary
command. ABORT remains single-shot because stopping must never wait for a
confirmation gesture.

ABORT and CONFIRM also share a measured onset: one validation trial left only
one 50 ms window between an onset CONFIRM run and the five-window ordinary
event threshold. A confirmed CONFIRM is therefore held for 250 ms while fresh
MCU liveness packets arrive. ABORT during that grace cancels it and publishes
immediately. NEXT_TARGET is not delayed. This host-side override avoids adding
200 ms to ABORT itself, which a longer global MCU onset hold-off would do.

The window is measured from both sides, not assumed. A live self-paced round
put consecutive deliberate events 3.80, 3.95 and 4.15 s apart, because each
one costs the MCU its onset hold-off, five agreeing windows and a REST re-arm
before the wearer's own relax time is counted; the original 3.0 s estimate sat
below that floor and paired nothing at all. The ceiling comes from the
ten-minute ordinary-activity recording, whose closest two same-command events
are 7.05 s apart. That pair is worth naming: at a wider window it is held
apart only by an ABORT landing 0.05 s later and clearing the pending half, so
anything at or above 7 s would be passing on interference rather than margin.
"""

from dataclasses import dataclass


REST = 0
NEXT_TARGET = 1
CONFIRM = 2
ABORT = 3
KNOWN_COMMANDS = (REST, NEXT_TARGET, CONFIRM, ABORT)
DOUBLE_CONFIRM_COMMANDS = (NEXT_TARGET, CONFIRM)


@dataclass(frozen=True)
class DeviceIntent:
    """One decoded MCU INTENT packet on an unwrapped device time base."""

    sequence: int
    timestamp_us: int
    command: int
    confidence: int
    signal_quality: int
    stream_discontinuity: bool = False

    def __post_init__(self):
        if self.command not in KNOWN_COMMANDS:
            raise ValueError(f"unknown command {self.command}")
        if not 0 <= self.confidence <= 255:
            raise ValueError("confidence must be in 0..255")
        if not 0 <= self.signal_quality <= 255:
            raise ValueError("signal_quality must be in 0..255")


@dataclass(frozen=True)
class ConfirmedIntent:
    """An event safe to publish on the source-independent ROS contract."""

    timestamp_us: int
    command: int
    confidence: int
    signal_quality: int


class IntentConfirmationGate:
    """Require two matching ordinary events; pass ABORT immediately."""

    def __init__(self, confirmation_window_sec=3.0,
                 abort_override_window_sec=0.25):
        window = float(confirmation_window_sec)
        override = float(abort_override_window_sec)
        if window <= 0.0:
            raise ValueError("confirmation_window_sec must be positive")
        if override < 0.0:
            raise ValueError("abort_override_window_sec must be non-negative")
        self.window_us = round(window * 1e6)
        self.abort_override_us = round(override * 1e6)
        self._pending = None
        self._deferred_confirm = None

    @property
    def pending_command(self):
        if self._pending is not None:
            return self._pending.command
        if self._deferred_confirm is not None:
            return self._deferred_confirm.command
        return None

    def invalidate(self):
        """Discard incomplete or not-yet-released ordinary evidence."""
        self._pending = None
        self._deferred_confirm = None

    def _expired(self, timestamp_us):
        return (
            self._pending is not None
            and timestamp_us - self._pending.timestamp_us > self.window_us
        )

    def _release_due(self, timestamp_us):
        if (
            self._deferred_confirm is None
            or timestamp_us - self._deferred_confirm.timestamp_us
            < self.abort_override_us
        ):
            return None
        confirmed = self._deferred_confirm
        self._deferred_confirm = None
        return confirmed

    @staticmethod
    def _confirmed(intent):
        return ConfirmedIntent(
            timestamp_us=intent.timestamp_us,
            command=intent.command,
            confidence=intent.confidence,
            signal_quality=intent.signal_quality,
        )

    def push(self, intent):
        """Consume one packet and return an event only when policy permits."""
        if not isinstance(intent, DeviceIntent):
            raise TypeError("intent must be DeviceIntent")
        if intent.stream_discontinuity:
            self.invalidate()
        if self._expired(intent.timestamp_us):
            self.invalidate()

        # ABORT is judged before any quality rule. signal_quality is a
        # diagnostic byte (255 minus saturations, or 0 when a frame was
        # missing), not evidence about the gesture, and the MCU has already
        # spent its activation threshold and five agreeing windows on this
        # decision. Dropping a stop request because a frame went missing
        # trades a certain hazard for a cosmetic one.
        if intent.command == ABORT:
            self.invalidate()
            return self._confirmed(intent)

        # Contact loss invalidates a pending half even when this packet says
        # REST. Pairing across an untrusted interval would not be fail-closed.
        if intent.signal_quality == 0:
            self.invalidate()
            return None

        # ABORT was checked first, including at the exact grace deadline.
        # A REST liveness packet releases CONFIRM after the override interval.
        # If packets stop, nothing is released: silence is not evidence that
        # no ABORT followed.
        due = self._release_due(intent.timestamp_us)
        if due is not None:
            return due

        # REST is link liveness, not a ROS event. It deliberately leaves one
        # pending pair intact because the MCU itself requires REST before it
        # can emit the second occurrence of a gesture.
        if intent.command == REST:
            return None

        # A second ordinary event during this short interval is outside the
        # measured event cadence. Fail closed instead of releasing an
        # unresolved CONFIRM or pairing across it.
        self._deferred_confirm = None

        if intent.command not in DOUBLE_CONFIRM_COMMANDS:
            raise ValueError(f"command {intent.command} has no bridge policy")

        if (
            self._pending is not None
            and self._pending.command == intent.command
            and intent.timestamp_us > self._pending.timestamp_us
        ):
            first = self._pending
            self._pending = None
            confirmed = ConfirmedIntent(
                timestamp_us=intent.timestamp_us,
                command=intent.command,
                confidence=min(first.confidence, intent.confidence),
                signal_quality=min(first.signal_quality, intent.signal_quality),
            )
            if intent.command == CONFIRM and self.abort_override_us > 0:
                self._deferred_confirm = confirmed
                return None
            return confirmed

        # A different command replaces the pending one. Publishing neither is
        # fail-closed: the user must repeat whichever action they intended.
        self._pending = intent
        return None
