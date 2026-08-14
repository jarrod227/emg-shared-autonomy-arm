#!/usr/bin/env python3
"""Pure timing and labelling core for guided sEMG collection.

The GUI and serial worker live in ``emg_guided_capture.py``.  This module has
no GUI, serial, NumPy, or Matplotlib dependency, so the rules that decide
which frames become training data can be tested deterministically.

Only the ACTIVE phase is labelled.  Preparation, movement transition, the
short post-action verification phase, recovery, and pause spans remain in the
raw byte log but are excluded from training.  A capture that contains lost
packets, parser errors, duplicated packets, reversed timestamps, insufficient
frames, or an electrode-contact loss is rejected and the same trial is
repeated.
"""

from dataclasses import dataclass
from enum import Enum
import math
import random


GESTURE_ACTIONS = {
    "REST": "RELAX",
    "NEXT_TARGET": "WRIST UP",
    "CONFIRM": "MAKE A FIST",
    "ABORT": "WRIST DOWN",
}
GESTURE_LABELS = tuple(GESTURE_ACTIONS)


class Phase(str, Enum):
    """The host-side collection state; only ACTIVE is training data."""

    IDLE = "idle"
    PREPARE = "prepare"
    TRANSITION = "transition"
    ACTIVE = "active"
    VERIFY = "verify"
    RECOVERY = "recovery"
    PAUSED = "paused"
    COMPLETE = "complete"
    STOPPED = "stopped"


RUNNING_PHASES = {
    Phase.PREPARE,
    Phase.TRANSITION,
    Phase.ACTIVE,
    Phase.VERIFY,
    Phase.RECOVERY,
}


@dataclass(frozen=True)
class TrialSpec:
    """One requested gesture in the balanced, randomized session plan."""

    index: int
    label: str
    action: str
    repetition: int

    def to_dict(self):
        return {
            "index": self.index,
            "label": self.label,
            "action": self.action,
            "repetition": self.repetition,
        }


def build_trial_plan(repetitions, seed):
    """Build balanced randomized blocks containing every label once.

    Randomizing within each complete block avoids collecting every gesture in
    one long run while still guaranteeing balance at every repetition count.
    """
    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    generator = random.Random(seed)
    trials = []
    for repetition in range(1, repetitions + 1):
        block = list(GESTURE_LABELS)
        generator.shuffle(block)
        for label in block:
            trials.append(
                TrialSpec(
                    index=len(trials),
                    label=label,
                    action=GESTURE_ACTIONS[label],
                    repetition=repetition,
                )
            )
    return tuple(trials)


@dataclass(frozen=True)
class StreamPosition:
    """Cumulative position in the raw log at one host-side boundary.

    ``frame_index`` is the primary label key.  The 32-bit device timestamp is
    retained as supporting metadata, but it wraps and therefore must not be
    the only way a later trainer selects frames.
    """

    frame_index: int
    timestamp_us: int | None = None
    detached_by_channel: tuple[int, ...] = ()
    lost_packets: int = 0
    malformed_packets: int = 0
    duplicated_packets: int = 0
    time_reversed_packets: int = 0

    def __post_init__(self):
        counters = (
            self.frame_index,
            self.lost_packets,
            self.malformed_packets,
            self.duplicated_packets,
            self.time_reversed_packets,
            *self.detached_by_channel,
        )
        if any(value < 0 for value in counters):
            raise ValueError("stream counters must be non-negative")
        if self.timestamp_us is not None and not 0 <= self.timestamp_us < (1 << 32):
            raise ValueError("timestamp_us must fit uint32")

    def quality_reasons_since(self, earlier):
        """Return every reason the half-open frame span is not trainable."""
        if not isinstance(earlier, StreamPosition):
            raise TypeError("earlier must be a StreamPosition")

        reasons = []
        if self.frame_index <= earlier.frame_index:
            reasons.append("no_frames")
        if self.frame_index < earlier.frame_index:
            reasons.append("counter_reset")

        width = max(
            len(self.detached_by_channel),
            len(earlier.detached_by_channel),
        )
        for channel in range(width):
            current = (
                self.detached_by_channel[channel]
                if channel < len(self.detached_by_channel)
                else 0
            )
            previous = (
                earlier.detached_by_channel[channel]
                if channel < len(earlier.detached_by_channel)
                else 0
            )
            if current < previous:
                reasons.append("counter_reset")
                break
            if current > previous:
                reasons.append("electrode_contact")
                break

        counter_names = (
            ("packet_loss", self.lost_packets, earlier.lost_packets),
            ("malformed_packet", self.malformed_packets, earlier.malformed_packets),
            ("duplicated_packet", self.duplicated_packets, earlier.duplicated_packets),
            (
                "time_reversed_packet",
                self.time_reversed_packets,
                earlier.time_reversed_packets,
            ),
        )
        for reason, current, previous in counter_names:
            if current < previous:
                if "counter_reset" not in reasons:
                    reasons.append("counter_reset")
            elif current > previous:
                reasons.append(reason)
        return tuple(dict.fromkeys(reasons))

    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "timestamp_us": self.timestamp_us,
            "detached_by_channel": list(self.detached_by_channel),
            "lost_packets": self.lost_packets,
            "malformed_packets": self.malformed_packets,
            "duplicated_packets": self.duplicated_packets,
            "time_reversed_packets": self.time_reversed_packets,
        }


@dataclass(frozen=True)
class LabelSegment:
    """One completed or rejected attempt and its exact raw-frame bounds."""

    trial: TrialSpec
    attempt: int
    start: StreamPosition
    end: StreamPosition
    start_host_sec: float
    end_host_sec: float
    include: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "trial": self.trial.to_dict(),
            "attempt": self.attempt,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "start_host_sec": round(self.start_host_sec, 6),
            "end_host_sec": round(self.end_host_sec, 6),
            "include": self.include,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PauseSpan:
    """An unlabelled span while the stream stayed open and draining."""

    reason: str
    start: StreamPosition
    end: StreamPosition
    start_host_sec: float
    end_host_sec: float

    def to_dict(self):
        return {
            "reason": self.reason,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "start_host_sec": round(self.start_host_sec, 6),
            "end_host_sec": round(self.end_host_sec, 6),
            "include": False,
        }


def _positive_duration(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite_time(now):
    now = float(now)
    if not math.isfinite(now):
        raise ValueError("time must be finite")
    return now


class GuidedSession:
    """Clock-driven collection protocol with fail-closed label quality."""

    def __init__(
        self,
        plan,
        *,
        prepare_seconds=2.0,
        transition_seconds=0.5,
        active_seconds=3.0,
        verification_seconds=0.1,
        recovery_seconds=1.5,
        sample_rate_hz=2000,
        min_active_fraction=0.9,
    ):
        self.plan = tuple(plan)
        if not self.plan:
            raise ValueError("plan must contain at least one trial")
        if any(not isinstance(trial, TrialSpec) for trial in self.plan):
            raise TypeError("plan entries must be TrialSpec values")

        self.prepare_seconds = _positive_duration(
            "prepare_seconds", prepare_seconds
        )
        self.transition_seconds = _positive_duration(
            "transition_seconds", transition_seconds
        )
        self.active_seconds = _positive_duration("active_seconds", active_seconds)
        self.verification_seconds = _positive_duration(
            "verification_seconds", verification_seconds
        )
        self.recovery_seconds = _positive_duration(
            "recovery_seconds", recovery_seconds
        )
        self.sample_rate_hz = self._validated_sample_rate(sample_rate_hz)
        self.min_active_fraction = float(min_active_fraction)
        if (
            not math.isfinite(self.min_active_fraction)
            or not 0.0 < self.min_active_fraction <= 1.0
        ):
            raise ValueError("min_active_fraction must be in (0, 1]")

        self.phase = Phase.IDLE
        self.trial_index = 0
        self.started_at = None
        self.phase_started_at = None
        self.deadline = None
        self.last_result = ""
        self.segments = []
        self.pause_spans = []
        self._attempt_counts = [0] * len(self.plan)
        self._capture_start = None
        self._capture_started_at = None
        self._capture_end = None
        self._capture_ended_at = None
        self._pause_start = None
        self._pause_started_at = None
        self._pause_reason = ""

    @property
    def current_trial(self):
        if 0 <= self.trial_index < len(self.plan):
            return self.plan[self.trial_index]
        return None

    @property
    def completed_trials(self):
        return sum(segment.include for segment in self.segments)

    @property
    def total_trials(self):
        return len(self.plan)

    @property
    def current_attempt(self):
        if self.current_trial is None:
            return 0
        return self._attempt_counts[self.trial_index]

    def _elapsed(self, now):
        return 0.0 if self.started_at is None else now - self.started_at

    @staticmethod
    def _validated_sample_rate(value):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        return value

    def set_sample_rate_hz(self, value):
        """Adopt the rate reported by INFO before any labelled timing starts."""
        if self.phase is not Phase.IDLE:
            raise RuntimeError("sample rate can only change while idle")
        self.sample_rate_hz = self._validated_sample_rate(value)

    def _set_phase(self, phase, now, duration=None):
        self.phase = phase
        self.phase_started_at = now
        self.deadline = None if duration is None else now + duration

    def _begin_attempt(self, now):
        self._attempt_counts[self.trial_index] += 1
        self._capture_start = None
        self._capture_started_at = None
        self._capture_end = None
        self._capture_ended_at = None
        self._set_phase(Phase.PREPARE, now, self.prepare_seconds)

    def start(self, now):
        """Start the first preparation countdown after stream preflight."""
        now = _finite_time(now)
        if self.phase is not Phase.IDLE:
            raise RuntimeError("session can only start from idle")
        self.started_at = now
        self._begin_attempt(now)

    def remaining_seconds(self, now):
        now = _finite_time(now)
        if self.deadline is None:
            return 0.0
        return max(0.0, self.deadline - now)

    def advance(self, now, position):
        """Advance at most one timed phase and return whether it changed."""
        now = _finite_time(now)
        if not isinstance(position, StreamPosition):
            raise TypeError("position must be a StreamPosition")
        if self.phase not in RUNNING_PHASES or now < self.deadline:
            return False

        if self.phase is Phase.PREPARE:
            self._set_phase(
                Phase.TRANSITION,
                now,
                self.transition_seconds,
            )
        elif self.phase is Phase.TRANSITION:
            self._capture_start = position
            self._capture_started_at = now
            self._set_phase(Phase.ACTIVE, now, self.active_seconds)
        elif self.phase is Phase.ACTIVE:
            self._capture_end = position
            self._capture_ended_at = now
            self._set_phase(Phase.VERIFY, now, self.verification_seconds)
        elif self.phase is Phase.VERIFY:
            # A later RAW packet is necessary because only that packet can
            # expose a sequence gap at the tail of the labelled interval.
            if (
                self._capture_end is None
                or position.frame_index <= self._capture_end.frame_index
            ):
                return False
            self._finish_capture(now, position)
        elif self.phase is Phase.RECOVERY:
            self._begin_attempt(now)
        return True

    def _finish_capture(self, now, verification_position, forced_reasons=()):
        reasons = tuple(forced_reasons)
        if self._capture_start is None or self._capture_started_at is None:
            reasons += ("missing_capture_boundary",)
            start = verification_position
            started_at = now
        else:
            start = self._capture_start
            started_at = self._capture_started_at
        if self._capture_end is None or self._capture_ended_at is None:
            end = verification_position
            ended_at = now
        else:
            end = self._capture_end
            ended_at = self._capture_ended_at

        # Contact and frame-count quality apply only to the labelled ACTIVE
        # span. Parser counters are observed through VERIFY so a missing final
        # packet cannot pass before the following packet reveals the gap.
        reasons += end.quality_reasons_since(start)
        verification_reasons = verification_position.quality_reasons_since(start)
        delayed_parser_reasons = {
            "packet_loss",
            "malformed_packet",
            "duplicated_packet",
            "time_reversed_packet",
            "counter_reset",
        }
        reasons += tuple(
            reason
            for reason in verification_reasons
            if reason in delayed_parser_reasons
        )
        minimum_frames = math.ceil(
            self.active_seconds * self.sample_rate_hz * self.min_active_fraction
        )
        if end.frame_index - start.frame_index < minimum_frames:
            reasons += ("insufficient_frames",)
        reasons = tuple(dict.fromkeys(reasons))
        include = not reasons
        trial = self.current_trial
        if trial is None:
            raise RuntimeError("capture ended without a current trial")

        segment = LabelSegment(
            trial=trial,
            attempt=self.current_attempt,
            start=start,
            end=end,
            start_host_sec=self._elapsed(started_at),
            end_host_sec=self._elapsed(ended_at),
            include=include,
            reasons=reasons,
        )
        self.segments.append(segment)
        self._capture_start = None
        self._capture_started_at = None
        self._capture_end = None
        self._capture_ended_at = None

        if include:
            self.last_result = f"accepted {trial.label}"
            self.trial_index += 1
        else:
            self.last_result = (
                f"repeat {trial.label}: " + ", ".join(reasons)
            )

        if include and self.trial_index >= len(self.plan):
            self._set_phase(Phase.COMPLETE, now)
        else:
            self._set_phase(Phase.RECOVERY, now, self.recovery_seconds)

    def pause(self, now, position, reason="manual_pause"):
        """Pause timing while continuing to drain and save the byte stream."""
        now = _finite_time(now)
        if not isinstance(position, StreamPosition):
            raise TypeError("position must be a StreamPosition")
        if self.phase not in RUNNING_PHASES:
            raise RuntimeError("only a running session can be paused")
        if not reason:
            raise ValueError("pause reason must not be empty")

        if self.phase is Phase.ACTIVE:
            self._capture_end = position
            self._capture_ended_at = now
            self._finish_capture(now, position, forced_reasons=(reason,))
            # `_finish_capture` uses RECOVERY; pausing supersedes that timer.
        elif self.phase is Phase.VERIFY:
            self._finish_capture(now, position, forced_reasons=(reason,))
            # `_finish_capture` uses RECOVERY; pausing supersedes that timer.
        self._pause_start = position
        self._pause_started_at = now
        self._pause_reason = reason
        self._set_phase(Phase.PAUSED, now)
        self.last_result = f"paused: {reason}"

    def resume(self, now, position):
        """Close the pause span and restart a full preparation countdown."""
        now = _finite_time(now)
        if not isinstance(position, StreamPosition):
            raise TypeError("position must be a StreamPosition")
        if self.phase is not Phase.PAUSED:
            raise RuntimeError("session is not paused")
        self._close_pause(now, position)
        self._begin_attempt(now)

    def _close_pause(self, now, position):
        if self._pause_start is None or self._pause_started_at is None:
            return
        self.pause_spans.append(
            PauseSpan(
                reason=self._pause_reason,
                start=self._pause_start,
                end=position,
                start_host_sec=self._elapsed(self._pause_started_at),
                end_host_sec=self._elapsed(now),
            )
        )
        self._pause_start = None
        self._pause_started_at = None
        self._pause_reason = ""

    def stop(self, now, position, reason="manual_stop"):
        """Stop early, preserving completed trials and rejecting an active one."""
        now = _finite_time(now)
        if not isinstance(position, StreamPosition):
            raise TypeError("position must be a StreamPosition")
        if self.phase in (Phase.COMPLETE, Phase.STOPPED):
            return
        if self.phase is Phase.ACTIVE:
            self._capture_end = position
            self._capture_ended_at = now
            self._finish_capture(now, position, forced_reasons=(reason,))
        elif self.phase is Phase.VERIFY:
            self._finish_capture(now, position, forced_reasons=(reason,))
        if self.phase is Phase.PAUSED:
            self._close_pause(now, position)
        self._set_phase(Phase.STOPPED, now)
        self.last_result = reason

    def to_manifest(self, *, seed, status):
        """Return the serializable experiment contract for the sidecar."""
        return {
            "schema_version": 1,
            "status": status,
            "gesture_actions": dict(GESTURE_ACTIONS),
            "seed": seed,
            "timing_seconds": {
                "prepare": self.prepare_seconds,
                "transition_unlabelled": self.transition_seconds,
                "active_labelled": self.active_seconds,
                "verification_unlabelled": self.verification_seconds,
                "recovery_unlabelled": self.recovery_seconds,
            },
            "sample_rate_hz": self.sample_rate_hz,
            "minimum_active_fraction": self.min_active_fraction,
            "schedule": [trial.to_dict() for trial in self.plan],
            "completed_trials": self.completed_trials,
            "total_trials": self.total_trials,
            "segments": [segment.to_dict() for segment in self.segments],
            "pause_spans": [span.to_dict() for span in self.pause_spans],
            "final_phase": self.phase.value,
            "last_result": self.last_result,
        }
