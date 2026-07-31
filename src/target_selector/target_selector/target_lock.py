"""Pure selected-track lock and last-seen watchdog logic."""

from dataclasses import dataclass
import math

from target_selector.candidate_stability import (
    CandidateGateDecision,
    CandidateMeasurement,
)


def _finite_float(value, name):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted


@dataclass(frozen=True)
class TargetLockConfig:
    """Timing limit for retaining the identity of a missing target."""

    last_seen_timeout_sec: float = 0.5

    def __post_init__(self):
        timeout = _finite_float(
            self.last_seen_timeout_sec,
            'last_seen_timeout_sec',
        )
        if timeout <= 0.0:
            raise ValueError(
                'last_seen_timeout_sec must be greater than zero'
            )
        object.__setattr__(self, 'last_seen_timeout_sec', timeout)


@dataclass(frozen=True)
class TargetLockDecision:
    """Observable selection state after one candidate or intent update."""

    selected_candidate: CandidateMeasurement | None
    selected_frame_id: str | None
    selected_visible: bool
    confirmed: bool
    stable_candidates: tuple[CandidateMeasurement, ...]
    last_seen_source_time_sec: float | None
    reason: str

    @property
    def ready(self):
        """Return whether the current target is safe to pass downstream."""
        return (
            self.selected_candidate is not None
            and self.selected_visible
            and self.confirmed
        )


class TargetLockManager:
    """Keep one stable track selected until intent or timeout changes it."""

    def __init__(self, config=None):
        config = config or TargetLockConfig()
        if not isinstance(config, TargetLockConfig):
            raise TypeError('config must be a TargetLockConfig')
        self._config = config
        self._selected_candidate = None
        self._selected_frame_id = None
        self._selected_visible = False
        self._confirmed = False
        self._last_seen_source_time_sec = None
        self._stable_candidates = ()
        self._stable_frame_id = None
        self._stable_source_time_sec = None

    def update(self, gate_decision, now_sec):
        """Consume the latest stability-gate result."""
        now_sec = _finite_float(now_sec, 'now_sec')
        if not isinstance(gate_decision, CandidateGateDecision):
            raise TypeError(
                'gate_decision must be a CandidateGateDecision'
            )

        expired = self._expire(now_sec)
        decision_frame_id = gate_decision.frame_id
        if (
            self._selected_frame_id is not None
            and decision_frame_id is not None
            and decision_frame_id != self._selected_frame_id
        ):
            self._clear_selection()
            expired = False

        stable_candidates = tuple(gate_decision.stable_candidates)
        if not stable_candidates:
            self._clear_stable_snapshot()
            self._selected_visible = False
            self._confirmed = False
            reason = 'lock_expired' if expired else 'no_stable_candidates'
            return self._decision(reason)

        source_time_sec = _finite_float(
            gate_decision.source_time_sec,
            'source_time_sec',
        )
        if (
            not isinstance(decision_frame_id, str)
            or not decision_frame_id.strip()
        ):
            raise ValueError(
                'stable candidates require a non-empty frame_id'
            )
        frame_id = decision_frame_id.strip()
        if any(
            not isinstance(candidate, CandidateMeasurement)
            for candidate in stable_candidates
        ):
            raise TypeError(
                'stable candidates must be CandidateMeasurement values'
            )
        stable_candidates = tuple(
            sorted(
                stable_candidates,
                key=lambda candidate: (
                    candidate.track_id,
                    candidate.class_label,
                ),
            )
        )
        self._stable_candidates = stable_candidates
        self._stable_frame_id = frame_id
        self._stable_source_time_sec = source_time_sec

        selected_index = self._selected_index()
        if selected_index is not None:
            self._select(
                stable_candidates[selected_index],
                frame_id,
                source_time_sec,
                keep_confirmation=True,
            )
            return self._decision('locked_target_visible')

        if self._selected_candidate is not None:
            self._selected_visible = False
            self._confirmed = False
            return self._decision('locked_target_missing')

        self._select(
            stable_candidates[0],
            frame_id,
            source_time_sec,
        )
        reason = 'target_relocked' if expired else 'target_locked'
        return self._decision(reason)

    def next_target(self, now_sec):
        """Cycle to the next currently stable target."""
        now_sec = _finite_float(now_sec, 'now_sec')
        expired = self._expire(now_sec)
        if not self._stable_candidates:
            reason = 'lock_expired' if expired else 'next_without_candidates'
            return self._decision(reason)

        selected_index = self._selected_index()
        if selected_index is None:
            next_index = 0
        else:
            next_index = (selected_index + 1) % len(
                self._stable_candidates
            )
        self._select(
            self._stable_candidates[next_index],
            self._stable_frame_id,
            self._stable_source_time_sec,
        )
        return self._decision('target_cycled')

    def confirm(self, now_sec):
        """Confirm only a currently visible, non-expired target."""
        now_sec = _finite_float(now_sec, 'now_sec')
        expired = self._expire(now_sec)
        if (
            self._selected_candidate is None
            or not self._selected_visible
            or self._selected_index() is None
        ):
            reason = 'lock_expired' if expired else 'confirm_without_target'
            return self._decision(reason)

        self._confirmed = True
        return self._decision('target_confirmed')

    def abort(self):
        """Clear candidate state and the current user confirmation."""
        self._clear_stable_snapshot()
        self._clear_selection()
        return self._decision('aborted')

    def tick(self, now_sec):
        """Advance the watchdog when no new candidate message arrives."""
        now_sec = _finite_float(now_sec, 'now_sec')
        expired = self._expire(now_sec)
        return self._decision(
            'lock_expired' if expired else 'watchdog_ok'
        )

    def _selected_index(self):
        if (
            self._selected_candidate is None
            or self._selected_frame_id != self._stable_frame_id
        ):
            return None
        for index, candidate in enumerate(self._stable_candidates):
            if (
                candidate.track_id
                == self._selected_candidate.track_id
                and candidate.class_label
                == self._selected_candidate.class_label
            ):
                return index
        return None

    def _select(
        self,
        candidate,
        frame_id,
        source_time_sec,
        *,
        keep_confirmation=False,
    ):
        self._selected_candidate = candidate
        self._selected_frame_id = frame_id
        self._selected_visible = True
        if not keep_confirmation:
            self._confirmed = False
        self._last_seen_source_time_sec = source_time_sec

    def _expire(self, now_sec):
        timeout = self._config.last_seen_timeout_sec
        if (
            self._stable_source_time_sec is not None
            and now_sec - self._stable_source_time_sec > timeout
        ):
            self._clear_stable_snapshot()
            self._selected_visible = False
            self._confirmed = False

        if (
            self._last_seen_source_time_sec is not None
            and now_sec - self._last_seen_source_time_sec > timeout
        ):
            self._clear_selection()
            return True
        return False

    def _clear_stable_snapshot(self):
        self._stable_candidates = ()
        self._stable_frame_id = None
        self._stable_source_time_sec = None

    def _clear_selection(self):
        self._selected_candidate = None
        self._selected_frame_id = None
        self._selected_visible = False
        self._confirmed = False
        self._last_seen_source_time_sec = None

    def _decision(self, reason):
        return TargetLockDecision(
            selected_candidate=self._selected_candidate,
            selected_frame_id=self._selected_frame_id,
            selected_visible=self._selected_visible,
            confirmed=self._confirmed,
            stable_candidates=self._stable_candidates,
            last_seen_source_time_sec=self._last_seen_source_time_sec,
            reason=reason,
        )
