"""Pure temporal-stability logic for markerless object candidates."""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral
import operator
from statistics import median


DEFAULT_OBJECT_CLASSES = (
    'bottle',
    'cup',
    'apple',
)


def _finite_float(value, name):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted


def _nonnegative_float(value, name):
    converted = _finite_float(value, name)
    if converted < 0.0:
        raise ValueError(f'{name} must be non-negative')
    return converted


def _confidence(value, name):
    converted = _finite_float(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return converted


def _position(value):
    try:
        position = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError('position must contain three numeric values') from error
    if len(position) != 3:
        raise ValueError('position must contain exactly three values')
    if not all(math.isfinite(component) for component in position):
        raise ValueError('position must contain only finite values')
    return position


@dataclass(frozen=True)
class CandidateMeasurement:
    """One candidate from one source stereo pair."""

    track_id: int
    class_label: str
    class_confidence: float
    position: tuple[float, float, float]
    localization_confidence: float


@dataclass(frozen=True)
class CandidateFrame:
    """Source metadata and candidates from one stereo observation."""

    source_time_sec: float
    frame_id: str
    valid: bool
    pair_skew_sec: float
    candidates: tuple[CandidateMeasurement, ...]


@dataclass(frozen=True)
class CandidateStabilityConfig:
    """Thresholds applied before a candidate becomes selectable."""

    required_frames: int = 3
    min_class_confidence: float = 0.5
    min_localization_confidence: float = 0.5
    max_pair_skew_sec: float = 0.05
    max_age_sec: float = 0.5
    future_tolerance_sec: float = 0.05
    max_frame_gap_sec: float = 0.2
    max_position_span_m: float = 0.03
    allowed_classes: tuple[str, ...] = DEFAULT_OBJECT_CLASSES

    def __post_init__(self):
        if isinstance(self.required_frames, bool):
            raise ValueError('required_frames must be an integer')
        try:
            required_frames = operator.index(self.required_frames)
        except TypeError as error:
            raise ValueError('required_frames must be an integer') from error
        if required_frames < 1:
            raise ValueError('required_frames must be at least one')

        allowed_classes = tuple(self.allowed_classes)
        if not allowed_classes:
            raise ValueError('allowed_classes must not be empty')
        if any(
            not isinstance(label, str) or not label.strip()
            for label in allowed_classes
        ):
            raise ValueError('allowed_classes must contain non-empty strings')
        allowed_classes = tuple(label.strip() for label in allowed_classes)
        if len(set(allowed_classes)) != len(allowed_classes):
            raise ValueError('allowed_classes must not contain duplicates')

        max_age_sec = _finite_float(self.max_age_sec, 'max_age_sec')
        if max_age_sec <= 0.0:
            raise ValueError('max_age_sec must be greater than zero')

        object.__setattr__(self, 'required_frames', required_frames)
        object.__setattr__(
            self,
            'min_class_confidence',
            _confidence(
                self.min_class_confidence,
                'min_class_confidence',
            ),
        )
        object.__setattr__(
            self,
            'min_localization_confidence',
            _confidence(
                self.min_localization_confidence,
                'min_localization_confidence',
            ),
        )
        object.__setattr__(
            self,
            'max_pair_skew_sec',
            _nonnegative_float(
                self.max_pair_skew_sec,
                'max_pair_skew_sec',
            ),
        )
        object.__setattr__(self, 'max_age_sec', max_age_sec)
        object.__setattr__(
            self,
            'future_tolerance_sec',
            _nonnegative_float(
                self.future_tolerance_sec,
                'future_tolerance_sec',
            ),
        )
        object.__setattr__(
            self,
            'max_frame_gap_sec',
            _finite_float(self.max_frame_gap_sec, 'max_frame_gap_sec'),
        )
        if self.max_frame_gap_sec <= 0.0:
            raise ValueError('max_frame_gap_sec must be greater than zero')
        object.__setattr__(
            self,
            'max_position_span_m',
            _nonnegative_float(
                self.max_position_span_m,
                'max_position_span_m',
            ),
        )
        object.__setattr__(self, 'allowed_classes', allowed_classes)


@dataclass(frozen=True)
class CandidateGateDecision:
    """Stable candidates plus compact state for logging and tests."""

    stable_candidates: tuple[CandidateMeasurement, ...]
    stable_counts: tuple[tuple[int, int], ...]
    reason: str
    source_time_sec: float | None
    frame_id: str | None


class CandidateStabilityGate:
    """Track every visible instance and expose only N-frame-stable candidates."""

    def __init__(self, config=None):
        config = config or CandidateStabilityConfig()
        if not isinstance(config, CandidateStabilityConfig):
            raise TypeError('config must be a CandidateStabilityConfig')
        self._config = config
        self._histories = {}
        self._last_source_time_sec = None
        self._frame_id = None

    def reset(self):
        """Forget all accumulated evidence."""
        self._histories.clear()
        self._last_source_time_sec = None
        self._frame_id = None

    def _decision(
        self,
        reason,
        stable_candidates=(),
        source_time_sec=None,
        frame_id=None,
    ):
        counts = tuple(
            (track_id, len(history))
            for track_id, history in sorted(self._histories.items())
        )
        return CandidateGateDecision(
            stable_candidates=tuple(stable_candidates),
            stable_counts=counts,
            reason=reason,
            source_time_sec=source_time_sec,
            frame_id=frame_id,
        )

    def _reject(self, reason):
        self.reset()
        return self._decision(reason)

    def update(self, frame, now_sec):
        """Consume one frame and return all candidates stable in this frame."""
        now_sec = _finite_float(now_sec, 'now_sec')
        if not isinstance(frame, CandidateFrame):
            raise TypeError('frame must be a CandidateFrame')

        if not isinstance(frame.valid, bool):
            return self._reject('invalid_frame')

        try:
            source_time_sec = _finite_float(
                frame.source_time_sec,
                'source_time_sec',
            )
            if not isinstance(frame.frame_id, str) or not frame.frame_id.strip():
                raise ValueError('frame_id must be a non-empty string')
            frame_id = frame.frame_id.strip()
            pair_skew_sec = _nonnegative_float(
                frame.pair_skew_sec,
                'pair_skew_sec',
            )
            candidates = tuple(frame.candidates)
        except (TypeError, ValueError):
            return self._reject('invalid_frame')

        if not frame.valid:
            reason = (
                'invalid_frame_has_candidates'
                if candidates
                else 'invalid_frame'
            )
            return self._reject(reason)
        if pair_skew_sec > self._config.max_pair_skew_sec:
            return self._reject('excessive_pair_skew')
        if source_time_sec > now_sec + self._config.future_tolerance_sec:
            return self._reject('future_timestamp')
        if now_sec - source_time_sec > self._config.max_age_sec:
            return self._reject('stale')

        reset_reason = None
        if self._frame_id is not None and frame_id != self._frame_id:
            self._histories.clear()
            self._last_source_time_sec = None
            reset_reason = 'frame_changed'
        if (
            self._last_source_time_sec is not None
            and source_time_sec <= self._last_source_time_sec
        ):
            return self._reject('non_increasing_timestamp')
        if (
            self._last_source_time_sec is not None
            and source_time_sec - self._last_source_time_sec
            > self._config.max_frame_gap_sec
        ):
            self._histories.clear()
            reset_reason = 'frame_gap'
        if not candidates:
            return self._reject('no_candidates')

        normalized = []
        seen_track_ids = set()
        try:
            for candidate in candidates:
                candidate = self._normalize_candidate(candidate)
                if candidate.track_id in seen_track_ids:
                    return self._reject('duplicate_track_id')
                seen_track_ids.add(candidate.track_id)
                normalized.append(candidate)
        except (TypeError, ValueError):
            return self._reject('invalid_candidate')

        eligible = tuple(
            candidate
            for candidate in normalized
            if (
                candidate.class_label in self._config.allowed_classes
                and candidate.class_confidence
                >= self._config.min_class_confidence
                and candidate.localization_confidence
                >= self._config.min_localization_confidence
            )
        )
        if not eligible:
            return self._reject('no_eligible_candidates')

        self._last_source_time_sec = source_time_sec
        self._frame_id = frame_id
        eligible_track_ids = {candidate.track_id for candidate in eligible}
        for missing_id in set(self._histories) - eligible_track_ids:
            del self._histories[missing_id]

        stable_candidates = []
        for candidate in eligible:
            history = self._histories.get(candidate.track_id)
            if history is None:
                history = deque(maxlen=self._config.required_frames)
                self._histories[candidate.track_id] = history
            elif (
                history[-1].class_label != candidate.class_label
                or any(
                    math.dist(previous.position, candidate.position)
                    > self._config.max_position_span_m
                    for previous in history
                )
            ):
                history.clear()

            history.append(candidate)
            if len(history) == self._config.required_frames:
                stable_candidates.append(self._aggregate(history))

        stable_candidates.sort(key=lambda candidate: candidate.track_id)
        if stable_candidates:
            reason = 'stable'
        else:
            reason = reset_reason or 'warming_up'
        return self._decision(
            reason,
            stable_candidates,
            source_time_sec,
            frame_id,
        )

    def _normalize_candidate(self, candidate):
        if not isinstance(candidate, CandidateMeasurement):
            raise TypeError('candidates must be CandidateMeasurement values')
        if (
            not isinstance(candidate.track_id, Integral)
            or isinstance(candidate.track_id, bool)
            or not 0 <= candidate.track_id <= 0xFFFFFFFF
        ):
            raise ValueError('track_id must fit uint32')
        if (
            not isinstance(candidate.class_label, str)
            or not candidate.class_label.strip()
        ):
            raise ValueError('class_label must be a non-empty string')
        return CandidateMeasurement(
            track_id=int(candidate.track_id),
            class_label=candidate.class_label.strip(),
            class_confidence=_confidence(
                candidate.class_confidence,
                'class_confidence',
            ),
            position=_position(candidate.position),
            localization_confidence=_confidence(
                candidate.localization_confidence,
                'localization_confidence',
            ),
        )

    def _aggregate(self, history):
        return CandidateMeasurement(
            track_id=history[-1].track_id,
            class_label=history[-1].class_label,
            class_confidence=min(
                candidate.class_confidence for candidate in history
            ),
            position=tuple(
                median(candidate.position[axis] for candidate in history)
                for axis in range(3)
            ),
            localization_confidence=min(
                candidate.localization_confidence for candidate in history
            ),
        )
