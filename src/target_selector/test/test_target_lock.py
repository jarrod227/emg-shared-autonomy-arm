"""Tests for markerless selected-track locking and expiry."""

import math

import pytest

from target_selector.candidate_stability import (
    CandidateGateDecision,
    CandidateMeasurement,
)
from target_selector.target_lock import (
    TargetLockConfig,
    TargetLockManager,
)


def candidate(track_id, class_label='bottle', position=(0.4, 0.1, 0.7)):
    return CandidateMeasurement(
        track_id=track_id,
        class_label=class_label,
        class_confidence=0.9,
        position=position,
        localization_confidence=0.85,
    )


def stable_decision(source_time_sec, *candidates, frame_id='stereo_frame'):
    return CandidateGateDecision(
        stable_candidates=tuple(candidates),
        stable_counts=tuple(
            (item.track_id, 3) for item in candidates
        ),
        reason='stable',
        source_time_sec=source_time_sec,
        frame_id=frame_id,
    )


def empty_decision(reason='no_candidates', frame_id=None):
    return CandidateGateDecision(
        stable_candidates=(),
        stable_counts=(),
        reason=reason,
        source_time_sec=None,
        frame_id=frame_id,
    )


def manager():
    return TargetLockManager(
        TargetLockConfig(last_seen_timeout_sec=0.5)
    )


def test_first_stable_target_is_selected_deterministically():
    lock = manager()

    result = lock.update(
        stable_decision(
            10.0,
            candidate(9, 'cup'),
            candidate(2, 'cell_phone'),
        ),
        now_sec=10.01,
    )

    assert result.reason == 'target_locked'
    assert result.selected_candidate.track_id == 2
    assert result.selected_visible
    assert not result.confirmed
    assert not result.ready


def test_lock_stays_on_same_track_when_other_candidates_change():
    lock = manager()
    lock.update(
        stable_decision(10.0, candidate(7), candidate(9, 'cup')),
        now_sec=10.01,
    )

    result = lock.update(
        stable_decision(
            10.1,
            candidate(1, 'medicine_box'),
            candidate(7, position=(0.41, 0.1, 0.7)),
        ),
        now_sec=10.11,
    )

    assert result.reason == 'locked_target_visible'
    assert result.selected_candidate.track_id == 7
    assert result.selected_candidate.position == pytest.approx(
        (0.41, 0.1, 0.7)
    )


def test_next_target_cycles_only_currently_stable_candidates():
    lock = manager()
    lock.update(
        stable_decision(
            10.0,
            candidate(3, 'cell_phone'),
            candidate(1),
            candidate(2, 'cup'),
        ),
        now_sec=10.01,
    )

    assert lock.next_target(10.02).selected_candidate.track_id == 2
    assert lock.next_target(10.03).selected_candidate.track_id == 3
    assert lock.next_target(10.04).selected_candidate.track_id == 1


def test_missing_lock_does_not_jump_to_another_target():
    lock = manager()
    lock.update(
        stable_decision(10.0, candidate(1), candidate(2, 'cup')),
        now_sec=10.01,
    )
    assert lock.confirm(10.02).ready

    missing = lock.update(
        stable_decision(10.1, candidate(2, 'cup')),
        now_sec=10.11,
    )

    assert missing.reason == 'locked_target_missing'
    assert missing.selected_candidate.track_id == 1
    assert not missing.selected_visible
    assert not missing.confirmed
    assert not missing.ready
    assert lock.next_target(10.12).selected_candidate.track_id == 2


def test_confirm_requires_currently_visible_target():
    lock = manager()

    assert lock.confirm(10.0).reason == 'confirm_without_target'
    lock.update(
        stable_decision(10.1, candidate(4)),
        now_sec=10.11,
    )
    confirmed = lock.confirm(10.12)

    assert confirmed.reason == 'target_confirmed'
    assert confirmed.ready

    missing = lock.update(empty_decision(), now_sec=10.2)
    assert not missing.confirmed
    assert not lock.confirm(10.21).ready


def test_watchdog_clears_lock_after_last_seen_timeout():
    lock = manager()
    lock.update(
        stable_decision(10.0, candidate(4)),
        now_sec=10.01,
    )

    still_locked = lock.tick(10.5)
    expired = lock.tick(10.5001)

    assert still_locked.selected_candidate.track_id == 4
    assert expired.reason == 'lock_expired'
    assert expired.selected_candidate is None
    assert expired.stable_candidates == ()


def test_reference_frame_change_invalidates_previous_identity():
    lock = manager()
    lock.update(
        stable_decision(10.0, candidate(4), frame_id='left_frame'),
        now_sec=10.01,
    )

    warming = lock.update(
        empty_decision(reason='frame_changed', frame_id='right_frame'),
        now_sec=10.1,
    )
    relocked = lock.update(
        stable_decision(
            10.2,
            candidate(4),
            frame_id='right_frame',
        ),
        now_sec=10.21,
    )

    assert warming.selected_candidate is None
    assert relocked.selected_frame_id == 'right_frame'
    assert not relocked.confirmed


def test_abort_clears_selection_and_candidate_snapshot():
    lock = manager()
    lock.update(
        stable_decision(10.0, candidate(1)),
        now_sec=10.01,
    )
    lock.confirm(10.02)

    result = lock.abort()

    assert result.reason == 'aborted'
    assert result.selected_candidate is None
    assert result.stable_candidates == ()
    assert not result.ready


@pytest.mark.parametrize('value', (0.0, -0.1, math.inf))
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(ValueError):
        TargetLockConfig(last_seen_timeout_sec=value)
