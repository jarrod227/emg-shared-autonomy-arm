"""Tests for markerless candidate freshness and N-frame stability."""

import math

import pytest

from target_selector.candidate_stability import (
    CandidateFrame,
    CandidateMeasurement,
    CandidateStabilityConfig,
    CandidateStabilityGate,
    DEFAULT_OBJECT_CLASSES,
)


CONFIG = CandidateStabilityConfig(
    required_frames=3,
    min_class_confidence=0.6,
    min_localization_confidence=0.7,
    max_pair_skew_sec=0.02,
    max_age_sec=0.2,
    future_tolerance_sec=0.01,
    max_frame_gap_sec=0.1,
    max_position_span_m=0.03,
)


def candidate(
    *,
    track_id=1,
    class_label='bottle',
    class_confidence=0.9,
    position=(0.4, 0.2, 0.8),
    localization_confidence=0.85,
):
    return CandidateMeasurement(
        track_id=track_id,
        class_label=class_label,
        class_confidence=class_confidence,
        position=position,
        localization_confidence=localization_confidence,
    )


def frame(
    source_time_sec,
    *candidates,
    frame_id='stereo_left_optical_frame',
    valid=True,
    pair_skew_sec=0.01,
):
    return CandidateFrame(
        source_time_sec=source_time_sec,
        frame_id=frame_id,
        valid=valid,
        pair_skew_sec=pair_skew_sec,
        candidates=tuple(candidates),
    )


def test_default_closed_set_matches_selected_project_objects():
    assert DEFAULT_OBJECT_CLASSES == (
        'bottle',
        'cup',
        'cell_phone',
        'medicine_box',
    )


def test_candidate_becomes_stable_on_third_consecutive_frame():
    gate = CandidateStabilityGate(CONFIG)
    points = (
        (0.40, 0.20, 0.80),
        (0.41, 0.19, 0.81),
        (0.395, 0.205, 0.795),
    )

    first = gate.update(
        frame(10.00, candidate(position=points[0])),
        now_sec=10.01,
    )
    second = gate.update(
        frame(10.03, candidate(position=points[1])),
        now_sec=10.04,
    )
    third = gate.update(
        frame(10.06, candidate(position=points[2])),
        now_sec=10.07,
    )

    assert first.reason == 'warming_up'
    assert first.stable_counts == ((1, 1),)
    assert second.stable_counts == ((1, 2),)
    assert third.reason == 'stable'
    assert third.stable_counts == ((1, 3),)
    assert third.source_time_sec == pytest.approx(10.06)
    assert third.frame_id == 'stereo_left_optical_frame'
    assert third.stable_candidates[0].position == pytest.approx(
        (0.40, 0.20, 0.80)
    )


def test_multiple_tracks_become_stable_independently():
    gate = CandidateStabilityGate(CONFIG)
    for index in range(3):
        source_time = 10.0 + index * 0.03
        decision = gate.update(
            frame(
                source_time,
                candidate(
                    track_id=2,
                    class_label='cell_phone',
                    position=(0.5, 0.1, 0.75),
                ),
                candidate(track_id=1, class_label='bottle'),
            ),
            now_sec=source_time + 0.01,
        )

    assert [item.track_id for item in decision.stable_candidates] == [1, 2]
    assert decision.stable_counts == ((1, 3), (2, 3))


def test_missing_track_loses_only_its_history():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(
        frame(
            10.00,
            candidate(track_id=1),
            candidate(track_id=2, class_label='cup'),
        ),
        now_sec=10.01,
    )
    decision = gate.update(
        frame(10.03, candidate(track_id=2, class_label='cup')),
        now_sec=10.04,
    )

    assert decision.stable_counts == ((2, 2),)


@pytest.mark.parametrize(
    'changed_candidate',
    (
        candidate(position=(0.44, 0.2, 0.8)),
        candidate(class_label='cup'),
    ),
)
def test_position_jump_or_class_change_restarts_track(changed_candidate):
    gate = CandidateStabilityGate(CONFIG)
    gate.update(frame(10.00, candidate()), now_sec=10.01)
    gate.update(frame(10.03, candidate()), now_sec=10.04)

    decision = gate.update(
        frame(10.06, changed_candidate),
        now_sec=10.07,
    )

    assert decision.reason == 'warming_up'
    assert decision.stable_counts == ((1, 1),)
    assert decision.stable_candidates == ()


def test_cumulative_drift_across_window_restarts_track():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(
        frame(10.00, candidate(position=(0.40, 0.20, 0.80))),
        now_sec=10.01,
    )
    gate.update(
        frame(10.03, candidate(position=(0.42, 0.20, 0.80))),
        now_sec=10.04,
    )

    decision = gate.update(
        frame(10.06, candidate(position=(0.44, 0.20, 0.80))),
        now_sec=10.07,
    )

    assert decision.reason == 'warming_up'
    assert decision.stable_counts == ((1, 1),)
    assert decision.stable_candidates == ()


def test_long_frame_gap_restarts_all_histories():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(frame(10.00, candidate()), now_sec=10.01)
    gate.update(frame(10.03, candidate()), now_sec=10.04)

    decision = gate.update(frame(10.20, candidate()), now_sec=10.21)

    assert decision.reason == 'frame_gap'
    assert decision.stable_counts == ((1, 1),)
    assert decision.source_time_sec == pytest.approx(10.20)


def test_reference_frame_change_restarts_history_and_updates_metadata():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(frame(10.00, candidate()), now_sec=10.01)
    gate.update(frame(10.03, candidate()), now_sec=10.04)

    decision = gate.update(
        frame(
            10.06,
            candidate(),
            frame_id='other_stereo_frame',
        ),
        now_sec=10.07,
    )

    assert decision.reason == 'frame_changed'
    assert decision.stable_counts == ((1, 1),)
    assert decision.frame_id == 'other_stereo_frame'
    assert decision.source_time_sec == pytest.approx(10.06)


def test_low_confidence_track_does_not_reset_other_track():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(
        frame(
            10.00,
            candidate(track_id=1),
            candidate(track_id=2, class_label='cup'),
        ),
        now_sec=10.01,
    )
    decision = gate.update(
        frame(
            10.03,
            candidate(track_id=1, class_confidence=0.59),
            candidate(track_id=2, class_label='cup'),
        ),
        now_sec=10.04,
    )

    assert decision.stable_counts == ((2, 2),)


@pytest.mark.parametrize(
    ('bad_frame', 'now_sec', 'reason'),
    (
        (frame(10.00, valid=False), 10.01, 'invalid_frame'),
        (
            frame(10.00, candidate(), valid=False),
            10.01,
            'invalid_frame_has_candidates',
        ),
        (frame(10.00), 10.01, 'no_candidates'),
        (
            frame(10.00, candidate(), frame_id=''),
            10.01,
            'invalid_frame',
        ),
        (
            frame(10.00, candidate(), pair_skew_sec=0.021),
            10.01,
            'excessive_pair_skew',
        ),
        (frame(9.80, candidate()), 10.01, 'stale'),
        (frame(10.02, candidate()), 10.00, 'future_timestamp'),
    ),
)
def test_invalid_or_empty_frame_clears_all_history(
    bad_frame,
    now_sec,
    reason,
):
    gate = CandidateStabilityGate(CONFIG)
    gate.update(frame(9.94, candidate()), now_sec=9.95)
    gate.update(frame(9.97, candidate()), now_sec=9.98)

    decision = gate.update(bad_frame, now_sec=now_sec)

    assert decision.reason == reason
    assert decision.stable_counts == ()
    assert decision.stable_candidates == ()


def test_duplicate_or_older_source_time_cannot_count_twice():
    gate = CandidateStabilityGate(CONFIG)
    gate.update(frame(10.00, candidate()), now_sec=10.01)

    duplicate = gate.update(frame(10.00, candidate()), now_sec=10.01)

    assert duplicate.reason == 'non_increasing_timestamp'
    assert duplicate.stable_counts == ()


@pytest.mark.parametrize(
    'bad_candidate',
    (
        candidate(track_id=-1),
        candidate(class_label=''),
        candidate(class_confidence=math.nan),
        candidate(position=(math.inf, 0.2, 0.8)),
        candidate(localization_confidence=1.1),
    ),
)
def test_malformed_candidate_fails_closed(bad_candidate):
    gate = CandidateStabilityGate(CONFIG)

    decision = gate.update(
        frame(10.00, bad_candidate),
        now_sec=10.01,
    )

    assert decision.reason == 'invalid_candidate'
    assert decision.stable_counts == ()


def test_duplicate_track_id_in_one_frame_fails_closed():
    gate = CandidateStabilityGate(CONFIG)

    decision = gate.update(
        frame(
            10.00,
            candidate(track_id=1),
            candidate(track_id=1, class_label='cup'),
        ),
        now_sec=10.01,
    )

    assert decision.reason == 'duplicate_track_id'


@pytest.mark.parametrize(
    ('field', 'bad_value'),
    (
        ('required_frames', 0),
        ('min_class_confidence', 1.1),
        ('min_localization_confidence', math.nan),
        ('max_pair_skew_sec', -0.1),
        ('max_age_sec', 0.0),
        ('future_tolerance_sec', -0.1),
        ('max_frame_gap_sec', 0.0),
        ('max_position_span_m', -0.1),
        ('allowed_classes', ()),
    ),
)
def test_bad_configuration_is_rejected(field, bad_value):
    values = {
        'required_frames': CONFIG.required_frames,
        'min_class_confidence': CONFIG.min_class_confidence,
        'min_localization_confidence': CONFIG.min_localization_confidence,
        'max_pair_skew_sec': CONFIG.max_pair_skew_sec,
        'max_age_sec': CONFIG.max_age_sec,
        'future_tolerance_sec': CONFIG.future_tolerance_sec,
        'max_frame_gap_sec': CONFIG.max_frame_gap_sec,
        'max_position_span_m': CONFIG.max_position_span_m,
        'allowed_classes': CONFIG.allowed_classes,
    }
    values[field] = bad_value

    with pytest.raises(ValueError):
        CandidateStabilityConfig(**values)
