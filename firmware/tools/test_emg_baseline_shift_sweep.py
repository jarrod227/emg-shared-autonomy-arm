"""Tests for the baseline-shift sweep.

The metrics are the point of this tool, so the tests pin the two that an
earlier version got wrong: the cold-start threshold step must be excluded, and
the two directions of pass error must not be pooled.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from emg_activation_ref import REST
from emg_baseline_shift_sweep import (
    format_rows,
    label_pass_counts,
    replay_session,
    sweep,
    total_mav,
)


def _timeline(mavs, predictions, labels=None, valid=None, session="s"):
    """A stand-in carrying only the fields the sweep reads."""

    rows = []
    for mav in mavs:
        row = []
        for channel in range(3):
            # (MAV, RMS, waveform length, zero crossings) per channel; only
            # the MAV columns are read, the rest are filler.
            row.extend([mav[channel], 0.0, 0.0, 0.0])
        rows.append(row)
    count = len(mavs)
    return SimpleNamespace(
        session_id=session,
        features=np.asarray(rows, dtype=np.float64),
        valid=np.ones(count, dtype=bool) if valid is None else np.asarray(valid),
        labels=np.asarray(
            [None] * count if labels is None else labels, dtype=object
        ),
    ), SimpleNamespace(
        timeline=None, predictions=np.asarray(predictions, dtype=object)
    )


def _fold(mavs, predictions, **kwargs):
    timeline, fold = _timeline(mavs, predictions, **kwargs)
    fold.timeline = timeline
    return fold


def test_total_mav_sums_the_three_mav_columns_only():
    fold = _fold([(10, 20, 30)], [REST])
    assert total_mav(fold.timeline.features).tolist() == [60]


def test_total_mav_rejects_a_short_feature_row():
    with pytest.raises(ValueError):
        total_mav(np.zeros((2, 4)))


def test_cold_start_step_is_excluded_from_the_largest_step():
    # One REST window initialises the accumulator exactly, so the threshold
    # steps from the floor to 3 x 100 in a single window. That step is an
    # artefact of initialisation and identical at every shift.
    fold = _fold([(100, 0, 0)] * 4, [REST] * 4)
    result = replay_session(fold, factor=3, shift=4, floor=110)
    assert result["threshold_max_step"] == 0


def test_floor_governing_everywhere_is_reported_as_such():
    fold = _fold([(1, 1, 1)] * 5, [REST] * 5)
    result = replay_session(fold, factor=3, shift=4, floor=110)
    assert result["relative_pct"] == 0.0
    rows = sweep([fold], factor=3, floor=110, shifts=(4,))
    assert "cannot discriminate" in format_rows(rows, factor=3, floor=110)


def test_pass_counts_keep_the_two_error_directions_apart():
    # A loud CONFIRM after rest passes; a rest window never counts as passed.
    labels = [REST, REST, "CONFIRM"]
    fold = _fold(
        [(10, 10, 10), (10, 10, 10), (400, 400, 400)],
        [REST, REST, "CONFIRM"],
        labels=labels,
    )
    result = replay_session(fold, factor=3, shift=4, floor=110)
    passes = result["passes"]
    assert passes["rest_total"] == 2
    assert passes["rest_passed"] == 0
    assert passes["gesture_total"] == 1
    assert passes["gesture_passed"] == 1


def test_invalid_windows_are_not_counted_as_labelled_evidence():
    labels = [REST, "CONFIRM"]
    fold = _fold(
        [(10, 10, 10), (400, 400, 400)],
        [REST, "CONFIRM"],
        labels=labels,
        valid=[True, False],
    )
    counts = label_pass_counts(fold.timeline, [REST, "CONFIRM"])
    assert counts["gesture_total"] == 0


def test_shipped_shift_is_the_reference_and_others_are_diffed_against_it():
    fold = _fold(
        [(30, 30, 30)] * 6 + [(150, 150, 150)] * 6,
        [REST] * 6 + ["CONFIRM"] * 6,
    )
    rows = sweep([fold], factor=3, floor=110, shifts=(1, 4))
    by_shift = {row["shift"]: row for row in rows}
    assert by_shift[4]["differing_windows"] == 0
    assert "ref" in format_rows(rows, factor=3, floor=110)
    assert by_shift[1]["windows"] == 12
