"""Tests for the per-donning activation calibration.

The decision rule is pure arithmetic over collected windows, so it is tested
against the two real donnings that motivated it rather than against invented
numbers. Those two are the whole evidence base: one that should calibrate
cleanly and one that should refuse to.
"""

import math

import numpy as np
import pytest

from emg_activation_ref import FROZEN_BASELINE_SHIFT, FROZEN_FACTOR
from emg_calibrate import (
    SEPARATION_MARGINAL,
    SEPARATION_PASS,
    SUSTAIN_WINDOWS,
    CalibrationError,
    confirm_applied,
    summarize,
    sustained_level,
    total_mav_windows,
    verdict_message,
)
from emg_features_ref import WINDOW
from emg_protocol import (
    SET_RESULT_ACCEPTED,
    SET_RESULT_REJECTED,
    ActivationState,
)


def constant(value, count=40):
    return np.full(count, value, dtype=np.int64)


def donning(rest, preparation, plateaus):
    """One donning's measured bands, as per-trial window arrays."""
    return {
        "rest": constant(rest),
        "preparation": [constant(preparation)],
        "gestures": {
            name: [constant(value)] for name, value in plateaus.items()
        },
    }


def late_trial(level, rest=20, total=77, hold=20):
    """A trial where the hold only starts near the end.

    The shape that broke the first version: a second or two of reaction time
    before a late, genuine contraction.
    """
    trial = np.full(total, rest, dtype=np.int64)
    trial[-hold:] = level
    return trial


# 2026-08-14: rest 35, preparation 79, gestures 318-736.
CLEAN = donning(35, 79, {"NEXT_TARGET": 400, "CONFIRM": 736, "ABORT": 318})
# 2026-08-15 after re-gelling: rest 6, preparation 73, gestures 145-197.
THIN = donning(6, 73, {"NEXT_TARGET": 173, "CONFIRM": 197, "ABORT": 145})


def run(measured):
    return summarize(measured["rest"], measured["preparation"],
                     measured["gestures"])


def test_the_clean_donning_passes_and_places_the_threshold_between_the_bands():
    summary = run(CLEAN)

    assert summary["verdict"] == "pass"
    assert summary["separation_ratio"] == pytest.approx(318 / 79, abs=0.01)
    assert summary["weakest_gesture"] == "ABORT"
    # Strictly between what must be suppressed and what must get through.
    assert 79 < summary["threshold_floor"] < 318


def test_the_re_gelled_donning_fails_rather_than_getting_a_clever_threshold():
    # 145 / 73 = 1.99. This is the donning where ABORT could not fire all
    # morning; the calibration must say "re-place the electrodes", not
    # produce a threshold wedged into the gap.
    summary = run(THIN)

    assert summary["verdict"] == "fail"
    assert summary["separation_ratio"] < SEPARATION_MARGINAL
    assert "electrode" in verdict_message(summary).lower()


@pytest.mark.parametrize(
    "weakest, separation, verdict",
    [
        (300, 3.0, "pass"),       # exactly at the pass edge
        (299, 2.99, "marginal"),
        (250, 2.5, "marginal"),   # exactly at the marginal edge
        (249, 2.49, "fail"),
    ],
)
def test_the_three_tiers_are_inclusive_at_their_lower_edges(
    weakest, separation, verdict
):
    # Preparation of 100 makes each ratio exact in integer MAV counts, so
    # the edges are probed rather than approached.
    measured = donning(6, 100, {"ABORT": weakest})
    summary = run(measured)

    assert summary["separation_ratio"] == separation
    assert summary["verdict"] == verdict
    assert SEPARATION_PASS == 3.0 and SEPARATION_MARGINAL == 2.5


def test_the_threshold_is_the_geometric_mean_not_the_arithmetic_one():
    summary = run(THIN)

    # The arithmetic mean of 73 and 145 is 109; the geometric mean is 103.
    # Geometric keeps the relative margin equal on both sides, which matters
    # precisely when separation is thin.
    assert summary["threshold_floor"] == round(math.sqrt(73 * 145))
    assert summary["threshold_floor"] < (73 + 145) / 2


def test_the_weakest_gesture_sets_the_bound_not_the_average():
    # A strong CONFIRM must not lift the threshold above a weak ABORT: the
    # stop command is the one that must never be blocked.
    summary = run(CLEAN)

    assert summary["weakest_gesture_plateau"] == 318
    assert summary["threshold_floor"] < 318


def test_the_weakest_trial_of_a_gesture_sets_its_plateau():
    measured = donning(6, 30, {"ABORT": 300})
    measured["gestures"]["ABORT"] = [constant(300), constant(120)]

    summary = summarize(measured["rest"], measured["preparation"],
                        measured["gestures"])

    # One weak repetition out of two is the one that must still get through.
    assert summary["gesture_plateaus"]["ABORT"] == 120


def test_k_and_shift_are_not_calibrated():
    # Only the floor is per-session. A K derived from this donning's rest
    # (103 / 6 = 17) would push the threshold to 340 as soon as rest drifted
    # to 20, blocking every real gesture including ABORT.
    summary = run(CLEAN)

    assert summary["factor"] == FROZEN_FACTOR
    assert summary["baseline_shift"] == FROZEN_BASELINE_SHIFT


def test_a_late_hold_is_measured_at_its_held_level():
    # The regression. On 2026-08-15 a real ABORT sustaining about 536 total
    # MAV was reported as 98, three calibrations failed in a row, and the
    # hardware was blamed. The cause was a 75th-percentile plateau over the
    # whole trial: with a second of reaction time the top quartile of
    # windows lands on the onset ramp, not the hold.
    measured = {
        "rest": constant(20),
        "preparation": [constant(70)],
        "gestures": {"ABORT": [late_trial(536)]},
    }

    summary = summarize(**{
        "rest_totals": measured["rest"],
        "preparation_trials": measured["preparation"],
        "gesture_totals": measured["gestures"],
    })

    assert summary["weakest_gesture_plateau"] == 536
    assert summary["separation_ratio"] == pytest.approx(536 / 70, abs=0.01)
    assert summary["verdict"] == "pass"


def test_a_twitch_shorter_than_the_gate_does_not_set_the_preparation_bound():
    # A movement too brief to fill the gate's stable run cannot fire an
    # event at any threshold, so it must not drag the threshold down.
    brief = np.full(40, 20, dtype=np.int64)
    brief[10:10 + SUSTAIN_WINDOWS - 1] = 400

    assert sustained_level(brief) == 20


def test_a_hold_exactly_as_long_as_the_gate_counts():
    trial = np.full(40, 20, dtype=np.int64)
    trial[10:10 + SUSTAIN_WINDOWS] = 400

    assert sustained_level(trial) == 400


def test_sustained_level_ignores_where_in_the_trial_the_hold_happened():
    early = np.full(60, 15, dtype=np.int64)
    early[2:22] = 300
    late = np.full(60, 15, dtype=np.int64)
    late[-20:] = 300

    assert sustained_level(early) == sustained_level(late) == 300


def test_sustained_level_refuses_a_trial_shorter_than_the_gate_run():
    with pytest.raises(CalibrationError, match="at least"):
        sustained_level(np.full(SUSTAIN_WINDOWS - 1, 100, dtype=np.int64))


def test_an_inert_floor_is_reported_rather_than_left_to_be_discovered():
    # If K x baseline already exceeds T_session, the firmware's relative
    # rule governs and the calibration changes nothing.
    loud_rest = summarize(constant(60), [constant(70)],
                          {"ABORT": [constant(300)]})
    quiet_rest = summarize(constant(10), [constant(70)],
                           {"ABORT": [constant(300)]})

    assert "inert_floor_warning" in loud_rest
    assert "inert_floor_warning" not in quiet_rest


def test_rest_is_reported_as_a_median_comparable_with_the_firmware_baseline():
    # A tail statistic here invited exactly the wrong comparison: the
    # firmware's baseline is an EMA of classified-REST windows, so a 95th
    # percentile of rest is not the number to multiply K by.
    rest = np.concatenate([constant(20, 90), constant(200, 10)])

    summary = summarize(rest, [constant(70)], {"ABORT": [constant(300)]})

    assert summary["rest_baseline"] == 20


def test_empty_or_missing_captures_are_refused():
    with pytest.raises(CalibrationError):
        summarize(np.asarray([]), [constant(70)], {"ABORT": [constant(300)]})
    with pytest.raises(CalibrationError):
        summarize(constant(30), [], {"ABORT": [constant(300)]})
    with pytest.raises(CalibrationError):
        summarize(constant(30), [constant(70)], {})
    with pytest.raises(CalibrationError):
        summarize(constant(30), [constant(70)], {"ABORT": []})


def test_silent_preparation_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(CalibrationError, match="electrode"):
        summarize(constant(0), [constant(0)], {"ABORT": [constant(300)]})


def test_window_totals_match_the_firmware_grid():
    rng = np.random.default_rng(7)
    samples = rng.integers(1900, 2200, size=(WINDOW + 300, 3), dtype=np.int64)

    totals = total_mav_windows(samples, 3)

    # 400-frame window, 100-frame hop: 300 extra frames give three more.
    assert len(totals) == 4
    assert all(value >= 0 for value in totals)


def test_window_totals_reject_a_capture_shorter_than_one_window():
    with pytest.raises(CalibrationError, match="at least"):
        total_mav_windows(np.zeros((WINDOW - 1, 3), dtype=np.int64), 3)


def test_window_totals_reject_a_channel_count_mismatch():
    with pytest.raises(CalibrationError, match="channel count"):
        total_mav_windows(np.zeros((WINDOW, 2), dtype=np.int64), 3)


def state(**overrides):
    fields = dict(source=1, factor=FROZEN_FACTOR,
                  baseline_shift=FROZEN_BASELINE_SHIFT,
                  last_result=SET_RESULT_ACCEPTED, threshold_floor=158,
                  applied_sequence=1)
    fields.update(overrides)
    return ActivationState(**fields)


def test_confirmation_requires_the_board_to_report_what_was_sent():
    summary = run(CLEAN)
    summary["threshold_floor"] = 158

    confirm_applied(state(), summary)

    with pytest.raises(CalibrationError, match="rejected"):
        confirm_applied(state(last_result=SET_RESULT_REJECTED), summary)
    # A board that accepted something else is not a calibrated board.
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(threshold_floor=110), summary)
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(factor=FROZEN_FACTOR + 1), summary)
