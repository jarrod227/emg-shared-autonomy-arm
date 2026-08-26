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
    CAPTURED_GESTURES,
    diagnose_failure,
    sendable_references,
    GESTURES,
    REFERENCE_GESTURES,
    SEPARATION_MARGINAL,
    SEPARATION_PASS,
    SUSTAIN_WINDOWS,
    ABORT_SUSTAIN_WINDOWS,
    CalibrationError,
    instantaneous_level,
    reference_levels,
    confirm_applied,
    gate_reachability,
    gesture_sustain_windows,
    longest_run_above,
    summarize,
    sustained_level,
    total_mav_windows,
    verdict_message,
)
from emg_event_gate_replay import VALIDATED_GATE
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
    left, right = sendable_references(summary)
    applied = dict(reference_left=left, reference_right=right)

    confirm_applied(state(**applied), summary)

    with pytest.raises(CalibrationError, match="rejected"):
        confirm_applied(state(last_result=SET_RESULT_REJECTED, **applied),
                        summary)
    # A board that accepted something else is not a calibrated board.
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(threshold_floor=110, **applied), summary)
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(factor=FROZEN_FACTOR + 1, **applied), summary)


def test_the_sustain_run_is_the_one_the_gate_actually_needs():
    # The comment always said "the number of consecutive windows the event
    # gate requires before it will emit anything" and the code used
    # stable_windows alone. The gate discards onset_holdoff windows first, so
    # a threshold fitted to a five-window hold sits about twice too high: 123
    # and 116 were shipped where the corrected rule gives 68 and 69, and at
    # the shipped values even the four trained gestures fired in 0-4 of 6.
    assert SUSTAIN_WINDOWS == (
        VALIDATED_GATE.onset_holdoff_windows + VALIDATED_GATE.stable_windows
    )


def test_abort_needs_a_shorter_run_than_the_ordinary_gestures():
    # abort_stable_windows is 1, so ABORT clears the gate in holdoff + 1.
    # Measuring it at the ordinary run understates it, and it is often the
    # weakest gesture -- the one that sets the threshold.
    assert ABORT_SUSTAIN_WINDOWS < SUSTAIN_WINDOWS
    assert ABORT_SUSTAIN_WINDOWS == (
        VALIDATED_GATE.onset_holdoff_windows
        + VALIDATED_GATE.abort_stable_windows
    )
    assert gesture_sustain_windows("ABORT") == ABORT_SUSTAIN_WINDOWS
    assert gesture_sustain_windows("CONFIRM") == SUSTAIN_WINDOWS


def test_a_gesture_that_only_holds_briefly_is_measured_lower_than_before():
    # A trial that spikes for six windows and then fades used to be credited
    # with the spike, because five windows was all the measure asked for.
    trial = np.concatenate([
        np.full(6, 400, dtype=np.int64),
        np.full(60, 90, dtype=np.int64),
    ])
    assert sustained_level(trial, 5) == 400
    assert sustained_level(trial, SUSTAIN_WINDOWS) == 90


def test_reachability_reports_what_the_gate_would_do_at_a_given_threshold():
    # Reported, not decisive: against a threshold derived from these same
    # windows both sides pass by construction. It is informative when the
    # threshold comes from elsewhere, which is how the 2026-08-18 defect was
    # found -- a threshold fitted to five windows, checked against seventeen.
    prep = [np.full(40, 60, dtype=np.int64)]
    gestures = {"ABORT": [np.full(40, 300, dtype=np.int64)],
                "CONFIRM": [np.full(40, 300, dtype=np.int64)]}

    workable = gate_reachability(prep, gestures, 100)
    assert workable["preparation_can_fire"] is False
    assert workable["all_gestures_fire"] is True

    # The shipped-threshold case: too high for anything to hold.
    too_high = gate_reachability(prep, gestures, 320)
    assert too_high["any_gesture_dead"] is True
    assert too_high["gestures"]["ABORT"]["windows_needed"] == (
        ABORT_SUSTAIN_WINDOWS
    )


def test_longest_run_restarts_after_a_single_dip():
    # One window below the threshold is rewritten to REST, which clears the
    # hold-off and the stable run together.
    dips = np.array([200] * 10 + [50] + [200] * 8, dtype=np.int64)
    assert longest_run_above(dips, 100) == 10


def test_every_reference_gesture_is_actually_captured():
    # The list that drives the prompts and the list the reference is read
    # from have to agree, or a reference silently goes missing.
    for name in REFERENCE_GESTURES:
        assert name in CAPTURED_GESTURES


def test_a_direction_only_class_cannot_set_the_threshold():
    """ULNAR never reaches the event gate, so it must not decide T_session.

    Made concrete: a weak ulnar deviation, weaker than every gate gesture,
    would otherwise become the "weakest gesture" and drag the threshold down
    to suppress a class the gate cannot act on anyway.
    """
    measured = donning(35, 79, {"NEXT_TARGET": 400, "CONFIRM": 736,
                                "ABORT": 318})
    measured["gestures"]["ULNAR"] = [constant(120)]

    summary = run(measured)

    assert summary["weakest_gesture"] == "ABORT"
    assert "ULNAR" not in summary["gesture_plateaus"]
    assert "ULNAR" not in summary["gate_reachability"]["gestures"]
    assert summary["threshold_floor"] == run(CLEAN)["threshold_floor"]
    # But it is still measured, which is the whole reason it was captured.
    assert summary["reference_levels"]["ULNAR"] == pytest.approx(120.0)


def test_the_reference_is_instantaneous_where_the_threshold_is_sustained():
    """The two statistics answer different questions about the same trial.

    A late hold: mostly rest, then a genuine contraction. The threshold takes
    the level held across the gate's whole run, which a late start pulls
    down; activation is computed per window, so its reference has to be what
    the windows actually reach. Setting a reference from a sustained level
    was tried, and the board saturated through most of every hold.
    """
    # A real hold is not flat. The sustained level is a run *minimum*, so a
    # hold that dips reports its dips; the windows themselves reach much
    # higher, and activation is computed from those windows.
    hold = np.array([700, 420, 680, 430, 660, 410, 690, 440, 670, 400,
                     700, 430, 680, 420, 660, 440, 690, 410, 670, 430,
                     700, 420, 680, 430, 660] * 2, dtype=np.int64)
    trial = np.concatenate([np.full(20, 25, dtype=np.int64), hold])

    sustained = sustained_level(trial)
    instantaneous = instantaneous_level(trial)

    assert sustained == pytest.approx(410.0), "not the run minimum"
    assert instantaneous > 1.6 * sustained, (
        f"instantaneous {instantaneous} should sit near the top of the hold, "
        f"not at its floor {sustained}"
    )


def test_the_reference_takes_the_median_trial_not_the_weakest():
    """A reference wants a typical effort; a threshold wants the feeblest.

    Taking the minimum here would put ordinary effort above full deflection
    and leave the wearer permanently saturated, which is the defect being
    fixed, reached by another route.
    """
    levels = reference_levels({
        "ULNAR": [constant(100), constant(300), constant(200)],
    })

    assert levels["ULNAR"] == pytest.approx(200.0)


def test_a_reference_below_the_threshold_warns_without_failing_the_donning():
    """The span proportional control maps is the one above the threshold.

    A reference inside it means an empty span, which is a misjudged trial
    rather than anything about electrode placement -- so it must not touch a
    separation verdict that says the placement is fine.
    """
    measured = donning(35, 79, {"NEXT_TARGET": 400, "CONFIRM": 736,
                                "ABORT": 318})
    measured["gestures"]["ULNAR"] = [constant(10)]

    summary = run(measured)

    assert summary["verdict"] == "pass"
    assert "ULNAR" in summary["reference_warning"]
    assert "T_session" in summary["reference_warning"]


def test_a_missing_direction_reference_says_the_firmware_will_fall_back():
    summary = run(CLEAN)

    assert "ULNAR" not in summary["reference_levels"]
    assert "ULNAR" in summary["reference_warning"]
    assert "compile-time" in summary["reference_warning"]


def test_the_reference_ratio_is_recorded_per_donning():
    # The frozen firmware constant assumed this ratio was 3 and stable. It is
    # recorded per donning so that assumption can be checked against data
    # rather than re-argued.
    summary = run(CLEAN)

    ratio = summary["reference_over_threshold"]["NEXT_TARGET"]
    assert ratio == pytest.approx(
        400.0 / summary["threshold_floor"], abs=0.01
    )


def test_a_rest_capture_with_a_gesture_in_it_is_flagged():
    """Stillness louder than deliberate movement is not a donning property.

    Measured 2026-08-23: a rest trial the wearer accidentally performed a
    gesture during reported baseline 80.5 against a preparation bound of
    48.0. The inconsistency was already printed, next to the number it
    contradicts, and went unremarked while the cause was looked for in the
    hardware instead.
    """
    measured = donning(80, 48, {"NEXT_TARGET": 174, "CONFIRM": 162,
                                "ABORT": 170})

    summary = run(measured)

    assert "rest_contamination_warning" in summary
    assert "80" in summary["rest_contamination_warning"]
    # The separation ratio measures electrode placement, which a botched rest
    # trial says nothing about, so it must not be overridden.
    assert summary["verdict"] == "pass"


def test_an_ordinary_donning_is_not_flagged_for_rest_contamination():
    # The clean capture from the same evening: rest 15, preparation 19.
    measured = donning(15, 19, {"NEXT_TARGET": 174, "CONFIRM": 162,
                                "ABORT": 170})

    assert "rest_contamination_warning" not in run(measured)


def test_a_high_resting_level_points_at_the_electrodes():
    # 2026-08-23, electrode motion: rest quadrupled while the gestures did
    # not move, and K x baseline would have governed the threshold outright.
    measured = donning(80, 48, {"NEXT_TARGET": 153, "CONFIRM": 114,
                                "ABORT": 191})

    hint = diagnose_failure(run(measured))

    assert "re-place the electrodes" in hint
    assert "resting level is high" in hint


def test_one_weak_gesture_points_at_that_muscle_not_the_electrodes():
    """The failure that used to print the opposite of the right advice.

    2026-08-23, after an evening of trials: rest identical to the passing
    capture at 15, CONFIRM and ULNAR *higher* than before, and only wrist
    extension down 36%. Re-placing electrodes on that reading is how a
    working donning gets thrown away.
    """
    measured = donning(15, 47, {"NEXT_TARGET": 112, "CONFIRM": 180,
                                "ABORT": 141})

    summary = run(measured)
    hint = diagnose_failure(summary)

    assert summary["verdict"] == "fail"
    assert "NEXT_TARGET" in hint
    assert "fatigued" in hint
    assert "re-place" not in hint.replace("re-placing working electrodes", "")


def test_bands_close_with_nothing_else_odd_points_at_preparation():
    # Rest normal, all three gestures within a quarter of each other: the
    # only thing left holding the bands together is preparation.
    measured = donning(15, 70, {"NEXT_TARGET": 160, "CONFIRM": 155,
                                "ABORT": 150})

    hint = diagnose_failure(run(measured))

    assert "preparation" in hint
    assert "smaller movements" in hint


def test_the_verdict_line_carries_the_diagnosis():
    measured = donning(15, 47, {"NEXT_TARGET": 112, "CONFIRM": 180,
                                "ABORT": 141})

    message = verdict_message(run(measured))

    assert message.startswith("FAIL")
    assert "NEXT_TARGET" in message


def test_confirmation_covers_the_references_not_only_the_threshold():
    """They were measured, written to the file, and then not sent.

    The tool printed "board confirmed" -- true of everything it compared, and
    false of the thing the measurement existed for. Caught on hardware
    2026-08-25: a passing calibration left the board reporting
    references 0/0.
    """
    summary = {"factor": 3, "baseline_shift": 4, "threshold_floor": 52,
               "reference_levels": {"NEXT_TARGET": 184.4, "ULNAR": 165.0}}

    assert sendable_references(summary) == (184, 165)

    confirm_applied(state(threshold_floor=52, reference_left=184,
                          reference_right=165), summary)
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(threshold_floor=52), summary)
    with pytest.raises(CalibrationError, match="expected"):
        confirm_applied(state(threshold_floor=52, reference_left=184),
                        summary)


def test_a_summary_without_references_sends_and_confirms_zero():
    # "Use the firmware fallback", not an error: older files stay usable.
    summary = {"factor": 3, "baseline_shift": 4, "threshold_floor": 52}

    assert sendable_references(summary) == (0, 0)
    confirm_applied(state(threshold_floor=52), summary)


def test_a_weak_gesture_is_reported_even_when_separation_passes():
    """Tonight's failure, and the reason it cost a session.

    2026-08-25: separation 3.24, a clean pass, with NEXT_TARGET at 94 against
    CONFIRM's 206. Three deliberate wrist extensions in the run that followed
    produced zero gate events. weakest_over_strongest was 0.46 -- computed,
    stored in the file, and printed nowhere, because the only thing that
    looked at it ran on failure.
    """
    measured = donning(21, 29, {"NEXT_TARGET": 94, "CONFIRM": 206,
                                "ABORT": 131})

    summary = run(measured)

    assert summary["verdict"] == "pass"
    assert summary["weakest_over_strongest"] == pytest.approx(0.46, abs=0.01)
    warning = summary["weak_gesture_warning"]
    assert "NEXT_TARGET" in warning and "94" in warning and "206" in warning


def test_gestures_of_similar_strength_raise_nothing():
    # The passing donning from two days earlier: 174 / 162 / 170.
    measured = donning(15, 19, {"NEXT_TARGET": 174, "CONFIRM": 162,
                                "ABORT": 170})

    summary = run(measured)

    assert summary["verdict"] == "pass"
    assert "weak_gesture_warning" not in summary
