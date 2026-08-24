#!/usr/bin/env python3
"""Measure one donning's activation threshold and send it to the board.

The compile-time constants cannot serve every donning. `K x rest baseline`
was falsified on 2026-08-15: re-gelling one noisy electrode dropped resting
total MAV from 32 to a drifting 6-43 while the band the threshold must sit
in barely moved (preparation 73, weakest deliberate gesture 145). Rest
amplitude is a contact-noise figure and the preparation/gesture band is set
by physiology and electrode placement, so no single multiple of rest places
a threshold between them across donnings.

This measures both bounds directly, in the same session, on the same
electrodes:

    rest        10 s of stillness                -> the noise floor
    preparation small non-command movements      -> what must be suppressed
    gestures    three repetitions of each command -> what must get through

Every band is measured as the level a trial *sustained* over the same number
of consecutive windows the event gate needs before it emits anything, so the
numbers describe what the gate can actually act on rather than a statistic
that depends on how promptly the wearer reacted to a prompt.

`T_session` is the geometric mean of the preparation upper bound and the
weakest gesture's plateau. Geometric because the quantity is
ratio-structured: the arithmetic mean skews toward the louder bound exactly
when separation is thin, which is the case that matters.

The separation ratio is itself the acceptance criterion, in three tiers. A
donning that cannot separate the two bands does not get a cleverer
threshold, it gets re-placed electrodes - today's 2.0 fails where
yesterday's 4.0 passes outright.

The tiers are provisional as of 2026-08-19. They were fitted against a
five-window sustained level, and the measurement has since been corrected to
the seventeen-window run the gate actually requires, which lowers every ratio:
the two donnings recorded on 2026-08-18 move from 4.36 and 3.30 to 3.16 and
2.38, so one of them now "fails" a tier it was never calibrated for while its
gestures fire with 46-78 windows of margin. Re-deriving the tiers needs more
donnings than exist. Until then the verdict stays fail-closed on the old
numbers and `gate_reachability` is reported beside it, so a marginal or
failing result can be inspected for what the gate would actually do rather
than only what the ratio says.

    emg_calibrate.py                      # guided capture, then send
    emg_calibrate.py --dry-run            # capture and report, send nothing
    emg_calibrate.py --send-file PATH     # re-send a stored calibration
    emg_calibrate.py --defaults           # un-calibrate the board

Why preparation is prompted rather than taken from gesture onsets: a
gesture's own ramp is a smaller copy of that gesture, not a sample of the
different low-amplitude class that caused the original defect (an
unconscious wrist extension before a fist). Measuring the ramp would
calibrate against the wrong thing.
"""

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

from emg_activation_ref import FROZEN_BASELINE_SHIFT, FROZEN_FACTOR
from emg_event_gate_replay import VALIDATED_GATE
from emg_features_ref import HOP, WINDOW, mean_absolute_value
from emg_filter_ref import design_emg_filter, filter_fixed, to_fixed
from emg_protocol import (
    RAW_HEADER_SIZE,
    SET_MODE_APPLY,
    SET_MODE_DEFAULTS,
    SET_RESULT_ACCEPTED,
    TYPE_ACTIVATION_STATE,
    TYPE_INFO,
    TYPE_RAW,
    PacketParser,
    decode_activation_state,
    decode_info,
    decode_raw,
    encode_set_activation,
)

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_STORE = pathlib.Path("datasets/emg_calibration")

REST_SECONDS = 10.0
PREPARATION_SECONDS = 4.0
# Four seconds, not three: a wearer needs a second or so to react to the
# prompt and ramp up, and a three-second trial left too little held signal
# behind it. The sustained-level measure below tolerates a late start, but
# it still needs the hold to actually happen inside the window.
GESTURE_SECONDS = 4.0
PREPARATION_TRIALS = 3
GESTURE_TRIALS = 3
GESTURES = ("NEXT_TARGET", "CONFIRM", "ABORT")

# The two gestures that steer the proportional view axis. NEXT_TARGET is in
# both lists: it is a gate command and it is also LEFT. ULNAR is here only --
# it is a classifier class that never reaches the event gate, so it must not
# be allowed to set T_session, and its gate reachability would mean nothing.
REFERENCE_GESTURES = ("NEXT_TARGET", "ULNAR")
CAPTURED_GESTURES = GESTURES + tuple(
    name for name in REFERENCE_GESTURES if name not in GESTURES
)

GESTURE_PROMPTS = {
    "NEXT_TARGET": "lift your wrist upward and hold",
    "CONFIRM": "make a fist and hold",
    "ABORT": "press your wrist downward and hold",
    "ULNAR": "tilt your wrist toward the little-finger side and hold",
}

# The level that maps to full deflection, per direction gesture.
#
# It cannot be derived from the threshold, which is what the firmware does
# today with a compile-time reference = 3 x threshold. Measured 2026-08-23
# across two donnings: the cleaner one calibrated to a *lower* threshold, 47
# against 59, which lowered the derived reference from 177 to 141 while the
# wearer's NEXT_TARGET amplitude did not move (146, then 152). The result is
# perverse -- better electrode contact produced worse control -- and it is not
# marginal: 58% of that session's LEFT commands sat pinned at full deflection,
# carrying no proportional information at all.
#
# Two references rather than one, because the two gestures are not equally
# strong and the asymmetry is systematic. Wrist extension ran 1.35x and 1.24x
# above ulnar deviation on two separate donnings, same direction both times. A
# single shared reference has to sacrifice one side: set to the stronger, the
# weaker cannot reach full speed; set to the weaker, the stronger saturates
# over its top quarter. Saturation costs resolution, which is the entire point
# of proportional control, so neither compromise is acceptable when the fix is
# one more field in a payload that has to grow anyway.
#
# Instantaneous, not sustained. activation is computed per window from that
# window's total MAV, so the reference has to be a statistic of the same
# thing. A 17-window sustained level is a run *minimum*, far below what the
# window-by-window signal reaches, and setting the reference from one was
# tried: the board saturated through most of every hold.
REFERENCE_PERCENTILE = 90.0

# Three tiers rather than one threshold: "recalibrate" and "re-place the
# electrodes" are different instructions and conflating them wastes the
# wearer's time on whichever one is wrong.
#
# These no longer decide the verdict; see gate_reachability below. They were
# fitted against a five-window sustained level, and once the measurement was
# corrected to the run the gate actually requires, the same recordings that
# fire every gesture cleanly score 2.38 and 3.16 -- one of them "failing" a
# tier it was never calibrated for. They stay as a reported placement-quality
# figure, which is what the wording above was always about.
SEPARATION_PASS = 3.0
SEPARATION_MARGINAL = 2.5

# Both bands are measured as the level a trial *sustained*, over the same
# number of consecutive windows the event gate requires before it will emit
# anything. That number is imported rather than written here so the two
# cannot drift apart.
#
# Percentiles were tried first and were wrong in a way that looked like bad
# hardware. A fixed percentile over a whole trial measures the plateau only
# when the gesture fills most of it; with a second of reaction time before a
# late start, the top quartile of windows lands on the onset ramp instead. A
# real ABORT sustaining about 536 total MAV was reported as 98 that way,
# while preparation - brief, so its own peak sat comfortably inside a 95th
# percentile - was reported near its true value. The two errors pushed the
# separation ratio in opposite directions and failed three calibrations in a
# row on hardware that was working.
#
# The sustained-level rule also states the right thing about preparation: a
# movement too short to occupy the gate's run cannot fire an event at any
# threshold, so it does not need suppressing and should not drag the
# threshold down.
#
# The run is holdoff + stable, not stable alone. The comment above always
# said "the number of consecutive windows the event gate requires before it
# will emit anything" and the code used five, but the gate discards
# onset_holdoff windows first: the first non-REST decision arms the hold-off,
# eleven more are consumed by it, and only then does the stable run begin. A
# level held for five windows and a level held for seventeen are different
# numbers, and fitting a threshold to the first places it about twice too
# high. Measured 2026-08-18: thresholds of 123 and 116 were shipped where the
# corrected rule gives 68 and 69, and at the shipped values even the four
# trained gestures fired in only 0-4 of 6 recorded trials.
#
# ABORT is its own number. abort_stable_windows is 1, so it needs
# holdoff + 1 = 13, and it is often the weakest gesture -- the one that sets
# the threshold. Measuring it at 17 understates it: on donning A it moved the
# weakest gesture from ABORT at 99 to CONFIRM at 120, and the threshold from
# 61 to 68.
SUSTAIN_WINDOWS = (
    VALIDATED_GATE.onset_holdoff_windows + VALIDATED_GATE.stable_windows
)
ABORT_SUSTAIN_WINDOWS = (
    VALIDATED_GATE.onset_holdoff_windows + VALIDATED_GATE.abort_stable_windows
)
GESTURE_SUSTAIN_WINDOWS = {"ABORT": ABORT_SUSTAIN_WINDOWS}


class CalibrationError(Exception):
    """A calibration that must not be sent to the board."""


def total_mav_windows(samples, channel_count):
    """Summed per-channel MAV per feature window, as the firmware computes it.

    Same filter, same 200 ms / 50 ms grid, same summation over channels, so
    the numbers this produces are directly comparable with the thresholds
    the firmware applies to them.
    """
    if samples.ndim != 2 or samples.shape[1] != channel_count:
        raise CalibrationError("sample matrix does not match the channel count")
    if len(samples) < WINDOW:
        raise CalibrationError(
            f"need at least {WINDOW} frames, got {len(samples)}"
        )
    sections = to_fixed(design_emg_filter())
    filtered = [
        filter_fixed(
            sections,
            np.clip(samples[:, channel], -32768, 32767).astype(np.int16),
        )
        for channel in range(channel_count)
    ]
    totals = []
    for end in range(WINDOW, len(samples) + 1, HOP):
        totals.append(sum(
            mean_absolute_value(column[end - WINDOW:end])
            for column in filtered
        ))
    return np.asarray(totals, dtype=np.int64)


def sustained_level(totals, windows=SUSTAIN_WINDOWS):
    """The highest level held for at least `windows` consecutive windows.

    Equivalently: the largest V for which some run of that many windows is
    entirely at or above V. Insensitive to when in the trial the hold
    happened, and to how much rest surrounds it, which is exactly what a
    fixed percentile is not.
    """
    totals = np.asarray(totals)
    if windows <= 0:
        raise CalibrationError("sustain window count must be positive")
    if len(totals) < windows:
        raise CalibrationError(
            f"trial has {len(totals)} windows, need at least {windows}"
        )
    return float(max(
        totals[start:start + windows].min()
        for start in range(len(totals) - windows + 1)
    ))


def instantaneous_level(totals, percentile=REFERENCE_PERCENTILE):
    """The level this trial's windows actually reach, as a high percentile.

    The maximum would be one window of a three-second hold and is hostage to
    a single artifact; a percentile describes the ceiling the wearer held.
    This is the statistic the frozen 3 x threshold reference was originally
    fitted against, so measuring it directly replaces a constant with the
    quantity it was standing in for.
    """
    totals = np.asarray(totals)
    if len(totals) == 0:
        raise CalibrationError("trial has no windows")
    return float(np.percentile(totals, percentile))


def reference_levels(gesture_totals):
    """Full-deflection level per direction gesture, or None if not captured.

    The median across trials, where the threshold takes each gesture's
    weakest repetition. The two want opposite statistics for the same reason:
    a threshold has to pass even the feeblest attempt, while a reference has
    to sit where ordinary effort lands. Taking the minimum here would put a
    typical contraction above full deflection and leave the wearer
    permanently saturated -- which is the bug being fixed, arrived at by a
    different route.
    """
    levels = {}
    for name in REFERENCE_GESTURES:
        trials = gesture_totals.get(name)
        if not trials:
            continue
        levels[name] = float(np.median(
            [instantaneous_level(trial) for trial in trials]
        ))
    return levels


def gesture_sustain_windows(name):
    """How long this gesture must hold above the threshold to fire an event."""
    return GESTURE_SUSTAIN_WINDOWS.get(name, SUSTAIN_WINDOWS)


def longest_run_above(totals, threshold):
    """The longest unbroken stretch of windows at or above a threshold.

    This is what the gate sees. A window below the threshold is rewritten to
    REST, which clears the hold-off and the stable run both, so a gesture that
    dips even once starts the count again from zero.
    """
    best = run = 0
    for value in totals:
        run = run + 1 if value >= threshold else 0
        best = max(best, run)
    return best


def gate_reachability(preparation_trials, gesture_totals, threshold):
    """Check the threshold against what it has to achieve, on both sides.

    The separation ratio is a proxy. What the threshold must actually do is
    two directly observable things, in the same windows already captured:
    preparation must not hold the gate's run above it, and every gesture must.
    The two disagree in practice -- a donning scoring 2.38 fired every gesture
    with 46-78 windows of margin while its preparation reached 2 -- so the
    verdict follows the measurement and the ratio is reported beside it.
    """
    preparation_runs = [
        longest_run_above(trial, threshold) for trial in preparation_trials
    ]
    gestures = {}
    for name, trials in sorted(gesture_totals.items()):
        needed = gesture_sustain_windows(name)
        runs = [longest_run_above(trial, threshold) for trial in trials]
        gestures[name] = {
            "runs": runs,
            "windows_needed": needed,
            "trials_firing": sum(1 for run in runs if run >= needed),
            "trials": len(runs),
        }
    return {
        "preparation_runs": preparation_runs,
        "preparation_windows_needed": SUSTAIN_WINDOWS,
        "preparation_can_fire": max(preparation_runs) >= SUSTAIN_WINDOWS,
        "gestures": gestures,
        "all_gestures_fire": all(
            item["trials_firing"] == item["trials"] for item in gestures.values()
        ),
        "any_gesture_dead": any(
            item["trials_firing"] == 0 for item in gestures.values()
        ),
    }


def summarize(rest_totals, preparation_trials, gesture_totals):
    """Turn measured windows into a threshold and a verdict.

    Pure arithmetic on already-collected numbers, so the decision rule is
    testable without a board, a wearer, or a serial port.

    `preparation_trials` and each gesture's trials are lists of per-trial
    window arrays. Preparation takes the loudest sustained level across its
    trials (the worst case that must be suppressed) and each gesture the
    quietest (the weakest repetition that must still get through).

    """
    if len(rest_totals) == 0 or not preparation_trials:
        raise CalibrationError("rest and preparation captures cannot be empty")
    if not gesture_totals:
        raise CalibrationError("at least one gesture must be captured")

    # Rest is reported for context and for the inert-floor check below, and
    # as a median rather than a tail statistic so it is comparable with the
    # firmware's own EMA baseline over classified-REST windows.
    rest_baseline = float(np.median(rest_totals))
    preparation_upper = max(
        sustained_level(trial) for trial in preparation_trials
    )
    plateaus = {}
    for name, trials in gesture_totals.items():
        if not trials:
            raise CalibrationError(f"gesture {name} has no trials")
        # Direction-only classes are captured for their reference level and
        # nothing else. ULNAR never reaches the event gate, so letting it
        # into the threshold would let a class the gate cannot act on decide
        # what the gate suppresses.
        if name not in GESTURES:
            continue
        windows = gesture_sustain_windows(name)
        plateaus[name] = min(
            sustained_level(trial, windows) for trial in trials
        )
    if not plateaus:
        raise CalibrationError("no gate gesture was captured")
    weakest_name = min(plateaus, key=plateaus.get)
    weakest = plateaus[weakest_name]

    if preparation_upper <= 0.0:
        raise CalibrationError("preparation measured zero; check the electrodes")
    separation = weakest / preparation_upper
    threshold = int(round(math.sqrt(preparation_upper * weakest)))

    # The ratio, not the reachability below, decides. Reachability against a
    # threshold derived from the same windows is vacuous: the threshold is the
    # geometric mean of preparation and the weakest gesture, so it always sits
    # between them, and "sustained_level < T" already means no run of that
    # length clears T. Both sides pass by construction. It is informative only
    # when the threshold comes from somewhere else -- re-verifying a stored
    # calibration, or the 2026-08-18 finding, where a threshold fitted to a
    # five-window level was checked against the seventeen the gate needs.
    #
    # What the ratio measures and reachability cannot is margin: three
    # preparation trials are a small sample, and the question is whether a
    # fourth, louder one would cross.
    if separation >= SEPARATION_PASS:
        verdict = "pass"
    elif separation >= SEPARATION_MARGINAL:
        verdict = "marginal"
    else:
        verdict = "fail"
    reachability = gate_reachability(
        preparation_trials,
        {n: t for n, t in gesture_totals.items() if n in GESTURES},
        threshold,
    )
    references = reference_levels(gesture_totals)

    summary = {
        "rest_baseline": round(rest_baseline, 1),
        "preparation_upper": round(preparation_upper, 1),
        "weakest_gesture": weakest_name,
        "weakest_gesture_plateau": round(weakest, 1),
        "gesture_plateaus": {
            name: round(value, 1) for name, value in sorted(plateaus.items())
        },
        "separation_ratio": round(separation, 2),
        "gate_reachability": reachability,
        "verdict": verdict,
        "threshold_floor": threshold,
        "weakest_over_strongest": (
            round(weakest / max(plateaus.values()), 2)
            if max(plateaus.values()) > 0 else None
        ),
        "factor": FROZEN_FACTOR,
        "baseline_shift": FROZEN_BASELINE_SHIFT,
        "sustain_windows": SUSTAIN_WINDOWS,
        "reference_levels": {
            name: round(value, 1) for name, value in sorted(references.items())
        },
        "reference_percentile": REFERENCE_PERCENTILE,
        # Recorded per donning so the question the frozen constant answered
        # badly -- is reference / threshold stable? -- can be settled from
        # data instead of assumed. It was 3 in the firmware; measured values
        # so far disagree with each other by more than that constant's own
        # margin.
        "reference_over_threshold": {
            name: round(value / threshold, 2)
            for name, value in sorted(references.items())
        } if threshold > 0 else {},
    }
    unusable = sorted(
        name for name, value in references.items() if value <= threshold
    )
    if unusable:
        summary["reference_warning"] = (
            f"reference level for {', '.join(unusable)} is at or below "
            f"T_session ({threshold}); proportional control maps the span "
            "above the threshold, so that span is empty. Re-run the trials "
            "for that gesture rather than trusting the number."
        )
    missing = [n for n in REFERENCE_GESTURES if n not in references]
    if missing:
        summary["reference_warning"] = (
            f"no reference measured for {', '.join(missing)}; the firmware "
            "will fall back to its compile-time reference for that direction"
        )
    # Stillness cannot be noisier than deliberate small movement. When it
    # measures that way the rest capture contains something that is not rest,
    # and every number downstream of it is describing the wrong thing --
    # including the K x baseline rule the firmware will actually judge with.
    #
    # Caught after the fact on 2026-08-23: a rest trial with a gesture in it
    # reported baseline 80.5 against a preparation bound of 48.0, and the
    # inconsistency sat in the output unremarked while the cause was looked
    # for in the hardware. The comparison is free and the two numbers were
    # already side by side.
    #
    # A warning rather than a verdict, because the separation ratio it would
    # override is measuring electrode placement, which a botched rest trial
    # says nothing about.
    if rest_baseline > preparation_upper:
        summary["rest_contamination_warning"] = (
            f"rest baseline ({rest_baseline:.0f}) is above the preparation "
            f"bound ({preparation_upper:.0f}); stillness cannot be louder "
            "than deliberate movement, so the rest capture probably contains "
            "movement. Re-run rather than trusting the threshold."
        )

    # The firmware judges on max(K x baseline, floor). If the relative rule
    # already sits above the calibrated floor, the calibration is inert and
    # says nothing about what the board will actually do - worth stating
    # rather than leaving to be rediscovered from a surprising threshold.
    relative = FROZEN_FACTOR * rest_baseline
    if relative > threshold:
        summary["inert_floor_warning"] = (
            f"K x rest baseline ({FROZEN_FACTOR} x {rest_baseline:.0f} = "
            f"{relative:.0f}) already exceeds T_session ({threshold}); the "
            f"relative rule, not this calibration, will govern"
        )
    return summary


# One gesture far below its peers is a statement about that muscle or that
# electrode; all of them close together is a statement about the donning.
# Provisional, like the separation tiers, and for the same reason -- it is cut
# from three captures. 2026-08-23: two donnings that failed for unrelated
# reasons both had a weakest/strongest of 0.93, and the fatigued one 0.62.
WEAK_GESTURE_FRACTION = 0.75


def diagnose_failure(summary):
    """Why the bands are close, when they are. None when they are not.

    The separation ratio says a donning failed; it does not say what to do,
    and the two remedies are opposite. On 2026-08-23 one failure was electrode
    motion -- rest quadrupled, gestures unchanged -- and another was fatigue in
    one muscle after an evening of trials, with rest identical to the passing
    capture. Both printed "Check electrode contact and gel". Re-placing good
    electrodes is how a working donning gets thrown away, and the numbers to
    tell the cases apart were already side by side in the output.
    """
    plateaus = summary["gesture_plateaus"]
    weakest = summary["weakest_gesture_plateau"]
    strongest = max(plateaus.values()) if plateaus else 0.0

    if "inert_floor_warning" in summary:
        # The noise floor is high enough to govern the threshold on its own.
        # Whatever the gestures do, the electrodes are what to fix.
        return (
            "the resting level is high enough that K x baseline, not this "
            "calibration, would set the threshold. That is a contact or "
            "interference problem: re-place the electrodes, and check that "
            "nothing mains-powered is connected."
        )
    if strongest > 0.0 and weakest / strongest < WEAK_GESTURE_FRACTION:
        return (
            f"the resting level is normal and the other gestures are not "
            f"weak, so this is specific to {summary['weakest_gesture']} "
            f"({weakest:.0f} against {strongest:.0f}). Either that muscle is "
            "fatigued -- rest it rather than re-placing working electrodes -- "
            "or the electrode over it is the one that moved."
        )
    return (
        "the resting level is normal and no single gesture stands out, so "
        "the bands are close from the preparation side. Repeat it with "
        "smaller movements before concluding anything about the electrodes."
    )


def verdict_message(summary):
    separation = summary["separation_ratio"]
    if summary["verdict"] == "pass":
        return f"PASS: separation {separation} (>= {SEPARATION_PASS})"
    if summary["verdict"] == "marginal":
        return (
            f"MARGINAL: separation {separation} is between "
            f"{SEPARATION_MARGINAL} and {SEPARATION_PASS}. A threshold fitted "
            f"here sits close to noise. Repeat the calibration; "
            f"{diagnose_failure(summary)}"
        )
    return (
        f"FAIL: separation {separation} is below {SEPARATION_MARGINAL}. A "
        f"threshold fitted to this donning would sit within noise of real "
        f"gestures. Looking at where the bands actually are: "
        f"{diagnose_failure(summary)}"
    )


class Capture:
    """Collect RAW frames from the port for fixed prompted intervals."""

    def __init__(self, connection):
        self._connection = connection
        self._parser = PacketParser()
        self.info = None

    def _pump(self, seconds, collect):
        deadline = time.perf_counter() + seconds
        frames = []
        while time.perf_counter() < deadline:
            chunk = self._connection.read(4096)
            if not chunk:
                continue
            for packet in self._parser.feed(chunk):
                if packet.type == TYPE_INFO and self.info is None:
                    self.info = decode_info(packet.payload)
                elif packet.type == TYPE_RAW and collect and self.info:
                    block = decode_raw(packet.payload, self.info.channel_count)
                    if block.all_attached:
                        frames.extend(block.frames)
        return frames

    def wait_for_info(self, seconds=5.0):
        self._pump(seconds if self.info is None else 0.0, collect=False)
        if self.info is None:
            raise CalibrationError(
                "no INFO packet: wrong port, or the firmware is not streaming"
            )
        return self.info

    def prompt(self, label, seconds):
        """Count in, capture, and return the window totals for one interval."""
        print(f"\n  {label}")
        for remaining in (3, 2, 1):
            print(f"    starting in {remaining}...", flush=True)
            time.sleep(1.0)
        print(f"    GO ({seconds:.0f} s)", flush=True)
        frames = self._pump(seconds, collect=True)
        print("    done", flush=True)
        if len(frames) < WINDOW:
            raise CalibrationError(
                f"only {len(frames)} fully-attached frames captured during "
                f"'{label}'; an electrode is loose"
            )
        return total_mav_windows(
            np.asarray(frames, dtype=np.int64), self.info.channel_count
        )


def send_calibration(connection, *, mode, factor, baseline_shift,
                     threshold_floor, sequence=1, timeout_sec=4.0):
    """Send one request and return the ACTIVATION_STATE the board reports.

    Waits for the state rather than assuming success: the reply is the only
    evidence the board applied anything, and a rejected request looks
    identical on the wire until it comes back.
    """
    connection.write(encode_set_activation(
        sequence, mode=mode, factor=factor, baseline_shift=baseline_shift,
        threshold_floor=threshold_floor,
    ))
    parser = PacketParser()
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        chunk = connection.read(4096)
        if not chunk:
            continue
        for packet in parser.feed(chunk):
            if packet.type != TYPE_ACTIVATION_STATE:
                continue
            state = decode_activation_state(packet.payload)
            if state.applied_sequence == sequence or state.last_result != 0:
                return state
    raise CalibrationError(
        "no ACTIVATION_STATE within the timeout; the board did not confirm"
    )


def confirm_applied(state, summary):
    """Check the board is judging with what was just sent."""
    if state.last_result != SET_RESULT_ACCEPTED:
        raise CalibrationError(
            f"board rejected the calibration (last_result={state.last_result})"
        )
    applied = (state.factor, state.baseline_shift, state.threshold_floor)
    wanted = (summary["factor"], summary["baseline_shift"],
              summary["threshold_floor"])
    if applied != wanted:
        raise CalibrationError(
            f"board reports {applied}, expected {wanted}"
        )


def store_path(directory, started):
    stamp = started.strftime("%Y%m%d_%H%M%S")
    return pathlib.Path(directory) / f"calibration_{stamp}.json"


def run_capture(connection):
    capture = Capture(connection)
    info = capture.wait_for_info()
    print(f"  board: {info.channel_count} channels at {info.sample_rate_hz} Hz")

    rest = capture.prompt(
        f"REST - sit still, arm relaxed ({REST_SECONDS:.0f} s)", REST_SECONDS
    )
    preparation = []
    for trial in range(1, PREPARATION_TRIALS + 1):
        preparation.append(capture.prompt(
            f"PREPARATION {trial}/{PREPARATION_TRIALS} - shift your wrist "
            f"slightly, as if adjusting position, NOT a command",
            PREPARATION_SECONDS,
        ))
    gestures = {}
    for name in CAPTURED_GESTURES:
        trials = []
        for trial in range(1, GESTURE_TRIALS + 1):
            trials.append(capture.prompt(
                f"{name} {trial}/{GESTURE_TRIALS} - "
                f"{GESTURE_PROMPTS[name]}",
                GESTURE_SECONDS,
            ))
        gestures[name] = trials
    return rest, preparation, gestures


def print_summary(summary):
    print("\n" + "=" * 58)
    print(f"  (levels are sustained over {summary['sustain_windows']} "
          f"consecutive windows, as the event gate requires)")
    print(f"  rest baseline           {summary['rest_baseline']}")
    print(f"  preparation upper bound {summary['preparation_upper']}")
    for name, value in summary["gesture_plateaus"].items():
        print(f"  {name:<23} {value}")
    print(f"  weakest gesture         {summary['weakest_gesture']} "
          f"({summary['weakest_gesture_plateau']})")
    print(f"  separation ratio        {summary['separation_ratio']}")
    for name, value in summary["reference_levels"].items():
        ratio = summary["reference_over_threshold"].get(name)
        suffix = f"  ({ratio}x T_session)" if ratio is not None else ""
        print(f"  reference {name:<13} {value}{suffix}")
    print(f"  T_session               {summary['threshold_floor']}")
    print(f"  K (fixed)               {summary['factor']}")
    reach = summary["gate_reachability"]
    print(f"  gate needs a run of     {reach['preparation_windows_needed']} "
          f"windows above T_session")
    print(f"  preparation reached     {reach['preparation_runs']} "
          f"-> {'CAN FIRE' if reach['preparation_can_fire'] else 'cannot fire'}")
    for name, item in reach["gestures"].items():
        print(f"  {name:<23} {item['runs']} "
              f"(needs {item['windows_needed']}) -> "
              f"{item['trials_firing']}/{item['trials']} fire")
    print("=" * 58)
    if "rest_contamination_warning" in summary:
        print(f"  WARNING: {summary['rest_contamination_warning']}")
    if "inert_floor_warning" in summary:
        print(f"  WARNING: {summary['inert_floor_warning']}")
    if "reference_warning" in summary:
        print(f"  WARNING: {summary['reference_warning']}")
    print(f"  {verdict_message(summary)}")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--store", default=str(DEFAULT_STORE),
                        help="directory for calibration JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and report, send nothing")
    parser.add_argument("--send-file", metavar="PATH",
                        help="re-send a stored calibration without measuring")
    parser.add_argument("--defaults", action="store_true",
                        help="tell the board to discard any calibration")
    parser.add_argument("--allow-marginal", action="store_true",
                        help="send a marginal result; a failing one is never sent")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    import serial  # Imported late so the pure logic is testable without it.

    try:
        connection = serial.Serial(arguments.port, 115200, timeout=0.2)
    except OSError as error:
        print(f"Could not open {arguments.port}: {error}")
        return 1

    with connection:
        try:
            if arguments.defaults:
                state = send_calibration(
                    connection, mode=SET_MODE_DEFAULTS, factor=FROZEN_FACTOR,
                    baseline_shift=FROZEN_BASELINE_SHIFT, threshold_floor=110,
                )
                print(f"  board reset to defaults: K={state.factor} "
                      f"shift={state.baseline_shift} "
                      f"floor={state.threshold_floor}")
                return 0

            if arguments.send_file:
                summary = json.loads(
                    pathlib.Path(arguments.send_file).read_text()
                )
            else:
                connection.reset_input_buffer()
                # Un-calibrate first: measuring through a previous session's
                # threshold is fine (this reads RAW, not decisions), but
                # leaving a stale calibration in place if the capture aborts
                # is not.
                send_calibration(
                    connection, mode=SET_MODE_DEFAULTS, factor=FROZEN_FACTOR,
                    baseline_shift=FROZEN_BASELINE_SHIFT, threshold_floor=110,
                )
                started = datetime.datetime.now().astimezone()
                rest, preparation, gestures = run_capture(connection)
                summary = summarize(rest, preparation, gestures)
                summary["started"] = started.isoformat()
                summary["port"] = arguments.port
                # Keep the windows the verdict was computed from. The first
                # version stored only the summary, and when a result was
                # disputed there was nothing left to re-analyse - the bug
                # that produced it was found from a screenshot instead.
                summary["windows"] = {
                    "rest": [int(value) for value in rest],
                    "preparation": [
                        [int(value) for value in trial]
                        for trial in preparation
                    ],
                    "gestures": {
                        name: [
                            [int(value) for value in trial]
                            for trial in trials
                        ]
                        for name, trials in gestures.items()
                    },
                }
                print_summary(summary)

                directory = pathlib.Path(arguments.store)
                directory.mkdir(parents=True, exist_ok=True)
                path = store_path(directory, started)
                path.write_text(json.dumps(summary, indent=2,
                                           sort_keys=True) + "\n")
                print(f"  wrote {path}")

            if summary["verdict"] == "fail":
                print("  not sending a failing calibration")
                return 1
            if summary["verdict"] == "marginal" and not arguments.allow_marginal:
                print("  not sending a marginal calibration "
                      "(--allow-marginal to override)")
                return 1
            if arguments.dry_run:
                print("  dry run: nothing sent")
                return 0

            state = send_calibration(
                connection, mode=SET_MODE_APPLY, factor=summary["factor"],
                baseline_shift=summary["baseline_shift"],
                threshold_floor=summary["threshold_floor"],
            )
            confirm_applied(state, summary)
            print(f"  board confirmed: K={state.factor} "
                  f"shift={state.baseline_shift} "
                  f"floor={state.threshold_floor}")
            return 0
        except CalibrationError as error:
            print(f"  calibration aborted: {error}")
            # Leave the board in a known state rather than half-configured.
            try:
                send_calibration(
                    connection, mode=SET_MODE_DEFAULTS, factor=FROZEN_FACTOR,
                    baseline_shift=FROZEN_BASELINE_SHIFT, threshold_floor=110,
                    sequence=2,
                )
                print("  board returned to compile-time defaults")
            except CalibrationError:
                print("  WARNING: could not confirm the board's state")
            return 1


if __name__ == "__main__":
    sys.exit(main())
