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
GESTURE_PROMPTS = {
    "NEXT_TARGET": "lift your wrist upward and hold",
    "CONFIRM": "make a fist and hold",
    "ABORT": "press your wrist downward and hold",
}

# Three tiers rather than one threshold: "recalibrate" and "re-place the
# electrodes" are different instructions and conflating them wastes the
# wearer's time on whichever one is wrong.
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
# movement too short to occupy the gate's stable run cannot fire an event at
# any threshold, so it does not need suppressing and should not drag the
# threshold down.
SUSTAIN_WINDOWS = VALIDATED_GATE.stable_windows


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
        plateaus[name] = min(sustained_level(trial) for trial in trials)
    weakest_name = min(plateaus, key=plateaus.get)
    weakest = plateaus[weakest_name]

    if preparation_upper <= 0.0:
        raise CalibrationError("preparation measured zero; check the electrodes")
    separation = weakest / preparation_upper
    threshold = int(round(math.sqrt(preparation_upper * weakest)))

    if separation >= SEPARATION_PASS:
        verdict = "pass"
    elif separation >= SEPARATION_MARGINAL:
        verdict = "marginal"
    else:
        verdict = "fail"

    summary = {
        "rest_baseline": round(rest_baseline, 1),
        "preparation_upper": round(preparation_upper, 1),
        "weakest_gesture": weakest_name,
        "weakest_gesture_plateau": round(weakest, 1),
        "gesture_plateaus": {
            name: round(value, 1) for name, value in sorted(plateaus.items())
        },
        "separation_ratio": round(separation, 2),
        "verdict": verdict,
        "threshold_floor": threshold,
        "factor": FROZEN_FACTOR,
        "baseline_shift": FROZEN_BASELINE_SHIFT,
        "sustain_windows": SUSTAIN_WINDOWS,
    }
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


def verdict_message(summary):
    separation = summary["separation_ratio"]
    if summary["verdict"] == "pass":
        return f"PASS: separation {separation} (>= {SEPARATION_PASS})"
    if summary["verdict"] == "marginal":
        return (
            f"MARGINAL: separation {separation} is between "
            f"{SEPARATION_MARGINAL} and {SEPARATION_PASS}. Repeat the "
            f"calibration; if it stays here, re-place the electrodes."
        )
    return (
        f"FAIL: separation {separation} is below {SEPARATION_MARGINAL}. "
        f"Check electrode contact and gel, then calibrate again. A threshold "
        f"fitted to this donning would sit within noise of real gestures."
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
    for name in GESTURES:
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
    print(f"  T_session               {summary['threshold_floor']}")
    print(f"  K (fixed)               {summary['factor']}")
    print("=" * 58)
    if "inert_floor_warning" in summary:
        print(f"  WARNING: {summary['inert_floor_warning']}")
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
