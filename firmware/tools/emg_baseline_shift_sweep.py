#!/usr/bin/env python3
"""Offline sweep of the activation baseline shift over recorded sessions.

`EMG_ACTIVATION_BASELINE_SHIFT` is 4 for a weaker reason than its neighbour
`EMG_ACTIVATION_FACTOR` is 3. emg_activation.h says so plainly: K was chosen by
a mechanism -- at K = 3 an entire preparatory movement falls below threshold,
at K = 2 four of its windows cross and are stopped only by fragmenting into
runs of two -- while shift 4 "remains the middle of the swept range rather than
a measured value".

This replays recordings through the activation stage at each shift in 1..8,
holding the factor and floor fixed, and reports what actually changes. It runs
entirely offline against `datasets/`: no board, no wearer.

It answers the prior question first. The threshold is max(K x baseline, floor),
so on any window where the floor already exceeds K x baseline the shift cannot
move the threshold at all. If that is most windows, the honest answer to "why
4" is "it barely matters here", which is worth knowing before tuning it. The
`relative_pct` column is that fraction.

Deliberately no "best shift" is nominated. emg_activation.h records a
run-length margin being tried as a selector and discarded, because it jumps
non-monotonically and so measures threshold proximity rather than robustness.
This prints numbers; choosing is a separate act.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics

import numpy as np

from emg_activation_ref import (
    FROZEN_BASELINE_SHIFT,
    FROZEN_FACTOR,
    REST,
    THRESHOLD_FLOOR,
    ActivationGate,
)
from emg_event_gate_replay import (
    ACTIVE_LABELS,
    load_event_gate_timelines,
    load_timelines,
    prepare_external_validation_folds,
    prepare_loso_folds,
)

# Channel features are laid out (MAV, RMS, waveform length, zero crossings)
# per channel, so the three MAVs are every fourth column from zero. The board
# sums exactly these three; see main.c.
MAV_COLUMNS = (0, 4, 8)

DEFAULT_SHIFTS = tuple(range(1, 9))
# There is deliberately no "recovery after a gesture" metric. The baseline
# updates only on windows the classifier calls REST, so it is frozen for the
# whole of a contraction and has nothing to recover from -- an early version of
# this tool measured that and reported 0 windows at every shift, which is
# arithmetic rather than a result.


def total_mav(features: np.ndarray) -> np.ndarray:
    """Sum the three per-channel MAVs, as the firmware does per window."""

    if features.ndim != 2 or features.shape[1] <= max(MAV_COLUMNS):
        raise ValueError(f"unexpected feature shape {features.shape}")
    return features[:, list(MAV_COLUMNS)].sum(axis=1).astype(np.int64)


def replay_session(fold, *, factor: int, shift: int, floor: int) -> dict:
    """Run one session through the activation stage at one shift.

    Records, per window, the baseline the gate held *before* the call and the
    threshold that baseline implies, because that is the pair the decision was
    actually made against.
    """

    timeline = fold.timeline
    totals = total_mav(timeline.features)
    gate = ActivationGate(
        factor=factor, baseline_shift=shift, threshold_floor=floor
    )

    thresholds: list[int] = []
    relative_governs = 0
    decisions: list[str] = []
    settled: list[bool] = []

    for index, prediction in enumerate(fold.predictions):
        relative = gate.baseline * factor
        thresholds.append(max(relative, floor))
        relative_governs += int(relative > floor)
        # The first REST window initialises the accumulator exactly, so the
        # threshold steps from the floor to 3 x baseline in one window at
        # every shift alike. Counting it makes the largest step a constant
        # that says nothing about steady-state jitter.
        settled.append(gate.has_baseline)
        decisions.append(gate.apply(
            str(prediction),
            valid=bool(timeline.valid[index]),
            total_mav=int(totals[index]),
        ))

    windows = len(decisions)
    steps = [
        abs(thresholds[i] - thresholds[i - 1])
        for i in range(1, windows)
        if settled[i] and settled[i - 1]
    ]
    return {
        "session": timeline.session_id,
        "windows": windows,
        "relative_pct": 100.0 * relative_governs / windows if windows else 0.0,
        "threshold_std": statistics.pstdev(thresholds) if windows > 1 else 0.0,
        "threshold_max_step": max(steps) if steps else 0,
        "decisions": decisions,
        "passes": label_pass_counts(timeline, decisions),
    }


def label_pass_counts(timeline, decisions: list[str]) -> dict:
    """How many labelled windows the activation stage let through.

    Split because the two directions of error are not interchangeable: a
    gesture window suppressed is a command lost, a rest window passed is a
    false trigger waiting for the event gate to catch it.
    """

    gesture_total = gesture_passed = rest_total = rest_passed = 0
    for index, decision in enumerate(decisions):
        label = timeline.labels[index]
        if label is None or not timeline.valid[index]:
            continue
        if label in ACTIVE_LABELS:
            gesture_total += 1
            gesture_passed += int(decision != REST)
        elif label == REST:
            rest_total += 1
            rest_passed += int(decision != REST)
    return {
        "gesture_total": gesture_total,
        "gesture_passed": gesture_passed,
        "rest_total": rest_total,
        "rest_passed": rest_passed,
    }


def sweep(folds, *, factor: int, floor: int, shifts) -> list[dict]:
    """Replay every session at every shift, and diff against the shipped one."""

    per_shift = {
        shift: [
            replay_session(fold, factor=factor, shift=shift, floor=floor)
            for fold in folds
        ]
        for shift in shifts
    }
    reference = per_shift.get(FROZEN_BASELINE_SHIFT)

    rows = []
    for shift in shifts:
        sessions = per_shift[shift]
        windows = sum(item["windows"] for item in sessions)
        differing = 0
        if reference is not None and shift != FROZEN_BASELINE_SHIFT:
            for item, base in zip(sessions, reference):
                differing += sum(
                    1 for a, b in zip(item["decisions"], base["decisions"])
                    if a != b
                )
        rows.append({
            "shift": shift,
            "windows": windows,
            "relative_pct": _weighted(sessions, "relative_pct"),
            "threshold_std": _weighted(sessions, "threshold_std"),
            "threshold_max_step": max(
                item["threshold_max_step"] for item in sessions
            ),
            "differing_windows": differing,
            "gesture_passed": sum(
                item["passes"]["gesture_passed"] for item in sessions
            ),
            "gesture_total": sum(
                item["passes"]["gesture_total"] for item in sessions
            ),
            "rest_passed": sum(
                item["passes"]["rest_passed"] for item in sessions
            ),
            "rest_total": sum(
                item["passes"]["rest_total"] for item in sessions
            ),
        })
    return rows


def _weighted(sessions, key: str) -> float:
    windows = sum(item["windows"] for item in sessions)
    if windows == 0:
        return 0.0
    return sum(item[key] * item["windows"] for item in sessions) / windows


def format_rows(rows, *, factor: int, floor: int) -> str:
    lines = [
        f"factor {factor} and floor {floor} held fixed; "
        f"shipped shift is {FROZEN_BASELINE_SHIFT}",
        "",
        f"{'shift':>5} {'relative%':>10} {'T std':>8} {'T step':>7} "
        f"{'differs':>8} {'gesture':>12} {'rest pass':>10}",
    ]
    for row in rows:
        differs = "ref" if row["shift"] == FROZEN_BASELINE_SHIFT else str(
            row["differing_windows"]
        )
        lines.append(
            f"{row['shift']:>5} {row['relative_pct']:>9.1f}% "
            f"{row['threshold_std']:>8.1f} {row['threshold_max_step']:>7} "
            f"{differs:>8} "
            f"{row['gesture_passed']:>5}/{row['gesture_total']:<6} "
            f"{row['rest_passed']:>4}/{row['rest_total']:<5}"
        )

    lines.append("")
    governs = max(row["relative_pct"] for row in rows) if rows else 0.0
    if governs < 1.0:
        lines.append(
            "The floor governs every window at every shift: this recording "
            "cannot discriminate between shifts, and nothing below is "
            "evidence about which one to ship."
        )
    else:
        lines.append(
            "relative%  windows where K x baseline exceeds the floor, so the "
            "shift can move the threshold at all"
        )
        lines.append(
            "T std      spread of the threshold; T step  its largest "
            "window-to-window jump"
        )
        lines.append(
            "           both measured only after the baseline exists, so the "
            "cold-start step from the floor is excluded"
        )
        lines.append(
            f"differs    windows decided differently from shift "
            f"{FROZEN_BASELINE_SHIFT}"
        )
    return "\n".join(lines)


def write_csv(rows, path: pathlib.Path) -> None:
    fields = [key for key in rows[0]] if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dataset_root", nargs="?", default="datasets/emg",
        help="classifier sessions the model is fitted on",
    )
    parser.add_argument(
        "--validation-root", type=pathlib.Path,
        help="event-gate sessions to replay against a model fitted on "
             "dataset_root, instead of leave-one-session-out within it. "
             "Event-gate sessions carry no ULNAR trials, so they cannot fit "
             "a model of their own",
    )
    parser.add_argument("--factor", type=int, default=FROZEN_FACTOR)
    parser.add_argument(
        "--floor", type=int, default=THRESHOLD_FLOOR,
        help="threshold floor in MAV counts; sessions ship a calibrated one, "
             "and the compile-time default is often high enough to govern "
             "every window and hide the shift entirely",
    )
    parser.add_argument(
        "--shifts", type=int, nargs="+", default=list(DEFAULT_SHIFTS),
    )
    parser.add_argument("--csv", type=pathlib.Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    for shift in args.shifts:
        if not 1 <= shift <= 8:
            raise SystemExit(f"shift {shift} is outside the protocol's 1..8")

    training, training_skipped = load_timelines(args.dataset_root)
    for path, reason in training_skipped:
        print(f"  skipped {path}: {reason}")

    if args.validation_root is None:
        folds = prepare_loso_folds(training)
        print(
            f"{len(training)} sessions, leave-one-session-out. Folds group by "
            "session, not by donning, so these are within-donning numbers."
        )
    else:
        validation, validation_skipped = load_event_gate_timelines(
            args.validation_root
        )
        for path, reason in validation_skipped:
            print(f"  skipped {path}: {reason}")
        folds = prepare_external_validation_folds(training, validation)
        print(
            f"trained on {len(training)} sessions, replaying "
            f"{len(validation)} independent event-gate session(s)."
        )
    rows = sweep(
        folds, factor=args.factor, floor=args.floor, shifts=args.shifts
    )
    print(format_rows(rows, factor=args.factor, floor=args.floor))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
