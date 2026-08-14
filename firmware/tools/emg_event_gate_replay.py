#!/usr/bin/env python3
"""Replay labelled EMG sessions through candidate discrete-event gates.

This tool does not choose deployment thresholds.  It reconstructs the full
400-sample / 100-sample feature timeline, uses fold-specific Q18 LDA models,
and compares candidate gates without leaking the held-out session into its
model.  Events in unlabelled prompt/transition gaps are diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
import json
import pathlib

import numpy as np

from emg_features_ref import HOP, WINDOW, compute_features
from emg_filter_ref import design_emg_filter, filter_fixed, to_fixed
from emg_protocol import (
    TYPE_INFO,
    TYPE_RAW,
    PacketParser,
    decode_info,
    decode_raw,
)
from emg_train_lda import (
    CHANNEL_COUNT,
    FEATURE_NAMES,
    LABELS,
    SCHEMA_VERSION,
    SessionFeatures,
    ZERO_CROSSING_THRESHOLD,
    fit_lda,
    quantize_lda,
    read_manifest,
    select_complete_manifests,
    validate_complete_manifest,
)


SAMPLE_RATE_HZ = 2000
ACTIVE_LABELS = tuple(label for label in LABELS if label != "REST")


@dataclass(frozen=True)
class GateConfig:
    """Candidate event-gate counts, expressed in 50 ms feature hops."""

    stable_windows: int
    rest_rearm_windows: int
    refractory_windows: int
    abort_stable_windows: int
    onset_holdoff_windows: int

    def __post_init__(self):
        if self.stable_windows <= 0:
            raise ValueError("stable_windows must be positive")
        if self.rest_rearm_windows <= 0:
            raise ValueError("rest_rearm_windows must be positive")
        if self.refractory_windows < 0:
            raise ValueError("refractory_windows must be non-negative")
        if self.abort_stable_windows <= 0:
            raise ValueError("abort_stable_windows must be positive")
        if self.onset_holdoff_windows < 0:
            raise ValueError("onset_holdoff_windows must be non-negative")


class EventGate:
    """Fail-closed, one-event-per-gesture candidate gate.

    Ordinary events require an armed gate and a stable run.  REST re-arms only
    after both its own stable run and the refractory period.  ABORT has an
    independent stable run and bypasses ordinary arming/refractory, but remains
    latched until stable REST so a held gesture cannot emit repeatedly.

    Leaving REST starts an onset hold-off.  Measured event-gate replay put
    essentially all cross-session confusion in the first few hundred
    milliseconds of a contraction, where a 200 ms feature window straddles rest
    and contraction and the classifier has never seen that mixture: the
    classifier sessions label only steady state.  Hold-off discards those
    windows instead of scoring them, so no evidence accumulates until the
    pattern has established.  It costs one hold-off period of latency on every
    event, and it is deliberately not a confidence threshold - the wrong
    predictions there are confident, not marginal.
    """

    def __init__(self, config):
        self.config = config
        self.invalidate()

    @property
    def armed(self):
        return self._armed

    @property
    def holding_off(self):
        return self._onset_holdoff > 0

    def invalidate(self):
        """Clear all evidence; valid REST is required before ordinary events."""
        self._armed = False
        self._candidate = None
        self._candidate_run = 0
        self._rest_run = 0
        self._abort_run = 0
        self._abort_latched = False
        self._refractory = 0
        self._onset_holdoff = 0
        # An interrupted stream is not evidence that the muscle is resting, so
        # the next contraction still counts as an onset.
        self._resting = True

    def push(self, prediction, *, valid=True):
        if prediction not in LABELS:
            raise ValueError(f"unknown prediction: {prediction}")
        if not valid:
            self.invalidate()
            return None

        # The refractory period is elapsed time, so it keeps counting down
        # through a hold-off.
        if self._refractory > 0:
            self._refractory -= 1

        if prediction == "REST":
            self._resting = True
            self._onset_holdoff = 0
            self._candidate = None
            self._candidate_run = 0
            self._abort_run = 0
            self._rest_run += 1
            if (
                self._rest_run >= self.config.rest_rearm_windows
                and self._refractory == 0
            ):
                self._armed = True
                self._abort_latched = False
            return None

        self._rest_run = 0
        if self._resting:
            self._resting = False
            self._onset_holdoff = self.config.onset_holdoff_windows
        if self._onset_holdoff > 0:
            self._onset_holdoff -= 1
            self._candidate = None
            self._candidate_run = 0
            self._abort_run = 0
            return None

        if prediction == "ABORT":
            self._candidate = None
            self._candidate_run = 0
            self._abort_run += 1
            if (
                self._abort_run >= self.config.abort_stable_windows
                and not self._abort_latched
            ):
                self._abort_latched = True
                self._armed = False
                self._refractory = self.config.refractory_windows
                return "ABORT"
            return None

        self._abort_run = 0
        if not self._armed or self._refractory > 0:
            self._candidate = None
            self._candidate_run = 0
            return None
        if prediction == self._candidate:
            self._candidate_run += 1
        else:
            self._candidate = prediction
            self._candidate_run = 1
        if self._candidate_run < self.config.stable_windows:
            return None

        event = self._candidate
        self._armed = False
        self._candidate = None
        self._candidate_run = 0
        self._refractory = self.config.refractory_windows
        return event


@dataclass(frozen=True)
class TrialSpan:
    trial_id: str
    label: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SessionTimeline:
    session_id: str
    sample_rate_hz: int
    features: np.ndarray
    frame_ends: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    trials: tuple[TrialSpan, ...]
    transition_frames: int = 0
    manifest_path: pathlib.Path | None = None

    def labelled_features(self):
        mask = self.labels != None  # noqa: E711 - intentional object sentinel
        return SessionFeatures(
            session_id=self.session_id,
            features=self.features[mask],
            labels=self.labels[mask],
            trial_ids=self.trial_ids[mask],
            manifest_path=self.manifest_path,
        )


@dataclass(frozen=True)
class PreparedFold:
    held_out_session: str
    training_sessions: tuple[str, ...]
    timeline: SessionTimeline
    predictions: np.ndarray
    q18_float_agreement: float


def _parser_errors(stats):
    return {
        "lost": stats.lost,
        "malformed": stats.malformed,
        "duplicated": stats.duplicated,
        "time_reversed": stats.time_reversed,
        "discarded_bytes": stats.discarded_bytes,
    }


def _trial_spans(manifest, included):
    session_id = manifest.get("session_id", "unknown_session")
    spans = []
    for segment in sorted(included, key=lambda item: item["start"]["frame_index"]):
        trial = segment["trial"]
        trial_id = (
            f"{session_id}:trial={trial['index']}:"
            f"attempt={segment.get('attempt', 1)}"
        )
        spans.append(TrialSpan(
            trial_id=trial_id,
            label=trial["label"],
            start_frame=segment["start"]["frame_index"],
            end_frame=segment["end"]["frame_index"],
        ))
    return tuple(spans)


def validate_event_gate_manifest(manifest, path=None):
    """Validate a complete alternating REST/event session without training it."""
    source = str(path) if path is not None else "manifest"
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{source}: expected schema_version {SCHEMA_VERSION}")
    if manifest.get("collection_protocol") != "event-gate":
        raise ValueError(f"{source}: collection_protocol must be event-gate")
    if manifest.get("status") != "complete":
        raise ValueError(f"{source}: session status must be complete")
    if manifest.get("final_phase") != "complete":
        raise ValueError(f"{source}: final_phase must be complete")

    included = [
        segment
        for segment in manifest.get("segments", ())
        if segment.get("include") is True
    ]
    completed = manifest.get("completed_trials")
    total = manifest.get("total_trials")
    if completed != total or completed != len(included):
        raise ValueError(
            f"{source}: completed/total/accepted trial counts disagree"
        )

    ordered = sorted(
        included,
        key=lambda item: item.get("start", {}).get("frame_index", -1),
    )
    labels = [segment.get("trial", {}).get("label") for segment in ordered]
    if len(labels) < 3 or len(labels) % 2 == 0:
        raise ValueError(f"{source}: event-gate plan must have odd trial count")
    if any(label != "REST" for label in labels[::2]):
        raise ValueError(f"{source}: every event must be surrounded by REST")
    if any(label not in ACTIVE_LABELS for label in labels[1::2]):
        raise ValueError(f"{source}: event slots must contain active gestures")

    active_counts = Counter(labels[1::2])
    if (
        set(active_counts) != set(ACTIVE_LABELS)
        or len(set(active_counts.values())) != 1
    ):
        raise ValueError(
            f"{source}: active events must be balanced across {ACTIVE_LABELS}; "
            f"got {dict(active_counts)}"
        )
    if labels.count("REST") != len(labels[1::2]) + 1:
        raise ValueError(f"{source}: REST/event count does not alternate")

    previous_end = -1
    for segment in ordered:
        start = segment.get("start", {}).get("frame_index")
        end = segment.get("end", {}).get("frame_index")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0:
            raise ValueError(f"{source}: invalid segment frame bounds")
        if end <= start:
            raise ValueError(f"{source}: segment end must follow its start")
        if start < previous_end:
            raise ValueError(f"{source}: accepted segments overlap")
        previous_end = end
    return ordered


def build_feature_timeline(
    filtered_columns,
    frame_valid,
    trials,
    *,
    threshold=ZERO_CROSSING_THRESHOLD,
):
    """Build global-grid features and attach only wholly contained labels."""
    columns = [np.asarray(column, dtype=np.int64) for column in filtered_columns]
    if len(columns) != CHANNEL_COUNT:
        raise ValueError(f"expected {CHANNEL_COUNT} channels")
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("channel lengths disagree")
    frame_count = lengths.pop()
    frame_valid = np.asarray(frame_valid, dtype=bool)
    if len(frame_valid) != frame_count:
        raise ValueError("frame_valid length disagrees with samples")
    if frame_count < WINDOW:
        raise ValueError("recording is shorter than one feature window")

    frame_ends = np.arange(WINDOW, frame_count + 1, HOP, dtype=np.int64)
    rows = []
    for end in frame_ends:
        row = []
        begin = int(end) - WINDOW
        for column in columns:
            row.extend(compute_features(column[begin:int(end)], threshold))
        rows.append(row)

    invalid_prefix = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(~frame_valid, dtype=np.int64),
    ))
    valid = (
        invalid_prefix[frame_ends]
        - invalid_prefix[frame_ends - WINDOW]
    ) == 0
    labels = np.full(len(frame_ends), None, dtype=object)
    trial_ids = np.full(len(frame_ends), None, dtype=object)
    for trial in trials:
        inside = (
            (frame_ends - WINDOW >= trial.start_frame)
            & (frame_ends <= trial.end_frame)
        )
        if np.any(labels[inside] != None):  # noqa: E711
            raise ValueError("trial spans overlap on the feature grid")
        labels[inside] = trial.label
        trial_ids[inside] = trial.trial_id
    return (
        np.asarray(rows, dtype=np.float64),
        frame_ends,
        valid,
        labels,
        trial_ids,
    )


def load_timeline(manifest_path, *, validator=validate_complete_manifest):
    path = pathlib.Path(manifest_path)
    manifest = read_manifest(path)
    included = validator(manifest, path)
    raw_log = manifest.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        raise ValueError(f"{path}: raw_log is missing")
    raw_path = pathlib.Path(raw_log)
    if not raw_path.is_absolute():
        raw_path = path.parent / raw_path

    parser = PacketParser()
    packets = parser.feed(raw_path.read_bytes())
    errors = _parser_errors(parser.stats)
    if any(errors.values()):
        raise ValueError(f"{path}: raw parser errors {errors}")
    info_packets = [decode_info(packet.payload)
                    for packet in packets if packet.type == TYPE_INFO]
    if not info_packets:
        raise ValueError(f"{path}: no INFO packet")
    info = info_packets[0]
    if any(item != info for item in info_packets[1:]):
        raise ValueError(f"{path}: INFO packets disagree")
    if info.channel_count != CHANNEL_COUNT or info.sample_rate_hz != SAMPLE_RATE_HZ:
        raise ValueError(f"{path}: unsupported INFO geometry")
    if int(manifest.get("sample_rate_hz", -1)) != info.sample_rate_hz:
        raise ValueError(f"{path}: manifest and INFO sample rates disagree")

    frames = []
    frame_valid = []
    for packet in packets:
        if packet.type != TYPE_RAW:
            continue
        block = decode_raw(packet.payload, info.channel_count)
        frames.extend(block.frames)
        frame_valid.extend([block.all_attached] * len(block.frames))
    samples = np.asarray(frames, dtype=np.int64)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"{path}: malformed RAW frame matrix")

    sections = to_fixed(design_emg_filter(rate_hz=float(info.sample_rate_hz)))
    filtered = [
        filter_fixed(
            sections,
            np.clip(samples[:, channel], -32768, 32767).astype(np.int16),
        )
        for channel in range(CHANNEL_COUNT)
    ]
    trials = _trial_spans(manifest, included)
    timing = manifest.get("timing_seconds", {})
    transition_seconds = timing.get("transition_unlabelled")
    if not isinstance(transition_seconds, (int, float)):
        raise ValueError(f"{path}: transition timing is missing")
    transition_frames = round(float(transition_seconds) * info.sample_rate_hz)
    if transition_frames < 0:
        raise ValueError(f"{path}: transition timing must be non-negative")
    features, ends, valid, labels, trial_ids = build_feature_timeline(
        filtered, frame_valid, trials
    )
    return SessionTimeline(
        session_id=manifest.get("session_id", path.parent.name),
        sample_rate_hz=info.sample_rate_hz,
        features=features,
        frame_ends=ends,
        valid=valid,
        labels=labels,
        trial_ids=trial_ids,
        trials=trials,
        transition_frames=transition_frames,
        manifest_path=path,
    )


def load_timelines(dataset_root):
    paths, skipped = select_complete_manifests(dataset_root)
    timelines = [load_timeline(path) for path in paths]
    if len(timelines) < 2:
        raise ValueError("LOSO replay needs at least two complete sessions")
    return timelines, skipped


def load_event_gate_timelines(dataset_root):
    paths, skipped = select_complete_manifests(dataset_root)
    timelines = [
        load_timeline(path, validator=validate_event_gate_manifest)
        for path in paths
    ]
    return timelines, skipped


def prepare_loso_folds(timelines, *, fraction_bits=18):
    sessions = [timeline.labelled_features() for timeline in timelines]
    folds = []
    for timeline, held_out in zip(timelines, sessions):
        training = [session for session in sessions if session is not held_out]
        model = fit_lda(
            np.vstack([session.features for session in training]),
            np.concatenate([session.labels for session in training]),
        )
        quantized = quantize_lda(model, fraction_bits)
        float_predictions = model.predict(timeline.features)
        fixed_predictions = quantized.predict(timeline.features)
        agreement = float(np.mean(float_predictions == fixed_predictions))
        folds.append(PreparedFold(
            held_out_session=held_out.session_id,
            training_sessions=tuple(item.session_id for item in training),
            timeline=timeline,
            predictions=fixed_predictions,
            q18_float_agreement=agreement,
        ))
    return folds


def prepare_external_validation_folds(
    training_timelines,
    validation_timelines,
    *,
    fraction_bits=18,
):
    """Fit once on classifier sessions and predict independent event sessions."""
    if not training_timelines:
        raise ValueError("at least one classifier-training session is required")
    if not validation_timelines:
        raise ValueError("at least one event-gate validation session is required")
    training = [timeline.labelled_features() for timeline in training_timelines]
    model = fit_lda(
        np.vstack([session.features for session in training]),
        np.concatenate([session.labels for session in training]),
    )
    quantized = quantize_lda(model, fraction_bits)
    folds = []
    training_ids = tuple(session.session_id for session in training)
    for timeline in validation_timelines:
        float_predictions = model.predict(timeline.features)
        fixed_predictions = quantized.predict(timeline.features)
        folds.append(PreparedFold(
            held_out_session=timeline.session_id,
            training_sessions=training_ids,
            timeline=timeline,
            predictions=fixed_predictions,
            q18_float_agreement=float(
                np.mean(float_predictions == fixed_predictions)
            ),
        ))
    return folds


def _event_trial(timeline, index):
    """Return (trial_id, phase) for ACTIVE or its preceding MOVE NOW span."""
    frame_end = int(timeline.frame_ends[index])
    for trial in timeline.trials:
        # Event ownership follows the decision timestamp.  Training remains
        # stricter: it uses only feature windows wholly contained in ACTIVE.
        if trial.start_frame <= frame_end <= trial.end_frame:
            return trial.trial_id, "active"
        if (
            trial.start_frame - timeline.transition_frames
            <= frame_end
            < trial.start_frame
        ):
            return trial.trial_id, "transition"
    return None, "unlabelled"


def evaluate_gate(folds, config):
    metrics = {
        "active_trials": 0,
        "clean_correct_trials": 0,
        "missed_active_trials": 0,
        "wrong_active_trials": 0,
        "duplicate_active_trials": 0,
        "rest_trials": 0,
        "rest_false_trials": 0,
        "rest_false_events": 0,
        "rest_valid_seconds": 0.0,
        "unlabelled_events": 0,
        "transition_events": 0,
        "invalid_windows": 0,
        "latency_seconds": [],
        "per_class": {
            label: {"trials": 0, "correct": 0} for label in ACTIVE_LABELS
        },
        "min_q18_float_agreement": min(
            fold.q18_float_agreement for fold in folds
        ),
    }
    for fold in folds:
        timeline = fold.timeline
        gate = EventGate(config)
        events = {trial.trial_id: [] for trial in timeline.trials}
        trial_by_id = {trial.trial_id: trial for trial in timeline.trials}
        for index, prediction in enumerate(fold.predictions):
            if not timeline.valid[index]:
                metrics["invalid_windows"] += 1
            event = gate.push(str(prediction), valid=bool(timeline.valid[index]))
            if event is None:
                continue
            trial_id, phase = _event_trial(timeline, index)
            if trial_id is None:
                metrics["unlabelled_events"] += 1
            else:
                metrics["transition_events"] += int(phase == "transition")
                events[trial_id].append(
                    (int(timeline.frame_ends[index]), event, phase)
                )

        rest_mask = (timeline.labels == "REST") & timeline.valid
        metrics["rest_valid_seconds"] += (
            int(np.count_nonzero(rest_mask)) * HOP / timeline.sample_rate_hz
        )
        for trial_id, trial_events in events.items():
            trial = trial_by_id[trial_id]
            commands = [command for _frame, command, _phase in trial_events]
            if trial.label == "REST":
                metrics["rest_trials"] += 1
                false_count = sum(command != "REST" for command in commands)
                metrics["rest_false_events"] += false_count
                metrics["rest_false_trials"] += int(false_count > 0)
                continue

            metrics["active_trials"] += 1
            metrics["per_class"][trial.label]["trials"] += 1
            correct = [item for item in trial_events if item[1] == trial.label]
            wrong = [item for item in trial_events if item[1] != trial.label]
            if correct:
                metrics["per_class"][trial.label]["correct"] += 1
                latency = (correct[0][0] - trial.start_frame) / timeline.sample_rate_hz
                metrics["latency_seconds"].append(latency)
            else:
                metrics["missed_active_trials"] += 1
            metrics["wrong_active_trials"] += int(bool(wrong))
            metrics["duplicate_active_trials"] += int(len(commands) > 1)
            metrics["clean_correct_trials"] += int(
                len(commands) == 1 and commands[0] == trial.label
            )

    rest_minutes = metrics["rest_valid_seconds"] / 60.0
    metrics["rest_false_events_per_min"] = (
        None if rest_minutes == 0.0
        else metrics["rest_false_events"] / rest_minutes
    )
    latencies = metrics.pop("latency_seconds")
    metrics["latency_p50_sec"] = (
        None if not latencies else float(np.percentile(latencies, 50))
    )
    metrics["latency_p95_sec"] = (
        None if not latencies else float(np.percentile(latencies, 95))
    )
    return metrics


def default_sweep_configs():
    # The hold-off grid stays coarse on purpose.  One scripted session carries
    # nine active trials, which cannot separate 0.70 s from 0.80 s; a finer
    # grid would only offer more ways to fit this session's noise.
    return [
        GateConfig(stable, rearm, refractory, abort, holdoff)
        for stable, rearm, refractory, abort, holdoff in product(
            range(2, 6), range(2, 7), (5, 10, 15, 20), range(1, 4), (0, 4, 8, 12, 16)
        )
    ]


def _rank(item):
    metrics = item["metrics"]
    unsafe = metrics["wrong_active_trials"] + metrics["rest_false_trials"]
    latency = metrics["latency_p95_sec"]
    return (
        unsafe,
        metrics["missed_active_trials"],
        metrics["duplicate_active_trials"],
        -metrics["clean_correct_trials"],
        float("inf") if latency is None else latency,
    )


def _validation_rank(item):
    metrics = item["metrics"]
    unsafe = (
        metrics["wrong_active_trials"]
        + metrics["rest_false_events"]
        + metrics["unlabelled_events"]
    )
    latency = metrics["latency_p95_sec"]
    return (
        unsafe,
        metrics["missed_active_trials"],
        metrics["duplicate_active_trials"],
        -metrics["clean_correct_trials"],
        float("inf") if latency is None else latency,
    )


def validation_sequence_passed(metrics):
    """Return only the scripted sequence-integrity result, not deployment approval."""
    return (
        metrics["active_trials"] > 0
        and metrics["clean_correct_trials"] == metrics["active_trials"]
        and metrics["missed_active_trials"] == 0
        and metrics["wrong_active_trials"] == 0
        and metrics["duplicate_active_trials"] == 0
        and metrics["rest_false_events"] == 0
        and metrics["unlabelled_events"] == 0
    )


def sweep_gates(folds, configs=None, *, validation_mode=False):
    candidates = []
    for config in configs or default_sweep_configs():
        candidates.append({
            "config": asdict(config),
            "metrics": evaluate_gate(folds, config),
        })
    rank = _validation_rank if validation_mode else _rank
    return sorted(candidates, key=rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="datasets/emg")
    parser.add_argument("--validation-root", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args(argv)
    if arguments.top <= 0:
        parser.error("--top must be positive")

    training_timelines, training_skipped = load_timelines(arguments.dataset_root)
    if arguments.validation_root is None:
        validation_timelines = []
        validation_skipped = []
        folds = prepare_loso_folds(training_timelines)
        method = "fold_specific_q18_loso_full_timeline_gate_sweep"
        limitations = [
            "unlabelled transition events are diagnostic, not false positives",
            "REST exposure is too short for a natural-use false-trigger claim",
            "freeze a candidate before evaluating a new independent session",
            "latency is relative to the ACTIVE boundary and may be negative",
        ]
        print(
            f"Loaded {len(training_timelines)} complete sessions; "
            f"skipped {len(training_skipped)}; "
            "using fold-specific LOSO replay."
        )
    else:
        validation_timelines, validation_skipped = load_event_gate_timelines(
            arguments.validation_root
        )
        folds = prepare_external_validation_folds(
            training_timelines,
            validation_timelines,
        )
        method = "fixed_training_q18_independent_event_gate_validation"
        limitations = [
            "one short scripted session is not a natural-use false-trigger claim",
            "validation samples are excluded from model fitting",
            "events outside accepted REST/active spans count as off-trial events",
            "sequence integrity alone is not deployment approval",
        ]
        print(
            f"Trained on {len(training_timelines)} classifier sessions; "
            f"validating {len(validation_timelines)} independent event session(s)."
        )
    validation_mode = arguments.validation_root is not None
    candidates = sweep_gates(folds, validation_mode=validation_mode)
    print(f"Swept {len(candidates)} configurations.")
    print(
        " N  M  refr  A hold | clean/active miss wrong dup rest_false off_trial "
        "p50_ms p95_ms"
    )
    for item in candidates[:arguments.top]:
        config = item["config"]
        metrics = item["metrics"]
        p50 = metrics["latency_p50_sec"]
        p95 = metrics["latency_p95_sec"]
        print(
            f"{config['stable_windows']:2d} "
            f"{config['rest_rearm_windows']:2d} "
            f"{config['refractory_windows']:5d} "
            f"{config['abort_stable_windows']:2d} "
            f"{config['onset_holdoff_windows']:4d} | "
            f"{metrics['clean_correct_trials']:2d}/"
            f"{metrics['active_trials']:2d} "
            f"{metrics['missed_active_trials']:4d} "
            f"{metrics['wrong_active_trials']:5d} "
            f"{metrics['duplicate_active_trials']:3d} "
            f"{metrics['rest_false_events']:10d} "
            f"{metrics['unlabelled_events']:9d} "
            f"{'n/a' if p50 is None else round(1000 * p50):>6} "
            f"{'n/a' if p95 is None else round(1000 * p95):>6}"
        )

    clean_candidates = sum(
        validation_sequence_passed(item["metrics"])
        for item in candidates
    ) if validation_mode else None
    if validation_mode:
        print(
            "Scripted sequence-clean candidates: "
            f"{clean_candidates}/{len(candidates)}"
        )

    payload = {
        "method": method,
        "limitations": limitations,
        "training_sessions": [
            timeline.session_id for timeline in training_timelines
        ],
        "validation_sessions": [
            timeline.session_id for timeline in validation_timelines
        ],
        "skipped_training_sessions": training_skipped,
        "skipped_validation_sessions": validation_skipped,
        "scripted_sequence_clean_candidates": clean_candidates,
        "candidates": candidates,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
