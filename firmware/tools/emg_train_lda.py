#!/usr/bin/env python3
"""Train and evaluate the Objective 3.5 intent LDA baseline.

The guided collector stores one continuous RAW stream plus exact half-open
frame ranges for every ACTIVE gesture.  This tool filters each channel once
over that continuous stream, then keeps only feature windows that fit wholly
inside an accepted labelled range.

Validation holds out one *donning* at a time wherever the sessions record
which electrode application they came from, and says so when they do not.
Overlapping 200 ms windows from one recording cannot leak into both train and
test data, and neither can two recordings of one electrode placement -- which
is the leak that made the previously accepted figure a within-donning number
while the quantity that matters is cross-donning.

Example::

    python3 firmware/tools/emg_train_lda.py datasets/emg --require-donning \
        --output datasets/emg/lda_model.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import pathlib
import sys

import numpy as np

from emg_analyze import load_session
from emg_features_ref import HOP, WINDOW, compute_features
from emg_filter_ref import design_emg_filter, filter_fixed, to_fixed
from emg_guided_session import GESTURE_LABELS


SCHEMA_VERSION = 2
# Version 1 sessions are still loadable and still worth training on; what they
# cannot do is carry a donning, so evaluate_loso reports what its folds are
# not rather than pretending they group. Refusing them outright would discard
# every recording made before 2026-08-27 to gain nothing.
ACCEPTED_SCHEMA_VERSIONS = (1, 2)
MODEL_SCHEMA_VERSION = 1
CHANNEL_COUNT = 3
ZERO_CROSSING_THRESHOLD = 10
DEFAULT_RIDGE = 1.0e-3
DEFAULT_QUANTIZATION_BITS = 18
LABELS = tuple(GESTURE_LABELS)
PROTOCOL_COMMAND_LABELS = ("REST", "NEXT_TARGET", "CONFIRM", "ABORT")
# Classes the firmware may hold that are not commands. They steer the
# proportional view channel and are rewritten to REST before the event gate,
# so they never reach the INTENT command field. Ordered after the protocol
# commands so the emitted coefficient rows keep the order the firmware's own
# enum gives them.
DIRECTION_ONLY_LABELS = ("ULNAR",)
FIRMWARE_LABELS = PROTOCOL_COMMAND_LABELS + DIRECTION_ONLY_LABELS
FEATURE_NAMES = tuple(
    f"ch{channel}_{feature}"
    for channel in range(CHANNEL_COUNT)
    for feature in ("mav", "rms", "waveform_length", "zero_crossings")
)


@dataclass(frozen=True)
class SessionFeatures:
    """Feature rows and provenance from one physical recording session."""

    session_id: str
    features: np.ndarray
    labels: np.ndarray
    trial_ids: np.ndarray
    manifest_path: pathlib.Path | None = None
    # Which electrode application this was recorded under. None for sessions
    # captured before the field existed; those cannot be grouped, and
    # evaluate_loso says so rather than guessing.
    donning: str | None = None

    def __post_init__(self):
        rows = len(self.features)
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (N, {len(FEATURE_NAMES)})"
            )
        if len(self.labels) != rows or len(self.trial_ids) != rows:
            raise ValueError("features, labels, and trial_ids must have equal rows")
        if rows == 0:
            raise ValueError("a session must contain at least one feature window")


@dataclass(frozen=True)
class LDAModel:
    """Standardized, ridge-regularized pooled-covariance LDA parameters."""

    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    ridge: float
    effective_ridge: float

    def scores(self, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have {len(self.feature_names)} columns"
            )
        standardized = (values - self.mean) / self.scale
        return standardized @ self.weights.T + self.intercept

    def predict(self, features):
        scores = self.scores(features)
        labels = np.asarray(self.labels, dtype=object)
        return labels[np.argmax(scores, axis=1)]

    def raw_affine(self):
        """Fold standardization into one affine score over integer features."""
        weights = self.weights / self.scale[np.newaxis, :]
        intercept = self.intercept - weights @ self.mean
        return weights, intercept

    def to_dict(self):
        return {
            "labels": list(self.labels),
            "feature_names": list(self.feature_names),
            "standardizer": {
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
            },
            "discriminant": {
                "weights": self.weights.tolist(),
                "intercept": self.intercept.tolist(),
                "ridge": self.ridge,
                "effective_ridge": self.effective_ridge,
                "priors": [1.0 / len(self.labels)] * len(self.labels),
            },
        }

    @classmethod
    def from_dict(cls, value):
        standardizer = value["standardizer"]
        discriminant = value["discriminant"]
        return cls(
            labels=tuple(value["labels"]),
            feature_names=tuple(value["feature_names"]),
            mean=np.asarray(standardizer["mean"], dtype=np.float64),
            scale=np.asarray(standardizer["scale"], dtype=np.float64),
            weights=np.asarray(discriminant["weights"], dtype=np.float64),
            intercept=np.asarray(discriminant["intercept"], dtype=np.float64),
            ridge=float(discriminant["ridge"]),
            effective_ridge=float(discriminant["effective_ridge"]),
        )


@dataclass(frozen=True)
class QuantizedLDAModel:
    """Raw-feature affine LDA with one shared binary fixed-point scale."""

    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    fraction_bits: int
    weights: np.ndarray
    intercept: np.ndarray

    def scores(self, features):
        values = np.asarray(features, dtype=np.int64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have {len(self.feature_names)} columns"
            )
        return values @ self.weights.T + self.intercept

    def predict(self, features):
        scores = self.scores(features)
        labels = np.asarray(self.labels, dtype=object)
        return labels[np.argmax(scores, axis=1)]

    def to_dict(self):
        return {
            "labels": list(self.labels),
            "feature_names": list(self.feature_names),
            "fraction_bits": self.fraction_bits,
            "weights": self.weights.tolist(),
            "intercept": self.intercept.tolist(),
            "score_units": f"Q{self.fraction_bits}",
            "accumulator": "signed_int64",
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            labels=tuple(value["labels"]),
            feature_names=tuple(value["feature_names"]),
            fraction_bits=int(value["fraction_bits"]),
            weights=np.asarray(value["weights"], dtype=np.int64),
            intercept=np.asarray(value["intercept"], dtype=np.int64),
        )


def read_manifest(path):
    manifest_path = pathlib.Path(path)
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error


def discover_manifests(dataset_root):
    root = pathlib.Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    manifests = sorted(root.glob("session_*/session.json"))
    if not manifests:
        raise ValueError(f"no session_*/session.json files under {root}")
    return manifests


def select_complete_manifests(dataset_root):
    """Return complete manifests and report non-complete sessions separately."""
    selected = []
    skipped = []
    for path in discover_manifests(dataset_root):
        manifest = read_manifest(path)
        status = manifest.get("status")
        session_id = manifest.get("session_id", path.parent.name)
        if status == "complete":
            selected.append(path)
        else:
            skipped.append({"session_id": session_id, "status": status or "missing"})
    if not selected:
        raise ValueError("no complete sessions are available for training")
    return selected, skipped


def manifest_labels(manifest, path=None):
    """The label set this session actually collected, in a stable order.

    Read from the session rather than from the module constant so an
    exploratory label set trains against its own labels. The module constant
    stays the four protocol commands, which is what the firmware emitter in
    ``write_c_model`` still checks against.
    """
    source = str(path) if path is not None else "manifest"
    actions = manifest.get("gesture_actions")
    if not isinstance(actions, dict) or not actions:
        raise ValueError(f"{source}: gesture_actions is missing or empty")
    for label, action in actions.items():
        if not isinstance(label, str) or not isinstance(action, str):
            raise ValueError(f"{source}: gesture_actions must map str to str")
        if not label.strip() or not action.strip():
            raise ValueError(f"{source}: gesture_actions has a blank entry")
    # Order matters and must not come from the JSON key order. Class order
    # fixes the row order of the emitted coefficient table, and the firmware
    # header is only emitted for exactly PROTOCOL_COMMAND_LABELS in exactly
    # that order. Sorting the protocol commands alphabetically instead would
    # silently stop the live model from being regenerable.
    extra = sorted(set(actions) - set(FIRMWARE_LABELS))
    return tuple(
        [label for label in FIRMWARE_LABELS if label in actions] + extra
    )


def validate_complete_manifest(manifest, path=None, labels=None):
    """Fail closed on metadata that could shift or mislabel training windows."""
    source = str(path) if path is not None else "manifest"
    expected = manifest_labels(manifest, path) if labels is None else tuple(labels)
    if manifest.get("schema_version") not in ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{source}: expected schema_version in "
            f"{sorted(ACCEPTED_SCHEMA_VERSIONS)}"
        )
    if manifest.get("status") != "complete":
        raise ValueError(f"{source}: session status must be complete")
    if manifest.get("final_phase") != "complete":
        raise ValueError(f"{source}: final_phase must be complete")

    included = [segment for segment in manifest.get("segments", ())
                if segment.get("include") is True]
    counts = Counter(segment.get("trial", {}).get("label") for segment in included)
    if set(counts) != set(expected) or len(set(counts.values())) != 1:
        raise ValueError(
            f"{source}: accepted trials must be balanced across {expected}; "
            f"got {dict(counts)}"
        )
    if not included or any(count <= 0 for count in counts.values()):
        raise ValueError(f"{source}: no accepted balanced trials")
    completed = manifest.get("completed_trials")
    total = manifest.get("total_trials")
    if completed != total or completed != len(included):
        raise ValueError(
            f"{source}: completed/total/accepted trial counts disagree"
        )

    previous_end = -1
    for segment in sorted(included, key=lambda item: item["start"]["frame_index"]):
        start = segment.get("start", {}).get("frame_index")
        end = segment.get("end", {}).get("frame_index")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0:
            raise ValueError(f"{source}: invalid segment frame bounds")
        if end <= start:
            raise ValueError(f"{source}: segment end must follow its start")
        if start < previous_end:
            raise ValueError(f"{source}: accepted segments overlap")
        previous_end = end
    return included


def _first_grid_end(segment_start, window=WINDOW, hop=HOP):
    earliest = segment_start + window
    return ((earliest + hop - 1) // hop) * hop


def extract_feature_windows(
    filtered_columns,
    manifest,
    *,
    labels=None,
    threshold=ZERO_CROSSING_THRESHOLD,
    window=WINDOW,
    hop=HOP,
):
    """Extract windows wholly contained in accepted segments on a global grid."""
    if window != WINDOW:
        raise ValueError(f"feature implementation requires window={WINDOW}")
    if hop <= 0:
        raise ValueError("hop must be positive")
    columns = [np.asarray(column, dtype=np.int64) for column in filtered_columns]
    if len(columns) != CHANNEL_COUNT:
        raise ValueError(f"expected {CHANNEL_COUNT} channels, got {len(columns)}")
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("all filtered channels must have equal lengths")
    total_frames = lengths.pop()
    included = validate_complete_manifest(manifest, labels=labels)
    session_id = manifest.get("session_id", "unknown_session")

    rows = []
    labels = []
    trial_ids = []
    for segment in sorted(included, key=lambda item: item["start"]["frame_index"]):
        start = segment["start"]["frame_index"]
        stop = segment["end"]["frame_index"]
        if stop > total_frames:
            raise ValueError(
                f"{session_id}: labelled frame {stop} exceeds raw length "
                f"{total_frames}"
            )
        trial = segment["trial"]
        trial_id = (
            f"{session_id}:trial={trial['index']}:"
            f"attempt={segment.get('attempt', 1)}"
        )
        for end in range(_first_grid_end(start, window, hop), stop + 1, hop):
            begin = end - window
            if begin < start:
                continue
            row = []
            for column in columns:
                row.extend(compute_features(column[begin:end], threshold))
            rows.append(row)
            labels.append(trial["label"])
            trial_ids.append(trial_id)

    if not rows:
        raise ValueError(f"{session_id}: no complete feature windows in labels")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=object),
        np.asarray(trial_ids, dtype=object),
    )


def load_feature_session(
    manifest_path, threshold=ZERO_CROSSING_THRESHOLD, expected_labels=None
):
    # Named ``expected_labels`` rather than ``labels`` because the local
    # ``labels`` below holds this session's per-window label array.
    path = pathlib.Path(manifest_path)
    manifest = read_manifest(path)
    validate_complete_manifest(manifest, path, labels=expected_labels)
    raw_log = manifest.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        raise ValueError(f"{path}: raw_log is missing")
    raw_path = pathlib.Path(raw_log)
    if not raw_path.is_absolute():
        raw_path = path.parent / raw_path
    if not raw_path.is_file():
        raise ValueError(f"{path}: raw log does not exist: {raw_path}")

    info, columns, stats, _wear_counts = load_session(raw_path)
    if info.channel_count != CHANNEL_COUNT:
        raise ValueError(
            f"{path}: expected {CHANNEL_COUNT} channels, got {info.channel_count}"
        )
    if int(manifest.get("sample_rate_hz", -1)) != info.sample_rate_hz:
        raise ValueError(f"{path}: manifest and INFO sample rates disagree")
    parser_errors = {
        "lost": stats.lost,
        "malformed": stats.malformed,
        "duplicated": stats.duplicated,
        "time_reversed": stats.time_reversed,
        "discarded_bytes": stats.discarded_bytes,
    }
    if any(parser_errors.values()):
        raise ValueError(f"{path}: raw parser errors {parser_errors}")

    sections = to_fixed(design_emg_filter(rate_hz=float(info.sample_rate_hz)))
    filtered = [
        filter_fixed(sections, np.clip(column, -32768, 32767).astype(np.int16))
        for column in columns
    ]
    features, labels, trial_ids = extract_feature_windows(
        filtered, manifest, labels=expected_labels, threshold=threshold
    )
    return SessionFeatures(
        session_id=manifest.get("session_id", path.parent.name),
        features=features,
        labels=labels,
        trial_ids=trial_ids,
        manifest_path=path,
        donning=manifest.get("donning"),
    )


def load_dataset(dataset_root, threshold=ZERO_CROSSING_THRESHOLD,
                 require_donning=False):
    """Load every complete session and the one label set they all share.

    Sessions that disagree on their labels cannot be pooled: a fold would
    train on classes the held-out session never collected, and the confusion
    matrix would silently gain an all-zero row. That is a hard error rather
    than an intersection, because dropping a label quietly would change what
    the reported accuracy is an accuracy *of*.
    """
    paths, skipped = select_complete_manifests(dataset_root)
    if require_donning:
        # Keeping only sessions that can be grouped by electrode application.
        # Without it a fold is held out by session, which puts the held-out
        # recording's own donning back into training and reports a
        # within-donning number as if it were a cross-donning one.
        kept = []
        for path in paths:
            manifest = read_manifest(path)
            if manifest.get("donning"):
                kept.append(path)
            else:
                skipped.append({
                    "session_id": manifest.get("session_id", path.parent.name),
                    "status": "no donning recorded",
                })
        paths = kept
    if not paths:
        raise ValueError("no complete sessions are available for training")
    labels = manifest_labels(read_manifest(paths[0]), paths[0])
    for path in paths[1:]:
        other = manifest_labels(read_manifest(path), path)
        if set(other) != set(labels):
            raise ValueError(
                f"sessions disagree on their label set: {paths[0]} has "
                f"{sorted(labels)}, {path} has {sorted(other)}"
            )
    sessions = [
        load_feature_session(path, threshold, expected_labels=labels)
        for path in paths
    ]
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out validation needs at least 2 sessions")
    return sessions, skipped, labels


def fit_lda(features, labels, *, class_order=LABELS, ridge=DEFAULT_RIDGE):
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=object)
    classes = tuple(class_order)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
    if len(targets) != len(values) or len(values) <= len(classes):
        raise ValueError("not enough labelled rows")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    unknown = set(targets) - set(classes)
    missing = set(classes) - set(targets)
    if unknown or missing:
        raise ValueError(f"class mismatch: missing={missing}, unknown={unknown}")

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (values - mean) / scale
    class_means = np.vstack([
        standardized[targets == label].mean(axis=0) for label in classes
    ])
    scatter = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for index, label in enumerate(classes):
        centered = standardized[targets == label] - class_means[index]
        scatter += centered.T @ centered
    covariance = scatter / (len(values) - len(classes))
    covariance_scale = float(np.trace(covariance) / values.shape[1])
    effective_ridge = ridge * (covariance_scale if covariance_scale > 0.0 else 1.0)
    covariance += effective_ridge * np.eye(values.shape[1])
    weights = np.linalg.solve(covariance, class_means.T).T
    intercept = -0.5 * np.sum(class_means * weights, axis=1)
    intercept += math.log(1.0 / len(classes))
    return LDAModel(
        labels=classes,
        feature_names=FEATURE_NAMES,
        mean=mean,
        scale=scale,
        weights=weights,
        intercept=intercept,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
    )


def quantize_lda(model, fraction_bits=DEFAULT_QUANTIZATION_BITS):
    """Quantize a fitted model after folding away runtime standardization."""
    if not isinstance(fraction_bits, int) or isinstance(fraction_bits, bool):
        raise TypeError("fraction_bits must be an integer")
    if not 0 <= fraction_bits <= 31:
        raise ValueError("fraction_bits must be in [0, 31]")
    if not np.all(np.isfinite(model.weights)):
        raise ValueError("model weights must be finite")
    if not np.all(np.isfinite(model.intercept)):
        raise ValueError("model intercept must be finite")
    if not np.all(np.isfinite(model.mean)):
        raise ValueError("model mean must be finite")
    if not np.all(np.isfinite(model.scale)) or np.any(model.scale <= 0.0):
        raise ValueError("model scale must be finite and positive")
    scale = 1 << fraction_bits
    raw_weights, raw_intercept = model.raw_affine()
    int32 = np.iinfo(np.int32)
    int64 = np.iinfo(np.int64)
    weights = _quantize_checked(
        raw_weights, scale, int32.min, int32.max, "weights"
    )
    intercept = _quantize_checked(
        raw_intercept, scale, int64.min, int64.max, "intercept"
    )
    return QuantizedLDAModel(
        labels=model.labels,
        feature_names=model.feature_names,
        fraction_bits=fraction_bits,
        weights=weights,
        intercept=intercept,
    )


def _quantize_checked(values, scale, minimum, maximum, name):
    """Round finite floats to int64 only after checking the target range."""
    source = np.asarray(values, dtype=np.float64)
    result = []
    for value in source.flat:
        scaled = float(value) * scale
        if not math.isfinite(scaled):
            raise ValueError(f"quantized {name} must be finite")
        rounded = round(scaled)
        if rounded < minimum or rounded > maximum:
            raise ValueError(f"quantized {name} do not fit target integer type")
        result.append(rounded)
    return np.asarray(result, dtype=np.int64).reshape(source.shape)


def theoretical_feature_bounds():
    """Absolute maxima implied by int16 samples and a 400-sample window."""
    per_channel = (32768, 32768, (WINDOW - 1) * 65535, WINDOW - 1)
    return np.asarray(per_channel * CHANNEL_COUNT, dtype=np.int64)


def theoretical_score_bound(model):
    """Conservative absolute accumulator bound for every class."""
    bounds = theoretical_feature_bounds()
    return np.abs(model.intercept) + np.abs(model.weights) @ bounds


def render_c_model_header(model, source_sessions=()):
    """Render deterministic C parameters; no floats or divisions at runtime."""
    # The four protocol commands, optionally followed by the direction-only
    # classes, in exactly that order. Class order fixes the row order of the
    # coefficient table and the firmware indexes it with its own enum, so a
    # model ordered differently would silently mean different gestures.
    labels = tuple(model.labels)
    if labels != PROTOCOL_COMMAND_LABELS and labels != FIRMWARE_LABELS:
        raise ValueError(
            f"model class order must be {list(PROTOCOL_COMMAND_LABELS)} "
            f"or {list(FIRMWARE_LABELS)}, got {list(labels)}"
        )
    if tuple(model.feature_names) != FEATURE_NAMES:
        raise ValueError("model feature order does not match firmware features")
    sessions = ", ".join(source_sessions) if source_sessions else "unspecified"
    lines = [
        "/* Generated by firmware/tools/emg_train_lda.py.",
        f" * Source sessions: {sessions}",
        " * Do not hand-edit coefficients; regenerate and rerun golden tests.",
        " */",
        "#ifndef EMG_CLASSIFIER_MODEL_H",
        "#define EMG_CLASSIFIER_MODEL_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define EMG_CLASSIFIER_MODEL_FRACTION_BITS {model.fraction_bits}u",
        f"#define EMG_CLASSIFIER_MODEL_CLASS_COUNT {len(model.labels)}u",
        f"#define EMG_CLASSIFIER_MODEL_FEATURE_COUNT {len(model.feature_names)}u",
        # Generated from the label list, not written out. These four lines
        # were literals, so a model with a fifth class declared CLASS_COUNT 5
        # and defined four commands -- and the C still compiled, because the
        # initializer list is length-inferred and the missing entry becomes
        # zero, silently making the new class REST.
        *[
            f"#define EMG_CLASSIFIER_MODEL_CLASS_{index}_COMMAND "
            f"{FIRMWARE_LABELS.index(label)}u /* {label} */"
            for index, label in enumerate(model.labels)
        ],
        "",
        "static const uint8_t emg_classifier_model_commands",
        "    [EMG_CLASSIFIER_MODEL_CLASS_COUNT] = {",
        *[
            f"    EMG_CLASSIFIER_MODEL_CLASS_{index}_COMMAND"
            + ("," if index + 1 < len(model.labels) else "")
            for index in range(len(model.labels))
        ],
        "};",
        "",
        "static const int32_t emg_classifier_model_weights",
        "    [EMG_CLASSIFIER_MODEL_CLASS_COUNT]",
        "    [EMG_CLASSIFIER_MODEL_FEATURE_COUNT] = {",
    ]
    for row in model.weights:
        lines.append("    {" + ", ".join(str(int(value)) for value in row) + "},")
    lines.extend([
        "};",
        "",
        "static const int64_t emg_classifier_model_intercept",
        "    [EMG_CLASSIFIER_MODEL_CLASS_COUNT] = {",
        "    " + ", ".join(str(int(value)) for value in model.intercept),
        "};",
        "",
        "#endif /* EMG_CLASSIFIER_MODEL_H */",
        "",
    ])
    return "\n".join(lines)


def confusion_matrix(actual, predicted, labels=LABELS):
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        if truth not in index or guess not in index:
            raise ValueError(f"unknown label in confusion matrix: {truth}, {guess}")
        matrix[index[truth], index[guess]] += 1
    return matrix


def matrix_metrics(matrix, labels=LABELS):
    values = np.asarray(matrix, dtype=np.int64)
    total = int(values.sum())
    correct = int(np.trace(values))
    recalls = {}
    for row, label in enumerate(labels):
        denominator = int(values[row].sum())
        recalls[label] = None if denominator == 0 else values[row, row] / denominator
    return {
        "accuracy": None if total == 0 else correct / total,
        "correct": correct,
        "total": total,
        "per_class_recall": recalls,
        "confusion": values.tolist(),
    }


def trial_predictions(actual, predicted, scores, trial_ids, labels=LABELS):
    """Majority vote per trial; summed scores break an exact vote tie."""
    actual = np.asarray(actual, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    scores = np.asarray(scores, dtype=np.float64)
    trial_ids = np.asarray(trial_ids, dtype=object)
    trial_actual = []
    trial_predicted = []
    seen = []
    for trial_id in trial_ids:
        if trial_id not in seen:
            seen.append(trial_id)
    for trial_id in seen:
        mask = trial_ids == trial_id
        truths = set(actual[mask])
        if len(truths) != 1:
            raise ValueError(f"trial {trial_id} contains multiple true labels")
        counts = Counter(predicted[mask])
        best_count = max(counts.values())
        tied = [label for label in labels if counts[label] == best_count]
        if len(tied) == 1:
            winner = tied[0]
        else:
            totals = scores[mask].sum(axis=0)
            winner = max(tied, key=lambda label: totals[labels.index(label)])
        trial_actual.append(next(iter(truths)))
        trial_predicted.append(winner)
    return np.asarray(trial_actual, dtype=object), np.asarray(trial_predicted, dtype=object)


def _evaluation_groups(sessions):
    """Group sessions by donning, or refuse to pretend they are independent.

    Holding out one session at a time puts a held-out session's own donning
    back into the training set whenever more than one session was recorded on
    it, and the number that comes out is within-donning accuracy. The quantity
    that decides whether a wearer can use the system tomorrow is cross-donning
    accuracy, and it was never what this reported: the deployed model's six
    training sessions span three days, and held-out donnings from 2026-08-14
    give NEXT_TARGET at 2%, 12%, 68%, 100% and 100%.

    Sessions recorded before the field existed carry no donning, and there is
    no way to recover it. They are grouped one per fold, as before, and the
    caller is told the result is not a cross-donning number.
    """
    if any(session.donning is None for session in sessions):
        return (
            [[session] for session in sessions],
            "leave_one_session_out",
            "at least one session has no donning recorded, so folds cannot be "
            "grouped by electrode application; this is a within-donning "
            "number wherever a donning contributed more than one session",
        )
    groups = {}
    for session in sessions:
        groups.setdefault(session.donning, []).append(session)
    return list(groups.values()), "leave_one_donning_out", None


# The gate used to be exact agreement on the training windows. It is a proxy
# for "quantizing did not change the model", and it failed on a model whose
# quantized held-out accuracy was identical to four decimal places: one window
# in 7111 crossed a decision boundary. Measured 2026-08-28, five donnings, the
# per-fold difference was 0 to 2 windows out of ~1421 and the mean held-out
# accuracy was 94.1% either way.
#
# Keeping the exact rule would have meant searching ridge and scale until the
# proxy happened to read zero -- the disagreement count moved between 0 and 6
# with no monotonicity in either parameter, so that search selects noise. The
# gate now measures the thing the proxy stood for, and keeps a loose agreement
# floor to catch a scale that is actually broken rather than rounding.
QUANTIZATION_MIN_AGREEMENT = 0.999
QUANTIZATION_MAX_ACCURACY_LOSS = 0.005


def evaluate_loso(sessions, *, ridge=DEFAULT_RIDGE, labels=LABELS,
                  quantize_bits=None):
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out validation needs at least 2 sessions")
    groups, method, caveat = _evaluation_groups(sessions)
    if len(groups) < 2:
        raise ValueError(
            "held-out validation needs at least two donnings; every session "
            "here was recorded on one electrode application, and a model "
            "cannot be validated against the variation it never saw"
        )
    folds = []
    all_window_actual = []
    all_window_predicted = []
    all_trial_actual = []
    all_trial_predicted = []
    for group in groups:
        held_out = _merge_sessions(group)
        training = [
            session for session in sessions
            if all(session is not member for member in group)
        ]
        train_features = np.vstack([session.features for session in training])
        train_labels = np.concatenate([session.labels for session in training])
        model = fit_lda(
            train_features, train_labels, class_order=labels, ridge=ridge
        )
        if quantize_bits is not None:
            # Scoring the model that actually ships, on the same folds.
            model = quantize_lda(model, quantize_bits)
        scores = model.scores(held_out.features)
        predicted = model.predict(held_out.features)
        trial_actual, trial_predicted = trial_predictions(
            held_out.labels, predicted, scores, held_out.trial_ids,
            labels=labels,
        )
        window_matrix = confusion_matrix(held_out.labels, predicted, labels)
        trial_matrix = confusion_matrix(trial_actual, trial_predicted, labels)
        folds.append({
            "held_out_session": held_out.session_id,
            "training_sessions": [session.session_id for session in training],
            "window": matrix_metrics(window_matrix, labels),
            "trial": matrix_metrics(trial_matrix, labels),
        })
        all_window_actual.extend(held_out.labels.tolist())
        all_window_predicted.extend(predicted.tolist())
        all_trial_actual.extend(trial_actual.tolist())
        all_trial_predicted.extend(trial_predicted.tolist())

    result = {
        "method": method,
        "folds": folds,
        "overall_window": matrix_metrics(
            confusion_matrix(all_window_actual, all_window_predicted, labels),
            labels,
        ),
        "overall_trial": matrix_metrics(
            confusion_matrix(all_trial_actual, all_trial_predicted, labels),
            labels,
        ),
    }
    if caveat:
        result["caveat"] = caveat
    return result


def _merge_sessions(group):
    """One fold's held-out sessions as a single session-shaped record."""
    if len(group) == 1:
        return group[0]
    return SessionFeatures(
        session_id="+".join(session.session_id for session in group),
        features=np.vstack([session.features for session in group]),
        labels=np.concatenate([session.labels for session in group]),
        # Trial ids are only unique within a session, so they are namespaced
        # before merging or two sessions' trial 0 would be scored as one.
        trial_ids=np.concatenate([
            np.char.add(f"{session.session_id}:",
                        session.trial_ids.astype(str))
            for session in group
        ]),
        donning=group[0].donning,
    )


def format_confusion(matrix, labels=LABELS):
    width = max(8, max(len(label) for label in labels) + 1)
    lines = [" " * width + "".join(f"{label:>{width}}" for label in labels)]
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>{width}}" + "".join(f"{value:>{width}d}" for value in row))
    return "\n".join(lines)


def _percent(value):
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def deployment_status(quantized_model):
    """Describe generated parameters without claiming live-loop deployment."""
    return (
        f"host_q{quantized_model.fraction_bits}_parameters_"
        "not_live_integrated"
    )


def build_output(
    sessions,
    skipped,
    validation,
    model,
    quantized_model,
    quantized_agreement,
    threshold,
    ridge,
    labels=LABELS,
):
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "standardized_ridge_lda",
        "intent_labels": list(labels),
        "source_sessions": [session.session_id for session in sessions],
        "skipped_sessions": skipped,
        "preprocessing": {
            "sample_rate_hz": 2000,
            "filter": "Q29 20-450 Hz band-pass + 50/150 Hz notches",
            "filter_state": "continuous_per_session",
            "window_samples": WINDOW,
            "hop_samples": HOP,
            "zero_crossing_threshold": threshold,
            "feature_order": list(FEATURE_NAMES),
        },
        "training": {
            "ridge": ridge,
            "window_count": sum(len(session.features) for session in sessions),
            "trial_count": sum(len(set(session.trial_ids)) for session in sessions),
        },
        "validation": validation,
        "model": model.to_dict(),
        "quantized_model": quantized_model.to_dict(),
        "quantization": {
            "float_prediction_agreement": quantized_agreement,
            "theoretical_abs_score_bound": theoretical_score_bound(
                quantized_model
            ).tolist(),
            "observed_max_abs_score": int(
                np.max(np.abs(quantized_model.scores(np.vstack([
                    session.features for session in sessions
                ]))))
            ),
        },
        "deployment_status": deployment_status(quantized_model),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="datasets/emg",
        help="directory containing session_*/session.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="model JSON (default: DATASET_ROOT/lda_model.json)",
    )
    parser.add_argument(
        "--require-donning",
        action="store_true",
        help=(
            "train only on sessions that record which electrode application "
            "they came from, so folds can be held out by donning. Sessions "
            "recorded before that field existed are listed as skipped."
        ),
    )
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument(
        "--fraction-bits",
        type=int,
        default=DEFAULT_QUANTIZATION_BITS,
        help="shared binary scale for raw-space LDA coefficients",
    )
    parser.add_argument(
        "--c-output",
        type=pathlib.Path,
        help="optional generated emg_classifier_model.h path",
    )
    parser.add_argument(
        "--zero-crossing-threshold",
        type=int,
        default=ZERO_CROSSING_THRESHOLD,
    )
    arguments = parser.parse_args(argv)
    if arguments.zero_crossing_threshold < 0:
        parser.error("--zero-crossing-threshold must be non-negative")

    root = pathlib.Path(arguments.dataset_root)
    output = arguments.output or root / "lda_model.json"
    try:
        sessions, skipped, labels = load_dataset(
            root,
            arguments.zero_crossing_threshold,
            require_donning=arguments.require_donning,
        )
        if tuple(labels) not in (PROTOCOL_COMMAND_LABELS, FIRMWARE_LABELS):
            print(
                f"Label set {list(labels)} is not the four protocol commands; "
                "this model can be scored but not emitted as firmware."
            )
        print(f"Loaded {len(sessions)} complete sessions:")
        for session in sessions:
            trials = len(set(session.trial_ids))
            print(f"  {session.session_id}: {len(session.features)} windows, "
                  f"{trials} trials")
        for item in skipped:
            print(f"Skipped {item['session_id']}: status={item['status']}")

        validation = evaluate_loso(
            sessions, ridge=arguments.ridge, labels=labels
        )
        print(f"\n{validation['method'].replace('_', '-')}:")
        if validation.get("caveat"):
            print(f"  NOTE: {validation['caveat']}")
        for fold in validation["folds"]:
            print(
                f"  {fold['held_out_session']}: "
                f"window={_percent(fold['window']['accuracy'])} "
                f"trial={_percent(fold['trial']['accuracy'])}"
            )
        print("\nOverall window confusion (rows=true, columns=predicted):")
        print(format_confusion(
            validation["overall_window"]["confusion"], labels
        ))
        print(f"window accuracy: "
              f"{_percent(validation['overall_window']['accuracy'])}")
        print("\nOverall trial confusion (rows=true, columns=predicted):")
        print(format_confusion(
            validation["overall_trial"]["confusion"], labels
        ))
        print(f"trial accuracy: "
              f"{_percent(validation['overall_trial']['accuracy'])}")

        all_features = np.vstack([session.features for session in sessions])
        all_labels = np.concatenate([session.labels for session in sessions])
        final_model = fit_lda(
            all_features, all_labels, class_order=labels, ridge=arguments.ridge
        )
        quantized_model = quantize_lda(final_model, arguments.fraction_bits)
        float_predictions = final_model.predict(all_features)
        quantized_predictions = quantized_model.predict(all_features)
        quantized_agreement = float(np.mean(
            float_predictions == quantized_predictions
        ))
        print(
            f"Q{arguments.fraction_bits} float agreement: "
            f"{100.0 * quantized_agreement:.3f}% "
            f"({int(np.count_nonzero(float_predictions == quantized_predictions))}"
            f"/{len(all_features)})"
        )
        quantized_validation = evaluate_loso(
            sessions, ridge=arguments.ridge, labels=labels,
            quantize_bits=arguments.fraction_bits,
        )
        float_held_out = validation["overall_window"]["accuracy"]
        quantized_held_out = quantized_validation["overall_window"]["accuracy"]
        print(
            f"held-out accuracy: float {_percent(float_held_out)}, "
            f"quantized {_percent(quantized_held_out)}"
        )
        if quantized_agreement < QUANTIZATION_MIN_AGREEMENT:
            raise ValueError(
                f"quantized model disagrees with the float model on "
                f"{100.0 * (1.0 - quantized_agreement):.2f}% of source "
                f"windows; that is a broken fixed-point scale, not rounding"
            )
        if quantized_held_out < float_held_out - QUANTIZATION_MAX_ACCURACY_LOSS:
            raise ValueError(
                f"quantizing costs held-out accuracy: "
                f"{_percent(float_held_out)} float against "
                f"{_percent(quantized_held_out)} quantized. The model that "
                f"ships is the quantized one, so this is a real loss."
            )
        payload = build_output(
            sessions,
            skipped,
            validation,
            final_model,
            quantized_model,
            quantized_agreement,
            arguments.zero_crossing_threshold,
            arguments.ridge,
            labels,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")
        if arguments.c_output is not None:
            arguments.c_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.c_output.write_text(
                render_c_model_header(
                    quantized_model,
                    [session.session_id for session in sessions],
                ),
                encoding="utf-8",
            )
            print(f"Wrote {arguments.c_output}")
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
