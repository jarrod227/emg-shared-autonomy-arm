"""Tests for labelled-window extraction and the host LDA baseline."""

from dataclasses import replace
import json

import numpy as np
import pytest

from emg_train_lda import (
    FEATURE_NAMES,
    LABELS,
    LDAModel,
    QuantizedLDAModel,
    SessionFeatures,
    deployment_status,
    evaluate_loso,
    extract_feature_windows,
    fit_lda,
    quantize_lda,
    render_c_model_header,
    select_complete_manifests,
    trial_predictions,
    theoretical_score_bound,
    validate_complete_manifest,
)


def manifest_for_segments(segments, *, session_id="session_test", status="complete"):
    included = [segment for segment in segments if segment["include"]]
    return {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "final_phase": status,
        "total_trials": len(included),
        "completed_trials": len(included),
        "segments": segments,
    }


def segment(index, label, start, end, *, include=True, attempt=1):
    return {
        "trial": {
            "index": index,
            "label": label,
            "action": label,
            "repetition": 1,
        },
        "attempt": attempt,
        "start": {"frame_index": start},
        "end": {"frame_index": end},
        "include": include,
        "reasons": [] if include else ["manual_pause"],
    }


def balanced_manifest(*, session_id="session_test"):
    segments = []
    for index, label in enumerate(LABELS):
        start = 1000 + 1000 * index
        segments.append(segment(index, label, start, start + 600))
    return manifest_for_segments(segments, session_id=session_id)


def test_manifest_requires_complete_balanced_labels():
    manifest = balanced_manifest()
    assert len(validate_complete_manifest(manifest)) == 4

    manifest["segments"][0]["include"] = False
    manifest["completed_trials"] -= 1
    manifest["total_trials"] -= 1
    with pytest.raises(ValueError, match="balanced"):
        validate_complete_manifest(manifest)


def test_manifest_rejects_overlapping_accepted_segments():
    manifest = balanced_manifest()
    manifest["segments"][1]["start"]["frame_index"] = 1500
    with pytest.raises(ValueError, match="overlap"):
        validate_complete_manifest(manifest)


def test_global_windows_stay_wholly_inside_included_segments():
    manifest = balanced_manifest()
    # A rejected attempt must never contribute even though it contains data.
    manifest["segments"].append(segment(9, "REST", 5200, 5800, include=False))
    samples = np.arange(6000, dtype=np.int64)
    columns = [samples, -samples, samples % 31 - 15]

    features, labels, trial_ids = extract_feature_windows(columns, manifest)

    # Each aligned 600-frame segment yields ends at +400, +500, +600.
    assert features.shape == (12, len(FEATURE_NAMES))
    assert {label: int(np.count_nonzero(labels == label)) for label in LABELS} == {
        label: 3 for label in LABELS
    }
    assert len(set(trial_ids)) == 4


def test_complete_selector_skips_stopped_session(tmp_path):
    complete = tmp_path / "session_complete"
    stopped = tmp_path / "session_stopped"
    complete.mkdir()
    stopped.mkdir()
    (complete / "session.json").write_text(
        json.dumps(balanced_manifest(session_id="session_complete")),
        encoding="utf-8",
    )
    stopped_manifest = balanced_manifest(session_id="session_stopped")
    stopped_manifest["status"] = "stopped"
    stopped_manifest["final_phase"] = "stopped"
    (stopped / "session.json").write_text(
        json.dumps(stopped_manifest), encoding="utf-8"
    )

    selected, skipped = select_complete_manifests(tmp_path)

    assert selected == [complete / "session.json"]
    assert skipped == [{"session_id": "session_stopped", "status": "stopped"}]


def separable_rows(session_index, repeats=5):
    rows = []
    labels = []
    trial_ids = []
    for class_index, label in enumerate(LABELS):
        centre = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        centre[class_index] = 20.0
        for repeat in range(repeats):
            row = centre.copy()
            row[4:] = session_index * 0.05
            row[class_index] += repeat * 0.01
            rows.append(row)
            labels.append(label)
            trial_ids.append(f"s{session_index}:{label}:{repeat // 2}")
    return SessionFeatures(
        session_id=f"session_{session_index}",
        features=np.asarray(rows),
        labels=np.asarray(labels, dtype=object),
        trial_ids=np.asarray(trial_ids, dtype=object),
    )


def quantization_fixture():
    """Well-conditioned affine model with a large, deterministic class margin."""
    mean = np.arange(len(FEATURE_NAMES), dtype=np.float64) * 10.0
    scale = np.arange(1, len(FEATURE_NAMES) + 1, dtype=np.float64)
    weights = np.zeros((len(LABELS), len(FEATURE_NAMES)), dtype=np.float64)
    for class_index in range(len(LABELS)):
        weights[class_index, class_index] = 1.0
    model = LDAModel(
        labels=LABELS,
        feature_names=FEATURE_NAMES,
        mean=mean,
        scale=scale,
        weights=weights,
        intercept=np.zeros(len(LABELS), dtype=np.float64),
        ridge=0.001,
        effective_ridge=0.001,
    )
    features = np.tile(mean, (len(LABELS), 1))
    for class_index in range(len(LABELS)):
        features[class_index, class_index] += 100.0 * scale[class_index]
    labels = np.asarray(LABELS, dtype=object)
    return model, features, labels


def test_lda_model_round_trip_preserves_predictions():
    session = separable_rows(0, repeats=8)
    model = fit_lda(session.features, session.labels)
    restored = LDAModel.from_dict(model.to_dict())

    assert np.array_equal(model.predict(session.features), session.labels)
    assert np.array_equal(restored.predict(session.features), session.labels)
    assert restored.feature_names == FEATURE_NAMES


def test_q18_raw_affine_quantization_preserves_separable_predictions():
    float_model, features, labels = quantization_fixture()
    quantized = quantize_lda(float_model, fraction_bits=18)
    restored = QuantizedLDAModel.from_dict(quantized.to_dict())

    expected = float_model.predict(features)
    assert np.array_equal(expected, labels)
    assert np.array_equal(quantized.predict(features), expected)
    assert np.array_equal(restored.predict(features), expected)
    assert quantized.weights.dtype == np.int64
    assert max(theoretical_score_bound(quantized)) < np.iinfo(np.int64).max


def test_c_header_contains_integer_model_and_source_provenance():
    float_model, _features, _labels = quantization_fixture()
    model = quantize_lda(float_model, 18)

    header = render_c_model_header(model, ["session_a", "session_b"])

    assert "EMG_CLASSIFIER_MODEL_FRACTION_BITS 18u" in header
    assert "int32_t emg_classifier_model_weights" in header
    assert "int64_t emg_classifier_model_intercept" in header
    assert "EMG_CLASSIFIER_MODEL_CLASS_3_COMMAND 3u /* ABORT */" in header
    assert "emg_classifier_model_commands" in header
    assert "session_a, session_b" in header


def test_c_header_rejects_class_order_that_disagrees_with_protocol():
    float_model, _features, _labels = quantization_fixture()
    model = quantize_lda(float_model, 18)
    reordered = replace(model, labels=tuple(reversed(model.labels)))

    with pytest.raises(ValueError, match="model class order"):
        render_c_model_header(reordered)


def test_quantization_rejects_non_finite_and_out_of_range_values():
    float_model, _features, _labels = quantization_fixture()
    bad_weights = float_model.weights.copy()
    bad_weights[0, 0] = np.inf
    with pytest.raises(ValueError, match="weights must be finite"):
        quantize_lda(replace(float_model, weights=bad_weights), 18)

    bad_intercept = float_model.intercept.copy()
    bad_intercept[0] = float(2**63) / float(1 << 18)
    with pytest.raises(ValueError, match="intercept do not fit"):
        quantize_lda(replace(float_model, intercept=bad_intercept), 18)


def test_deployment_status_uses_selected_fraction_bits():
    float_model, _features, _labels = quantization_fixture()

    assert deployment_status(quantize_lda(float_model, 16)) == (
        "host_q16_parameters_not_live_integrated"
    )
    assert deployment_status(quantize_lda(float_model, 18)) == (
        "host_q18_parameters_not_live_integrated"
    )


def test_loso_never_trains_on_held_out_session():
    sessions = [separable_rows(index) for index in range(3)]

    result = evaluate_loso(sessions)

    assert result["overall_window"]["accuracy"] == pytest.approx(1.0)
    assert result["overall_trial"]["accuracy"] == pytest.approx(1.0)
    for fold in result["folds"]:
        assert fold["held_out_session"] not in fold["training_sessions"]
        assert len(fold["training_sessions"]) == 2


def test_trial_vote_uses_majority_and_score_for_tie():
    actual = np.asarray(["REST"] * 4, dtype=object)
    predicted = np.asarray(["REST", "REST", "ABORT", "ABORT"], dtype=object)
    scores = np.zeros((4, len(LABELS)), dtype=np.float64)
    scores[:, LABELS.index("ABORT")] = 2.0
    ids = np.asarray(["trial"] * 4, dtype=object)

    trial_actual, trial_predicted = trial_predictions(
        actual, predicted, scores, ids
    )

    assert trial_actual.tolist() == ["REST"]
    assert trial_predicted.tolist() == ["ABORT"]
