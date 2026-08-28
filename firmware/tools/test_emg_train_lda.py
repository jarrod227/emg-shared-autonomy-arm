"""Tests for labelled-window extraction and the host LDA baseline."""

from dataclasses import replace
import json

import numpy as np
import pytest

from emg_train_lda import (
    SessionFeatures,
    evaluate_loso,
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
    FIRMWARE_LABELS,
    PROTOCOL_COMMAND_LABELS,
    load_dataset,
    manifest_labels,
    render_c_model_header,
    select_complete_manifests,
    trial_predictions,
    theoretical_score_bound,
    validate_complete_manifest,
)


def manifest_for_segments(segments, *, session_id="session_test", status="complete",
                         gesture_actions=None):
    included = [segment for segment in segments if segment["include"]]
    if gesture_actions is None:
        # Derived from the planned trials, exactly as GuidedSession.to_manifest
        # does, so a fixture cannot claim a label set the segments never used.
        gesture_actions = {
            item["trial"]["label"]: item["trial"]["action"] for item in segments
        }
    return {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "final_phase": status,
        "gesture_actions": gesture_actions,
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


def balanced_manifest(*, session_id="session_test", labels=LABELS):
    segments = []
    for index, label in enumerate(labels):
        start = 1000 + 1000 * index
        segments.append(segment(index, label, start, start + 600))
    return manifest_for_segments(segments, session_id=session_id)


def test_manifest_requires_complete_balanced_labels():
    manifest = balanced_manifest()
    assert len(validate_complete_manifest(manifest)) == len(LABELS)

    manifest["segments"][0]["include"] = False
    manifest["completed_trials"] -= 1
    manifest["total_trials"] -= 1
    with pytest.raises(ValueError, match="balanced"):
        validate_complete_manifest(manifest)


# ULNAR is a model class now, so the screening set adds only the two
# candidates that never made it in.
CANDIDATE_LABELS = LABELS + ("RADIAL", "PRONATE")


def test_labels_come_from_the_session_not_the_module_constant():
    # Candidate-gesture screening records label sets the firmware model does
    # not contain. Validating against the module constant would reject those
    # sessions as unbalanced even though they are perfectly balanced.
    manifest = balanced_manifest(labels=CANDIDATE_LABELS)

    # Order is a separate contract, covered below; what matters here is that
    # every recorded label reaches the trainer.
    assert set(manifest_labels(manifest)) == set(CANDIDATE_LABELS)
    assert len(validate_complete_manifest(manifest)) == len(CANDIDATE_LABELS)


def test_protocol_commands_keep_their_frozen_class_order():
    # Class order fixes the row order of the emitted coefficient table, and
    # render_c_model_header only accepts PROTOCOL_COMMAND_LABELS in exactly
    # that order. Taking the order from the manifest's JSON keys silently
    # alphabetized it, which left the live firmware model unregenerable while
    # every test still passed.
    alphabetical = {
        label: label for label in sorted(PROTOCOL_COMMAND_LABELS)
    }
    manifest = balanced_manifest()
    manifest["gesture_actions"] = alphabetical

    assert manifest_labels(manifest) == PROTOCOL_COMMAND_LABELS


def test_candidate_labels_sort_after_the_firmware_classes():
    # Any deterministic position works for exploratory labels; what must not
    # move is where the classes the firmware holds sit, because class order
    # fixes the row order of the emitted coefficient table and the firmware
    # indexes it with its own enum.
    manifest = balanced_manifest(labels=CANDIDATE_LABELS)
    ordered = manifest_labels(manifest)

    assert ordered[:len(FIRMWARE_LABELS)] == FIRMWARE_LABELS
    assert ordered[len(FIRMWARE_LABELS):] == ("PRONATE", "RADIAL")


def test_a_direction_only_class_may_be_emitted_as_firmware():
    # ULNAR steers the proportional channel and is rewritten to REST before
    # the gate, so it is a class the firmware holds without being a command.
    # The emitter has to accept it and must still reject anything else.
    assert FIRMWARE_LABELS == PROTOCOL_COMMAND_LABELS + ("ULNAR",)
    assert "ULNAR" not in PROTOCOL_COMMAND_LABELS


def test_a_manifest_without_a_recorded_label_set_is_rejected():
    # Assuming the four protocol commands would let a session whose real label
    # set is unknown train silently against the wrong classes.
    manifest = balanced_manifest()
    del manifest["gesture_actions"]
    with pytest.raises(ValueError, match="gesture_actions"):
        validate_complete_manifest(manifest)

    manifest["gesture_actions"] = {}
    with pytest.raises(ValueError, match="gesture_actions"):
        validate_complete_manifest(manifest)


def test_an_explicit_label_set_overrides_what_the_session_recorded():
    # This is how load_dataset enforces agreement: every session is validated
    # against the set resolved from the first one.
    manifest = balanced_manifest(labels=CANDIDATE_LABELS)
    with pytest.raises(ValueError, match="balanced across"):
        validate_complete_manifest(manifest, labels=LABELS)


def test_sessions_that_disagree_on_labels_cannot_be_pooled(tmp_path):
    # Pooling them would train a fold on classes the held-out session never
    # collected, and the confusion matrix would gain a silent all-zero row.
    for name, labels in (
        ("session_a", LABELS),
        ("session_b", CANDIDATE_LABELS),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "session.json").write_text(
            json.dumps(balanced_manifest(session_id=name, labels=labels)),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="disagree on their label set"):
        load_dataset(tmp_path)


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
    assert features.shape == (3 * len(LABELS), len(FEATURE_NAMES))
    assert {label: int(np.count_nonzero(labels == label)) for label in LABELS} == {
        label: 3 for label in LABELS
    }
    assert len(set(trial_ids)) == len(LABELS)


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


def features_for(session_id, donning, rows=8, label_cycle=None):
    labels = np.array(
        [(label_cycle or LABELS)[i % len(label_cycle or LABELS)]
         for i in range(rows)]
    )
    rng = np.random.default_rng(abs(hash(session_id)) % (2 ** 32))
    return SessionFeatures(
        session_id=session_id,
        features=rng.normal(size=(rows, len(FEATURE_NAMES))),
        labels=labels,
        trial_ids=np.arange(rows),
        donning=donning,
    )


def test_folds_are_held_out_by_donning_not_by_session():
    """Two sessions on one donning must never straddle a fold.

    Leave-one-session-out put a held-out session's own donning back into the
    training set, so the accepted number measured within-donning performance.
    Held-out donnings from 2026-08-14 give NEXT_TARGET at 2%, 12%, 68%, 100%
    and 100% -- the spread the reported figure could not contain.
    """
    sessions = [
        features_for("a1", "donningA"),
        features_for("a2", "donningA"),
        features_for("b1", "donningB"),
    ]

    result = evaluate_loso(sessions)

    assert result["method"] == "leave_one_donning_out"
    assert len(result["folds"]) == 2
    held = {fold["held_out_session"] for fold in result["folds"]}
    assert "a1+a2" in held and "b1" in held
    for fold in result["folds"]:
        assert not (set(fold["training_sessions"]) & {"a1", "a2"}) or \
            fold["held_out_session"] != "a1+a2"


def test_sessions_without_a_donning_are_reported_as_not_cross_donning():
    sessions = [features_for("a1", None), features_for("b1", None)]

    result = evaluate_loso(sessions)

    assert result["method"] == "leave_one_session_out"
    assert "cannot be grouped" in result["caveat"]


def test_a_single_donning_cannot_be_validated_at_all():
    # Every session on one electrode application says nothing about the
    # variation the wearer will actually meet.
    sessions = [
        features_for("a1", "donningA"),
        features_for("a2", "donningA"),
    ]

    with pytest.raises(ValueError, match="two donnings"):
        evaluate_loso(sessions)
