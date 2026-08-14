"""Tests for the offline Objective 3.5 discrete-event gate replay."""

import numpy as np
import pytest

import emg_event_gate_replay as replay
from emg_event_gate_replay import (
    EventGate,
    GateConfig,
    PreparedFold,
    SessionTimeline,
    TrialSpan,
    build_feature_timeline,
    default_sweep_configs,
    evaluate_gate,
    prepare_external_validation_folds,
    validate_event_gate_manifest,
    validation_sequence_passed,
)


def config(**changes):
    values = {
        "stable_windows": 2,
        "rest_rearm_windows": 2,
        "refractory_windows": 2,
        "abort_stable_windows": 2,
        "onset_holdoff_windows": 0,
    }
    values.update(changes)
    return GateConfig(**values)


def event_manifest(labels=None):
    labels = labels or (
        "REST",
        "NEXT_TARGET",
        "REST",
        "CONFIRM",
        "REST",
        "ABORT",
        "REST",
    )
    segments = []
    for index, label in enumerate(labels):
        start = index * 1000
        segments.append({
            "trial": {"index": index, "label": label},
            "attempt": 1,
            "start": {"frame_index": start},
            "end": {"frame_index": start + 800},
            "include": True,
            "reasons": [],
        })
    return {
        "schema_version": 1,
        "collection_protocol": "event-gate",
        "status": "complete",
        "final_phase": "complete",
        "completed_trials": len(segments),
        "total_trials": len(segments),
        "segments": segments,
    }


def test_event_gate_manifest_requires_balanced_rest_wrapped_events():
    included = validate_event_gate_manifest(event_manifest(), "session.json")

    assert [item["trial"]["label"] for item in included] == [
        "REST",
        "NEXT_TARGET",
        "REST",
        "CONFIRM",
        "REST",
        "ABORT",
        "REST",
    ]


@pytest.mark.parametrize(
    "manifest,reason",
    [
        (
            {**event_manifest(), "collection_protocol": "classifier"},
            "collection_protocol",
        ),
        (
            event_manifest((
                "REST",
                "NEXT_TARGET",
                "CONFIRM",
                "REST",
                "ABORT",
                "REST",
                "NEXT_TARGET",
            )),
            "surrounded by REST",
        ),
        (
            event_manifest(("REST", "NEXT_TARGET", "REST")),
            "balanced",
        ),
    ],
)
def test_event_gate_manifest_rejects_unsafe_protocol_shapes(manifest, reason):
    with pytest.raises(ValueError, match=reason):
        validate_event_gate_manifest(manifest, "session.json")


def test_gate_requires_rest_then_emits_only_once_for_held_gesture():
    gate = EventGate(config())

    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("REST") is None
    assert gate.push("REST") is None
    assert gate.armed
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") == "NEXT_TARGET"
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") is None


def test_gate_rearms_only_after_rest_and_refractory():
    gate = EventGate(config(refractory_windows=3))
    for prediction in ("REST", "REST", "CONFIRM", "CONFIRM"):
        event = gate.push(prediction)
    assert event == "CONFIRM"

    assert gate.push("REST") is None
    assert gate.push("REST") is None
    assert not gate.armed
    assert gate.push("REST") is None
    assert gate.armed
    assert gate.push("CONFIRM") is None
    assert gate.push("CONFIRM") == "CONFIRM"


def test_abort_bypasses_ordinary_arming_but_latches_until_rest():
    gate = EventGate(config(abort_stable_windows=2))

    assert gate.push("ABORT") is None
    assert gate.push("ABORT") == "ABORT"
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") is None
    assert gate.push("REST") is None
    assert gate.push("REST") is None
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") == "ABORT"


def test_invalid_window_clears_partial_evidence_and_requires_rearm():
    gate = EventGate(config())
    gate.push("REST")
    gate.push("REST")
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET", valid=False) is None
    assert not gate.armed
    assert gate.push("NEXT_TARGET") is None
    gate.push("REST")
    gate.push("REST")
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") == "NEXT_TARGET"


def test_onset_holdoff_discards_the_windows_that_straddle_rest_and_contraction():
    gate = EventGate(config(onset_holdoff_windows=3))
    gate.push("REST")
    gate.push("REST")
    assert gate.armed

    # The first three contraction windows are dropped whatever they say, so a
    # confident onset misclassification cannot claim the event.
    assert gate.push("CONFIRM") is None
    assert gate.holding_off
    assert gate.push("CONFIRM") is None
    assert gate.push("CONFIRM") is None
    assert not gate.holding_off
    # Evidence starts from scratch afterwards: hold-off never counts toward the
    # stable run.
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") == "NEXT_TARGET"


def test_onset_holdoff_restarts_only_after_returning_to_rest():
    gate = EventGate(config(onset_holdoff_windows=2, refractory_windows=0))
    gate.push("REST")
    gate.push("REST")
    gate.push("NEXT_TARGET")
    gate.push("NEXT_TARGET")
    assert gate.push("NEXT_TARGET") is None
    assert gate.push("NEXT_TARGET") == "NEXT_TARGET"

    # A held gesture stays out of hold-off; only a rest window re-arms it.
    assert not gate.holding_off
    gate.push("REST")
    gate.push("REST")
    assert gate.push("CONFIRM") is None
    assert gate.holding_off


def test_rest_during_holdoff_cancels_it_without_consuming_the_gesture():
    gate = EventGate(config(onset_holdoff_windows=4))
    gate.push("REST")
    gate.push("REST")
    assert gate.push("CONFIRM") is None
    assert gate.holding_off

    # A twitch that falls back to rest must not leave a half-spent hold-off
    # that would let the next real onset through unfiltered.
    gate.push("REST")
    assert not gate.holding_off
    gate.push("REST")
    for _ in range(4):
        assert gate.push("CONFIRM") is None
    assert gate.push("CONFIRM") is None
    assert gate.push("CONFIRM") == "CONFIRM"


def test_onset_holdoff_also_covers_abort():
    gate = EventGate(config(abort_stable_windows=2, onset_holdoff_windows=3))

    # ABORT bypasses arming and refractory, but not the hold-off - onset
    # windows are unreadable for every class, not just the ordinary ones.
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") is None
    assert gate.push("ABORT") == "ABORT"


def test_invalid_window_makes_the_next_contraction_an_onset_again():
    gate = EventGate(config(onset_holdoff_windows=2))
    gate.push("REST")
    gate.push("REST")
    gate.push("CONFIRM")
    gate.push("CONFIRM")
    assert not gate.holding_off

    gate.push("CONFIRM", valid=False)
    assert gate.push("CONFIRM") is None
    assert gate.holding_off


@pytest.mark.parametrize(
    "field,value",
    [
        ("stable_windows", 0),
        ("rest_rearm_windows", 0),
        ("refractory_windows", -1),
        ("abort_stable_windows", 0),
        ("onset_holdoff_windows", -1),
    ],
)
def test_gate_config_rejects_invalid_counts(field, value):
    values = {
        "stable_windows": 2,
        "rest_rearm_windows": 2,
        "refractory_windows": 2,
        "abort_stable_windows": 2,
        "onset_holdoff_windows": 2,
    }
    values[field] = value
    with pytest.raises(ValueError):
        GateConfig(**values)


def test_global_feature_grid_and_wear_validity_are_preserved():
    columns = [
        np.arange(600, dtype=np.int64),
        -np.arange(600, dtype=np.int64),
        np.arange(600, dtype=np.int64) % 17,
    ]
    valid = np.ones(600, dtype=bool)
    valid[450] = False
    trial = TrialSpan("trial", "CONFIRM", 0, 600)

    features, ends, windows_valid, labels, trial_ids = build_feature_timeline(
        columns, valid, (trial,)
    )

    assert features.shape == (3, 12)
    assert ends.tolist() == [400, 500, 600]
    assert windows_valid.tolist() == [True, False, False]
    assert labels.tolist() == ["CONFIRM"] * 3
    assert trial_ids.tolist() == ["trial"] * 3


def small_fold():
    rest = TrialSpan("rest", "REST", 0, 500)
    active = TrialSpan("active", "NEXT_TARGET", 500, 1000)
    timeline = SessionTimeline(
        session_id="held",
        sample_rate_hz=2000,
        features=np.zeros((10, 12), dtype=np.float64),
        frame_ends=np.arange(100, 1100, 100, dtype=np.int64),
        valid=np.ones(10, dtype=bool),
        labels=np.asarray(["REST"] * 5 + ["NEXT_TARGET"] * 5, dtype=object),
        trial_ids=np.asarray(["rest"] * 5 + ["active"] * 5, dtype=object),
        trials=(rest, active),
    )
    predictions = np.asarray(
        ["REST"] * 5 + ["NEXT_TARGET"] * 5, dtype=object
    )
    return PreparedFold("held", ("train_a", "train_b"), timeline, predictions, 1.0)


def test_gate_metrics_keep_rest_and_active_trials_separate():
    metrics = evaluate_gate([small_fold()], config(refractory_windows=0))

    assert metrics["active_trials"] == 1
    assert metrics["clean_correct_trials"] == 1
    assert metrics["missed_active_trials"] == 0
    assert metrics["wrong_active_trials"] == 0
    assert metrics["rest_trials"] == 1
    assert metrics["rest_false_events"] == 0
    assert metrics["min_q18_float_agreement"] == 1.0
    assert validation_sequence_passed(metrics)

    metrics["unlabelled_events"] = 1
    assert not validation_sequence_passed(metrics)


def test_external_validation_samples_never_enter_model_fit(monkeypatch):
    training = SessionTimeline(
        session_id="classifier_train",
        sample_rate_hz=2000,
        features=np.ones((4, 12), dtype=np.float64),
        frame_ends=np.arange(400, 800, 100, dtype=np.int64),
        valid=np.ones(4, dtype=bool),
        labels=np.asarray(["REST", "NEXT_TARGET", "CONFIRM", "ABORT"], dtype=object),
        trial_ids=np.asarray(["r", "n", "c", "a"], dtype=object),
        trials=(),
    )
    validation = SessionTimeline(
        session_id="event_validation",
        sample_rate_hz=2000,
        features=np.full((3, 12), 999.0, dtype=np.float64),
        frame_ends=np.arange(400, 700, 100, dtype=np.int64),
        valid=np.ones(3, dtype=bool),
        labels=np.asarray(["REST", "NEXT_TARGET", "REST"], dtype=object),
        trial_ids=np.asarray(["r0", "n", "r1"], dtype=object),
        trials=(),
    )
    fitted = {}

    class FakeModel:
        def predict(self, features):
            return np.asarray(["REST"] * len(features), dtype=object)

    def fake_fit(features, labels):
        fitted["features"] = features.copy()
        fitted["labels"] = labels.copy()
        return FakeModel()

    monkeypatch.setattr(replay, "fit_lda", fake_fit)
    monkeypatch.setattr(replay, "quantize_lda", lambda _model, _bits: FakeModel())

    folds = prepare_external_validation_folds((training,), (validation,))

    assert fitted["features"].shape == (4, 12)
    assert np.all(fitted["features"] == 1.0)
    assert folds[0].training_sessions == ("classifier_train",)
    assert folds[0].held_out_session == "event_validation"


def test_move_now_transition_event_belongs_to_following_trial():
    trial = TrialSpan("active", "NEXT_TARGET", 500, 1000)
    timeline = SessionTimeline(
        session_id="held",
        sample_rate_hz=2000,
        features=np.zeros((10, 12), dtype=np.float64),
        frame_ends=np.arange(100, 1100, 100, dtype=np.int64),
        valid=np.ones(10, dtype=bool),
        labels=np.asarray([None] * 5 + ["NEXT_TARGET"] * 5, dtype=object),
        trial_ids=np.asarray([None] * 5 + ["active"] * 5, dtype=object),
        trials=(trial,),
        transition_frames=200,
    )
    predictions = np.asarray(
        ["REST", "REST", "NEXT_TARGET", "NEXT_TARGET"]
        + ["NEXT_TARGET"] * 6,
        dtype=object,
    )
    fold = PreparedFold("held", ("train",), timeline, predictions, 1.0)

    metrics = evaluate_gate([fold], config(refractory_windows=0))

    assert metrics["clean_correct_trials"] == 1
    assert metrics["transition_events"] == 1
    assert metrics["unlabelled_events"] == 0
    assert metrics["latency_p50_sec"] == pytest.approx(-0.05)


def test_straddling_feature_window_uses_event_time_for_active_ownership():
    trial = TrialSpan("active", "NEXT_TARGET", 500, 1000)
    timeline = SessionTimeline(
        session_id="held",
        sample_rate_hz=2000,
        features=np.zeros((6, 12), dtype=np.float64),
        frame_ends=np.arange(100, 700, 100, dtype=np.int64),
        valid=np.ones(6, dtype=bool),
        labels=np.asarray([None] * 6, dtype=object),
        trial_ids=np.asarray([None] * 6, dtype=object),
        trials=(trial,),
        transition_frames=200,
    )
    predictions = np.asarray(
        ["REST", "REST", "REST", "NEXT_TARGET", "NEXT_TARGET", "NEXT_TARGET"],
        dtype=object,
    )
    fold = PreparedFold("held", ("train",), timeline, predictions, 1.0)

    metrics = evaluate_gate([fold], config(refractory_windows=0))

    assert metrics["clean_correct_trials"] == 1
    assert metrics["transition_events"] == 0
    assert metrics["unlabelled_events"] == 0
    assert metrics["latency_p50_sec"] == pytest.approx(0.0)


def test_default_sweep_is_fixed_and_reviewable():
    candidates = default_sweep_configs()

    assert len(candidates) == 1200
    assert candidates[0] == GateConfig(2, 2, 5, 1, 0)
    assert candidates[-1] == GateConfig(5, 6, 20, 3, 16)
    # Hold-off must be swept, not silently defaulted off.
    assert {item.onset_holdoff_windows for item in candidates} == {0, 4, 8, 12, 16}
