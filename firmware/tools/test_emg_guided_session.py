"""Headless tests for the guided Objective 3.5 collection protocol."""

from dataclasses import replace
import datetime
import pathlib
from types import SimpleNamespace

import pytest

import emg_guided_capture as capture
from emg_guided_session import (
    EVENT_GATE_LABELS,
    CLASSIFIER_PROTOCOL,
    EVENT_GATE_PROTOCOL,
    GESTURE_ACTIONS,
    GESTURE_LABELS,
    GuidedSession,
    Phase,
    StreamPosition,
    TrialSpec,
    build_collection_plan,
    build_event_gate_plan,
    build_trial_plan,
)


def position(
    frame,
    *,
    timestamp=None,
    detached=(0, 0, 0),
    lost=0,
    malformed=0,
    duplicated=0,
    reversed_time=0,
):
    return StreamPosition(
        frame_index=frame,
        timestamp_us=timestamp,
        detached_by_channel=detached,
        lost_packets=lost,
        malformed_packets=malformed,
        duplicated_packets=duplicated,
        time_reversed_packets=reversed_time,
    )


def one_trial(label="CONFIRM"):
    return (
        TrialSpec(
            index=0,
            label=label,
            action=GESTURE_ACTIONS[label],
            repetition=1,
        ),
    )


def enter_active(session, start_frame=100):
    session.start(0.0)
    assert session.advance(2.0, position(20))
    assert session.phase is Phase.TRANSITION
    assert session.advance(2.5, position(start_frame))
    assert session.phase is Phase.ACTIVE


def finish_active_and_verify(session, active_end, verify_end=None):
    """End the labelled span, then supply the packet that verifies its tail."""
    assert session.advance(5.5, active_end)
    assert session.phase is Phase.VERIFY
    if verify_end is None:
        verify_end = replace(active_end, frame_index=active_end.frame_index + 32)
    assert session.advance(5.61, verify_end)


def test_accepted_gesture_mapping_is_frozen():
    """Frozen against the deployed model, not against habit.

    ULNAR joined the model on 2026-08-23 and was missing here until 2026-08-27,
    so every session recorded in between omitted the class the model was
    newest and least validated on.
    """
    assert GESTURE_ACTIONS == {
        "REST": "RELAX",
        "NEXT_TARGET": "WRIST UP",
        "CONFIRM": "MAKE A FIST",
        "ABORT": "WRIST DOWN",
        "ULNAR": "TILT TOWARD THE LITTLE FINGER",
    }


def test_the_collectable_gestures_are_the_ones_the_model_has():
    # A class the model judges but this tool cannot record is a class that can
    # only ever be trained on data from somewhere else.
    from emg_train_lda import LABELS

    assert set(GESTURE_ACTIONS) == set(LABELS)


def test_plan_is_balanced_by_randomized_repetition_block():
    first = build_trial_plan(4, seed=123)
    second = build_trial_plan(4, seed=123)

    assert first == second
    assert len(first) == 4 * len(GESTURE_LABELS)
    for start in range(0, len(first), len(GESTURE_LABELS)):
        block = first[start:start + len(GESTURE_LABELS)]
        assert {trial.label for trial in block} == set(GESTURE_LABELS)
        assert {trial.repetition for trial in block} == {
            start // len(GESTURE_LABELS) + 1
        }


def test_event_gate_plan_wraps_every_event_in_labelled_rest():
    plan = build_event_gate_plan(3, seed=123)

    assert len(plan) == 1 + 2 * 3 * (len(EVENT_GATE_LABELS) - 1)
    assert plan[0].label == plan[-1].label == "REST"
    for index, trial in enumerate(plan):
        assert trial.index == index
        if trial.label != "REST":
            assert plan[index - 1].label == "REST"
            assert plan[index + 1].label == "REST"
    for repetition in range(1, 4):
        labels = {
            trial.label
            for trial in plan
            if trial.repetition == repetition and trial.label != "REST"
        }
        # Direction-only classes are absent: they produce no event for a
        # REST bracket to wrap.
        assert labels == set(EVENT_GATE_LABELS) - {"REST"}


CANDIDATE_GESTURES = {
    "REST": "RELAX",
    "NEXT_TARGET": "WRIST UP",
    "CONFIRM": "MAKE A FIST",
    "ABORT": "WRIST DOWN",
    "RADIAL": "TILT WRIST THUMB SIDE",
    "ULNAR": "TILT WRIST LITTLE-FINGER SIDE",
    "PRONATE": "ROTATE PALM DOWN",
}


def test_an_injected_gesture_set_replaces_the_default_labels():
    # Candidate-gesture screening needs labels the trained firmware does not
    # have. Replacing rather than extending is deliberate: an exploratory run
    # that silently kept a label it meant to drop would report an accuracy for
    # a class set nobody chose.
    plan = build_trial_plan(2, seed=7, gestures=CANDIDATE_GESTURES)

    assert len(plan) == 2 * len(CANDIDATE_GESTURES)
    assert {trial.label for trial in plan} == set(CANDIDATE_GESTURES)
    for trial in plan:
        assert trial.action == CANDIDATE_GESTURES[trial.label]


def test_the_default_gesture_set_is_unchanged_by_the_new_parameter():
    # The four protocol commands are what the live firmware model was trained
    # on; adding the parameter must not move them.
    assert build_trial_plan(3, seed=5) == build_trial_plan(3, seed=5, gestures=None)
    assert {trial.label for trial in build_trial_plan(1, seed=5)} == set(
        GESTURE_LABELS
    )


def test_an_injected_set_flows_through_the_collection_plan_and_manifest():
    plan = build_collection_plan(
        CLASSIFIER_PROTOCOL, 1, 9, CANDIDATE_GESTURES
    )
    session = GuidedSession(plan)

    manifest = session.to_manifest(seed=9, status="aborted")

    assert manifest["gesture_actions"] == CANDIDATE_GESTURES


def test_event_gate_plan_requires_rest_in_the_gesture_set():
    # Every event trial is bracketed by a labelled REST, so a set without it
    # cannot express the protocol at all.
    with pytest.raises(ValueError, match="must define REST"):
        build_event_gate_plan(2, seed=1, gestures={"PRONATE": "ROTATE PALM DOWN"})
    with pytest.raises(ValueError, match="at least one non-REST"):
        build_event_gate_plan(2, seed=1, gestures={"REST": "RELAX"})


@pytest.mark.parametrize("gestures", [
    (),
    {},
    {"PRONATE": 3},
    {4: "ROTATE PALM DOWN"},
    {"PRONATE": "   "},
    {"  ": "ROTATE PALM DOWN"},
])
def test_malformed_gesture_sets_are_rejected(gestures):
    with pytest.raises((TypeError, ValueError)):
        build_trial_plan(1, seed=1, gestures=gestures)


def test_collection_plan_preserves_classifier_and_event_protocols():
    assert build_collection_plan(CLASSIFIER_PROTOCOL, 1, 9) == build_trial_plan(
        1, 9
    )
    assert build_collection_plan(EVENT_GATE_PROTOCOL, 1, 9) == (
        build_event_gate_plan(1, 9)
    )
    with pytest.raises(ValueError, match="unsupported collection protocol"):
        build_collection_plan("unknown", 1, 9)


def test_only_active_span_is_labelled_and_last_trial_auto_completes():
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    finish_active_and_verify(
        session,
        position(6100, timestamp=123456),
    )

    assert session.phase is Phase.COMPLETE
    assert session.completed_trials == 1
    assert len(session.segments) == 1
    segment = session.segments[0]
    assert segment.include
    assert segment.start.frame_index == 100
    assert segment.end.frame_index == 6100
    assert segment.start_host_sec == pytest.approx(2.5)
    assert segment.end_host_sec == pytest.approx(5.5)


def test_advance_never_shortens_a_phase_when_gui_ticks_late():
    session = GuidedSession(one_trial())
    session.start(0.0)

    session.advance(10.0, position(100))
    assert session.phase is Phase.TRANSITION
    assert session.remaining_seconds(10.0) == pytest.approx(0.5)


def test_verify_waits_for_a_later_raw_packet_and_stays_unlabelled():
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    assert session.advance(5.5, position(6100, timestamp=1000))
    assert session.phase is Phase.VERIFY
    assert not session.advance(5.61, position(6100, timestamp=1000))
    assert session.segments == []
    assert session.advance(5.62, position(6132, timestamp=17000))

    segment = session.segments[0]
    assert segment.include
    assert segment.end.frame_index == 6100
    assert segment.end_host_sec == pytest.approx(5.5)


def test_pause_during_active_rejects_attempt_and_resume_restarts_countdown():
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    session.pause(3.0, position(1100), reason="manual_pause")

    assert session.phase is Phase.PAUSED
    assert session.trial_index == 0
    assert session.completed_trials == 0
    assert session.segments[0].include is False
    assert "manual_pause" in session.segments[0].reasons

    session.resume(10.0, position(2000))
    assert session.phase is Phase.PREPARE
    assert session.current_attempt == 2
    assert session.pause_spans[0].start.frame_index == 1100
    assert session.pause_spans[0].end.frame_index == 2000
    assert session.pause_spans[0].to_dict()["include"] is False


def test_pause_during_prepare_does_not_create_a_label_segment():
    session = GuidedSession(one_trial())
    session.start(0.0)

    session.pause(0.8, position(300))
    session.resume(5.0, position(900))

    assert session.segments == []
    assert session.phase is Phase.PREPARE
    assert session.remaining_seconds(5.0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("end", "reason"),
    [
        (position(6100, detached=(0, 32, 0)), "electrode_contact"),
        (position(6100, lost=1), "packet_loss"),
        (position(6100, malformed=1), "malformed_packet"),
        (position(6100, duplicated=1), "duplicated_packet"),
        (position(6100, reversed_time=1), "time_reversed_packet"),
    ],
)
def test_bad_capture_is_excluded_and_same_trial_is_repeated(end, reason):
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    finish_active_and_verify(
        session,
        end,
        replace(end, frame_index=end.frame_index + 32),
    )

    assert session.phase is Phase.RECOVERY
    assert session.trial_index == 0
    assert session.completed_trials == 0
    assert not session.segments[0].include
    assert reason in session.segments[0].reasons
    session.advance(7.2, end)
    assert session.phase is Phase.PREPARE
    assert session.current_attempt == 2


def test_packet_loss_first_visible_during_verify_rejects_the_active_span():
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    finish_active_and_verify(
        session,
        position(6100, lost=0),
        position(6132, lost=1),
    )

    assert session.phase is Phase.RECOVERY
    assert not session.segments[0].include
    assert "packet_loss" in session.segments[0].reasons


def test_active_span_with_less_than_ninety_percent_frames_is_repeated():
    session = GuidedSession(one_trial())
    enter_active(session, start_frame=100)

    finish_active_and_verify(session, position(5000))

    assert session.phase is Phase.RECOVERY
    assert not session.segments[0].include
    assert "insufficient_frames" in session.segments[0].reasons


def test_frame_index_remains_primary_across_device_timestamp_wrap():
    session = GuidedSession(one_trial())
    session.start(0.0)
    session.advance(2.0, position(20))
    session.advance(2.5, position(100, timestamp=(1 << 32) - 1000))

    finish_active_and_verify(
        session,
        position(6100, timestamp=500000),
    )

    assert session.phase is Phase.COMPLETE
    assert session.segments[0].include
    assert session.segments[0].start.timestamp_us > session.segments[0].end.timestamp_us


def test_manual_stop_preserves_prior_work_and_rejects_active_attempt():
    plan = (
        TrialSpec(0, "REST", GESTURE_ACTIONS["REST"], 1),
        TrialSpec(1, "ABORT", GESTURE_ACTIONS["ABORT"], 1),
    )
    session = GuidedSession(plan)
    enter_active(session, start_frame=100)
    finish_active_and_verify(session, position(6100))
    assert session.phase is Phase.RECOVERY
    session.advance(7.2, position(7000))
    session.advance(9.2, position(7100))
    session.advance(9.7, position(7200))
    assert session.phase is Phase.ACTIVE

    session.stop(10.0, position(8200), reason="manual_stop")

    assert session.phase is Phase.STOPPED
    assert session.completed_trials == 1
    assert session.segments[-1].include is False
    assert "manual_stop" in session.segments[-1].reasons


def test_manifest_separates_schedule_labels_and_unlabelled_timing():
    session = GuidedSession(one_trial("NEXT_TARGET"))
    enter_active(session)
    finish_active_and_verify(session, position(6100))

    manifest = session.to_manifest(seed=42, status="complete")

    assert manifest["status"] == "complete"
    assert manifest["collection_protocol"] == CLASSIFIER_PROTOCOL
    # The manifest reports the gestures the plan actually contained, not the
    # module default. This plan has one trial, and a manifest that claimed all
    # four would tell a later trainer to expect labels no segment carries.
    assert manifest["gesture_actions"] == {
        "NEXT_TARGET": GESTURE_ACTIONS["NEXT_TARGET"]
    }
    assert manifest["timing_seconds"]["transition_unlabelled"] == 0.5
    assert manifest["timing_seconds"]["verification_unlabelled"] == 0.1
    assert manifest["timing_seconds"]["recovery_unlabelled"] == 1.5
    assert manifest["sample_rate_hz"] == 2000
    assert manifest["minimum_active_fraction"] == 0.9
    assert manifest["segments"][0]["include"] is True
    assert manifest["schedule"][0]["label"] == "NEXT_TARGET"


@pytest.mark.parametrize("bad", (0, -1, 1.5, True))
def test_plan_rejects_invalid_repetition_count(bad):
    expected = TypeError if isinstance(bad, (float, bool)) else ValueError
    with pytest.raises(expected):
        build_trial_plan(bad, seed=1)


def test_session_rejects_non_positive_or_non_finite_timing():
    with pytest.raises(ValueError):
        GuidedSession(one_trial(), active_seconds=0)
    with pytest.raises(ValueError):
        GuidedSession(one_trial(), recovery_seconds=float("inf"))
    with pytest.raises(ValueError, match="unsupported collection protocol"):
        GuidedSession(one_trial(), protocol="unknown")


def test_gesture_cli_builds_a_replacement_set_and_defaults_to_none():
    parsed = capture.parse_arguments([
        "--donning", "d1",
        "--gesture", "REST=RELAX",
        "--gesture", "PRONATE=ROTATE PALM DOWN",
    ])

    assert parsed.gestures == {"REST": "RELAX", "PRONATE": "ROTATE PALM DOWN"}
    # No --gesture means None, which the plan builders resolve to the four
    # protocol commands. It must not become an empty dict, which would be
    # rejected as a malformed set instead of meaning "unchanged".
    assert capture.parse_arguments(["--donning", "d1"]).gestures is None


@pytest.mark.parametrize("entry", [
    "PRONATE",          # no separator
    "=ROTATE PALM DOWN",  # no label
    "PRONATE=",         # no action
])
def test_malformed_gesture_arguments_exit_rather_than_collect(entry):
    # argparse errors exit(2); collecting a session against a half-parsed
    # label set would waste the wearer's time and produce unusable data.
    with pytest.raises(SystemExit):
        capture.parse_arguments(["--donning", "d1", "--gesture", entry])


def test_a_repeated_gesture_label_is_an_error_not_a_silent_overwrite():
    with pytest.raises(SystemExit):
        capture.parse_arguments([
            "--donning", "d1",
            "--gesture", "PRONATE=ROTATE PALM DOWN",
            "--gesture", "PRONATE=TURN PALM DOWN",
        ])


def test_event_gate_cli_uses_short_independent_collection_defaults():
    classifier = capture.parse_arguments(["--donning", "d1"])
    event_gate = capture.parse_arguments(["--donning", "d1",
                                          "--protocol",
                                          EVENT_GATE_PROTOCOL])

    assert classifier.out_root == "datasets/emg"
    assert classifier.repetitions == 5
    assert classifier.prepare_seconds == 2.0
    assert classifier.active_seconds == 3.0
    assert classifier.recovery_seconds == 1.5
    assert event_gate.out_root == "datasets/emg_event_gate"
    assert event_gate.repetitions == 3
    assert event_gate.prepare_seconds == 0.5
    assert event_gate.active_seconds == 2.0
    assert event_gate.recovery_seconds == 0.5


def test_info_sample_rate_can_only_be_adopted_before_start():
    session = GuidedSession(one_trial())
    session.set_sample_rate_hz(1000)
    assert session.sample_rate_hz == 1000

    session.start(0.0)
    with pytest.raises(RuntimeError):
        session.set_sample_rate_hz(2000)


@pytest.mark.parametrize("bad", (0, -1, 2000.0, True))
def test_session_rejects_invalid_sample_rate(bad):
    with pytest.raises(ValueError):
        GuidedSession(one_trial(), sample_rate_hz=bad)


@pytest.mark.parametrize("bad", (0, -0.1, 1.1, float("inf")))
def test_session_rejects_invalid_minimum_active_fraction(bad):
    with pytest.raises(ValueError):
        GuidedSession(one_trial(), min_active_fraction=bad)


class FakeConnection:
    def __init__(self):
        self.closed = False

    def cancel_read(self):
        return None

    def close(self):
        self.closed = True


class FakeRecording:
    def __init__(self):
        self.info = SimpleNamespace(donning='d1', channel_count=3)
        self.raw_frames = 6000
        self.last_raw_timestamp_us = 3000000

    def summary(self, _elapsed):
        return {
            "frames": self.raw_frames,
            "frames_detached_by_channel": {},
            "parser": {
                "accepted": 10,
                "lost": 0,
                "malformed": 0,
                "duplicated": 0,
                "time_reversed": 0,
            },
        }


class FlakyLog:
    def __init__(self, *, flush_failures=0, close_failures=0,
                 close_before_raise=False):
        self.flush_failures = flush_failures
        self.close_failures = close_failures
        self.close_before_raise = close_before_raise
        self.closed = False
        self.flush_calls = 0
        self.close_calls = 0

    def flush(self):
        self.flush_calls += 1
        if self.flush_failures:
            self.flush_failures -= 1
            raise OSError("injected flush failure")

    def close(self):
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            if self.close_before_raise:
                self.closed = True
            raise OSError("injected close failure")
        self.closed = True


def worker_with_log(log):
    worker = capture.SerialWorker(FakeConnection(), "/dev/fake", 1.0)
    worker._session_log = log
    worker._session_recording = FakeRecording()
    worker._session_started_perf = capture.time.perf_counter()
    worker._session_started_wall = datetime.datetime.now().astimezone()
    return worker


def test_worker_finish_retries_flush_without_accepting_more_bytes():
    log = FlakyLog(flush_failures=1)
    worker = worker_with_log(log)

    with pytest.raises(OSError, match="flush"):
        worker.finish_recording()

    assert worker._session_log is None
    assert worker._pending_finish is not None
    summary, final_position = worker.finish_recording()
    assert summary["frames"] == 6000
    assert final_position.frame_index == 6000
    assert log.flush_calls == 2
    assert worker.finish_recording() == (summary, final_position)


def test_worker_finish_recovers_when_close_raised_after_closing():
    log = FlakyLog(close_failures=1, close_before_raise=True)
    worker = worker_with_log(log)

    with pytest.raises(OSError, match="close"):
        worker.finish_recording()

    summary, final_position = worker.finish_recording()
    assert summary["frames"] == final_position.frame_index == 6000
    assert log.close_calls == 1


def test_worker_close_can_be_retried_after_log_flush_failure():
    log = FlakyLog(flush_failures=1)
    worker = worker_with_log(log)

    with pytest.raises(OSError, match="flush"):
        worker.close()
    assert not worker._closed

    worker.close()
    assert worker._closed
    assert worker.connection.closed
    assert log.closed


class FakeFinishWorker:
    def __init__(self):
        self.finish_calls = 0

    def finish_recording(self):
        self.finish_calls += 1
        return (
            {"started": "2026-08-14T12:00:00-04:00", "frames": 6000},
            position(6000),
        )

    def snapshot(self):
        return SimpleNamespace(donning='d1', position=position(6000))


def bare_capture_app(tmp_path):
    app = object.__new__(capture.CaptureApp)
    app.worker = FakeFinishWorker()
    app.session = GuidedSession(one_trial())
    app.seed = 7
    app.arguments = SimpleNamespace(donning='d1', port="/dev/fake")
    app.session_id = "test_session"
    app.bin_path = pathlib.Path(tmp_path) / "session.bin"
    app.json_path = pathlib.Path(tmp_path) / "session.json"
    app.recording_started = True
    app.waiting_for_preflight = False
    app.finalized = False
    app._final_summary = None
    app._final_position = None
    app._final_status = None
    app._final_error = ""
    return app


def test_manifest_write_failure_retries_without_refinishing_raw(
    tmp_path,
    monkeypatch,
):
    app = bare_capture_app(tmp_path)
    writes = []

    def flaky_write(_path, payload):
        writes.append(payload)
        if len(writes) == 1:
            raise OSError("injected manifest failure")

    monkeypatch.setattr(capture, "write_manifest", flaky_write)

    with pytest.raises(OSError, match="manifest"):
        app._save_session("stopped")
    assert app.worker.finish_calls == 1
    assert not app.recording_started
    assert not app.finalized

    app._save_session("stopped")
    assert app.worker.finish_calls == 1
    assert app.finalized
    assert len(writes) == 2


def test_emergency_finalize_stops_and_saves_without_tk(
    tmp_path,
    monkeypatch,
):
    app = bare_capture_app(tmp_path)
    manifests = []
    monkeypatch.setattr(
        capture,
        "write_manifest",
        lambda _path, data: manifests.append(data),
    )
    monkeypatch.setattr(capture, "print_summary", lambda _summary: None)

    assert app.emergency_finalize("keyboard_interrupt")

    assert app.finalized
    assert app.session.phase is Phase.STOPPED
    assert manifests[0]["status"] == "stopped"
    assert manifests[0]["error"] == "keyboard_interrupt"
