import pytest

from emg_intent_bridge.confirmation import (
    ABORT,
    CONFIRM,
    NEXT_TARGET,
    REST,
    DeviceIntent,
    IntentConfirmationGate,
)


def intent(command, timestamp_us, *, confidence=200, quality=255,
           discontinuity=False):
    return DeviceIntent(
        sequence=timestamp_us // 50_000,
        timestamp_us=timestamp_us,
        command=command,
        confidence=confidence,
        signal_quality=quality,
        stream_discontinuity=discontinuity,
    )


@pytest.mark.parametrize("command", (NEXT_TARGET, CONFIRM))
def test_next_and_confirm_require_two_matching_events(command):
    gate = IntentConfirmationGate(confirmation_window_sec=3.0)

    assert gate.push(intent(command, 1_000_000, confidence=210)) is None
    assert gate.push(intent(REST, 1_050_000)) is None
    result = gate.push(intent(command, 2_500_000, confidence=180))

    assert result.command == command
    assert result.timestamp_us == 2_500_000
    assert result.confidence == 180
    assert gate.pending_command is None


def test_pair_expires_and_current_event_becomes_the_new_first_half():
    gate = IntentConfirmationGate(confirmation_window_sec=3.0)

    assert gate.push(intent(CONFIRM, 1_000_000)) is None
    assert gate.push(intent(CONFIRM, 4_000_001)) is None
    assert gate.pending_command == CONFIRM


def test_different_command_replaces_pending_without_publishing():
    gate = IntentConfirmationGate()

    assert gate.push(intent(NEXT_TARGET, 1_000_000)) is None
    assert gate.push(intent(CONFIRM, 2_000_000)) is None
    assert gate.pending_command == CONFIRM


def test_abort_is_immediate_and_clears_pending_pair():
    gate = IntentConfirmationGate()
    gate.push(intent(CONFIRM, 1_000_000))

    result = gate.push(intent(ABORT, 1_500_000, confidence=99))

    assert result.command == ABORT
    assert result.confidence == 99
    assert gate.pending_command is None


def test_abort_survives_zero_signal_quality():
    # signal_quality is a diagnostic byte, not evidence about the gesture.
    # A stop request that the MCU already committed to must not be dropped
    # because one frame went missing in the same window.
    gate = IntentConfirmationGate()
    gate.push(intent(CONFIRM, 1_000_000))

    result = gate.push(intent(ABORT, 1_500_000, quality=0, confidence=2))

    assert result.command == ABORT
    assert result.signal_quality == 0
    assert gate.pending_command is None


def test_abort_survives_a_stream_discontinuity():
    gate = IntentConfirmationGate()
    gate.push(intent(NEXT_TARGET, 1_000_000))

    result = gate.push(intent(ABORT, 2_000_000, discontinuity=True))

    assert result.command == ABORT
    assert gate.pending_command is None


def test_low_margin_never_blocks_an_event():
    # Live events measured margins from 2 to 162. The bridge must not
    # reintroduce a threshold on a value the protocol calls diagnostic.
    gate = IntentConfirmationGate()

    assert gate.push(intent(CONFIRM, 1_000_000, confidence=0)) is None
    result = gate.push(intent(CONFIRM, 2_000_000, confidence=1))

    assert result.command == CONFIRM
    assert result.confidence == 0


def test_zero_quality_event_is_fail_closed():
    gate = IntentConfirmationGate()
    gate.push(intent(CONFIRM, 1_000_000))

    assert gate.push(intent(CONFIRM, 2_000_000, quality=0)) is None
    assert gate.pending_command is None


def test_zero_quality_rest_also_clears_pending_pair():
    gate = IntentConfirmationGate()
    gate.push(intent(CONFIRM, 1_000_000))

    assert gate.push(intent(REST, 1_050_000, quality=0)) is None
    assert gate.pending_command is None


def test_stream_gap_prevents_pairing_across_missing_packets():
    gate = IntentConfirmationGate()
    gate.push(intent(NEXT_TARGET, 1_000_000))

    assert gate.push(intent(
        NEXT_TARGET, 2_000_000, discontinuity=True
    )) is None
    assert gate.pending_command == NEXT_TARGET


def test_invalid_configuration_and_input_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        IntentConfirmationGate(0)
    with pytest.raises(TypeError, match="DeviceIntent"):
        IntentConfirmationGate().push(object())


def test_window_admits_the_measured_deliberate_cadence():
    # 4.15 s was the slowest gap between two deliberate MCU events in a live
    # self-paced round. The default window must clear it, or the bridge
    # publishes nothing a wearer can actually produce.
    gate = IntentConfirmationGate(confirmation_window_sec=5.5)

    assert gate.push(intent(NEXT_TARGET, 1_000_000)) is None
    result = gate.push(intent(NEXT_TARGET, 5_150_000))

    assert result is not None
    assert result.command == NEXT_TARGET


def test_window_stays_below_the_measured_ordinary_activity_gap():
    # The closest two same-command events in ten minutes of ordinary activity
    # were 7.05 s apart, and only an ABORT arriving 0.05 s later kept them
    # from pairing. The window must reject that gap on its own.
    gate = IntentConfirmationGate(confirmation_window_sec=5.5)

    assert gate.push(intent(CONFIRM, 1_000_000)) is None
    assert gate.push(intent(CONFIRM, 8_050_000)) is None
    assert gate.pending_command == CONFIRM
