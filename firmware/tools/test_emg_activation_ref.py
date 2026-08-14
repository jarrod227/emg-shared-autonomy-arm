"""Cross-check the host-C activation threshold against the Python mirror.

Offline sweeps and firmware-versus-host replays will use the Python mirror,
so the mirror is only trustworthy to the extent it reproduces the C module
the firmware links. The C test emits a fixture containing the inputs, every
decision, and the baseline after every step; this replays the same inputs
and requires exact agreement on both outputs — baselines included, because
an EMA divergence shows up there many steps before it flips a decision.
"""

import pathlib
import struct

import pytest

from emg_activation_ref import (
    FROZEN_BASELINE_SHIFT,
    FROZEN_FACTOR,
    ActivationGate,
)
from emg_event_gate_replay import LABELS


ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "firmware" / "test" / "activation.bin"


def read_fixture(payload):
    steps = struct.unpack_from("<i", payload, 0)[0]
    factor, shift = struct.unpack_from("<2H", payload, 4)
    offset = 8
    predictions = struct.unpack_from(f"<{steps}B", payload, offset)
    offset += steps
    valid = struct.unpack_from(f"<{steps}B", payload, offset)
    offset += steps
    totals = struct.unpack_from(f"<{steps}i", payload, offset)
    offset += 4 * steps
    decisions = struct.unpack_from(f"<{steps}B", payload, offset)
    offset += steps
    baselines = struct.unpack_from(f"<{steps}i", payload, offset)
    offset += 4 * steps
    assert offset == len(payload), "fixture has trailing bytes"
    return steps, factor, shift, predictions, valid, totals, decisions, baselines


def test_label_order_matches_the_c_command_enum():
    # The fixture stores commands as emg_command_t values and this module
    # reads them as indices into LABELS; nothing else here would notice if
    # the two stopped agreeing.
    assert LABELS == ("REST", "NEXT_TARGET", "CONFIRM", "ABORT")


def test_c_activation_matches_python_step_for_step():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate activation.bin")

    steps, factor, shift, predictions, valid, totals, decisions, baselines = (
        read_fixture(FIXTURE.read_bytes())
    )
    # A drifted constant should report as a configuration mismatch, not
    # surface later as a confusing decision difference.
    assert (factor, shift) == (FROZEN_FACTOR, FROZEN_BASELINE_SHIFT)

    gate = ActivationGate(factor, shift)
    python_decisions = []
    python_baselines = []
    for index in range(steps):
        decision = gate.apply(
            LABELS[predictions[index]],
            valid=bool(valid[index]),
            total_mav=totals[index],
        )
        python_decisions.append(LABELS.index(decision))
        python_baselines.append(gate.baseline)

    assert python_decisions == list(decisions)
    assert python_baselines == list(baselines)


def test_fixture_actually_exercises_the_stage():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate activation.bin")

    steps, _factor, _shift, predictions, valid, _totals, decisions, _baselines = (
        read_fixture(FIXTURE.read_bytes())
    )
    assert steps >= 512
    # Agreement on a fixture that never suppresses proves nothing.
    suppressed = [
        index for index in range(steps)
        if valid[index] and predictions[index] != 0 and decisions[index] == 0
    ]
    assert len(suppressed) >= 20
    # Every non-REST class must both pass somewhere and reach the output,
    # or a class-specific exemption could slip in unnoticed.
    passed_classes = {
        decisions[index] for index in range(steps) if decisions[index] != 0
    }
    assert passed_classes == {1, 2, 3}
    # The pre-seed passthrough is a deliberate fail-open choice; keep it
    # exercised so removing it is a visible decision.
    assert predictions[0] != 0 and decisions[0] == predictions[0]
    assert 0 in set(valid), "fixture never exercises the invalid-window path"
