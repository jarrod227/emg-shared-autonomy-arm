"""Cross-check the host-C event gate against the Python EventGate.

The frozen counts were validated with the Python implementation, so the C port
is only trustworthy to the extent it reproduces it. The C test emits a fixture
containing both the decision sequence and the events it produced; this replays
the same sequence through EventGate and requires them to agree exactly.
"""

import pathlib
import struct

import pytest

from emg_event_gate_replay import LABELS, EventGate, GateConfig, VALIDATED_GATE


ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "firmware" / "test" / "gate.bin"


def read_fixture(payload):
    steps = struct.unpack_from("<i", payload, 0)[0]
    counts = struct.unpack_from("<5H", payload, 4)
    offset = 14
    predictions = struct.unpack_from(f"<{steps}B", payload, offset)
    offset += steps
    valid = struct.unpack_from(f"<{steps}B", payload, offset)
    offset += steps
    event_count = struct.unpack_from("<i", payload, offset)[0]
    offset += 4
    events = []
    for _ in range(event_count):
        index, command = struct.unpack_from("<iB", payload, offset)
        offset += 5
        events.append((index, command))
    assert offset == len(payload), "fixture has trailing bytes"
    return steps, counts, predictions, valid, events


def test_label_order_matches_the_c_command_enum():
    # The fixture stores commands as emg_command_t values and this module reads
    # them as indices into LABELS. That is only sound while the two agree, and
    # nothing else in this file would notice if they stopped.
    assert LABELS == ("REST", "NEXT_TARGET", "CONFIRM", "ABORT")


def test_c_gate_matches_python_event_for_event():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate gate.bin")

    steps, counts, predictions, valid, c_events = read_fixture(
        FIXTURE.read_bytes()
    )
    config = GateConfig(*counts)
    # A count mismatch is reported as itself rather than surfacing later as a
    # confusing event difference.
    assert config == VALIDATED_GATE

    gate = EventGate(config)
    python_events = []
    for index in range(steps):
        event = gate.push(
            LABELS[predictions[index]], valid=bool(valid[index])
        )
        if event is not None:
            python_events.append((index, LABELS.index(event)))

    assert python_events == c_events


def test_fixture_actually_exercises_the_gate():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate gate.bin")

    steps, _counts, predictions, valid, events = read_fixture(
        FIXTURE.read_bytes()
    )
    # Agreement on a fixture that never fires proves nothing, and a fixture
    # that only fires ABORT would miss the arming and stable-run paths
    # entirely, since ABORT bypasses both.
    assert steps >= 512
    assert len(events) >= 6
    fired = {LABELS[command] for _index, command in events}
    assert fired == {"NEXT_TARGET", "CONFIRM", "ABORT"}
    assert "REST" not in fired
    assert 0 in set(valid), "fixture never exercises the invalid-window path"
    assert set(predictions) == {0, 1, 2, 3}
