"""Cross-check host-C Q18 classifier scores against the deployed model.

Read from the generated header the firmware is built from, not from a JSON in
a dataset directory. Those two drifted the moment the model was retrained with
a fifth class: the fixture had five classes and the JSON four, and what the
test had really been checking was that two unrelated models happened to agree.
"""

import json
import pathlib
import struct

import numpy as np
import pytest

from emg_runtime_compare import load_deployed_model


ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "firmware" / "test" / "classifier.bin"
MODEL = ROOT / "firmware" / "src" / "emg_classifier_model.h"


def test_c_classifier_matches_python_q18_exactly():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate classifier.bin")
    if not MODEL.exists():
        pytest.skip("generated model header is not present")

    payload = FIXTURE.read_bytes()
    rows, feature_count, class_count = struct.unpack_from("<iii", payload, 0)
    model = load_deployed_model(MODEL)
    assert feature_count == len(model.feature_names)
    assert class_count == len(model.labels)

    offset = 12
    features = []
    c_scores = []
    c_commands = []
    for _ in range(rows):
        row = struct.unpack_from(f"<{feature_count}i", payload, offset)
        offset += 4 * feature_count
        scores = struct.unpack_from(f"<{class_count}q", payload, offset)
        offset += 8 * class_count
        command = struct.unpack_from("<B", payload, offset)[0]
        offset += 1
        features.append(row)
        c_scores.append(scores)
        c_commands.append(command)
    assert offset == len(payload)

    python_scores = model.scores(np.asarray(features, dtype=np.int64))
    assert np.array_equal(python_scores, np.asarray(c_scores, dtype=np.int64))
    python_commands = np.argmax(python_scores, axis=1)
    assert np.array_equal(python_commands, np.asarray(c_commands))
    assert len(set(c_commands)) >= 2
