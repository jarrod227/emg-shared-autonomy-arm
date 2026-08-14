"""Cross-check host-C Q18 classifier scores against the generated JSON."""

import json
import pathlib
import struct

import numpy as np
import pytest

from emg_train_lda import QuantizedLDAModel


ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "firmware" / "test" / "classifier.bin"
MODEL = ROOT / "datasets" / "emg" / "lda_model.json"


def test_c_classifier_matches_python_q18_exactly():
    if not FIXTURE.exists():
        pytest.skip("run `make -C firmware/test check` to generate classifier.bin")
    if not MODEL.exists():
        pytest.skip("local labelled dataset/model is not present")

    payload = FIXTURE.read_bytes()
    rows, feature_count, class_count = struct.unpack_from("<iii", payload, 0)
    model_payload = json.loads(MODEL.read_text(encoding="utf-8"))
    model = QuantizedLDAModel.from_dict(model_payload["quantized_model"])
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
