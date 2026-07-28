"""Optional real-model smoke test using a user-supplied one-hand image."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_hand_observer.hand_detector import (
    MediaPipeHandDetector,
    draw_hand_landmarks,
)


def required_asset(environment_name):
    """Return an explicitly configured asset or skip the real-model test."""
    value = os.environ.get(environment_name)
    if not value:
        pytest.skip(f"set {environment_name} to run the real-model smoke test")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{environment_name} does not name a file: {path}")
    return path


def test_real_model_detects_and_draws_one_complete_hand(tmp_path):
    pytest.importorskip("mediapipe")
    model_path = required_asset("MEDIAPIPE_HAND_MODEL")
    image_path = required_asset("MEDIAPIPE_ONE_HAND_IMAGE")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr is not None, "the configured hand image is unreadable"

    detector = MediaPipeHandDetector(model_path)
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        hand = detector.detect(image_rgb)
    finally:
        detector.close()

    assert hand is not None, "expected exactly one detected hand"
    assert len(hand.landmarks) == 21
    annotated = draw_hand_landmarks(image_bgr, hand)
    assert np.any(annotated != image_bgr)
    assert cv2.imwrite(str(tmp_path / "one_hand_annotated.png"), annotated)
