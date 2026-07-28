"""Offline one-image demo for the Objective 4.2 hand detector."""

import argparse
from pathlib import Path

import cv2

from stereo_hand_observer.hand_detector import (
    MediaPipeHandDetector,
    draw_hand_landmarks,
)


def annotate_one_hand(model_path, image_bgr):
    """Return an annotated image or None when no unique hand is detected."""
    detector = MediaPipeHandDetector(model_path)
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        hand = detector.detect(image_rgb)
    finally:
        detector.close()
    if hand is None:
        return None
    return draw_hand_landmarks(image_bgr, hand)


def main(args=None):
    """Detect and draw one complete hand in a still image."""
    parser = argparse.ArgumentParser(
        description="Draw all 21 MediaPipe hand landmarks in one image."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(args)

    image_bgr = cv2.imread(str(options.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        parser.error(f"cannot read input image: {options.image}")
    if not options.output.parent.is_dir():
        parser.error(f"output directory does not exist: {options.output.parent}")

    annotated = annotate_one_hand(options.model, image_bgr)
    if annotated is None:
        parser.error("expected exactly one hand, but detection was missing or ambiguous")
    if not cv2.imwrite(str(options.output), annotated):
        parser.error(f"failed to write annotated image: {options.output}")
    print(f"wrote complete-hand overlay to {options.output}")
