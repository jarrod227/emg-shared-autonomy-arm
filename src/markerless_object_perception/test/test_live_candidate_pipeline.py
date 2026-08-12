"""Tests for the live 2D-mask to aligned-XYZ candidate pipeline."""

from markerless_object_perception.candidate_builder import (
    InstanceMaskDetection,
)
from markerless_object_perception.live_candidate_pipeline import (
    LiveCandidatePipeline,
)
import numpy as np
import pytest


class StubSegmenter:
    """Return configured detections while recording whether inference ran."""

    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def track(self, frame):
        self.calls += 1
        return self.detections


def test_process_calls_segmenter_then_localizes_mask_points():
    mask = np.ones((5, 5), dtype=bool)
    segmenter = StubSegmenter(
        (
            InstanceMaskDetection(
                class_label='bottle',
                confidence=0.9,
                track_id=4,
                mask=mask,
            ),
        )
    )
    pipeline = LiveCandidatePipeline(segmenter)
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    xyz = np.full((5, 5, 3), [0.2, -0.1, 0.7], dtype=np.float32)

    frame_result = pipeline.process_with_detections(image, xyz)
    result = frame_result.build_result

    assert segmenter.calls == 1
    assert frame_result.detections == segmenter.detections
    assert result.rejections == ()
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.class_label == 'bottle'
    assert candidate.track_id == 4
    assert candidate.point == pytest.approx((0.2, -0.1, 0.7))


def test_no_2d_detection_is_a_valid_empty_build_result():
    segmenter = StubSegmenter(())
    pipeline = LiveCandidatePipeline(segmenter)

    result = pipeline.process(
        np.zeros((5, 5, 3), dtype=np.uint8),
        np.full((5, 5, 3), [0.0, 0.0, 0.8], dtype=np.float32),
    )

    assert segmenter.calls == 1
    assert result.candidates == ()
    assert result.rejections == ()


def test_float32_xyz_preserves_candidate_result():
    mask = np.ones((5, 5), dtype=bool)
    segmenter = StubSegmenter(
        (
            InstanceMaskDetection(
                class_label='apple',
                confidence=0.8,
                track_id=7,
                mask=mask,
            ),
        )
    )
    pipeline = LiveCandidatePipeline(segmenter)
    xyz = np.full((5, 5, 3), [0.15, 0.05, 0.9], dtype=np.float32)

    result = pipeline.process(
        np.zeros((5, 5, 3), dtype=np.uint8),
        xyz,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].point == pytest.approx((0.15, 0.05, 0.9))


def test_shape_mismatch_is_rejected_before_yolo_inference():
    segmenter = StubSegmenter(())
    pipeline = LiveCandidatePipeline(segmenter)

    with pytest.raises(ValueError, match='image shapes must match'):
        pipeline.process(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.float32),
        )

    assert segmenter.calls == 0
