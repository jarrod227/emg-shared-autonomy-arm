"""Tests for the Ultralytics-to-project segmentation adapter."""

from types import SimpleNamespace

from markerless_object_perception import webcam_segmentation_demo
from markerless_object_perception.candidate_builder import (
    CandidateBuilder,
)
from markerless_object_perception.masked_point_localizer import (
    MaskedPointLocalizer,
    MaskedPointLocalizerConfig,
)
from markerless_object_perception.yolo_segmenter import (
    YoloInstanceSegmenter,
    YoloSegmenterConfig,
)
import numpy as np
import pytest


class FakeBoxes:
    """Small stand-in for one Ultralytics Boxes result."""

    def __init__(self, classes, confidences, track_ids):
        self.cls = np.asarray(classes)
        self.conf = np.asarray(confidences)
        self.id = None if track_ids is None else np.asarray(track_ids)

    def __len__(self):
        return len(self.cls)


class FakeModel:
    """Record tracking arguments and return one configured result."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def fake_result(*, classes, confidences, track_ids, masks, names=None):
    """Construct only the result fields consumed by the adapter."""
    return SimpleNamespace(
        boxes=FakeBoxes(classes, confidences, track_ids),
        masks=None if masks is None else SimpleNamespace(data=np.asarray(masks)),
        names=names or {0: 'cup', 1: 'bottle'},
    )


def test_track_converts_masks_metadata_and_persistent_ids():
    masks = np.zeros((2, 4, 6), dtype=np.float32)
    masks[0, :, :3] = 1.0
    masks[1, :, 3:] = 1.0
    model = FakeModel(
        fake_result(
            classes=[0, 1],
            confidences=[0.8, 0.9],
            track_ids=[12, 15],
            masks=masks,
        )
    )
    segmenter = YoloInstanceSegmenter(model=model)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    detections = segmenter.track(frame)

    assert [item.class_label for item in detections] == ['cup', 'bottle']
    assert [item.track_id for item in detections] == [12, 15]
    assert detections[0].confidence == pytest.approx(0.8)
    assert detections[0].mask.dtype == np.bool_
    assert np.all(detections[0].mask[:, :3])
    assert model.calls[0]['persist'] is True
    assert model.calls[0]['retina_masks'] is True
    assert model.calls[0]['tracker'] == 'bytetrack.yaml'
    assert model.calls[0]['device'] == 'cpu'


def test_non_retina_mask_shape_is_rejected():
    mask = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    model = FakeModel(
        fake_result(
            classes=[0],
            confidences=[0.8],
            track_ids=[3],
            masks=mask,
        )
    )
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match='mask shape'):
        YoloInstanceSegmenter(model=model).track(frame)


def test_yolo_detection_is_accepted_by_candidate_builder():
    mask = np.ones((1, 4, 4), dtype=np.float32)
    model = FakeModel(
        fake_result(
            classes=[1],
            confidences=[0.9],
            track_ids=[8],
            masks=mask,
        )
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    detections = YoloInstanceSegmenter(model=model).track(frame)
    xyz = np.full((4, 4, 3), [0.2, 0.1, 0.7])
    builder = CandidateBuilder(
        MaskedPointLocalizer(
            MaskedPointLocalizerConfig(min_valid_points=4)
        )
    )

    result = builder.build(detections, xyz)

    assert not result.rejections
    assert result.candidates[0].class_label == 'bottle'
    assert result.candidates[0].track_id == 8
    assert result.candidates[0].point == pytest.approx((0.2, 0.1, 0.7))


def test_empty_scene_returns_empty_detection_tuple():
    result = fake_result(
        classes=[],
        confidences=[],
        track_ids=[],
        masks=None,
    )

    detections = YoloInstanceSegmenter(
        model=FakeModel(result)
    ).track(np.zeros((4, 4, 3), dtype=np.uint8))

    assert detections == ()


def test_boxes_without_segmentation_masks_are_rejected():
    result = fake_result(
        classes=[0],
        confidences=[0.8],
        track_ids=[1],
        masks=None,
    )

    with pytest.raises(RuntimeError, match='use a -seg model'):
        YoloInstanceSegmenter(
            model=FakeModel(result)
        ).track(np.zeros((4, 4, 3), dtype=np.uint8))


def test_unconfirmed_track_is_empty_until_an_id_is_available():
    unconfirmed = fake_result(
        classes=[0],
        confidences=[0.8],
        track_ids=None,
        masks=np.ones((1, 4, 4)),
    )
    confirmed = fake_result(
        classes=[0],
        confidences=[0.8],
        track_ids=[9],
        masks=np.ones((1, 4, 4)),
    )
    model = FakeModel(unconfirmed)
    segmenter = YoloInstanceSegmenter(model=model)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert segmenter.track(frame) == ()
    model.result = confirmed
    detections = segmenter.track(frame)

    assert len(detections) == 1
    assert detections[0].track_id == 9


def test_camera_is_released_when_open_fails(monkeypatch):
    class ClosedCapture:
        def __init__(self):
            self.released = False

        def set(self, *_args):  # noqa: A003 - mirrors cv2.VideoCapture
            return False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = ClosedCapture()
    destroyed = []
    monkeypatch.setattr(
        webcam_segmentation_demo,
        'YoloInstanceSegmenter',
        lambda _config: object(),
    )
    monkeypatch.setattr(
        webcam_segmentation_demo.cv2,
        'VideoCapture',
        lambda *_args: capture,
    )
    monkeypatch.setattr(
        webcam_segmentation_demo.cv2,
        'destroyAllWindows',
        lambda: destroyed.append(True),
    )
    args = SimpleNamespace(
        model='fake-seg.pt',
        confidence=0.5,
        tracker='bytetrack.yaml',
        inference_device='cpu',
        camera=0,
        width=1280,
        height=720,
        fps=30,
    )

    with pytest.raises(RuntimeError, match='could not open'):
        webcam_segmentation_demo.run_demo(args)

    assert capture.released
    assert destroyed == [True]


@pytest.mark.parametrize(
    'frame',
    [
        np.zeros((4, 4)),
        np.zeros((4, 4, 1)),
        np.zeros((0, 4, 3)),
    ],
)
def test_invalid_camera_frame_shape_is_rejected(frame):
    with pytest.raises(ValueError, match='HxWx3'):
        YoloInstanceSegmenter(model=FakeModel(None)).track(frame)


@pytest.mark.parametrize(
    'overrides',
    [
        {'model_path': ''},
        {'min_confidence': -0.1},
        {'min_confidence': float('nan')},
        {'iou_threshold': 1.1},
        {'tracker': ''},
        {'device': ''},
    ],
)
def test_invalid_yolo_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        YoloSegmenterConfig(**overrides)
