"""Tests for live image/PointCloud2 candidate observation adaptation."""

from cv_bridge import CvBridge
from markerless_object_perception.candidate_builder import (
    InstanceMaskDetection,
)
from markerless_object_perception.live_candidate_pipeline import (
    LiveCandidatePipeline,
)
from markerless_object_perception.live_candidate_publisher import (
    _sensor_qos,
    candidate_observation_from_pair,
)
import numpy as np
import pytest
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


class StubSegmenter:
    """Return one fixed set of model-independent detections."""

    def __init__(self, detections):
        self.detections = detections

    def track(self, _frame):
        return self.detections


def _image(height=5, width=5, *, frame_id='stereo_left_optical_frame'):
    message = CvBridge().cv2_to_imgmsg(
        np.zeros((height, width, 3), dtype=np.uint8),
        encoding='bgr8',
    )
    message.header.frame_id = frame_id
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 345_000_000
    return message


def _point_cloud(
    height=5,
    width=5,
    *,
    frame_id='stereo_left_optical_frame',
):
    xyz = np.full((height, width, 3), [0.2, -0.1, 0.7], dtype='<f4')
    message = PointCloud2()
    message.header.frame_id = frame_id
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 345_000_000
    message.height = height
    message.width = width
    message.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = width * message.point_step
    message.data = xyz.tobytes()
    message.is_dense = True
    return message


def _observe(image, cloud, detections):
    return candidate_observation_from_pair(
        image,
        cloud,
        bridge=CvBridge(),
        pipeline=LiveCandidatePipeline(StubSegmenter(detections)),
    )


def test_exact_aligned_pair_produces_source_preserving_candidate():
    detection = InstanceMaskDetection(
        class_label='cup',
        confidence=0.85,
        track_id=8,
        mask=np.ones((5, 5), dtype=bool),
    )

    observation = _observe(_image(), _point_cloud(), (detection,))

    assert observation.error is None
    assert observation.message.valid
    assert observation.message.header.frame_id == 'stereo_left_optical_frame'
    assert observation.message.header.stamp.sec == 12
    assert observation.message.header.stamp.nanosec == 345_000_000
    assert observation.message.pair_skew_sec == 0.0
    assert observation.bgr_image.shape == (5, 5, 3)
    assert observation.detections == (detection,)
    assert observation.cloud_decode_sec is not None
    assert observation.cloud_decode_sec >= 0.0
    assert observation.pipeline_sec is not None
    assert observation.pipeline_sec >= 0.0
    assert observation.processing_sec >= (
        observation.cloud_decode_sec + observation.pipeline_sec
    )
    assert len(observation.message.candidates) == 1
    assert observation.message.candidates[0].class_label == 'cup'
    assert observation.message.candidates[0].position.z == pytest.approx(0.7)


def test_no_detection_produces_valid_empty_observation():
    observation = _observe(_image(), _point_cloud(), ())

    assert observation.error is None
    assert observation.message.valid
    assert observation.message.candidates == []
    assert observation.detections == ()


def test_image_point_cloud_shape_mismatch_fails_closed():
    observation = _observe(_image(), _point_cloud(height=4), ())

    assert 'shape does not match' in observation.error
    assert not observation.message.valid
    assert observation.message.candidates == []


def test_source_stamp_mismatch_fails_closed_and_reports_skew():
    image = _image()
    cloud = _point_cloud()
    cloud.header.stamp.nanosec += 2_000_000

    observation = _observe(image, cloud, ())

    assert 'source stamps must match' in observation.error
    assert not observation.message.valid
    assert observation.message.pair_skew_sec == pytest.approx(0.002)


def test_frame_id_mismatch_fails_closed():
    observation = _observe(
        _image(frame_id='left_frame'),
        _point_cloud(frame_id='different_frame'),
        (),
    )

    assert 'frame IDs must match' in observation.error
    assert not observation.message.valid


def test_sensor_qos_depth_matches_synchronizer_window():
    qos = _sensor_qos(15)

    assert qos.depth == 15
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE
