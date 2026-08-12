"""Publish live mask-localized candidates from rectified image/point-cloud pairs."""

from dataclasses import dataclass
import math
import operator
import time

from assistive_interfaces.msg import ObjectCandidateArray
import cv2
from cv_bridge import CvBridge
from markerless_object_perception.candidate_builder import (
    CandidateBuilder,
    CandidateBuilderConfig,
)
from markerless_object_perception.live_candidate_pipeline import (
    LiveCandidatePipeline,
)
from markerless_object_perception.point_cloud_adapter import (
    organized_xyz_from_point_cloud,
)
from markerless_object_perception.ros_adapter import (
    object_candidate_array_from_result,
)
from markerless_object_perception.webcam_segmentation_demo import (
    draw_detections,
)
from markerless_object_perception.yolo_segmenter import (
    DEFAULT_OBJECT_CLASSES,
    YoloInstanceSegmenter,
    YoloSegmenterConfig,
)
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, PointCloud2


_NANOSECONDS_PER_SECOND = 1_000_000_000
_WINDOW_NAME = 'Objective 3.2 live stereo candidates - Q/ESC closes GUI'
_PERFORMANCE_LOG_PERIOD_SEC = 5.0


@dataclass(frozen=True)
class CandidateObservation:
    """One publishable observation plus an optional fail-closed diagnostic."""

    message: ObjectCandidateArray
    error: str | None
    bgr_image: np.ndarray | None
    detections: tuple
    cloud_decode_sec: float | None
    pipeline_sec: float | None
    processing_sec: float


def candidate_observation_from_pair(
    image_message,
    point_cloud_message,
    *,
    bridge,
    pipeline,
):
    """Convert one exact-time image/cloud pair into the ROS candidate contract."""
    processing_start = time.perf_counter()
    image_time = _stamp_nanoseconds(image_message.header.stamp)
    cloud_time = _stamp_nanoseconds(point_cloud_message.header.stamp)
    pair_skew_sec = abs(image_time - cloud_time) / _NANOSECONDS_PER_SECOND
    image_frame = str(image_message.header.frame_id).strip()
    cloud_frame = str(point_cloud_message.header.frame_id).strip()
    output_frame = cloud_frame or image_frame or 'invalid_stereo_frame'
    bgr_image = None
    detections = ()
    cloud_decode_sec = None
    pipeline_sec = None

    try:
        if image_time != cloud_time:
            raise ValueError('image and point cloud source stamps must match')
        if not image_frame or not cloud_frame:
            raise ValueError('image and point cloud frame IDs must not be empty')
        if image_frame != cloud_frame:
            raise ValueError('image and point cloud frame IDs must match')

        bgr_image = np.ascontiguousarray(
            bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding='bgr8',
            )
        )
        cloud_decode_start = time.perf_counter()
        xyz_points = organized_xyz_from_point_cloud(
            point_cloud_message,
            expected_shape=bgr_image.shape[:2],
        )
        cloud_decode_sec = time.perf_counter() - cloud_decode_start
        pipeline_start = time.perf_counter()
        frame_result = pipeline.process_with_detections(
            bgr_image,
            xyz_points,
        )
        pipeline_sec = time.perf_counter() - pipeline_start
        detections = frame_result.detections
        message = object_candidate_array_from_result(
            frame_result.build_result,
            cloud_frame,
            source_time_nanoseconds=image_time,
            pair_skew_sec=pair_skew_sec,
        )
        return CandidateObservation(
            message=message,
            error=None,
            bgr_image=bgr_image,
            detections=detections,
            cloud_decode_sec=cloud_decode_sec,
            pipeline_sec=pipeline_sec,
            processing_sec=time.perf_counter() - processing_start,
        )
    except Exception as error:  # Runtime boundary must fail closed per frame.
        message = object_candidate_array_from_result(
            None,
            output_frame,
            source_time_nanoseconds=image_time,
            pair_skew_sec=pair_skew_sec,
            input_valid=False,
        )
        return CandidateObservation(
            message=message,
            error=f'{type(error).__name__}: {error}',
            bgr_image=bgr_image,
            detections=detections,
            cloud_decode_sec=cloud_decode_sec,
            pipeline_sec=pipeline_sec,
            processing_sec=time.perf_counter() - processing_start,
        )


class LiveObjectCandidatePublisher(Node):
    """Synchronize live stereo products and publish source-preserving candidates."""

    def __init__(self, *, parameter_overrides=None, pipeline=None, bridge=None):
        super().__init__(
            'live_object_candidate_publisher',
            parameter_overrides=parameter_overrides,
        )
        self._declare_parameters()

        image_topic = _string_parameter(self, 'image_topic')
        point_cloud_topic = _string_parameter(self, 'point_cloud_topic')
        candidate_topic = _string_parameter(self, 'candidate_topic')
        sync_queue_size = _positive_integer_parameter(
            self,
            'sync_queue_size',
        )

        if pipeline is None:
            min_confidence = _probability_parameter(
                self,
                'min_confidence',
            )
            segmenter = YoloInstanceSegmenter(
                YoloSegmenterConfig(
                    model_path=_string_parameter(self, 'model_path'),
                    min_confidence=min_confidence,
                    iou_threshold=_probability_parameter(
                        self,
                        'iou_threshold',
                    ),
                    tracker=_string_parameter(self, 'tracker'),
                    device=_string_parameter(self, 'device'),
                    inference_size=_positive_integer_parameter(
                        self,
                        'inference_size',
                    ),
                    allowed_classes=_string_sequence_parameter(
                        self,
                        'allowed_classes',
                    ),
                    filter_classes=True,
                )
            )
            pipeline = LiveCandidatePipeline(
                segmenter,
                CandidateBuilder(
                    config=CandidateBuilderConfig(
                        min_detection_confidence=min_confidence,
                    )
                ),
            )
        if not callable(getattr(pipeline, 'process_with_detections', None)):
            raise TypeError(
                'pipeline must provide a callable process_with_detections method'
            )

        self._bridge = bridge or CvBridge()
        self._pipeline = pipeline
        self._last_error = None
        self._show_gui = self.get_parameter('show_gui').value
        if not isinstance(self._show_gui, bool):
            raise ValueError('show_gui must be a boolean')
        self._previous_gui_time = None
        self._performance_window_start = time.perf_counter()
        self._performance_samples = []
        self._publisher = self.create_publisher(
            ObjectCandidateArray,
            candidate_topic,
            _candidate_qos(),
        )
        sensor_qos = _sensor_qos(sync_queue_size)
        self._image_subscriber = Subscriber(
            self,
            Image,
            image_topic,
            qos_profile=sensor_qos,
        )
        self._point_cloud_subscriber = Subscriber(
            self,
            PointCloud2,
            point_cloud_topic,
            qos_profile=sensor_qos,
        )
        self._synchronizer = TimeSynchronizer(
            [self._image_subscriber, self._point_cloud_subscriber],
            sync_queue_size,
        )
        self._synchronizer.registerCallback(self._on_synchronized_pair)
        self.get_logger().info(
            'live Objective 3.2 candidate publisher started: '
            f'{image_topic} + {point_cloud_topic} -> {candidate_topic}'
        )

    def _declare_parameters(self):
        self.declare_parameter('image_topic', '/stereo/left/image_rect')
        self.declare_parameter('point_cloud_topic', '/stereo/points2')
        self.declare_parameter('candidate_topic', '/object_candidates')
        self.declare_parameter('sync_queue_size', 15)
        self.declare_parameter('model_path', 'yolo26n-seg.pt')
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('iou_threshold', 0.7)
        self.declare_parameter('tracker', 'bytetrack.yaml')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('inference_size', 480)
        self.declare_parameter('show_gui', False)
        self.declare_parameter(
            'allowed_classes',
            list(DEFAULT_OBJECT_CLASSES),
        )

    def _on_synchronized_pair(self, image_message, point_cloud_message):
        observation = candidate_observation_from_pair(
            image_message,
            point_cloud_message,
            bridge=self._bridge,
            pipeline=self._pipeline,
        )
        if observation.error is not None:
            if observation.error != self._last_error:
                self.get_logger().warning(
                    'publishing invalid candidate observation: '
                    f'{observation.error}'
                )
            self._last_error = observation.error
        else:
            self._last_error = None
        self._publisher.publish(observation.message)
        self._record_performance(observation)
        self._show_candidate_observation(observation)

    def _record_performance(self, observation):
        """Log rolling callback costs without changing candidate semantics."""
        source_time = _stamp_nanoseconds(observation.message.header.stamp)
        source_age_sec = (
            self.get_clock().now().nanoseconds - source_time
        ) / _NANOSECONDS_PER_SECOND
        self._performance_samples.append(
            (
                observation.cloud_decode_sec,
                observation.pipeline_sec,
                observation.processing_sec,
                source_age_sec,
            )
        )
        now = time.perf_counter()
        if now - self._performance_window_start < _PERFORMANCE_LOG_PERIOD_SEC:
            return

        samples = self._performance_samples
        cloud_values = [item[0] for item in samples if item[0] is not None]
        pipeline_values = [item[1] for item in samples if item[1] is not None]
        total_values = [item[2] for item in samples]
        age_values = [item[3] for item in samples]
        self.get_logger().info(
            'live candidate performance: '
            f'pairs={len(samples)} '
            f'cloud_avg={_mean_milliseconds(cloud_values):.1f}ms '
            f'pipeline_avg={_mean_milliseconds(pipeline_values):.1f}ms '
            f'total_avg={_mean_milliseconds(total_values):.1f}ms '
            f'source_age_avg={_mean_milliseconds(age_values):.1f}ms'
        )
        self._performance_samples = []
        self._performance_window_start = now

    def _show_candidate_observation(self, observation):
        if not self._show_gui or observation.bgr_image is None:
            return

        now = time.perf_counter()
        fps = 0.0
        if self._previous_gui_time is not None:
            fps = 1.0 / max(now - self._previous_gui_time, 1.0e-6)
        self._previous_gui_time = now
        status = (
            f'LIVE STEREO XYZ | valid={observation.message.valid} | '
            f'3D={len(observation.message.candidates)} | {fps:.1f} FPS'
        )
        try:
            annotated = draw_detections(
                observation.bgr_image,
                observation.detections,
                fps,
                status_text=status,
            )
            cv2.imshow(_WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow(_WINDOW_NAME)
                self._show_gui = False
                self.get_logger().info('live candidate GUI closed')
        except cv2.error as error:
            self._show_gui = False
            self.get_logger().error(
                f'disabling live candidate GUI: {error}'
            )

    def destroy_node(self):
        if self._show_gui:
            try:
                cv2.destroyWindow(_WINDOW_NAME)
            except cv2.error:
                pass
        return super().destroy_node()


def _candidate_qos():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _sensor_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _mean_milliseconds(values):
    if not values:
        return 0.0
    return 1000.0 * sum(values) / len(values)


def _stamp_nanoseconds(stamp):
    try:
        seconds = operator.index(stamp.sec)
        nanoseconds = operator.index(stamp.nanosec)
    except (AttributeError, TypeError) as error:
        raise ValueError('source stamp must contain integer sec/nanosec') from error
    if seconds < 0 or not 0 <= nanoseconds < _NANOSECONDS_PER_SECOND:
        raise ValueError('source stamp is outside the ROS time range')
    return seconds * _NANOSECONDS_PER_SECOND + nanoseconds


def _string_parameter(node, name):
    value = node.get_parameter(name).value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _string_sequence_parameter(node, name):
    value = node.get_parameter(name).value
    if isinstance(value, str):
        raise ValueError(f'{name} must be a string sequence')
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a string sequence') from error
    if not values or any(
        not isinstance(item, str) or not item.strip()
        for item in values
    ):
        raise ValueError(f'{name} must contain non-empty strings')
    return tuple(item.strip() for item in values)


def _probability_parameter(node, name):
    try:
        value = float(node.get_parameter(name).value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return value


def _positive_integer_parameter(node, name):
    raw_value = node.get_parameter(name).value
    try:
        value = operator.index(raw_value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if isinstance(raw_value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def main(args=None):
    """Run the live Objective 3.2 candidate publisher."""
    rclpy.init(args=args)
    node = LiveObjectCandidatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
