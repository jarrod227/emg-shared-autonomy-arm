"""Connect live instance segmentation to mask-filtered stereo localization."""

from dataclasses import dataclass

from markerless_object_perception.candidate_builder import (
    CandidateBuilder,
    CandidateBuildResult,
    InstanceMaskDetection,
)
import numpy as np


@dataclass(frozen=True)
class LiveCandidateFrameResult:
    """Retain 2D masks for diagnostics alongside the 3D build result."""

    detections: tuple[InstanceMaskDetection, ...]
    build_result: CandidateBuildResult


class LiveCandidatePipeline:
    """Preserve tracker state while converting one aligned RGB/XYZ frame."""

    def __init__(self, segmenter, builder: CandidateBuilder | None = None):
        if not callable(getattr(segmenter, 'track', None)):
            raise TypeError('segmenter must provide a callable track method')
        self._segmenter = segmenter
        self._builder = builder or CandidateBuilder()
        if not isinstance(self._builder, CandidateBuilder):
            raise TypeError('builder must be a CandidateBuilder')

    def process(self, bgr_image, xyz_points):
        """Run 2D tracking, then localize every accepted instance mask."""
        return self.process_with_detections(
            bgr_image,
            xyz_points,
        ).build_result

    def process_with_detections(self, bgr_image, xyz_points):
        """Return both diagnostic masks and the corresponding 3D result."""
        image = np.asarray(bgr_image)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError('bgr_image must have shape HxWx3')

        try:
            xyz = np.asarray(xyz_points)
        except (TypeError, ValueError) as error:
            raise ValueError('xyz_points must contain numeric values') from error
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError('xyz_points must have shape HxWx3')
        if xyz.shape[:2] != image.shape[:2]:
            raise ValueError('bgr_image and xyz_points image shapes must match')
        if not np.issubdtype(xyz.dtype, np.number):
            try:
                xyz = np.asarray(xyz_points, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    'xyz_points must contain numeric values'
                ) from error

        detections = tuple(self._segmenter.track(image))
        return LiveCandidateFrameResult(
            detections=detections,
            build_result=self._builder.build(detections, xyz),
        )
