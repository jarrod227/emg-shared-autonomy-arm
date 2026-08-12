"""Pure OpenCV rectification helpers for calibrated stereo images."""

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np


_SUPPORTED_DISTORTION_MODELS = {"plumb_bob", "rational_polynomial"}
_SUPPORTED_DISTORTION_LENGTHS = {4, 5, 8, 12, 14}


@dataclass(frozen=True)
class RectificationMaps:
    """Precomputed OpenCV maps for one calibrated camera."""

    map_x: np.ndarray
    map_y: np.ndarray
    width: int
    height: int


def _matrix(values, shape, name):
    array = np.asarray(values, dtype=np.float64)
    if array.size != int(np.prod(shape)):
        raise ValueError(
            f"{name} must contain {int(np.prod(shape))} values"
        )
    array = array.reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def build_rectification_maps(camera_info):
    """Build reusable rectification maps from one ROS CameraInfo message."""
    width = int(camera_info.width)
    height = int(camera_info.height)
    if width <= 0 or height <= 0:
        raise ValueError("CameraInfo width and height must be positive")
    if camera_info.distortion_model not in _SUPPORTED_DISTORTION_MODELS:
        raise ValueError(
            "unsupported CameraInfo distortion_model: "
            f"{camera_info.distortion_model!r}"
        )

    intrinsic = _matrix(camera_info.k, (3, 3), "CameraInfo.k")
    rectification = _matrix(camera_info.r, (3, 3), "CameraInfo.r")
    projection = _matrix(camera_info.p, (3, 4), "CameraInfo.p")
    distortion = np.asarray(camera_info.d, dtype=np.float64).reshape(-1)
    if distortion.size not in _SUPPORTED_DISTORTION_LENGTHS:
        raise ValueError(
            "CameraInfo.d must contain 4, 5, 8, 12, or 14 values"
        )
    if not np.all(np.isfinite(distortion)):
        raise ValueError("CameraInfo.d must contain only finite values")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("CameraInfo.k must contain positive focal lengths")
    if projection[0, 0] <= 0.0 or projection[1, 1] <= 0.0:
        raise ValueError("CameraInfo.p must contain positive focal lengths")

    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic,
        distortion,
        rectification,
        projection[:, :3],
        (width, height),
        cv2.CV_32FC1,
    )
    if map_x.shape != (height, width) or map_y.shape != (height, width):
        raise ValueError("OpenCV returned rectification maps with wrong size")
    if not np.all(np.isfinite(map_x)) or not np.all(np.isfinite(map_y)):
        raise ValueError("OpenCV returned non-finite rectification maps")
    return RectificationMaps(map_x, map_y, width, height)


def _rectify_image(image, maps, side):
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{side} image must use uint8 pixels")
    if array.ndim not in (2, 3):
        raise ValueError(f"{side} image must have shape HxW or HxWxC")
    if array.shape[:2] != (maps.height, maps.width):
        raise ValueError(
            f"{side} image size must be {maps.width}x{maps.height}, "
            f"got {array.shape[1]}x{array.shape[0]}"
        )
    return cv2.remap(
        array,
        maps.map_x,
        maps.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def rectify_pair(left_image, right_image, left_maps, right_maps):
    """Rectify one stereo pair without changing its left/right association."""
    return (
        _rectify_image(left_image, left_maps, "left"),
        _rectify_image(right_image, right_maps, "right"),
    )


def resize_rectified_pair(left_image, right_image, width, height):
    """Resize a rectified pair identically without changing eye ordering."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("output width and height must be positive integers")
    left = np.asarray(left_image)
    right = np.asarray(right_image)
    if left.shape != right.shape or left.ndim not in (2, 3):
        raise ValueError("left and right images must have the same image shape")
    interpolation = (
        cv2.INTER_AREA
        if width <= left.shape[1] and height <= left.shape[0]
        else cv2.INTER_LINEAR
    )
    return (
        cv2.resize(left, (width, height), interpolation=interpolation),
        cv2.resize(right, (width, height), interpolation=interpolation),
    )


def scale_camera_info(camera_info, width, height):
    """Scale a calibrated CameraInfo for a resized full-frame image."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("output width and height must be positive integers")
    source_width = int(camera_info.width)
    source_height = int(camera_info.height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("CameraInfo width and height must be positive")

    scale_x = width / source_width
    scale_y = height / source_height
    scaled = deepcopy(camera_info)
    scaled.width = width
    scaled.height = height
    intrinsic = np.asarray(
        camera_info.k,
        dtype=np.float64,
    ).reshape(3, 3).copy()
    projection = np.asarray(
        camera_info.p,
        dtype=np.float64,
    ).reshape(3, 4).copy()
    intrinsic[0, :] *= scale_x
    intrinsic[1, :] *= scale_y
    projection[0, :] *= scale_x
    projection[1, :] *= scale_y
    scaled.k = intrinsic.reshape(-1).tolist()
    scaled.p = projection.reshape(-1).tolist()

    roi = scaled.roi
    roi.x_offset = int(roi.x_offset * scale_x)
    roi.y_offset = int(roi.y_offset * scale_y)
    roi.width = int(roi.width * scale_x)
    roi.height = int(roi.height * scale_y)
    return scaled
