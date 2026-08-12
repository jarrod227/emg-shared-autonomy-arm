"""Tests for calibrated stereo image rectification helpers."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo

from stereo_hand_observer.stereo_rectification import (
    build_rectification_maps,
    rectify_pair,
    resize_rectified_pair,
    scale_camera_info,
)


def camera_info(width=8, height=6):
    """Return an identity-rectification pinhole camera model."""
    focal_length = 100.0
    principal_x = (width - 1) / 2.0
    principal_y = (height - 1) / 2.0
    intrinsic = [
        focal_length,
        0.0,
        principal_x,
        0.0,
        focal_length,
        principal_y,
        0.0,
        0.0,
        1.0,
    ]
    message = CameraInfo()
    message.width = width
    message.height = height
    message.distortion_model = "plumb_bob"
    message.d = [0.0] * 5
    message.k = intrinsic
    message.r = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    message.p = [
        focal_length,
        0.0,
        principal_x,
        0.0,
        0.0,
        focal_length,
        principal_y,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return message


def test_identity_rectification_preserves_pixels():
    info = camera_info()
    maps = build_rectification_maps(info)
    image = np.arange(
        info.height * info.width * 3,
        dtype=np.uint8,
    ).reshape(info.height, info.width, 3)

    left, right = rectify_pair(image, image.copy(), maps, maps)

    assert np.array_equal(left, image)
    assert np.array_equal(right, image)


def test_maps_have_calibration_size_and_finite_coordinates():
    info = camera_info(width=12, height=10)

    maps = build_rectification_maps(info)

    assert (maps.width, maps.height) == (12, 10)
    assert maps.map_x.shape == (10, 12)
    assert maps.map_y.shape == (10, 12)
    assert np.all(np.isfinite(maps.map_x))
    assert np.all(np.isfinite(maps.map_y))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda info: setattr(info, "width", 0), "must be positive"),
        (
            lambda info: setattr(info, "distortion_model", "equidistant"),
            "unsupported.*distortion_model",
        ),
        (lambda info: setattr(info, "d", []), "must contain"),
        (lambda info: setattr(info, "k", [0.0] * 9), "focal lengths"),
    ],
)
def test_bad_camera_info_is_rejected(mutate, message):
    info = camera_info()
    mutate(info)

    with pytest.raises(ValueError, match=message):
        build_rectification_maps(info)


def test_rectify_pair_rejects_wrong_image_size_and_type():
    maps = build_rectification_maps(camera_info())

    with pytest.raises(ValueError, match="left image size"):
        rectify_pair(
            np.zeros((5, 8, 3), dtype=np.uint8),
            np.zeros((6, 8, 3), dtype=np.uint8),
            maps,
            maps,
        )
    with pytest.raises(ValueError, match="right image must use uint8"):
        rectify_pair(
            np.zeros((6, 8, 3), dtype=np.uint8),
            np.zeros((6, 8, 3), dtype=np.float32),
            maps,
            maps,
        )


def test_resize_and_camera_info_scaling_preserve_stereo_geometry():
    info = camera_info(width=8, height=6)
    info.p[3] = -6.4
    info.binning_x = 3
    info.binning_y = 2
    info.roi.x_offset = 2
    info.roi.y_offset = 2
    info.roi.width = 4
    info.roi.height = 2
    info.roi.do_rectify = True
    original_k = np.asarray(info.k).copy()
    original_p = np.asarray(info.p).copy()
    original_baseline = -info.p[3] / info.p[0]

    scaled = scale_camera_info(info, 4, 3)

    assert (scaled.width, scaled.height) == (4, 3)
    assert scaled.k == pytest.approx(
        [50.0, 0.0, 1.75, 0.0, 50.0, 1.25, 0.0, 0.0, 1.0]
    )
    assert scaled.p == pytest.approx(
        [50.0, 0.0, 1.75, -3.2, 0.0, 50.0, 1.25, 0.0,
         0.0, 0.0, 1.0, 0.0]
    )
    assert np.array_equal(scaled.r, info.r)
    assert np.array_equal(scaled.d, info.d)
    assert np.array_equal(info.k, original_k)
    assert np.array_equal(info.p, original_p)
    assert -scaled.p[3] / scaled.p[0] == pytest.approx(original_baseline)
    assert (scaled.binning_x, scaled.binning_y) == (3, 2)
    assert (
        scaled.roi.x_offset,
        scaled.roi.y_offset,
        scaled.roi.width,
        scaled.roi.height,
        scaled.roi.do_rectify,
    ) == (1, 1, 2, 1, True)

    left = np.full((6, 8, 3), 11, dtype=np.uint8)
    right = np.full((6, 8, 3), 22, dtype=np.uint8)
    resized_left, resized_right = resize_rectified_pair(left, right, 4, 3)
    assert resized_left.shape == (3, 4, 3)
    assert resized_right.shape == (3, 4, 3)
    assert np.all(resized_left == 11)
    assert np.all(resized_right == 22)
