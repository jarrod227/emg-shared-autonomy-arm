"""Tests for organized PointCloud2 to aligned XYZ conversion."""

import struct

from markerless_object_perception.point_cloud_adapter import (
    organized_xyz_from_point_cloud,
)
import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField


def make_cloud(xyz, *, row_padding=0, big_endian=False):
    """Build one organized FLOAT32 XYZ cloud with optional row padding."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError('xyz must have shape HxWx3')

    height, width = xyz.shape[:2]
    point_step = 12
    row_step = width * point_step + row_padding
    data = bytearray(row_step * height)
    format_string = '>fff' if big_endian else '<fff'
    for row in range(height):
        for column in range(width):
            struct.pack_into(
                format_string,
                data,
                row * row_step + column * point_step,
                *xyz[row, column],
            )

    cloud = PointCloud2()
    cloud.height = height
    cloud.width = width
    cloud.fields = [
        PointField(
            name=name,
            offset=index * 4,
            datatype=PointField.FLOAT32,
            count=1,
        )
        for index, name in enumerate(('x', 'y', 'z'))
    ]
    cloud.is_bigendian = big_endian
    cloud.point_step = point_step
    cloud.row_step = row_step
    cloud.data = bytes(data)
    cloud.is_dense = bool(np.all(np.isfinite(xyz)))
    return cloud


def test_converts_organized_cloud_and_preserves_nan():
    expected = np.array(
        [
            [[0.1, -0.2, 0.7], [np.nan, np.nan, np.nan]],
            [[0.3, 0.0, 0.8], [0.4, 0.1, 0.9]],
        ],
        dtype=np.float32,
    )

    actual = organized_xyz_from_point_cloud(
        make_cloud(expected),
        expected_shape=(2, 2),
    )

    assert actual.shape == (2, 2, 3)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_preserves_float64_when_cloud_fields_are_float64():
    expected = np.array(
        [
            [[0.1, -0.2, 0.7], [0.3, 0.0, 0.8]],
            [[0.4, 0.1, 0.9], [0.5, 0.2, 1.0]],
        ],
        dtype='<f8',
    )
    cloud = PointCloud2()
    cloud.height = 2
    cloud.width = 2
    cloud.fields = [
        PointField(
            name=name,
            offset=index * 8,
            datatype=PointField.FLOAT64,
            count=1,
        )
        for index, name in enumerate(('x', 'y', 'z'))
    ]
    cloud.is_bigendian = False
    cloud.point_step = 24
    cloud.row_step = cloud.width * cloud.point_step
    cloud.data = expected.tobytes()

    actual = organized_xyz_from_point_cloud(cloud)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)


def test_handles_row_padding_without_shifting_pixels():
    expected = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float32,
    )

    actual = organized_xyz_from_point_cloud(
        make_cloud(expected, row_padding=16)
    )

    np.testing.assert_allclose(actual, expected)


def test_converts_big_endian_cloud():
    expected = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
        dtype=np.float32,
    )

    actual = organized_xyz_from_point_cloud(
        make_cloud(expected, big_endian=True)
    )

    np.testing.assert_allclose(actual, expected)


def test_rejects_unorganized_cloud():
    cloud = make_cloud(np.zeros((1, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match='must be organized'):
        organized_xyz_from_point_cloud(cloud)


def test_rejects_missing_xyz_field():
    cloud = make_cloud(np.zeros((2, 2, 3), dtype=np.float32))
    cloud.fields = cloud.fields[:2]

    with pytest.raises(ValueError, match="exactly one 'z' field"):
        organized_xyz_from_point_cloud(cloud)


def test_rejects_non_floating_xyz_field():
    cloud = make_cloud(np.zeros((2, 2, 3), dtype=np.float32))
    cloud.fields[0].datatype = PointField.UINT32

    with pytest.raises(ValueError, match="field 'x' must be FLOAT32"):
        organized_xyz_from_point_cloud(cloud)


def test_rejects_image_shape_mismatch():
    cloud = make_cloud(np.zeros((2, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match='does not match'):
        organized_xyz_from_point_cloud(
            cloud,
            expected_shape=(3, 2),
        )


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda cloud: setattr(cloud, 'row_step', 1),
            'row_step is smaller',
        ),
        (
            lambda cloud: setattr(cloud, 'data', cloud.data[:-1]),
            'data is shorter',
        ),
    ],
)
def test_rejects_malformed_binary_layout(mutation, message):
    cloud = make_cloud(np.zeros((2, 2, 3), dtype=np.float32))
    mutation(cloud)

    with pytest.raises(ValueError, match=message):
        organized_xyz_from_point_cloud(cloud)


@pytest.mark.parametrize(
    'expected_shape',
    ['2x2', (2,), (2, 2, 1), (0, 2), (True, 2)],
)
def test_rejects_invalid_expected_shape(expected_shape):
    cloud = make_cloud(np.zeros((2, 2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match='expected_shape'):
        organized_xyz_from_point_cloud(
            cloud,
            expected_shape=expected_shape,
        )
