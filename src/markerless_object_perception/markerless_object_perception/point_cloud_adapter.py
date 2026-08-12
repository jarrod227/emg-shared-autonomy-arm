"""Convert an organized ROS PointCloud2 into an image-aligned XYZ array."""

from numbers import Integral
import sys

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import dtype_from_fields


_XYZ_FIELD_NAMES = ('x', 'y', 'z')
_FLOAT_DATATYPES = (PointField.FLOAT32, PointField.FLOAT64)


def organized_xyz_from_point_cloud(cloud, *, expected_shape=None):
    """Return HxWx3 XYZ while preserving invalid points as NaN values."""
    if not isinstance(cloud, PointCloud2):
        raise TypeError('cloud must be a sensor_msgs/PointCloud2 message')

    width = int(cloud.width)
    height = int(cloud.height)
    if width <= 0 or height <= 1:
        raise ValueError(
            'point cloud must be organized with positive width and height > 1'
        )
    expected_height, expected_width = _validated_expected_shape(
        expected_shape,
        default=(height, width),
    )
    if (height, width) != (expected_height, expected_width):
        raise ValueError(
            'point cloud shape does not match the aligned image: '
            f'{height}x{width} != {expected_height}x{expected_width}'
        )

    point_step = int(cloud.point_step)
    row_step = int(cloud.row_step)
    if point_step <= 0:
        raise ValueError('PointCloud2.point_step must be positive')
    packed_row_size = width * point_step
    if row_step < packed_row_size:
        raise ValueError(
            'PointCloud2.row_step is smaller than width * point_step'
        )
    required_bytes = row_step * height
    if len(cloud.data) < required_bytes:
        raise ValueError('PointCloud2.data is shorter than row_step * height')

    _validate_xyz_fields(cloud.fields)
    try:
        point_dtype = dtype_from_fields(
            cloud.fields,
            point_step=point_step,
        )
        points = np.ndarray(
            shape=(height, width),
            dtype=point_dtype,
            buffer=cloud.data,
            strides=(row_step, point_step),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'PointCloud2 field layout is invalid: {error}'
        ) from error

    byte_order_differs = (
        bool(sys.byteorder != 'little') != bool(cloud.is_bigendian)
    )
    components = []
    for name in _XYZ_FIELD_NAMES:
        values = points[name]
        if byte_order_differs:
            values = values.byteswap()
        components.append(np.asarray(values))

    return np.ascontiguousarray(np.stack(components, axis=2))


def _validate_xyz_fields(fields):
    for name in _XYZ_FIELD_NAMES:
        matching = [field for field in fields if field.name == name]
        if len(matching) != 1:
            raise ValueError(
                f'PointCloud2 must contain exactly one {name!r} field'
            )
        field = matching[0]
        if field.count != 1:
            raise ValueError(
                f'PointCloud2 field {name!r} must have count 1'
            )
        if field.datatype not in _FLOAT_DATATYPES:
            raise ValueError(
                f'PointCloud2 field {name!r} must be FLOAT32 or FLOAT64'
            )


def _validated_expected_shape(value, *, default):
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        raise ValueError('expected_shape must contain height and width')
    try:
        shape = tuple(value)
    except TypeError as error:
        raise ValueError(
            'expected_shape must contain height and width'
        ) from error
    if len(shape) != 2:
        raise ValueError('expected_shape must contain height and width')
    if any(
        not isinstance(item, Integral)
        or isinstance(item, bool)
        or item <= 0
        for item in shape
    ):
        raise ValueError('expected_shape values must be positive integers')
    return int(shape[0]), int(shape[1])
