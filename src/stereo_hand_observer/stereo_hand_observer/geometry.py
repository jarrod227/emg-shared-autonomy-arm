"""Pure stereo geometry for Objective 4.2 hand-keypoint observations."""

from dataclasses import dataclass

import numpy as np


_EPSILON = 1e-12


class StereoGeometryError(ValueError):
    """Report a stereo pair that cannot produce a trustworthy 3D point."""


@dataclass(frozen=True)
class TriangulationResult:
    """A recovered 3D point together with its image-space quality metrics."""

    point: np.ndarray
    epipolar_error_px: float
    left_reprojection_error_px: float
    right_reprojection_error_px: float

    @property
    def max_reprojection_error_px(self):
        """Return the worse of the two per-camera reprojection errors."""
        return max(
            self.left_reprojection_error_px,
            self.right_reprojection_error_px,
        )


def fundamental_from_projections(left_projection, right_projection):
    """Derive a fundamental matrix from two calibrated projections."""
    left_matrix = _finite_array(
        left_projection,
        (3, 4),
        "left_projection",
    )
    right_matrix = _finite_array(
        right_projection,
        (3, 4),
        "right_projection",
    )

    _, _, left_singular_vectors = np.linalg.svd(left_matrix)
    left_camera_center = left_singular_vectors[-1]
    right_epipole = right_matrix @ left_camera_center
    epipole_norm = np.linalg.norm(right_epipole)
    if epipole_norm <= _EPSILON:
        raise StereoGeometryError(
            "projection matrices do not define a stereo baseline"
        )

    epipole_cross_product = np.array(
        [
            [0.0, -right_epipole[2], right_epipole[1]],
            [right_epipole[2], 0.0, -right_epipole[0]],
            [-right_epipole[1], right_epipole[0], 0.0],
        ]
    )
    fundamental = (
        epipole_cross_product
        @ right_matrix
        @ np.linalg.pinv(left_matrix)
    )
    fundamental_norm = np.linalg.norm(fundamental)
    if fundamental_norm <= _EPSILON:
        raise StereoGeometryError(
            "projection matrices produced a degenerate fundamental matrix"
        )
    return fundamental / fundamental_norm


def _finite_array(value, expected_shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_limit(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def project_point(projection_matrix, point):
    """Project one 3D point into an image and return its pixel coordinates."""
    matrix = _finite_array(projection_matrix, (3, 4), "projection_matrix")
    point = _finite_array(point, (3,), "point")
    homogeneous_point = np.append(point, 1.0)
    image_point = matrix @ homogeneous_point
    scale = image_point[2]
    if abs(scale) <= _EPSILON:
        raise StereoGeometryError("3D point projects to infinity")
    pixel = image_point[:2] / scale
    if not np.all(np.isfinite(pixel)):
        raise StereoGeometryError("3D point produced a non-finite pixel")
    return pixel


def epipolar_error(fundamental_matrix, left_pixel, right_pixel):
    """Return the worse point-to-epipolar-line distance in pixels."""
    matrix = _finite_array(
        fundamental_matrix,
        (3, 3),
        "fundamental_matrix",
    )
    if np.linalg.norm(matrix) <= _EPSILON:
        raise ValueError("fundamental_matrix must not be all zeros")

    left = _finite_array(left_pixel, (2,), "left_pixel")
    right = _finite_array(right_pixel, (2,), "right_pixel")
    left_h = np.append(left, 1.0)
    right_h = np.append(right, 1.0)

    right_line = matrix @ left_h
    left_line = matrix.T @ right_h
    right_norm = np.hypot(right_line[0], right_line[1])
    left_norm = np.hypot(left_line[0], left_line[1])
    if right_norm <= _EPSILON or left_norm <= _EPSILON:
        raise StereoGeometryError("epipolar line is degenerate")

    residual = abs(right_h @ matrix @ left_h)
    return float(max(residual / right_norm, residual / left_norm))


def triangulate_keypoint(
    left_projection,
    right_projection,
    fundamental_matrix,
    left_pixel,
    right_pixel,
    *,
    max_epipolar_error_px=1.5,
    max_reprojection_error_px=1.5,
    min_depth_m=1e-6,
):
    """Triangulate and reject a keypoint pair that fails geometry checks."""
    left_matrix = _finite_array(
        left_projection,
        (3, 4),
        "left_projection",
    )
    right_matrix = _finite_array(
        right_projection,
        (3, 4),
        "right_projection",
    )
    left = _finite_array(left_pixel, (2,), "left_pixel")
    right = _finite_array(right_pixel, (2,), "right_pixel")
    epipolar_limit = _nonnegative_limit(
        max_epipolar_error_px,
        "max_epipolar_error_px",
    )
    reprojection_limit = _nonnegative_limit(
        max_reprojection_error_px,
        "max_reprojection_error_px",
    )
    min_depth = float(min_depth_m)
    if not np.isfinite(min_depth) or min_depth < 0.0:
        raise ValueError("min_depth_m must be finite and non-negative")

    pair_error = epipolar_error(fundamental_matrix, left, right)
    if pair_error > epipolar_limit:
        raise StereoGeometryError(
            "keypoint pair exceeds epipolar-error limit: "
            f"{pair_error:.3f} px > {epipolar_limit:.3f} px"
        )

    system = np.vstack(
        (
            left[0] * left_matrix[2] - left_matrix[0],
            left[1] * left_matrix[2] - left_matrix[1],
            right[0] * right_matrix[2] - right_matrix[0],
            right[1] * right_matrix[2] - right_matrix[1],
        )
    )
    _, _, right_singular_vectors = np.linalg.svd(system)
    homogeneous_point = right_singular_vectors[-1]
    scale = homogeneous_point[3]
    spatial_norm = np.linalg.norm(homogeneous_point[:3])
    if abs(scale) <= _EPSILON * max(1.0, spatial_norm):
        raise StereoGeometryError(
            "keypoint rays do not produce a finite 3D point"
        )

    point = homogeneous_point[:3] / scale
    if not np.all(np.isfinite(point)):
        raise StereoGeometryError("triangulation produced a non-finite point")

    point_h = np.append(point, 1.0)
    left_depth = float((left_matrix @ point_h)[2])
    right_depth = float((right_matrix @ point_h)[2])
    if left_depth <= min_depth or right_depth <= min_depth:
        raise StereoGeometryError(
            "triangulated point is behind or too close to a camera"
        )

    left_reprojected = project_point(left_matrix, point)
    right_reprojected = project_point(right_matrix, point)
    left_error = float(np.linalg.norm(left_reprojected - left))
    right_error = float(np.linalg.norm(right_reprojected - right))
    worst_error = max(left_error, right_error)
    if worst_error > reprojection_limit:
        raise StereoGeometryError(
            "triangulation exceeds reprojection-error limit: "
            f"{worst_error:.3f} px > {reprojection_limit:.3f} px"
        )

    return TriangulationResult(
        point=point,
        epipolar_error_px=pair_error,
        left_reprojection_error_px=left_error,
        right_reprojection_error_px=right_error,
    )
