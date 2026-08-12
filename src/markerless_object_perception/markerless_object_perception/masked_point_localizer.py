"""Robustly localize one segmented object in an aligned stereo XYZ image."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MaskedPointLocalizerConfig:
    """Quality limits for mask-filtered stereo localization."""

    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    min_valid_points: int = 20
    outlier_mad_scale: float = 3.5
    min_outlier_scale_m: float = 1.0e-4
    max_spread_m: float = 0.12

    def __post_init__(self) -> None:
        """Normalize numeric values and reject unsafe configurations."""
        min_depth_m = _finite_float(self.min_depth_m, 'min_depth_m')
        max_depth_m = _finite_float(self.max_depth_m, 'max_depth_m')
        if min_depth_m < 0.0:
            raise ValueError('min_depth_m must be non-negative')
        if max_depth_m <= min_depth_m:
            raise ValueError('max_depth_m must be greater than min_depth_m')

        if (
            not isinstance(self.min_valid_points, int)
            or isinstance(self.min_valid_points, bool)
            or self.min_valid_points <= 0
        ):
            raise ValueError('min_valid_points must be a positive integer')

        outlier_mad_scale = _positive_float(
            self.outlier_mad_scale, 'outlier_mad_scale'
        )
        min_outlier_scale_m = _positive_float(
            self.min_outlier_scale_m, 'min_outlier_scale_m'
        )
        max_spread_m = _positive_float(self.max_spread_m, 'max_spread_m')

        object.__setattr__(self, 'min_depth_m', min_depth_m)
        object.__setattr__(self, 'max_depth_m', max_depth_m)
        object.__setattr__(self, 'outlier_mad_scale', outlier_mad_scale)
        object.__setattr__(
            self, 'min_outlier_scale_m', min_outlier_scale_m
        )
        object.__setattr__(self, 'max_spread_m', max_spread_m)


@dataclass(frozen=True)
class MaskedPointLocalizationResult:
    """One localization result with explicit validity and quality evidence."""

    valid: bool
    point: tuple[float, float, float] | None
    reason: str
    masked_point_count: int
    valid_point_count: int
    inlier_count: int
    spread_m: float | None


class MaskedPointLocalizer:
    """Estimate a robust 3D center from one mask and aligned XYZ image."""

    def __init__(self, config: MaskedPointLocalizerConfig | None = None):
        self._config = config or MaskedPointLocalizerConfig()
        if not isinstance(self._config, MaskedPointLocalizerConfig):
            raise TypeError('config must be a MaskedPointLocalizerConfig')

    def localize(self, mask, xyz_points) -> MaskedPointLocalizationResult:
        """Filter invalid/background points and return a robust object center."""
        mask_array = np.asarray(mask)
        if mask_array.ndim != 2:
            raise ValueError('mask must have shape HxW')
        if mask_array.dtype != np.bool_:
            raise ValueError('mask must be a boolean array')

        try:
            xyz_array = np.asarray(xyz_points)
        except (TypeError, ValueError) as error:
            raise ValueError('xyz_points must contain numeric values') from error
        if xyz_array.ndim != 3 or xyz_array.shape[2] != 3:
            raise ValueError('xyz_points must have shape HxWx3')
        if xyz_array.shape[:2] != mask_array.shape:
            raise ValueError('mask and xyz_points image shapes must match')
        if not np.issubdtype(xyz_array.dtype, np.number):
            try:
                xyz_array = np.asarray(xyz_points, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    'xyz_points must contain numeric values'
                ) from error

        masked_point_count = int(np.count_nonzero(mask_array))
        masked_points = np.asarray(
            xyz_array[mask_array],
            dtype=np.float64,
        )
        finite = np.all(np.isfinite(masked_points), axis=1)
        depth = masked_points[:, 2]
        valid_mask = (
            finite
            & (depth >= self._config.min_depth_m)
            & (depth <= self._config.max_depth_m)
        )
        valid_points = masked_points[valid_mask]
        valid_point_count = int(valid_points.shape[0])
        if valid_point_count < self._config.min_valid_points:
            return _invalid_result(
                'insufficient_valid_points',
                masked_point_count=masked_point_count,
                valid_point_count=valid_point_count,
            )

        initial_center = np.median(valid_points, axis=0)
        distances = np.linalg.norm(valid_points - initial_center, axis=1)
        median_distance = float(np.median(distances))
        distance_mad = float(
            np.median(np.abs(distances - median_distance))
        )
        robust_scale = max(
            1.4826 * distance_mad,
            self._config.min_outlier_scale_m,
        )
        outlier_threshold = (
            median_distance
            + self._config.outlier_mad_scale * robust_scale
        )
        inlier_points = valid_points[distances <= outlier_threshold]
        inlier_count = int(inlier_points.shape[0])
        if inlier_count < self._config.min_valid_points:
            return _invalid_result(
                'insufficient_inliers',
                masked_point_count=masked_point_count,
                valid_point_count=valid_point_count,
                inlier_count=inlier_count,
            )

        center = np.median(inlier_points, axis=0)
        center_distances = np.linalg.norm(inlier_points - center, axis=1)
        spread_m = float(np.percentile(center_distances, 90.0))
        if spread_m > self._config.max_spread_m:
            return _invalid_result(
                'excessive_spread',
                masked_point_count=masked_point_count,
                valid_point_count=valid_point_count,
                inlier_count=inlier_count,
                spread_m=spread_m,
            )

        return MaskedPointLocalizationResult(
            valid=True,
            point=tuple(float(value) for value in center),
            reason='localized',
            masked_point_count=masked_point_count,
            valid_point_count=valid_point_count,
            inlier_count=inlier_count,
            spread_m=spread_m,
        )


def _invalid_result(
    reason: str,
    *,
    masked_point_count: int,
    valid_point_count: int,
    inlier_count: int = 0,
    spread_m: float | None = None,
) -> MaskedPointLocalizationResult:
    """Build a fail-closed result without a usable 3D point."""
    return MaskedPointLocalizationResult(
        valid=False,
        point=None,
        reason=reason,
        masked_point_count=masked_point_count,
        valid_point_count=valid_point_count,
        inlier_count=inlier_count,
        spread_m=spread_m,
    )


def _finite_float(value, name: str) -> float:
    """Convert one configuration value to a finite float."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted


def _positive_float(value, name: str) -> float:
    """Convert one configuration value to a finite positive float."""
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f'{name} must be positive')
    return converted
