"""Tests for model-independent mask and stereo-XYZ candidate fusion."""

from markerless_object_perception.candidate_builder import (
    CandidateBuilder,
    CandidateBuilderConfig,
    InstanceMaskDetection,
)
from markerless_object_perception.masked_point_localizer import (
    MaskedPointLocalizer,
    MaskedPointLocalizerConfig,
)
import numpy as np
import pytest


def make_builder(min_confidence=0.5):
    """Create a candidate builder with small synthetic-test thresholds."""
    localizer = MaskedPointLocalizer(
        MaskedPointLocalizerConfig(
            min_depth_m=0.1,
            max_depth_m=2.0,
            min_valid_points=4,
            max_spread_m=0.05,
        )
    )
    return CandidateBuilder(
        localizer,
        CandidateBuilderConfig(
            min_detection_confidence=min_confidence,
        ),
    )


def detection(label, confidence, track_id, mask):
    """Build one concise synthetic model detection."""
    return InstanceMaskDetection(
        class_label=label,
        confidence=confidence,
        track_id=track_id,
        mask=mask,
    )


def test_builds_candidate_and_preserves_model_identity():
    xyz = np.full((4, 4, 3), [0.25, -0.1, 0.7], dtype=np.float64)
    mask = np.ones((4, 4), dtype=bool)

    result = make_builder().build(
        [detection('bottle', 0.91, 7, mask)],
        xyz,
    )

    assert not result.rejections
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.class_label == 'bottle'
    assert candidate.class_confidence == pytest.approx(0.91)
    assert candidate.track_id == 7
    assert candidate.point == pytest.approx((0.25, -0.1, 0.7))
    assert candidate.localization_confidence == pytest.approx(1.0)
    assert candidate.valid_point_count == 16
    assert candidate.inlier_count == 16
    assert candidate.localization_spread_m == pytest.approx(0.0)


def test_two_masks_produce_two_independent_3d_candidates():
    xyz = np.zeros((4, 8, 3), dtype=np.float64)
    xyz[:, :4, :] = [0.2, 0.0, 0.6]
    xyz[:, 4:, :] = [0.5, 0.1, 0.9]
    left_mask = np.zeros((4, 8), dtype=bool)
    left_mask[:, :4] = True
    right_mask = np.zeros((4, 8), dtype=bool)
    right_mask[:, 4:] = True

    result = make_builder().build(
        [
            detection('cup', 0.8, 2, left_mask),
            detection('box', 0.75, 3, right_mask),
        ],
        xyz,
    )

    assert not result.rejections
    assert [item.track_id for item in result.candidates] == [2, 3]
    assert result.candidates[0].point == pytest.approx((0.2, 0.0, 0.6))
    assert result.candidates[1].point == pytest.approx((0.5, 0.1, 0.9))


def test_low_detection_confidence_is_rejected_before_localization():
    bad_shape_mask = np.ones((2, 2), dtype=bool)

    result = make_builder(min_confidence=0.7).build(
        [detection('cup', 0.69, 1, bad_shape_mask)],
        np.zeros((5, 5, 3)),
    )

    assert not result.candidates
    assert len(result.rejections) == 1
    assert result.rejections[0].track_id == 1
    assert result.rejections[0].reason == 'low_detection_confidence'


def test_invalid_stereo_points_become_explicit_rejection():
    xyz = np.full((4, 4, 3), [np.nan, np.nan, np.nan])
    mask = np.ones((4, 4), dtype=bool)

    result = make_builder().build(
        [detection('bottle', 0.9, 4, mask)],
        xyz,
    )

    assert not result.candidates
    assert result.rejections[0].reason == 'insufficient_valid_points'


def test_localization_confidence_counts_invalid_mask_depth_as_missing_support():
    xyz = np.full((4, 4, 3), [0.1, 0.0, 0.6], dtype=np.float64)
    xyz[:2, :, :] = np.nan
    mask = np.ones((4, 4), dtype=bool)

    result = make_builder().build(
        [detection('cup', 0.9, 4, mask)],
        xyz,
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.valid_point_count == 8
    assert candidate.inlier_count == 8
    assert candidate.localization_confidence == pytest.approx(0.5)


def test_fresh_empty_detection_set_produces_empty_result():
    result = make_builder().build([], np.zeros((4, 4, 3)))

    assert result.candidates == ()
    assert result.rejections == ()


def test_duplicate_track_id_in_one_frame_is_rejected():
    xyz = np.full((4, 4, 3), [0.0, 0.0, 0.5])
    mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match='duplicate track_id'):
        make_builder().build(
            [
                detection('cup', 0.8, 5, mask),
                detection('box', 0.8, 5, mask),
            ],
            xyz,
        )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'class_label': '', 'confidence': 0.8, 'track_id': 1},
        {'class_label': 'cup', 'confidence': -0.1, 'track_id': 1},
        {'class_label': 'cup', 'confidence': float('nan'), 'track_id': 1},
        {'class_label': 'cup', 'confidence': 0.8, 'track_id': -1},
    ],
)
def test_invalid_model_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        InstanceMaskDetection(
            mask=np.ones((2, 2), dtype=bool),
            **kwargs,
        )


@pytest.mark.parametrize('threshold', [-0.1, 1.1, float('inf')])
def test_invalid_confidence_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        CandidateBuilderConfig(min_detection_confidence=threshold)
