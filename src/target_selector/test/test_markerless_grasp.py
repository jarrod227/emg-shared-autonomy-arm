"""Tests for planning-frame markerless grasp templates."""

import math
import os

import pytest

from target_selector.markerless_grasp import (
    build_markerless_grasp_pose,
    grasp_template_for_class,
    load_markerless_grasp_templates,
    MarkerlessGraspTemplate,
)


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'config',
    'markerless_grasp_templates.yaml',
)


def test_template_offset_is_applied_in_planning_frame():
    template = MarkerlessGraspTemplate(
        position_offset_m=(0.02, -0.03, 0.10),
    )

    pose = build_markerless_grasp_pose((0.40, 0.20, 0.70), template)

    assert pose.position.x == pytest.approx(0.42)
    assert pose.position.y == pytest.approx(0.17)
    assert pose.position.z == pytest.approx(0.80)
    assert pose.orientation.w == pytest.approx(1.0)


def test_fixed_orientation_is_normalized_before_use():
    template = MarkerlessGraspTemplate(
        orientation_xyzw=(0.0, 0.0, 0.0, 2.0),
    )

    pose = build_markerless_grasp_pose((0.0, 0.0, 0.0), template)

    assert (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_project_config_uses_fixed_simulation_default():
    templates = load_markerless_grasp_templates(CONFIG_PATH)
    template = grasp_template_for_class(templates, 'cup')

    assert template.position_offset_m == pytest.approx((0.0, 0.0, 0.0))
    assert math.sqrt(
        sum(value * value for value in template.orientation_xyzw)
    ) == pytest.approx(1.0)


def test_class_template_overrides_default():
    default = MarkerlessGraspTemplate()
    cup = MarkerlessGraspTemplate(position_offset_m=(0.0, 0.0, 0.1))
    templates = {'default': default, 'classes': {'cup': cup}}

    assert grasp_template_for_class(templates, 'cup') is cup
    assert grasp_template_for_class(templates, 'bottle') is default


@pytest.mark.parametrize(
    'bad_position',
    (
        (0.1, 0.2),
        (0.1, 0.2, math.nan),
    ),
)
def test_invalid_planning_position_is_rejected(bad_position):
    with pytest.raises(ValueError):
        build_markerless_grasp_pose(
            bad_position,
            MarkerlessGraspTemplate(),
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('position_offset_m', (0.0, 0.0)),
        ('position_offset_m', (0.0, math.inf, 0.0)),
        ('orientation_xyzw', (0.0, 0.0, 0.0, 0.0)),
        ('orientation_xyzw', (0.0, 0.0, math.nan, 1.0)),
    ),
)
def test_invalid_template_is_rejected(field, value):
    with pytest.raises(ValueError):
        MarkerlessGraspTemplate(**{field: value})


def test_builder_requires_validated_template():
    with pytest.raises(TypeError):
        build_markerless_grasp_pose(
            (0.0, 0.0, 0.0),
            {'position_offset_m': (0.0, 0.0, 0.0)},
        )
