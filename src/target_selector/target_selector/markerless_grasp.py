"""Build a markerless grasp pose after localization reaches the planning frame."""

from dataclasses import dataclass
import math

from geometry_msgs.msg import Pose
import yaml


def _finite_tuple(value, length, name):
    try:
        converted = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must contain numeric values') from error
    if len(converted) != length:
        raise ValueError(f'{name} must contain exactly {length} values')
    if not all(math.isfinite(component) for component in converted):
        raise ValueError(f'{name} must contain only finite values')
    return converted


@dataclass(frozen=True)
class MarkerlessGraspTemplate:
    """Fixed grasp offset and orientation expressed in the planning frame."""

    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def __post_init__(self):
        offset = _finite_tuple(
            self.position_offset_m,
            3,
            'position_offset_m',
        )
        orientation = _finite_tuple(
            self.orientation_xyzw,
            4,
            'orientation_xyzw',
        )
        norm = math.sqrt(
            sum(component * component for component in orientation)
        )
        if norm <= 1e-12:
            raise ValueError('orientation_xyzw must not be a zero quaternion')

        object.__setattr__(self, 'position_offset_m', offset)
        object.__setattr__(
            self,
            'orientation_xyzw',
            tuple(component / norm for component in orientation),
        )


def _template_from_mapping(mapping, name):
    if not isinstance(mapping, dict):
        raise ValueError(f'{name} must be a mapping')
    try:
        position_offset_m = mapping['position_offset_m']
        orientation_xyzw = mapping['orientation_xyzw']
    except KeyError as error:
        raise ValueError(
            f'{name} requires position_offset_m and orientation_xyzw'
        ) from error
    return MarkerlessGraspTemplate(
        position_offset_m=position_offset_m,
        orientation_xyzw=orientation_xyzw,
    )


def load_markerless_grasp_templates(path):
    """Load a default planning-frame template and optional class overrides."""
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('grasp-template config must be a mapping')

    default = _template_from_mapping(config.get('default'), 'default')
    class_config = config.get('classes') or {}
    if not isinstance(class_config, dict):
        raise ValueError('classes must be a mapping')

    classes = {}
    for class_label, mapping in class_config.items():
        if not isinstance(class_label, str) or not class_label.strip():
            raise ValueError('class labels must be non-empty strings')
        normalized_label = class_label.strip()
        classes[normalized_label] = _template_from_mapping(
            mapping,
            f'class {normalized_label}',
        )
    return {'default': default, 'classes': classes}


def grasp_template_for_class(templates, class_label):
    """Return a class override or the configured fixed default."""
    if not isinstance(class_label, str) or not class_label.strip():
        raise ValueError('class_label must be a non-empty string')
    try:
        default = templates['default']
        classes = templates['classes']
    except (KeyError, TypeError) as error:
        raise ValueError('templates must contain default and classes') from error
    if not isinstance(default, MarkerlessGraspTemplate):
        raise ValueError('default must be a MarkerlessGraspTemplate')
    if not isinstance(classes, dict):
        raise ValueError('classes must be a mapping')
    return classes.get(class_label.strip(), default)


def build_markerless_grasp_pose(planning_position, template):
    """Apply one planning-frame template to a transformed candidate point."""
    if not isinstance(template, MarkerlessGraspTemplate):
        raise TypeError('template must be a MarkerlessGraspTemplate')
    position = _finite_tuple(
        planning_position,
        3,
        'planning_position',
    )

    pose = Pose()
    (
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ) = tuple(
        coordinate + offset
        for coordinate, offset in zip(
            position,
            template.position_offset_m,
        )
    )
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = template.orientation_xyzw
    return pose
