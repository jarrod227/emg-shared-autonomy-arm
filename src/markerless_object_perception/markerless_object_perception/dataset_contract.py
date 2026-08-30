"""Validated, leakage-safe dataset contract for Objective 3.2.

Nothing here is on the path that ships, and none of it ever ran against
real data.
Objective 3.2 uses official COCO-pretrained instance-segmentation weights for
`bottle`, `cup` and `apple`; the four-class collection, polygon annotation,
frozen-bundle and fine-tuning plan this module belongs to was abandoned in
favour of them, and reopening it is a deliberate scope decision rather than a
next step. Kept as historical implementation: the logic is unit-tested with
fakes, but no training run, no dataset bundle and no weights were ever
produced from it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Sequence


CLASS_NAMES = ('bottle', 'cup', 'cell_phone', 'medicine_box')
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ('train', 'val', 'test')
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 3201
NEGATIVE_CATEGORY = '__negative__'
DEFAULT_MANIFEST = 'source_manifest.jsonl'
DEFAULT_MIN_POLYGON_AREA = 1.0e-6
IMAGE_SUFFIXES = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}

_PORTABLE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


class DatasetContractError(ValueError):
    """Report one or more violations of the frozen dataset contract."""


@dataclass(frozen=True)
class PolygonAnnotation:
    """One normalized YOLO instance-segmentation polygon."""

    class_id: int
    points: tuple[tuple[float, float], ...]
    area: float


@dataclass(frozen=True)
class DatasetSample:
    """One fully validated source sample."""

    sample_id: str
    image: str
    label: str
    session_id: str
    instances: tuple[tuple[str, str], ...]
    image_sha256: str
    label_sha256: str
    annotations: tuple[PolygonAnnotation, ...]

    @property
    def is_negative(self) -> bool:
        """Return whether the sample intentionally has no instances."""
        return not self.annotations


@dataclass(frozen=True)
class ValidatedDataset:
    """A source root and its validated, ordered samples."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    samples: tuple[DatasetSample, ...]


@dataclass(frozen=True)
class LeakageGroup:
    """Samples that must remain together in one dataset split."""

    group_id: str
    sample_ids: tuple[str, ...]
    categories: frozenset[str]


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_yolo_polygon_file(
    path: Path,
    *,
    min_area: float = DEFAULT_MIN_POLYGON_AREA,
) -> tuple[PolygonAnnotation, ...]:
    """Parse and validate one YOLO segmentation label file."""
    if not math.isfinite(min_area) or min_area <= 0.0:
        raise DatasetContractError('minimum polygon area must be positive')

    annotations: list[PolygonAnnotation] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetContractError(f'cannot read label {path}: {error}') from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise DatasetContractError(
                f'{path}:{line_number}: expected class plus at least '
                'three x/y polygon points'
            )
        try:
            class_id = int(tokens[0])
        except ValueError as error:
            raise DatasetContractError(
                f'{path}:{line_number}: class id must be an integer'
            ) from error
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise DatasetContractError(
                f'{path}:{line_number}: class id {class_id} is outside 0..3'
            )

        try:
            coordinates = tuple(float(value) for value in tokens[1:])
        except ValueError as error:
            raise DatasetContractError(
                f'{path}:{line_number}: polygon coordinates must be numbers'
            ) from error
        if any(not math.isfinite(value) for value in coordinates):
            raise DatasetContractError(
                f'{path}:{line_number}: polygon coordinates must be finite'
            )
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise DatasetContractError(
                f'{path}:{line_number}: normalized coordinates must be in [0, 1]'
            )

        points = tuple(zip(coordinates[::2], coordinates[1::2]))
        if len(set(points)) < 3:
            raise DatasetContractError(
                f'{path}:{line_number}: polygon needs three distinct points'
            )
        twice_area = sum(
            x_value * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * y_value
            for index, (x_value, y_value) in enumerate(points)
        )
        area = abs(twice_area) * 0.5
        if area < min_area:
            raise DatasetContractError(
                f'{path}:{line_number}: polygon area {area:.8g} is below '
                f'{min_area:.8g}'
            )
        annotations.append(PolygonAnnotation(class_id, points, area))
    return tuple(annotations)


def validate_source_dataset(
    source_root: Path | str,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    min_polygon_area: float = DEFAULT_MIN_POLYGON_AREA,
) -> ValidatedDataset:
    """Validate a source ``images/`` + ``labels/`` + JSONL dataset."""
    root = Path(source_root).resolve()
    images_root = root / 'images'
    labels_root = root / 'labels'
    manifest_relative = _portable_path(manifest_name, 'manifest')
    manifest_path = root / manifest_relative
    errors: list[str] = []

    if not images_root.is_dir():
        errors.append(f'missing images directory: {images_root}')
    if not labels_root.is_dir():
        errors.append(f'missing labels directory: {labels_root}')
    if not manifest_path.is_file() or manifest_path.is_symlink():
        errors.append(f'missing regular JSONL manifest: {manifest_path}')
    if errors:
        raise DatasetContractError(_format_errors(errors))

    records = _read_manifest(manifest_path, errors)
    samples: list[DatasetSample] = []
    sample_ids: set[str] = set()
    image_names: set[str] = set()
    label_names: set[str] = set()
    image_hashes: dict[str, str] = {}
    physical_classes: dict[str, str] = {}

    for line_number, record in records:
        sample = _validate_record(
            record,
            line_number=line_number,
            images_root=images_root,
            labels_root=labels_root,
            min_polygon_area=min_polygon_area,
            errors=errors,
        )
        if sample is None:
            continue
        if sample.sample_id in sample_ids:
            errors.append(f'duplicate sample_id: {sample.sample_id}')
        sample_ids.add(sample.sample_id)
        if sample.image in image_names:
            errors.append(f'duplicate image path: {sample.image}')
        image_names.add(sample.image)
        if sample.label in label_names:
            errors.append(f'duplicate derived label path: {sample.label}')
        label_names.add(sample.label)
        previous_hash = image_hashes.get(sample.image_sha256)
        if previous_hash is not None:
            errors.append(
                'duplicate image content: '
                f'{sample.sample_id} matches {previous_hash}'
            )
        image_hashes[sample.image_sha256] = sample.sample_id
        for instance_id, class_name in sample.instances:
            previous_class = physical_classes.setdefault(instance_id, class_name)
            if previous_class != class_name:
                errors.append(
                    f'physical instance {instance_id!r} changes class from '
                    f'{previous_class!r} to {class_name!r}'
                )
        samples.append(sample)

    actual_images = _relative_files(images_root, IMAGE_SUFFIXES)
    actual_labels = _relative_files(labels_root, {'.txt'})
    for orphan in sorted(actual_images - image_names):
        errors.append(f'orphan image not listed in manifest: {orphan}')
    for orphan in sorted(actual_labels - label_names):
        errors.append(f'orphan label without a manifest image: {orphan}')

    if not samples:
        errors.append('manifest contains no valid samples')
    if errors:
        raise DatasetContractError(_format_errors(errors))

    return ValidatedDataset(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
    )


def build_leakage_groups(
    samples: Sequence[DatasetSample],
) -> tuple[LeakageGroup, ...]:
    """Union samples sharing a session or physical object instance."""
    if not samples:
        return ()
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    by_session: dict[str, int] = {}
    by_instance: dict[str, int] = {}
    for index, sample in enumerate(samples):
        previous = by_session.setdefault(sample.session_id, index)
        union(index, previous)
        for instance_id, _class_name in sample.instances:
            previous = by_instance.setdefault(instance_id, index)
            union(index, previous)

    members: dict[int, list[DatasetSample]] = {}
    for index, sample in enumerate(samples):
        members.setdefault(find(index), []).append(sample)

    groups: list[LeakageGroup] = []
    for grouped_samples in members.values():
        sample_ids = tuple(sorted(item.sample_id for item in grouped_samples))
        categories = {
            class_name
            for item in grouped_samples
            for _instance_id, class_name in item.instances
        }
        if any(item.is_negative for item in grouped_samples):
            categories.add(NEGATIVE_CATEGORY)
        group_hash = hashlib.sha256('\n'.join(sample_ids).encode()).hexdigest()
        groups.append(
            LeakageGroup(
                group_id=f'group-{group_hash[:16]}',
                sample_ids=sample_ids,
                categories=frozenset(categories),
            )
        )
    return tuple(sorted(groups, key=lambda item: item.group_id))


def deterministic_group_split(
    samples: Sequence[DatasetSample],
    *,
    seed: int = SPLIT_SEED,
    ratios: Sequence[float] = SPLIT_RATIOS,
) -> dict[str, str]:
    """Assign whole leakage groups to reproducible train/val/test splits."""
    _validate_frozen_split(seed, ratios)
    groups = build_leakage_groups(samples)
    categories = (*CLASS_NAMES, NEGATIVE_CATEGORY)
    if not groups:
        raise DatasetContractError('cannot split an empty dataset')
    for category in categories:
        support = sum(category in group.categories for group in groups)
        if support < len(SPLIT_NAMES):
            raise DatasetContractError(
                f'category {category!r} appears in only {support} leakage '
                'groups; at least 3 are required'
            )

    category_bits = {
        category: 1 << index for index, category in enumerate(categories)
    }
    full_mask = (1 << len(categories)) - 1
    masks = [
        sum(category_bits[item] for item in group.categories)
        for group in groups
    ]
    sizes = [len(group.sample_ids) for group in groups]
    targets = [len(samples) * value for value in ratios]
    assignments = [-1] * len(groups)
    coverage = [0] * len(SPLIT_NAMES)
    counts = [0] * len(SPLIT_NAMES)
    nodes_visited = 0

    def feasible() -> bool:
        for bit in category_bits.values():
            missing = sum(not (coverage[index] & bit) for index in range(3))
            available = sum(
                assignments[index] < 0 and bool(masks[index] & bit)
                for index in range(len(groups))
            )
            if available < missing:
                return False
        return True

    def choose_need() -> tuple[int, int, list[int]]:
        needs: list[tuple[int, float, int, int, list[int]]] = []
        for split_index in range(3):
            for bit in category_bits.values():
                if coverage[split_index] & bit:
                    continue
                candidates = [
                    index
                    for index in range(len(groups))
                    if assignments[index] < 0 and masks[index] & bit
                ]
                needs.append(
                    (
                        len(candidates),
                        ratios[split_index],
                        split_index,
                        bit,
                        candidates,
                    )
                )
        _candidate_count, _ratio, split_index, bit, candidates = min(needs)
        return split_index, bit, candidates

    def cover() -> bool:
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > 250000:
            return False
        if all(value == full_mask for value in coverage):
            return True
        if not feasible():
            return False
        split_index, _bit, candidates = choose_need()
        missing_mask = full_mask ^ coverage[split_index]
        candidates.sort(
            key=lambda index: (
                -(masks[index] & missing_mask).bit_count(),
                max(0.0, counts[split_index] + sizes[index]
                    - targets[split_index]),
                abs(counts[split_index] + sizes[index]
                    - targets[split_index]),
                _stable_rank(seed, split_index, groups[index].group_id),
            )
        )
        for group_index in candidates:
            old_coverage = coverage[split_index]
            assignments[group_index] = split_index
            coverage[split_index] |= masks[group_index]
            counts[split_index] += sizes[group_index]
            if cover():
                return True
            counts[split_index] -= sizes[group_index]
            coverage[split_index] = old_coverage
            assignments[group_index] = -1
        return False

    if not cover():
        raise DatasetContractError(
            'cannot find disjoint leakage groups that cover all classes and '
            'negative samples in every split'
        )

    remaining = [
        index for index, split_index in enumerate(assignments)
        if split_index < 0
    ]
    remaining.sort(
        key=lambda index: (
            -sizes[index],
            _stable_rank(seed, 99, groups[index].group_id),
        )
    )
    for group_index in remaining:
        scored: list[tuple[float, int, int]] = []
        for split_index in range(3):
            projected = counts.copy()
            projected[split_index] += sizes[group_index]
            score = sum(
                ((projected[index] - targets[index])
                 / max(targets[index], 1.0)) ** 2
                for index in range(3)
            )
            scored.append(
                (
                    score,
                    _stable_rank(seed, split_index, groups[group_index].group_id),
                    split_index,
                )
            )
        _score, _rank, chosen = min(scored)
        assignments[group_index] = chosen
        counts[chosen] += sizes[group_index]

    result: dict[str, str] = {}
    for group, split_index in zip(groups, assignments):
        split_name = SPLIT_NAMES[split_index]
        for sample_id in group.sample_ids:
            result[sample_id] = split_name
    validate_split_coverage(samples, result)
    return dict(sorted(result.items()))


def validate_split_coverage(
    samples: Sequence[DatasetSample],
    assignments: dict[str, str],
) -> None:
    """Require every split to contain all four classes and a negative image."""
    expected_ids = {sample.sample_id for sample in samples}
    if set(assignments) != expected_ids:
        raise DatasetContractError('split assignments do not match sample ids')
    coverage = {name: set() for name in SPLIT_NAMES}
    for sample in samples:
        split_name = assignments[sample.sample_id]
        if split_name not in coverage:
            raise DatasetContractError(f'unknown split name: {split_name!r}')
        coverage[split_name].update(
            class_name for _instance_id, class_name in sample.instances
        )
        if sample.is_negative:
            coverage[split_name].add(NEGATIVE_CATEGORY)
    expected = {*CLASS_NAMES, NEGATIVE_CATEGORY}
    for split_name, present in coverage.items():
        missing = sorted(expected - present)
        if missing:
            raise DatasetContractError(
                f'{split_name} split is missing categories: {missing}'
            )


def _validate_record(
    record: dict[str, Any],
    *,
    line_number: int,
    images_root: Path,
    labels_root: Path,
    min_polygon_area: float,
    errors: list[str],
) -> DatasetSample | None:
    context = f'manifest line {line_number}'
    required = {'sample_id', 'image', 'session_id', 'instances'}
    missing = required - set(record)
    extra = set(record) - required
    if missing:
        errors.append(f'{context}: missing fields {sorted(missing)}')
    if extra:
        errors.append(f'{context}: unknown fields {sorted(extra)}')
    if missing or extra:
        return None

    sample_id = record['sample_id']
    session_id = record['session_id']
    if not isinstance(sample_id, str) or not _PORTABLE_ID.fullmatch(sample_id):
        errors.append(f'{context}: sample_id must be a portable filename token')
        return None
    if (
        not isinstance(session_id, str)
        or not session_id
        or session_id != session_id.strip()
    ):
        errors.append(
            f'{context}: session_id must be non-empty without edge whitespace'
        )
        return None
    try:
        image = _portable_path(record['image'], f'{context} image')
    except DatasetContractError as error:
        errors.append(str(error))
        return None
    if image.suffix.lower() not in IMAGE_SUFFIXES:
        errors.append(f'{context}: unsupported image suffix {image.suffix!r}')
        return None
    label = image.with_suffix('.txt')

    instances = _validate_instances(record['instances'], context, errors)
    if instances is None:
        return None

    image_path = images_root.joinpath(*image.parts)
    label_path = labels_root.joinpath(*label.parts)
    if not image_path.is_file():
        errors.append(f'{context}: image does not exist: {image.as_posix()}')
        return None
    if not label_path.is_file():
        errors.append(
            f'{context}: same-name label does not exist: {label.as_posix()}'
        )
        return None
    if image_path.is_symlink() or label_path.is_symlink():
        errors.append(f'{context}: source image and label may not be symlinks')
        return None

    actual_hash = sha256_file(image_path)
    try:
        _validate_decodable_image(image_path)
        annotations = parse_yolo_polygon_file(
            label_path,
            min_area=min_polygon_area,
        )
    except DatasetContractError as error:
        errors.append(str(error))
        return None

    label_counts = Counter(
        CLASS_NAMES[annotation.class_id] for annotation in annotations
    )
    instance_counts = Counter(class_name for _key, class_name in instances)
    if label_counts != instance_counts:
        errors.append(
            f'{context}: instances counts {dict(instance_counts)} do not '
            f'match label counts {dict(label_counts)}'
        )
    return DatasetSample(
        sample_id=sample_id,
        image=image.as_posix(),
        label=label.as_posix(),
        session_id=session_id,
        instances=instances,
        image_sha256=actual_hash,
        label_sha256=sha256_file(label_path),
        annotations=annotations,
    )


def _validate_instances(
    value: Any,
    context: str,
    errors: list[str],
) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(value, dict):
        errors.append(f'{context}: instances must be an object mapping id to class')
        return None
    result: list[tuple[str, str]] = []
    for instance_id, class_name in value.items():
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id != instance_id.strip()
        ):
            errors.append(
                f'{context}: physical instance ids must be non-empty '
                'without edge whitespace'
            )
            return None
        if not isinstance(class_name, str) or class_name not in CLASS_TO_ID:
            errors.append(
                f'{context}: instance {instance_id!r} has unknown class '
                f'{class_name!r}'
            )
            return None
        result.append((instance_id, class_name))
    return tuple(sorted(result))


def _validate_decodable_image(path: Path) -> None:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise DatasetContractError(
            'OpenCV is required to validate source images'
        ) from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise DatasetContractError(f'image cannot be decoded: {path}')


def _read_manifest(
    path: Path,
    errors: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetContractError(f'cannot read manifest {path}: {error}') from error
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, DatasetContractError) as error:
            errors.append(f'manifest line {line_number}: invalid JSON: {error}')
            continue
        if not isinstance(record, dict):
            errors.append(f'manifest line {line_number}: record must be an object')
            continue
        records.append((line_number, record))
    return records


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetContractError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def _portable_path(value: Any, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\\' in value:
        raise DatasetContractError(f'{description} must be a POSIX relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value.startswith('./'):
        raise DatasetContractError(f'{description} must stay inside its root')
    if path.as_posix() in {'.', ''}:
        raise DatasetContractError(f'{description} may not be empty')
    return path


def _relative_files(root: Path, suffixes: set[str]) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _stable_rank(seed: int, split_index: int, value: str) -> int:
    payload = f'{seed}:{split_index}:{value}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def _validate_frozen_split(seed: int, ratios: Sequence[float]) -> None:
    if seed != SPLIT_SEED:
        raise DatasetContractError(f'split seed is frozen at {SPLIT_SEED}')
    if len(ratios) != 3 or any(
        not math.isclose(value, expected, abs_tol=1.0e-12)
        for value, expected in zip(ratios, SPLIT_RATIOS)
    ):
        raise DatasetContractError('split ratios are frozen at 0.70/0.15/0.15')


def _format_errors(errors: Sequence[str]) -> str:
    return 'dataset contract violations:\n- ' + '\n- '.join(errors)
