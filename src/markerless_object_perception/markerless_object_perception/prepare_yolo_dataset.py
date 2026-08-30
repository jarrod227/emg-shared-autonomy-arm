"""Prepare and verify a portable Objective 3.2 YOLO dataset bundle.

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

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any, Sequence

from markerless_object_perception.dataset_contract import (
    build_leakage_groups,
    CLASS_NAMES,
    CLASS_TO_ID,
    DatasetContractError,
    DatasetSample,
    DEFAULT_MANIFEST,
    DEFAULT_MIN_POLYGON_AREA,
    deterministic_group_split,
    parse_yolo_polygon_file,
    sha256_file,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    SPLIT_SEED,
    validate_source_dataset,
    validate_split_coverage,
)
import tomllib


BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetPreparationConfig:
    """Frozen configuration used to create a reproducible dataset bundle."""

    manifest: str = DEFAULT_MANIFEST
    seed: int = SPLIT_SEED
    ratios: tuple[float, float, float] = SPLIT_RATIOS
    min_polygon_area: float = DEFAULT_MIN_POLYGON_AREA


@dataclass(frozen=True)
class BundleSummary:
    """Summary returned after a successful bundle creation or verification."""

    root: Path
    sample_count: int
    split_counts: tuple[tuple[str, int], ...]
    manifest_sha256: str


def load_dataset_config(path: Path | str) -> DatasetPreparationConfig:
    """Load the checked-in TOML file and enforce the frozen class/split values."""
    config_path = Path(path)
    try:
        with config_path.open('rb') as stream:
            content = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DatasetContractError(
            f'cannot load dataset config {config_path}: {error}'
        ) from error

    expected_top = {
        'schema_version',
        'classes',
        'manifest',
        'minimum_polygon_area',
        'split',
    }
    if set(content) != expected_top:
        raise DatasetContractError(
            'dataset config fields must be exactly '
            f'{sorted(expected_top)}'
        )
    if content['schema_version'] != BUNDLE_SCHEMA_VERSION:
        raise DatasetContractError('unsupported dataset config schema version')
    if content['classes'] != list(CLASS_NAMES):
        raise DatasetContractError(
            f'classes are frozen as {list(CLASS_NAMES)}'
        )
    manifest = content['manifest']
    if not isinstance(manifest, str) or not manifest:
        raise DatasetContractError('manifest must be a non-empty relative path')
    manifest_path = PurePosixPath(manifest)
    if manifest_path.is_absolute() or '..' in manifest_path.parts:
        raise DatasetContractError('manifest must stay inside the source root')

    split = content['split']
    if not isinstance(split, dict) or set(split) != {
        'seed', 'train', 'val', 'test'
    }:
        raise DatasetContractError(
            'split config requires seed, train, val, and test only'
        )
    if split['seed'] != SPLIT_SEED:
        raise DatasetContractError(f'split seed is frozen at {SPLIT_SEED}')
    ratios = tuple(split[name] for name in SPLIT_NAMES)
    if any(
        not isinstance(value, (int, float))
        or not math.isclose(float(value), expected, abs_tol=1.0e-12)
        for value, expected in zip(ratios, SPLIT_RATIOS)
    ):
        raise DatasetContractError('split ratios are frozen at 0.70/0.15/0.15')
    min_area = content['minimum_polygon_area']
    if not isinstance(min_area, (int, float)) or not math.isfinite(min_area):
        raise DatasetContractError('minimum_polygon_area must be finite')
    if min_area <= 0.0:
        raise DatasetContractError('minimum_polygon_area must be positive')
    return DatasetPreparationConfig(
        manifest=manifest,
        seed=SPLIT_SEED,
        ratios=SPLIT_RATIOS,
        min_polygon_area=float(min_area),
    )


def prepare_yolo_dataset(
    source_root: Path | str,
    output_root: Path | str,
    *,
    config: DatasetPreparationConfig | None = None,
) -> BundleSummary:
    """Create a new portable bundle and refuse any existing output path."""
    effective = config or DatasetPreparationConfig()
    source = validate_source_dataset(
        source_root,
        manifest_name=effective.manifest,
        min_polygon_area=effective.min_polygon_area,
    )
    assignments = deterministic_group_split(
        source.samples,
        seed=effective.seed,
        ratios=effective.ratios,
    )
    groups = build_leakage_groups(source.samples)
    group_by_sample = {
        sample_id: group.group_id
        for group in groups
        for sample_id in group.sample_ids
    }

    requested_output = Path(output_root)
    if requested_output.is_symlink():
        raise FileExistsError(
            f'output is a symlink; refusing overwrite: {requested_output}'
        )
    output = requested_output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output already exists; refusing overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        for kind in ('images', 'labels'):
            for split_name in SPLIT_NAMES:
                (output / kind / split_name).mkdir(parents=True)

        bundle_records: list[dict[str, Any]] = []
        for sample in source.samples:
            split_name = assignments[sample.sample_id]
            image_suffix = PurePosixPath(sample.image).suffix.lower()
            image_relative = PurePosixPath(
                'images', split_name, sample.sample_id + image_suffix
            )
            label_relative = PurePosixPath(
                'labels', split_name, sample.sample_id + '.txt'
            )
            source_image = source.root / 'images' / sample.image
            source_label = source.root / 'labels' / sample.label
            destination_image = output.joinpath(*image_relative.parts)
            destination_label = output.joinpath(*label_relative.parts)
            shutil.copyfile(source_image, destination_image)
            shutil.copyfile(source_label, destination_label)
            bundle_records.append(
                {
                    'sample_id': sample.sample_id,
                    'split': split_name,
                    'session_id': sample.session_id,
                    'instances': dict(sample.instances),
                    'leakage_group': group_by_sample[sample.sample_id],
                    'source_image': sample.image,
                    'image': image_relative.as_posix(),
                    'label': label_relative.as_posix(),
                    'image_sha256': sample.image_sha256,
                    'label_sha256': sample.label_sha256,
                }
            )

        (output / 'data.yaml').write_text(
            _data_yaml(),
            encoding='utf-8',
        )
        manifest = {
            'schema_version': BUNDLE_SCHEMA_VERSION,
            'classes': [
                {'id': class_id, 'name': class_name}
                for class_name, class_id in CLASS_TO_ID.items()
            ],
            'split': {
                'seed': effective.seed,
                'ratios': {
                    name: effective.ratios[index]
                    for index, name in enumerate(SPLIT_NAMES)
                },
            },
            'minimum_polygon_area': effective.min_polygon_area,
            'source_manifest_sha256': source.manifest_sha256,
            'samples': sorted(
                bundle_records,
                key=lambda item: item['sample_id'],
            ),
        }
        manifest_bytes = _canonical_json(manifest)
        (output / 'manifest.json').write_bytes(manifest_bytes)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        (output / 'manifest.sha256').write_text(
            f'{digest}  manifest.json\n',
            encoding='ascii',
        )
        return verify_bundle(output)
    except BaseException:
        shutil.rmtree(output)
        raise


def verify_bundle(
    bundle_root: Path | str,
) -> BundleSummary:
    """Verify manifest integrity, file hashes, labels, coverage, and leakage."""
    root = Path(bundle_root).resolve()
    manifest_path = root / 'manifest.json'
    checksum_path = root / 'manifest.sha256'
    data_path = root / 'data.yaml'
    for required in (manifest_path, checksum_path, data_path):
        if not required.is_file() or required.is_symlink():
            raise DatasetContractError(f'missing regular bundle file: {required}')

    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_tokens = checksum_path.read_text(encoding='ascii').split()
    if checksum_tokens != [manifest_digest, 'manifest.json']:
        raise DatasetContractError('manifest.sha256 does not match manifest.json')
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetContractError(f'invalid bundle manifest: {error}') from error
    if _canonical_json(manifest) != manifest_bytes:
        raise DatasetContractError('manifest.json is not canonical')
    min_polygon_area = _validate_bundle_header(manifest)
    if data_path.read_text(encoding='utf-8') != _data_yaml():
        raise DatasetContractError('data.yaml differs from the portable contract')

    records = manifest.get('samples')
    if not isinstance(records, list) or not records:
        raise DatasetContractError('bundle manifest must contain samples')
    samples: list[DatasetSample] = []
    assignments: dict[str, str] = {}
    expected_images: set[str] = set()
    expected_labels: set[str] = set()
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    group_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    instance_splits: dict[str, str] = {}
    instance_classes: dict[str, str] = {}

    for record in records:
        sample = _verify_bundle_record(
            root,
            record,
            min_polygon_area=min_polygon_area,
        )
        if sample.sample_id in sample_ids:
            raise DatasetContractError(
                f'duplicate bundle sample_id: {sample.sample_id}'
            )
        sample_ids.add(sample.sample_id)
        if sample.image_sha256 in image_hashes:
            raise DatasetContractError('duplicate image content in bundle')
        image_hashes.add(sample.image_sha256)
        split_name = record['split']
        assignments[sample.sample_id] = split_name
        expected_images.add(sample.image)
        expected_labels.add(sample.label)
        _require_one_split(
            group_splits,
            record['leakage_group'],
            split_name,
            'leakage group',
        )
        _require_one_split(
            session_splits,
            sample.session_id,
            split_name,
            'session',
        )
        for instance_id, _class_name in sample.instances:
            previous_class = instance_classes.setdefault(
                instance_id, _class_name
            )
            if previous_class != _class_name:
                raise DatasetContractError(
                    f'physical instance {instance_id!r} changes class'
                )
            _require_one_split(
                instance_splits,
                instance_id,
                split_name,
                'physical instance',
            )
        samples.append(sample)

    expected_groups = {
        sample_id: group.group_id
        for group in build_leakage_groups(samples)
        for sample_id in group.sample_ids
    }
    for record in records:
        sample_id = record['sample_id']
        if record['leakage_group'] != expected_groups[sample_id]:
            raise DatasetContractError(
                f'leakage group differs from deterministic value for {sample_id}'
            )
    expected_assignments = deterministic_group_split(samples)
    if assignments != expected_assignments:
        raise DatasetContractError(
            'bundle splits differ from the deterministic seed-3201 assignment'
        )

    actual_images = {
        path.relative_to(root).as_posix()
        for path in (root / 'images').rglob('*')
        if path.is_file()
    }
    actual_labels = {
        path.relative_to(root).as_posix()
        for path in (root / 'labels').rglob('*')
        if path.is_file()
    }
    if actual_images != expected_images:
        raise DatasetContractError('bundle contains missing or orphan image files')
    if actual_labels != expected_labels:
        raise DatasetContractError('bundle contains missing or orphan label files')
    validate_split_coverage(samples, assignments)
    counts = Counter(assignments.values())
    return BundleSummary(
        root=root,
        sample_count=len(samples),
        split_counts=tuple((name, counts[name]) for name in SPLIT_NAMES),
        manifest_sha256=manifest_digest,
    )


def _verify_bundle_record(
    root: Path,
    record: Any,
    *,
    min_polygon_area: float,
) -> DatasetSample:
    required = {
        'sample_id',
        'split',
        'session_id',
        'instances',
        'leakage_group',
        'source_image',
        'image',
        'label',
        'image_sha256',
        'label_sha256',
    }
    if not isinstance(record, dict) or set(record) != required:
        raise DatasetContractError('bundle sample has invalid fields')
    sample_id = record['sample_id']
    split_name = record['split']
    if not isinstance(sample_id, str) or not sample_id:
        raise DatasetContractError('bundle sample_id must be non-empty')
    if split_name not in SPLIT_NAMES:
        raise DatasetContractError(f'invalid bundle split: {split_name!r}')
    image = _bundle_path(record['image'])
    label = _bundle_path(record['label'])
    expected_image_prefix = PurePosixPath('images', split_name)
    expected_label_prefix = PurePosixPath('labels', split_name)
    if image.parent != expected_image_prefix:
        raise DatasetContractError('bundle image is outside its split directory')
    if label.parent != expected_label_prefix:
        raise DatasetContractError('bundle label is outside its split directory')
    if image.stem != sample_id or label.stem != sample_id:
        raise DatasetContractError('bundle filenames must use sample_id')
    image_path = root.joinpath(*image.parts)
    label_path = root.joinpath(*label.parts)
    for path in (image_path, label_path):
        if not path.is_file() or path.is_symlink():
            raise DatasetContractError(f'missing regular sample file: {path}')
    for field, path in (
        ('image_sha256', image_path),
        ('label_sha256', label_path),
    ):
        expected = record[field]
        if not isinstance(expected, str) or len(expected) != 64:
            raise DatasetContractError(f'invalid {field} for {sample_id}')
        if sha256_file(path) != expected:
            raise DatasetContractError(f'{field} mismatch for {sample_id}')

    instances_value = record['instances']
    if not isinstance(instances_value, dict):
        raise DatasetContractError(f'invalid instances for {sample_id}')
    instances: list[tuple[str, str]] = []
    for instance_id, class_name in instances_value.items():
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id != instance_id.strip()
        ):
            raise DatasetContractError(f'invalid physical instance for {sample_id}')
        if class_name not in CLASS_TO_ID:
            raise DatasetContractError(f'invalid instance class for {sample_id}')
        instances.append((instance_id, class_name))
    annotations = parse_yolo_polygon_file(
        label_path,
        min_area=min_polygon_area,
    )
    label_counts = Counter(
        CLASS_NAMES[annotation.class_id] for annotation in annotations
    )
    instance_counts = Counter(class_name for _key, class_name in instances)
    if label_counts != instance_counts:
        raise DatasetContractError(f'instance count mismatch for {sample_id}')
    session_id = record['session_id']
    if (
        not isinstance(session_id, str)
        or not session_id
        or session_id != session_id.strip()
    ):
        raise DatasetContractError(f'invalid session_id for {sample_id}')
    if not isinstance(record['leakage_group'], str) or not record[
        'leakage_group'
    ]:
        raise DatasetContractError(f'invalid leakage_group for {sample_id}')
    _bundle_path(record['source_image'])
    return DatasetSample(
        sample_id=sample_id,
        image=image.as_posix(),
        label=label.as_posix(),
        session_id=session_id,
        instances=tuple(sorted(instances)),
        image_sha256=record['image_sha256'],
        label_sha256=record['label_sha256'],
        annotations=annotations,
    )


def _validate_bundle_header(manifest: Any) -> float:
    if not isinstance(manifest, dict):
        raise DatasetContractError('bundle manifest must be an object')
    expected_fields = {
        'schema_version',
        'classes',
        'split',
        'minimum_polygon_area',
        'source_manifest_sha256',
        'samples',
    }
    if set(manifest) != expected_fields:
        raise DatasetContractError('bundle manifest fields differ from schema')
    if manifest.get('schema_version') != BUNDLE_SCHEMA_VERSION:
        raise DatasetContractError('unsupported bundle schema version')
    expected_classes = [
        {'id': index, 'name': name}
        for index, name in enumerate(CLASS_NAMES)
    ]
    if manifest.get('classes') != expected_classes:
        raise DatasetContractError('bundle classes differ from frozen classes')
    expected_split = {
        'seed': SPLIT_SEED,
        'ratios': {
            name: SPLIT_RATIOS[index]
            for index, name in enumerate(SPLIT_NAMES)
        },
    }
    if manifest.get('split') != expected_split:
        raise DatasetContractError('bundle split contract is not frozen')
    min_polygon_area = manifest.get('minimum_polygon_area')
    if (
        not isinstance(min_polygon_area, (int, float))
        or isinstance(min_polygon_area, bool)
        or not math.isfinite(min_polygon_area)
        or min_polygon_area <= 0.0
    ):
        raise DatasetContractError('invalid minimum polygon area')
    source_hash = manifest.get('source_manifest_sha256')
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in '0123456789abcdef' for character in source_hash)
    ):
        raise DatasetContractError('invalid source manifest SHA-256')
    return float(min_polygon_area)


def _require_one_split(
    owners: dict[str, str],
    key: str,
    split_name: str,
    description: str,
) -> None:
    previous = owners.setdefault(key, split_name)
    if previous != split_name:
        raise DatasetContractError(
            f'{description} {key!r} leaks across {previous} and {split_name}'
        )


def _bundle_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\\' in value:
        raise DatasetContractError('bundle paths must be POSIX relative paths')
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or '..' in path.parts
        or path.as_posix() in {'.', ''}
    ):
        raise DatasetContractError('bundle path escapes its root')
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        + '\n'
    ).encode('utf-8')


def _data_yaml() -> str:
    lines = [
        'train: images/train',
        'val: images/val',
        'test: images/test',
        '',
        f'nc: {len(CLASS_NAMES)}',
        'names:',
    ]
    lines.extend(
        f'  {index}: {name}' for index, name in enumerate(CLASS_NAMES)
    )
    return '\n'.join(lines) + '\n'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prepare or verify an Objective 3.2 YOLO dataset bundle.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='create a new bundle')
    prepare.add_argument('source_root', type=Path)
    prepare.add_argument('output_root', type=Path)
    prepare.add_argument('--config', type=Path)
    verify = subparsers.add_parser('verify', help='verify an existing bundle')
    verify.add_argument('bundle_root', type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the non-overwriting dataset preparation command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            config = (
                load_dataset_config(args.config)
                if args.config is not None
                else DatasetPreparationConfig()
            )
            summary = prepare_yolo_dataset(
                args.source_root,
                args.output_root,
                config=config,
            )
        else:
            summary = verify_bundle(args.bundle_root)
    except (DatasetContractError, FileExistsError, OSError) as error:
        parser.exit(2, f'error: {error}\n')
    print(
        json.dumps(
            {
                'root': str(summary.root),
                'sample_count': summary.sample_count,
                'split_counts': dict(summary.split_counts),
                'manifest_sha256': summary.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
