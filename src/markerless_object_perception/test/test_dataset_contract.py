"""Focused tests for the Objective 3.2 dataset contract."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from markerless_object_perception.dataset_contract import (
    build_leakage_groups,
    CLASS_NAMES,
    DatasetContractError,
    deterministic_group_split,
    NEGATIVE_CATEGORY,
    parse_yolo_polygon_file,
    validate_source_dataset,
)
import numpy as np
import pytest


def _add_sample(
    root: Path,
    records: list[dict[str, object]],
    sample_id: str,
    session_id: str,
    value: int,
    class_name: str | None,
    instance_id: str | None = None,
) -> None:
    relative = Path(session_id) / f'{sample_id}.png'
    image_path = root / 'images' / relative
    label_path = (root / 'labels' / relative).with_suffix('.txt')
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    instances: dict[str, str] = {}
    label = ''
    if class_name is not None:
        class_id = CLASS_NAMES.index(class_name)
        label = f'{class_id} 0.1 0.1 0.9 0.1 0.5 0.9\n'
        instances[instance_id or f'object_{sample_id}'] = class_name
    label_path.write_text(label, encoding='utf-8')
    records.append(
        {
            'sample_id': sample_id,
            'image': relative.as_posix(),
            'session_id': session_id,
            'instances': instances,
        }
    )


def _write_manifest(root: Path, records: list[dict[str, object]]) -> None:
    (root / 'source_manifest.jsonl').write_text(
        ''.join(json.dumps(record) + '\n' for record in records),
        encoding='utf-8',
    )


def _balanced_source(root: Path) -> None:
    records: list[dict[str, object]] = []
    categories: tuple[str | None, ...] = (*CLASS_NAMES, None)
    value = 1
    for category_index, class_name in enumerate(categories):
        for group_index in range(3):
            sample_id = f'c{category_index}_g{group_index}'
            _add_sample(
                root,
                records,
                sample_id,
                f'session_{sample_id}',
                value,
                class_name,
            )
            value += 1
    _write_manifest(root, records)


@pytest.mark.parametrize(
    'label',
    [
        '0 0.1 0.1 1.1 0.1 0.5 0.9\n',
        '0 0.1 0.1 0.2 0.2 0.3 0.3\n',
        '4 0.1 0.1 0.9 0.1 0.5 0.9\n',
    ],
)
def test_polygon_contract_rejects_invalid_geometry(
    tmp_path: Path,
    label: str,
) -> None:
    """Coordinates, area, and class IDs are validated before training."""
    path = tmp_path / 'sample.txt'
    path.write_text(label, encoding='utf-8')

    with pytest.raises(DatasetContractError):
        parse_yolo_polygon_file(path)


def test_session_and_physical_instance_links_are_transitive(
    tmp_path: Path,
) -> None:
    """Samples connected by either leakage key remain in one union."""
    records: list[dict[str, object]] = []
    _add_sample(tmp_path, records, 'a', 'shared', 20, 'bottle')
    _add_sample(
        tmp_path, records, 'b', 'shared', 21, 'cup', 'shared_cup'
    )
    _add_sample(
        tmp_path, records, 'c', 'other', 22, 'cup', 'shared_cup'
    )
    _add_sample(tmp_path, records, 'd', 'isolated', 23, None)
    _write_manifest(tmp_path, records)

    groups = build_leakage_groups(validate_source_dataset(tmp_path).samples)

    assert sorted(group.sample_ids for group in groups) == [
        ('a', 'b', 'c'),
        ('d',),
    ]


def test_split_is_deterministic_and_covers_classes_and_negatives(
    tmp_path: Path,
) -> None:
    """All three group-level splits receive four classes and negatives."""
    _balanced_source(tmp_path)
    samples = validate_source_dataset(tmp_path).samples

    first = deterministic_group_split(samples)
    second = deterministic_group_split(tuple(reversed(samples)))

    assert first == second
    for split_name in ('train', 'val', 'test'):
        selected = [
            sample for sample in samples
            if first[sample.sample_id] == split_name
        ]
        present = {
            class_name
            for sample in selected
            for _instance_id, class_name in sample.instances
        }
        if any(sample.is_negative for sample in selected):
            present.add(NEGATIVE_CATEGORY)
        assert present == {*CLASS_NAMES, NEGATIVE_CATEGORY}
