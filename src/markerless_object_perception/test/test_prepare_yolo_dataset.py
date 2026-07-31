"""Focused tests for portable YOLO dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from markerless_object_perception.dataset_contract import (
    CLASS_NAMES,
    DatasetContractError,
)
from markerless_object_perception.prepare_yolo_dataset import (
    prepare_yolo_dataset,
    verify_bundle,
)
from markerless_object_perception.yolo_training import (
    run_training,
    YoloTrainingConfig,
)
import numpy as np
import pytest


def _source(root: Path, *, mismatch: bool = False) -> None:
    records: list[dict[str, object]] = []
    categories: tuple[str | None, ...] = (*CLASS_NAMES, None)
    value = 40
    for category_index, class_name in enumerate(categories):
        for group_index in range(3):
            sample_id = f'c{category_index}_g{group_index}'
            session_id = f'session_{sample_id}'
            relative = Path(session_id) / f'{sample_id}.png'
            image_path = root / 'images' / relative
            label_path = (root / 'labels' / relative).with_suffix('.txt')
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((8, 8, 3), value, dtype=np.uint8)
            value += 1
            assert cv2.imwrite(str(image_path), image)
            instances: dict[str, str] = {}
            label = ''
            if class_name is not None:
                class_id = CLASS_NAMES.index(class_name)
                label = f'{class_id} 0.1 0.1 0.9 0.1 0.5 0.9\n'
                instances[f'object_{sample_id}'] = class_name
            label_path.write_text(label, encoding='utf-8')
            records.append(
                {
                    'sample_id': sample_id,
                    'image': relative.as_posix(),
                    'session_id': session_id,
                    'instances': instances,
                }
            )
    if mismatch:
        records[0]['instances'] = {}
    (root / 'source_manifest.jsonl').write_text(
        ''.join(json.dumps(record) + '\n' for record in records),
        encoding='utf-8',
    )


def test_prepare_verify_portability_tamper_and_no_overwrite(
    tmp_path: Path,
) -> None:
    """A bundle is portable, hashed, verified, and never overwritten."""
    source = tmp_path / 'source'
    output = tmp_path / 'bundle'
    _source(source)

    summary = prepare_yolo_dataset(source, output)

    assert summary.sample_count == 15
    assert verify_bundle(output) == summary
    yaml_text = (output / 'data.yaml').read_text(encoding='utf-8')
    assert 'path:' not in yaml_text
    assert 'train: images/train' in yaml_text
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        prepare_yolo_dataset(source, output)

    weights = tmp_path / 'local_initial.pt'
    weights.write_bytes(b'local only')
    plan = run_training(
        YoloTrainingConfig(
            bundle_dir=output,
            weights=weights,
            project=tmp_path / 'runs',
        )
    )
    assert plan['mode'] == 'plan'
    assert plan['bundle_verification']['sample_count'] == 15
    assert not (tmp_path / 'runs').exists()

    image = next((output / 'images/train').glob('*.png'))
    image.write_bytes(image.read_bytes() + b'tamper')
    with pytest.raises(DatasetContractError, match='image_sha256 mismatch'):
        verify_bundle(output)


@pytest.mark.parametrize('fault', ['orphan', 'mismatch'])
def test_prepare_rejects_invalid_source(tmp_path: Path, fault: str) -> None:
    """Orphan files and manifest-to-label count mismatches fail closed."""
    source = tmp_path / 'source'
    _source(source, mismatch=fault == 'mismatch')
    if fault == 'orphan':
        orphan = source / 'images/orphan.png'
        assert cv2.imwrite(
            str(orphan),
            np.full((8, 8, 3), 250, dtype=np.uint8),
        )

    with pytest.raises(DatasetContractError):
        prepare_yolo_dataset(source, tmp_path / 'bundle')
