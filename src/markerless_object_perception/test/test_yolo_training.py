"""Tests for the explicit Objective 3.2 training boundary."""

from pathlib import Path

from markerless_object_perception.yolo_training import (
    run_training,
    summarize_test_metrics,
    YoloTrainingConfig,
)
import pytest


def _config(tmp_path: Path, **overrides) -> YoloTrainingConfig:
    bundle = tmp_path / 'bundle'
    bundle.mkdir(parents=True)
    (bundle / 'data.yaml').write_text('names: [bottle]\n', encoding='utf-8')
    weights = tmp_path / 'local.pt'
    weights.write_bytes(b'local')
    values = {
        'bundle_dir': bundle,
        'weights': weights,
        'project': tmp_path / 'runs',
    }
    values.update(overrides)
    return YoloTrainingConfig(**values)


def _verify(bundle: Path):
    return {'verified': True, 'bundle': str(bundle)}


@pytest.mark.parametrize(
    'field,value',
    [
        ('epochs', 0),
        ('patience', -1),
        ('batch', 0),
        ('imgsz', 0),
        ('workers', -1),
        ('seed', -1),
        ('device', ' '),
        ('device', 'garbage'),
        ('name', 'nested/run'),
        ('metrics_filename', '../metrics.json'),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, field, value):
    with pytest.raises(ValueError):
        _config(tmp_path, **{field: value})


def test_default_plan_has_no_model_or_output_side_effect(tmp_path):
    config = _config(tmp_path)
    model_calls = []

    result = run_training(
        config,
        bundle_verifier=_verify,
        model_factory=lambda weights: model_calls.append(weights),
    )

    assert result['mode'] == 'plan'
    assert result['train_kwargs']['device'] == '0'
    assert result['train_kwargs']['epochs'] == 100
    assert result['test_kwargs']['split'] == 'test'
    assert model_calls == []
    assert not config.run_dir.exists()


def test_plan_requires_local_weights_and_frozen_bundle(tmp_path):
    config = _config(tmp_path)
    config.weights.unlink()
    with pytest.raises(FileNotFoundError, match='local file'):
        run_training(config, bundle_verifier=_verify)

    config = _config(tmp_path / 'second')
    with pytest.raises(RuntimeError, match='not frozen'):
        run_training(
            config,
            bundle_verifier=lambda unused: (_ for _ in ()).throw(
                RuntimeError('bundle is not frozen')
            ),
        )


def test_existing_exact_run_directory_is_rejected(tmp_path):
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='already exists'):
        run_training(config, bundle_verifier=_verify)


def test_cuda_failure_occurs_before_model_load(tmp_path):
    config = _config(tmp_path)
    model_calls = []

    with pytest.raises(RuntimeError, match='no model was loaded'):
        run_training(
            config,
            execute=True,
            bundle_verifier=_verify,
            cuda_checker=lambda: False,
            model_factory=lambda weights: model_calls.append(weights),
        )

    assert model_calls == []
    assert not config.run_dir.exists()


class _SegmentResults:

    def class_result(self, index):
        return (0.80 + index, 0.70, 0.60, 0.50)


class _Metrics:

    names = {
        0: 'bottle',
        1: 'cup',
        2: 'cell_phone',
        3: 'medicine_box',
    }
    seg = _SegmentResults()
    results_dict = {
        'metrics/precision(M)': 0.8,
        'metrics/mAP50-95(M)': 0.5,
    }


class _Trainer:

    def __init__(self, run_dir):
        self.save_dir = run_dir
        self.best = run_dir / 'weights' / 'best.pt'
        self.last = run_dir / 'weights' / 'last.pt'


class _Model:

    def __init__(self, run_dir):
        self.train_calls = []
        self.val_calls = []
        self.trainer = _Trainer(run_dir)

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return _Metrics()

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return _Metrics()


def test_execute_passes_train_and_test_arguments_and_writes_metrics(tmp_path):
    config = _config(tmp_path, device='cpu')
    model = _Model(config.run_dir)
    writes = []

    result = run_training(
        config,
        execute=True,
        bundle_verifier=_verify,
        model_factory=lambda weights: model,
        metrics_writer=lambda path, payload: writes.append((path, payload)),
    )

    assert model.train_calls[0]['data'].endswith('/bundle/data.yaml')
    assert model.train_calls[0]['exist_ok'] is False
    assert model.val_calls[0]['split'] == 'test'
    assert model.val_calls[0]['save_json'] is True
    assert writes[0][0] == config.run_dir.resolve() / (
        'objective32_test_metrics.json'
    )
    assert result['artifacts']['best'].endswith('/weights/best.pt')
    assert result['test_metrics']['per_class_mask'][1] == {
        'class_id': 1,
        'class_name': 'cup',
        'mask_precision': 1.8,
        'mask_recall': 0.7,
        'mask_map50': 0.6,
        'mask_map50_95': 0.5,
    }


def test_incomplete_or_nonfinite_metrics_fail_closed():
    class IncompleteMetrics:
        names = {0: 'bottle'}
        seg = _SegmentResults()
        results_dict = {}

    with pytest.raises(RuntimeError, match='four frozen classes'):
        summarize_test_metrics(IncompleteMetrics())

    class NonfiniteMetrics(_Metrics):
        results_dict = {'fitness': float('nan')}

    with pytest.raises(ValueError, match='NaN'):
        summarize_test_metrics(NonfiniteMetrics())
