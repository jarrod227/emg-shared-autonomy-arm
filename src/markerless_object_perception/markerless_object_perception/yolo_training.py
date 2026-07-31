"""Reproducible, explicit training entry point for Objective 3.2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .dataset_contract import CLASS_NAMES


DEFAULT_PROJECT = Path('runs/objective32')
DEFAULT_RUN_NAME = 'yolo26n_seg_4class'
DEFAULT_METRICS_NAME = 'objective32_test_metrics.json'
_DEVICE = re.compile(
    r'(?:cpu|mps|cuda(?::[0-9]+)?|-1|[0-9]+(?:,[0-9]+)*)'
)


@dataclass(frozen=True)
class YoloTrainingConfig:
    """Validated Ultralytics training and held-out-test configuration."""

    bundle_dir: Path
    weights: Path
    project: Path = DEFAULT_PROJECT
    name: str = DEFAULT_RUN_NAME
    epochs: int = 100
    patience: int = 20
    batch: int = 16
    imgsz: int = 640
    device: str = '0'
    workers: int = 8
    seed: int = 3201
    deterministic: bool = True
    cache: bool = False
    amp: bool = True
    val: bool = True
    plots: bool = True
    save: bool = True
    metrics_filename: str = DEFAULT_METRICS_NAME

    def __post_init__(self) -> None:
        object.__setattr__(self, 'bundle_dir', Path(self.bundle_dir))
        object.__setattr__(self, 'weights', Path(self.weights))
        object.__setattr__(self, 'project', Path(self.project))

        positive = {
            'epochs': self.epochs,
            'batch': self.batch,
            'imgsz': self.imgsz,
        }
        for field, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{field} must be a positive integer')

        nonnegative = {
            'patience': self.patience,
            'workers': self.workers,
            'seed': self.seed,
        }
        for field, value in nonnegative.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f'{field} must be a non-negative integer')

        for field in (
            'deterministic',
            'cache',
            'amp',
            'val',
            'plots',
            'save',
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f'{field} must be a boolean')

        if not _DEVICE.fullmatch(self.device.strip().lower()):
            raise ValueError(
                'device must be cpu, mps, cuda, cuda:N, -1, or GPU indices'
            )
        if not self.name or Path(self.name).name != self.name:
            raise ValueError('name must be one path component')
        if self.name in {'.', '..'}:
            raise ValueError('name must not be dot or dot-dot')
        if not self.metrics_filename.endswith('.json'):
            raise ValueError('metrics_filename must end with .json')
        if Path(self.metrics_filename).name != self.metrics_filename:
            raise ValueError('metrics_filename must be one file name')

    @property
    def data_yaml(self) -> Path:
        """Return the portable Ultralytics dataset description."""
        return self.bundle_dir / 'data.yaml'

    @property
    def run_dir(self) -> Path:
        """Return the exact output directory reserved for this run."""
        return self.project / self.name

    def train_kwargs(self) -> dict[str, Any]:
        """Return the fully specified Ultralytics train arguments."""
        return {
            'data': str(self.data_yaml.resolve()),
            'epochs': self.epochs,
            'patience': self.patience,
            'batch': self.batch,
            'imgsz': self.imgsz,
            'device': self.device,
            'workers': self.workers,
            'project': str(self.project.resolve()),
            'name': self.name,
            'exist_ok': False,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'cache': self.cache,
            'amp': self.amp,
            'val': self.val,
            'plots': self.plots,
            'save': self.save,
        }

    def test_kwargs(self) -> dict[str, Any]:
        """Return arguments for the independent held-out test split."""
        return {
            'data': str(self.data_yaml.resolve()),
            'split': 'test',
            'batch': self.batch,
            'imgsz': self.imgsz,
            'device': self.device,
            'workers': self.workers,
            'project': str(self.run_dir.resolve()),
            'name': 'test',
            'exist_ok': False,
            'plots': self.plots,
            'save_json': True,
        }


def _default_bundle_verifier(bundle_dir: Path) -> Any:
    """Late-import dataset validation so dry-run never imports ML libraries."""
    from .prepare_yolo_dataset import verify_bundle
    return verify_bundle(bundle_dir)


def _json_safe(value: Any) -> Any:
    """Convert paths, dataclasses, and NumPy-like scalars to JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('metrics contain NaN or infinity')
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, 'item', None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def build_training_plan(
    config: YoloTrainingConfig,
    *,
    bundle_verifier: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Validate all local inputs without importing or constructing YOLO."""
    verifier = bundle_verifier or _default_bundle_verifier
    bundle_dir = config.bundle_dir.resolve()
    weights = config.weights.resolve()
    run_dir = config.run_dir.resolve()

    if not bundle_dir.is_dir():
        raise FileNotFoundError(f'prepared bundle does not exist: {bundle_dir}')
    if not config.data_yaml.is_file():
        raise FileNotFoundError(f'dataset YAML does not exist: {config.data_yaml}')
    verification = verifier(bundle_dir)
    if not weights.is_file():
        raise FileNotFoundError(
            'initial weights must be an existing local file; automatic '
            f'downloads are disabled: {weights}'
        )
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(
            f'training output already exists; choose a new --name: {run_dir}'
        )

    return {
        'mode': 'plan',
        'bundle': str(bundle_dir),
        'bundle_verification': _json_safe(verification),
        'weights': str(weights),
        'run_dir': str(run_dir),
        'train_kwargs': _json_safe(config.train_kwargs()),
        'test_kwargs': _json_safe(config.test_kwargs()),
    }


def _requests_cuda(device: str) -> bool:
    normalized = device.strip().lower()
    return normalized not in {'cpu', 'mps'}


def _default_cuda_checker() -> bool:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            'PyTorch is not installed; install a CUDA-enabled build first'
        ) from error
    return bool(torch.cuda.is_available())


def _default_model_factory(weights: str) -> Any:
    from ultralytics import YOLO
    return YOLO(weights)


def _metric_class_names(metrics: Any) -> dict[int, str]:
    names = getattr(metrics, 'names', {})
    if isinstance(names, Mapping):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, str):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def _per_class_from_segment_metrics(metrics: Any) -> list[dict[str, Any]]:
    segment = getattr(metrics, 'seg', None)
    class_result = getattr(segment, 'class_result', None)
    names = _metric_class_names(metrics)
    expected_names = dict(enumerate(CLASS_NAMES))
    if not callable(class_result) or names != expected_names:
        raise RuntimeError(
            'held-out metrics must contain exactly the four frozen classes'
        )

    rows = []
    for index, name in sorted(names.items()):
        values = list(class_result(index))
        if len(values) < 4:
            raise RuntimeError(f'incomplete mask metrics for class {name}')
        rows.append({
            'class_id': index,
            'class_name': name,
            'mask_precision': _json_safe(values[0]),
            'mask_recall': _json_safe(values[1]),
            'mask_map50': _json_safe(values[2]),
            'mask_map50_95': _json_safe(values[3]),
        })
    return rows


def summarize_test_metrics(metrics: Any) -> dict[str, Any]:
    """Extract aggregate and per-class mask metrics from Ultralytics."""
    aggregate = _json_safe(getattr(metrics, 'results_dict', {}))
    per_class = _per_class_from_segment_metrics(metrics)
    return {
        'aggregate': aggregate,
        'per_class_mask': per_class,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def execute_training(
    config: YoloTrainingConfig,
    *,
    bundle_verifier: Callable[[Path], Any] | None = None,
    cuda_checker: Callable[[], bool] | None = None,
    model_factory: Callable[[str], Any] | None = None,
    metrics_writer: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train only after explicit local and accelerator preflight succeeds."""
    plan = build_training_plan(config, bundle_verifier=bundle_verifier)
    if _requests_cuda(config.device):
        checker = cuda_checker or _default_cuda_checker
        if not checker():
            raise RuntimeError(
                f'CUDA device {config.device!r} was requested, but CUDA is '
                'not available; no model was loaded and training did not start'
            )

    factory = model_factory or _default_model_factory
    model = factory(str(config.weights.resolve()))
    model.train(**config.train_kwargs())
    test_metrics = model.val(**config.test_kwargs())

    trainer = getattr(model, 'trainer', None)
    save_dir = Path(getattr(trainer, 'save_dir', config.run_dir)).resolve()
    artifacts = {
        'save_dir': str(save_dir),
        'best': str(Path(getattr(trainer, 'best', '')).resolve())
        if getattr(trainer, 'best', None) else None,
        'last': str(Path(getattr(trainer, 'last', '')).resolve())
        if getattr(trainer, 'last', None) else None,
    }
    result = {
        'mode': 'executed',
        'plan': plan,
        'artifacts': artifacts,
        'test_metrics': summarize_test_metrics(test_metrics),
    }
    metrics_path = save_dir / config.metrics_filename
    writer = metrics_writer or _write_json
    writer(metrics_path, result)
    result['metrics_json'] = str(metrics_path)
    return _json_safe(result)


def run_training(
    config: YoloTrainingConfig,
    *,
    execute: bool = False,
    bundle_verifier: Callable[[Path], Any] | None = None,
    cuda_checker: Callable[[], bool] | None = None,
    model_factory: Callable[[str], Any] | None = None,
    metrics_writer: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return a zero-side-effect plan unless execution is explicit."""
    if not execute:
        return build_training_plan(config, bundle_verifier=bundle_verifier)
    return execute_training(
        config,
        bundle_verifier=bundle_verifier,
        cuda_checker=cuda_checker,
        model_factory=model_factory,
        metrics_writer=metrics_writer,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Plan Objective 3.2 YOLO segmentation training. Training starts '
            'only when --execute is supplied.'
        )
    )
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--name', default=DEFAULT_RUN_NAME)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3201)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--execute', action='store_true')
    return parser


def main() -> None:
    """Run the command-line planner or explicitly requested training."""
    parser = _parser()
    args = parser.parse_args()
    try:
        result = run_training(
            YoloTrainingConfig(
                bundle_dir=args.bundle,
                weights=args.weights,
                project=args.project,
                name=args.name,
                epochs=args.epochs,
                patience=args.patience,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                seed=args.seed,
                cache=args.cache,
                amp=not args.no_amp,
            ),
            execute=args.execute,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
