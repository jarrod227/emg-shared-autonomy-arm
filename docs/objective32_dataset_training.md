# Objective 3.2 Dataset and GPU Training Guide

> **Scope status (2026-08-11): optional legacy capability.** Objective 3.2 now
> uses official COCO-pretrained instance-segmentation weights for the
> provisional runtime set `bottle`, `cup`, and `apple` at a `0.50` threshold.
> Real-apple validation remains pending. `medicine_box`, custom dataset
> collection, frozen-
> bundle generation, and CUDA training are no longer MVP requirements. The
> guide below documents the already implemented and tested four-class tooling;
> use it only if measured real-camera performance later justifies fine-tuning.

## Current boundary (2026-07-31)

The reproducible data-preparation and training entry points are implemented
and tested. `markerless_object_perception` builds and passes 85 tests. No real
Objective 3.2 source dataset has been accepted yet, no real frozen bundle has
been generated, and GPU training has not started.

The current project virtual environment is CPU-only:

```text
torch=2.13.0+cpu
cuda_available=False
```

At that checkpoint, the planned next work was to collect and annotate real
images, assign capture-session and physical-object identities, review the
manifest, generate the frozen bundle, and then move it to a CUDA machine. The
2026-08-09 scope update supersedes that requirement; the workflow below is now
optional and should be reopened only after a measured pretrained-model failure.

## Legacy implemented four-class contract

Class names and IDs must not be reordered:

| ID | Class |
| ---: | --- |
| 0 | `bottle` |
| 1 | `cup` |
| 2 | `cell_phone` |
| 3 | `medicine_box` |

Background is not a fifth class. A negative image has an empty `.txt` label
and an empty `instances` object in the source manifest.

## Source dataset contract

Use this layout before freezing:

```text
objective32_source/
├── images/
│   └── <relative image paths>
├── labels/
│   └── <same relative paths, .txt suffix>
└── source_manifest.jsonl
```

Every image must have one same-name label, including negatives. Every image
and label must be listed exactly once; missing files, duplicate content,
orphan files, symlinks, malformed polygons, and class/count mismatches fail
closed.

Each non-empty label line uses normalized YOLO segmentation polygon syntax:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> [...]
```

Coordinates must be finite and within `[0, 1]`, with at least three distinct
points and non-zero area.

Each JSONL row uses this shape:

```json
{"sample_id":"s01_bottle_0001","image":"s01/bottle_0001.jpg","session_id":"session_01","instances":{"blue_bottle_01":"bottle"}}
```

- `sample_id` is a unique portable filename token.
- `image` is relative to `images/`; the label path is derived automatically.
- `session_id` identifies one capture session. Adjacent video frames and one
  stop-and-look burst should share it.
- `instances` maps each physical-object identity to its frozen class. Reuse
  the exact identity whenever the same real object reappears.
- IDs may not have leading or trailing whitespace.
- An image with two annotated objects needs two label lines and two instance
  entries of matching classes.

The checked-in schema example is
`src/markerless_object_perception/config/source_manifest.example.jsonl`.

## Leakage rule and collection target

The split unit is a connected leakage group, not an individual image. Two
samples are forced into the same split when they share either:

1. a `session_id`; or
2. any physical-object identity.

This rule is transitive. If A shares a session with B and B shares an object
with C, A/B/C remain together. It prevents nearby frames or the same cup,
phone, bottle, or medicine box from appearing in both training and held-out
evaluation.

The hard contract requires each of the four classes and the negative category
to occur in at least three independent leakage groups, so train/val/test can
all contain them. For a meaningful MVP, collect at least 5 distinct physical
objects per class plus 5 independent negative groups, with varied views,
lighting, backgrounds, partial occlusion, and confusers. More independent
objects are preferable to many nearly identical frames of one object.

The deterministic policy targets 70/15/15 with seed `3201`. Group integrity
and per-split category coverage take precedence, so a small dataset may not
land on the exact percentages.

## Prepare and verify the frozen bundle

Build and source the package first:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select markerless_object_perception
source install/setup.bash
```

Create a new bundle. The command refuses an existing or symlink output path:

```bash
ros2 run markerless_object_perception prepare_yolo_dataset prepare \
  /absolute/path/objective32_source \
  /absolute/path/objective32_v1 \
  --config src/markerless_object_perception/config/objective32_dataset.toml
```

Verify it again at any time:

```bash
ros2 run markerless_object_perception prepare_yolo_dataset verify \
  /absolute/path/objective32_v1
```

The portable output is:

```text
objective32_v1/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── manifest.json
└── manifest.sha256
```

`data.yaml` deliberately has no machine-specific `path:`. Verification checks
the exact class order, split policy, polygon threshold, group assignment,
session/object leakage, category coverage, file set, and every image/label
hash. A changed file or manifest fails before training.

## Dry-run on this computer

An initial segmentation weight must already exist locally; bare model names
are rejected so Ultralytics cannot download implicitly. The existing local
candidate is `.venv/models/yolo26n-seg.pt`.

Without `--execute`, the command only verifies the bundle, local weight,
configuration, and unused exact output directory, then prints train/test
arguments. It does not import Torch/Ultralytics, construct YOLO, create a run
directory, or train:

```bash
ros2 run markerless_object_perception train_yolo_segmenter \
  --bundle /absolute/path/objective32_v1 \
  --weights /absolute/path/yolo26n-seg.pt \
  --project /absolute/path/runs/objective32 \
  --name yolo26n_seg_4class
```

## Move to a CUDA machine and train

The GPU machine needs the training code, the verified frozen bundle, and the
local initial `.pt` file. It does not need the unfrozen source dataset for
training. The simplest reproducible handoff is the repository checkout plus
those two external artifacts.

Create a separate GPU environment. Install the CUDA-enabled PyTorch build that
matches that machine's driver first, then install:

```bash
python -m pip install -r \
  src/markerless_object_perception/requirements-yolo-gpu.txt
python -m pip install -e src/markerless_object_perception
```

Run the command once without `--execute` on the GPU machine. After reviewing
the printed paths and parameters, add the explicit flag:

```bash
train_yolo_segmenter \
  --bundle /absolute/path/objective32_v1 \
  --weights /absolute/path/yolo26n-seg.pt \
  --project /absolute/path/runs/objective32 \
  --name yolo26n_seg_4class \
  --device 0 \
  --execute
```

The default run is 100 epochs, patience 20, batch 16, image size 640, seed
3201, deterministic mode, AMP enabled, and first CUDA GPU. The wrapper checks
CUDA before constructing the model and never silently falls back to CPU.

Training uses train/val, then independently evaluates the frozen `test` split.
The run must report all four classes. It writes Ultralytics artifacts including
`weights/best.pt` and `weights/last.pt`, plus
`objective32_test_metrics.json` containing aggregate results and per-class
mask precision, recall, mAP50, and mAP50-95. Missing classes, incomplete
metrics, NaN, or infinity fail closed.

## What remains after training

GPU training does not complete Objective 3.2 by itself. After selecting and
freezing one model, the remaining runtime work is live left/right image and
`CameraInfo` subscription, calibrated/rectified stereo depth, real
mask-filtered 3D candidate publication, measured class-specific grasp offsets,
and integration with the already verified selector contract. The ArUco path
remains a separate regression and fallback.
