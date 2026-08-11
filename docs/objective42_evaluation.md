# Objective 4.2 — Stereo Hand Observation Evaluation

Measured performance of the DECXIN stereo hand-position cue
(`stereo_hand_observer`), taken on the fixed bench on 2026-08-11.

## Role in the current roadmap

Objective 4.2 supplies the `/hand_observation` readiness cue that the
Objective 4.1 handoff controller gates `READY -> RELEASE` on. It is a
**non-safety-rated cue**, not certified human–robot separation monitoring,
and nothing below changes that. Objective 3.1 ArUco localization has its own
evaluation in `objective3_evaluation.md`; the two are independent paths.

## What is and isn't measured here

- **Precision (real, reported).** Scatter of the recovered palm point with
  the hand held still. This is repeatability of the whole chain.
- **Scale accuracy (real, reported, bounded).** A tape-measured displacement,
  which validates the calibration baseline. Absolute position accuracy is
  **not** separable from "where exactly is the palm centre", so it is not
  claimed.
- **Exposure skew (real, reported as an upper bound).** Bounded, not proven
  zero. The pair is **not** described as hardware-synchronized.
- **False/missed readiness-gate rates (real, reported).** Measured against a
  scripted protocol; see the gate-rate section below, including the ground
  truth that had to be discarded.
- **Pose-induced position bias (NOT measured).** Three attempts failed
  because holding a hand at one fixed location while only changing its
  orientation turned out to be beyond what a human can execute reliably.
  This is an open gap, not a finding of "no bias".

## Setup

- Camera: DECXIN one-board stereo module, one 2560×960 side-by-side MJPG
  composite stream over a single USB interface.
- Calibration: accepted run-2 candidate — 31 checkerboard pairs, 8×6 inner
  corners, 0.025 m squares, stereo RMS 1.0385 px, calibration-derived
  baseline **0.064529 m**. Rectified projections share fx = fy = 914.165.
- Detector: MediaPipe Tasks hand landmarker, one complete 21-landmark hand
  required per view.
- Geometry: the four palm knuckles (landmarks 5, 9, 13, 17) are triangulated
  independently; at least 3 must pass, outliers beyond 0.12 m of the cluster
  median are dropped, and the palm point is the median of the survivors.
- Hand pose: what actually matters is that the hand is **open and steady**,
  not which way the palm faces. See the gate-rate section — an early reading
  that palm orientation was the driver did not survive measurement.

## Why the original thresholds could never pass

The gate shipped with `max_epipolar_error_px = 1.5`, a value calibrated for
**checkerboard corners**, which localize to sub-pixel accuracy. Learned hand
keypoints do not. A 21-landmark sweep on the live stream measured:

| landmark group | median epipolar error |
| --- | --- |
| palm knuckles (5, 9, 13, 17) | 1.8–3.2 px |
| fingertips (12, 16) | 8–66 px |
| checkerboard, same rectification | 0.19 px |

Calibration was never at fault. The threshold was applying a geometric-feature
noise model to a learned-keypoint measurement.

Single-frame epipolar values fluctuate hard — a ~2 px median coexists with
~9 px peaks — so **no conclusion should be drawn from one frame**.

## Threshold selection (60 frames, hand still at z = 0.412 m)

Candidate limits were judged on the recovered 3D point, not on how often the
gate opened:

| epipolar limit | frames reaching quorum | palm scatter |
| ---: | ---: | ---: |
| 3.0 px | 50.0% | 5.9 mm |
| 4.0 px | 76.7% | 5.9 mm |
| **6.0 px** | **100.0%** | **3.9 mm** |
| 8.0 px | 100.0% | 3.6 mm |

Tightening the limit does not protect accuracy — it starves the median, since
fewer surviving knuckles make the aggregate noisier. 8.0 px gained a further
0.3 mm but exceeds the largest error ever observed (6.25 px), which would
leave the gate unable to reject a genuine left/right mismatch. **6.0 px** is
the working point. Reprojection stays at 3.0 px against a measured p95 of
1.837 px, so it remains active.

## Results

### Precision — hand held still

| distance | scatter (x, y, z) | total |
| ---: | --- | ---: |
| 0.42 m | 0.7, 3.1, 3.3 mm | ~4.6 mm |
| 0.72 m | 1.2, 1.4, 4.1 mm | ~4.5 mm |

Depth degrades more slowly with distance than the z² sensitivity law predicts
(9.7 mm expected at 0.72 m, 4.1 mm observed), so the usable working volume is
larger than a worst-case estimate would suggest.

### Scale accuracy — 300 mm tape displacement

| | x | y | z |
| --- | ---: | ---: | ---: |
| A (0 cm) | +0.0727 | +0.0440 | 0.4216 m |
| B (30 cm) | +0.0496 | −0.0191 | 0.7178 m |
| displacement | −23.1 | −63.1 | **+296.1 mm** |

Measured magnitude **303.6 mm** against a 300 mm tape reading. The lateral
components cannot be separated from tape misalignment, so the honest bound is
a **scale error within ±1.2%**. This confirms the 0.064529 m baseline: a 5%
baseline error would have read 285 or 315 mm.

### Exposure skew — 137 frames of vertical hand motion

A static scene cannot reveal exposure skew. With the hand moving vertically at
image speed v, a skew dt shifts what the second sensor sees, so the signed
vertical mismatch behaves as `dy = bias + v·dt`; fitting dy against v puts the
static calibration residual in the intercept and leaves the slope as the skew.

- Vertical speed spanned −1607 to +1622 px/s (544 samples)
- Static dy bias: **+0.36 px** — an independent confirmation of rectification
- Slope: **−0.188 ± 0.185 ms**, not resolved above noise
- **Upper bound: |skew| < 0.56 ms**

At a 1 m/s hand speed this contributes under 0.6 mm, well inside the precision
floor. The pair behaves as if simultaneously exposed **to within this bound**;
it is still not evidence of hardware synchronization.

### Readiness-gate rates

Scored against the running node, so these reflect the shipped gate including
its 3-frame stability requirement. The first and last 3 s of every segment
are discarded so a human moving into or out of position is never scored.

**False readiness (the safety-critical direction).** With both hands fully
out of frame, **0 valid frames in ~400** (~40 s). With zero events the
rule-of-three 95% upper bound is **< 0.75%**.

A first attempt at this segment scored 9.4% valid and was **discarded**: a
follow-up run showed part of an arm was still in frame, so the ground-truth
label, not the gate, was wrong. Re-run carefully it produced zero.

**Missed readiness.** Reported per condition, because a single number here
is misleading — the controller only needs one valid frame while a hand is
held out, so time-to-first-valid matters more than duty cycle:

| condition | duty cycle | first valid | longest gap |
| --- | ---: | ---: | ---: |
| upright, palm to camera, steady | 100.0% | 0.00 s | 0 s |
| **palm up, fingers spread, steady** (natural receiving pose) | **97.3%** | **0.00 s** | **0.48 s** |
| continuously rotating between orientations | 28.7% | 0.72 s | 13.39 s |
| hand pressed flat on the table, back up | degraded — depth scatter 23.5 mm vs 4.8 mm | | |

The 28.7% row is **not** a natural-use figure. That subject was deliberately
cycling through orientations for the whole segment; its dominant refusal was
`insufficient_consensus`, consistent with repeatedly passing through
ambiguous mid-rotation poses.

The operational conclusion is that **an open, steady hand works regardless of
palm orientation** — the natural receiving pose scores 97.3% with a worst gap
under half a second. An earlier hypothesis that palm-up was inherently the
worst case, and that it was also the ergonomically natural one, was tested
and **disproved**.

Degradation appears when the hand is closed or pressed flat so that fingers
occlude the knuckles: one view then fails to detect at all, or the knuckle
positions get inferred rather than observed and depth scatter rises about 5x
(4.8 mm to 23.5 mm). Even that degraded scatter is ~17x inside the delivery
volume's 400 mm radius, though only ~2x inside the 50 mm stability step.

### Timing and throughput

The raw composite is 7.37 MB per frame. Publishing it through DDS backed up
badly — observations carried a mean age of **1133 ms** (p95 2567) even though
the composite topic itself was only 76 ms old at a subscriber, and the entire
split/rectify/dual-MediaPipe chain measures 125 ms. Two cheaper explanations
were tested and rejected: reordering the gstreamer pipeline to drop frames
before `jpegdec` made age worse (2053 ms), and disabling the debug overlay
also made it worse (1670 ms) while the rate rose — a faster consumer draining
a backlog.

Capturing the device directly (`input_mode: direct`, one-frame driver buffer)
removed the transport entirely:

| | DDS path | direct capture |
| --- | ---: | ---: |
| observation age (mean) | 1133 ms | **72.9 ms** |
| age p95 / max | 2567 / 3067 ms | **79.6 / 86.2 ms** |
| rate, no hand present | 4.41 Hz | **9.80 Hz** |
| rate, hand present | — | ~6.8 Hz |

Per-stage compute: MediaPipe 2 × ~59 ms, rectification 4.6 ms, split 2.6 ms.
ROS pair skew is structurally zero — both halves come from one frame.

## Error budget against task tolerance

| source | magnitude |
| --- | ---: |
| precision (hand still) | 3.3–4.1 mm |
| scale error | ±1.2% |
| exposure skew at 1 m/s | < 0.6 mm |
| observation age | 73 ms |
| **delivery-volume gate radius** | **0.4 m** |
| **stability step limit** | **0.05 m** |

The measured errors sit roughly two orders of magnitude inside what the gate
actually decides on.

## Still open

These are gaps, not solved problems. Do not read the results above as
covering them.

- **This objective does not yet connect to Objective 4.1.** The observer
  publishes in `stereo_left_optical_frame`; the handoff controller requires
  `delivery_frame` (`world`) and refuses on any mismatch rather than
  guessing. Bridging them needs a real camera-to-world extrinsic, which is
  still a simulation placeholder and is deferred to Objective 5. The refusal
  is correct fail-closed behaviour, but perception-to-control is **not**
  demonstrated end to end.
- **Pose-induced position bias was never measured.** Three protocols failed:
  holding a hand at a fixed location while changing only its orientation is
  not something a person executes reliably, and every attempt either changed
  the location too or failed to reach the intended pose. Precision under each
  pose is reported; systematic offset between poses is unknown.
- **The camera-frame delivery volume barely discriminates.** With centre
  (0, 0, 0.5) and radius 0.4 m, and a 70 degree horizontal field of view, the
  frame is only 0.70 m wide at z = 0.5 m — so a hand cannot leave the sphere
  sideways while staying visible. The gate currently costs a full 3D estimate
  while rejecting almost nothing. Tightening it is a design decision tied to
  what the handoff should actually require.
- **image_proc rectify-node exit -11 on Ctrl-C.** Only affects the gscam-based
  `decxin_atomic_hand.yaml` path; `direct` mode does not use image_proc.
- **Absolute position accuracy.** Deliberately not claimed — the palm centre
  has no tape-measurable physical referent. Displacement is the honest test.

## Reproducing

```bash
colcon build --symlink-install --packages-select stereo_hand_observer
source install/setup.bash
ros2 launch stereo_hand_observer decxin_direct_hand_observer.launch.py \
  model_path:=$HOME/.cache/assistive_robot/mediapipe/hand_landmarker.task
```

Hold one hand upright with the palm toward the camera. A second hand anywhere
in frame makes the detector fail closed by design, since it requires exactly
one hand.
