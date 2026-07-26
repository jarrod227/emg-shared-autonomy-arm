# Objective 3 — Marker Perception Evaluation (M6)

Pose-estimation accuracy of the ArUco marker pipeline
(`marker_pose_provider`), measured against tape-measured ground truth.

## What is and isn't measured here

Per the layered-metrics principle, only what can be honestly measured is
reported:

- **Camera-frame accuracy (real, reported below).** Distance from the camera
  to the marker, compared against a tape measure. This reflects the perception
  pipeline itself (detection + calibration + PnP).
- **Full-chain / world-frame accuracy (deferred).** The `world -> camera`
  extrinsic is an invented simulation placeholder (the fake Panda planning
  frame has no physical relation to the desk camera), so a world-frame error
  number would be dominated by that fabricated transform and would say nothing
  about perception. It is deliberately **not** reported. On real SO-ARM101
  hardware (Phase 2) the extrinsic becomes a measured hand-eye calibration and
  full-chain accuracy becomes meaningful.

## Setup

- Camera: integrated laptop webcam, 640×480.
- Marker: printed `DICT_4X4_50` id 0, 51 mm black-square side (`marker_length`
  = 0.051).
- Marker fixed (leaning, not handheld), roughly frontal, tape-measured from the
  lens to the marker centre.
- Metric: Euclidean distance ‖(x, y, z)‖ in the camera frame (the tape measures
  the straight-line lens-to-marker distance, i.e. the norm, not the optical-axis
  z alone).
- Each bin: 10 s of samples (~290 image frames).
- **Measurement uncertainty ≈ ±5 mm**: a webcam's optical centre sits a few mm
  behind the bezel glass at an unknown depth, and tape/read-off adds error. Sub-
  cm distance errors below are at the floor of what the tape can resolve.

## Results (after re-calibration)

| Tape distance | Measured distance | Distance error | Distance std | Lateral std (x/y) | Detection |
| --- | --- | --- | --- | --- | --- |
| 0.30 m | 0.308 m | +8 mm | 0.2 mm | 0.02 / 0.04 mm | reliable |
| 0.50 m | 0.509 m | +9 mm | 0.3 mm | 0.02 / 0.08 mm | reliable |
| 0.80 m | ~0.70–0.75 m | ~−50 to −100 mm | 16 mm | 0.5 / 1.4 mm | detected, depth unreliable |
| 1.20 m | — | — | — | — | **0 detections** |

### Interpretation

- **≤ 0.5 m: reliable.** Distance error within ~1 cm (at the tape's own
  resolution floor), depth repeatability sub-millimetre, lateral repeatability
  sub-millimetre.
- **0.8 m: depth-accuracy cliff.** Detection still succeeds and lateral position
  stays millimetre-accurate, but depth becomes unreliable — std ~16 mm and the
  mean swings ~5 cm between runs. At 0.8 m the 51 mm marker spans only ~35 px,
  and depth sensitivity to per-pixel corner error grows with distance squared,
  so small pixel noise becomes large depth noise. Lateral position is unaffected,
  which is why x/y stays sharp while z scatters.
- **1.2 m: out of range.** The marker spans ~23 px and the detector rejects it
  entirely (0/295 frames).
- **Usable working envelope: depth reliable to ≤ 0.5 m; detection to < 0.8 m.**
  For Phase 2 this constrains camera placement (mount near the workspace) or
  argues for a larger marker.

## Repeatability vs accuracy

The distance-error column is *accuracy* (vs tape ground truth). The std and
lateral-std columns are *repeatability* (frame-to-frame scatter, no ground
truth needed) and stay valid regardless of calibration. Rotation was recorded
only as spread around the first sample (repeatability, not accuracy): a
near-frontal planar marker has an inherent two-solution PnP ambiguity, so
absolute rotation accuracy is not claimed here.

## Calibration debugging note (why the numbers were wrong first)

The first calibration used a **printed checkerboard that was slightly warped**.
A curved board violates the planar assumption calibration relies on and biased
the focal length low: fx = 437.5. That produced a *multiplicative* distance
error — measured/true ≈ 0.89, constant across distances — of roughly −11 %
(e.g. −34 mm at 0.30 m, −51 mm at 0.50 m).

A one-off pixel-geometry check confirmed the cause: from the marker's pixel
edge length and the known 51 mm side, the focal length implied by a correct
tape distance was ~490, not 437.

Re-calibrating with a **flat iPad screen** (glass is a true plane), covering the
image corners and using large tilt angles, gave fx = 501.5, fy = 502.3, and a
principal point (cx = 317.8, cy = 228.8) much closer to image centre than the
first calibration's cx = 287. Re-measuring, the multiplicative bias was gone and
errors dropped to +8 mm / +9 mm at 0.30 / 0.50 m.

Takeaway: perception accuracy is calibration-limited, not algorithm-limited —
the synthetic ground-truth tests (M3.5) already showed the PnP math itself is
accurate to ~0.2 mm. A flat calibration target is not optional.

## Known measurement artifacts

- The accuracy probe reports detection "success rate" as PoseArray callbacks
  over image frames; when detection is very stable this can exceed 100 % because
  a single image can trigger more than one callback. It is a counting
  artifact, not a real >100 % rate; the meaningful signal is "reliably detected"
  vs "fails to detect". (To be tidied in M7.)
