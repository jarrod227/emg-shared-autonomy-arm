# Objective 4.2 — Debugging Log

How the stereo hand observer went from never producing a valid observation to
97% availability in the natural receiving pose, on 2026-08-11.

This is kept separate from `objective42_evaluation.md` on purpose. That file
is a clean reference for the results; this one keeps the wrong turns, because
the wrong turns are where the method shows.

## Symptom

The live GUI overlay reported `valid=false stable_frames=0` continuously.
A single captured frame showed a complete green hand skeleton in both views
and the diagnostic `keypoint pair exceeds epipolar-error limit: 12.635 px >
1.500 px`.

Confusing part: a static checkerboard, through the same rectification and the
same calibration, measured 0.194 px mean epipolar error. So the calibration
looked excellent and the hand looked terrible.

## Hypotheses that were tested and killed

Each of these was plausible enough to act on. Each was killed by a
measurement, not by argument.

**H1: the calibration drifted or the module moved.**
Killed immediately — the checkerboard through the same maps was 0.19 px. A
broken calibration cannot be selectively bad for hands.

**H2: landmark 9 specifically is a bad choice.**
Killed by measuring all 21 landmarks over 20 frames instead of one frame.
Landmark 9's median was 2.19 px; the 12.635 px reading was a transient spike
(its own max over the window was 8.9 px). The single frame that started the
investigation was simply noise.

**H3: the pinky side of the hand localizes worse (foreshortening).**
Formed from single-frame readings where landmarks 13 and 17 failed while 5
and 9 passed. Killed by a 60-frame distribution: the pinky knuckle was the
**best** landmark at 1.79 px median. There was no side gradient — only
frame-to-frame noise being read as structure.

**H4: the ~900 ms latency is upstream, in capture and MJPEG decode.**
Killed by measuring the composite topic directly: frames arrived only 76 ms
old. The delay was downstream of that, inside the observer.

**H5: the gstreamer pipeline decodes at 30 FPS before dropping to 5, wasting
work and buffering.**
Acted on — reordered so `videorate` ran before `jpegdec`, plus a leaky queue.
Age got **worse**: 2053 ms versus 1133 ms. Reverted.

**H6: the debug overlay is eating the frame budget.**
Killed by disabling it: age got worse again (1670 ms) while the frame rate
*rose*. A faster consumer producing older data is the signature of draining a
backlog, which is what finally pointed at transport queueing rather than
per-frame cost.

**H7: palm-up is inherently the worst pose, and unluckily also the natural
receiving pose.**
This felt like an important product insight. Killed by measuring it: palm up
with fingers spread and held steady scored **97.3%**. The real variable was
whether the hand is open and steady, not which way the palm faces.

## Root causes

Two independent ones, neither of which was the first suspect.

**1. The epipolar limit encoded the wrong noise model.**
`max_epipolar_error_px = 1.5` was calibrated for checkerboard corners, which
localize to sub-pixel accuracy. Independently-run MediaPipe keypoints do not:
palm knuckles measure 1.8–3.2 px median with peaks near 9 px. The gate was
therefore below the noise floor of the thing it was gating and could
essentially never latch.

The fix was not simply a larger number. A single landmark gives no way to
tell whether the left and right detectors marked the *same physical point* —
the epipolar residual is the only available evidence, and one point yields
one weak bit of it. Triangulating four palm knuckles independently and
requiring a quorum turns that into a real consistency check, and the median
of the survivors is also more stable than any single point.

Measured while choosing the limit, judging on recovered 3D scatter rather
than on how often the gate opened:

| limit | quorum reached | palm scatter |
| ---: | ---: | ---: |
| 3.0 px | 50.0% | 5.9 mm |
| 6.0 px | 100.0% | 3.9 mm |

Tightening the limit *worsened* accuracy, because it starved the median of
points. That result is the opposite of the intuition that started the search.

**2. A 7.37 MB raw frame per cycle through DDS.**
The uncompressed composite backed up in transport: observations carried a
1133 ms mean age although the entire compute chain measures 125 ms. Giving
the observer a `direct` input mode that opens the device itself, with a
one-frame driver buffer, took that to 72.9 ms — and `cap.read()` staying flat
at 25 ms is the evidence that no backlog forms.

## Ground truth failed three times

More measurements were invalidated by bad labelling than by bad code.

1. A "no hand present" segment scored 9.4% false positives. A careful re-run
   scored zero — part of an arm had been in frame.
2. A "hand held naturally" segment scored 28.7%. The subject was in fact
   cycling continuously through orientations, so the label described
   something other than natural use.
3. A pose-flip experiment reported a 3.2 mm difference, which would have been
   a strong result. The condition had not actually been reached — confirmed
   only afterwards, by accident.

The fix was to make the measurement carry its own evidence: the position
script now reports the epipolar median alongside the result, so a face-on run
(~2 px) is distinguishable from an edge-on run (~5 px) from the data itself
rather than from a human's recollection. The very next run reported 1.00 px
and disproved its own label without needing a separate check.

## Lessons

- **Never conclude from one frame** when the quantity's own spread is larger
  than the effect. Two hypotheses here died to this alone.
- **A threshold has to match the noise model of what it measures.** 1.5 px
  was not wrong in general; it was wrong for learned keypoints.
- **Validate a threshold change against the physical quantity it protects**,
  not against the indicator turning green. Here that meant 3D scatter, and it
  reversed the expected direction of the fix.
- **A faster consumer returning older data means queueing, not slow code.**
- **Human-executed conditions need machine-checkable evidence.** If a run
  cannot prove which condition it was in, its result is not usable.
- **Measure before rearchitecting.** The direct-capture rewrite was only
  started after a throwaway prototype demonstrated 147 ms versus 1133 ms; two
  earlier "obvious" fixes had already made things worse.
