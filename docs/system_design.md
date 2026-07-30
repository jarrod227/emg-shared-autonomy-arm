# System Design

## Current Verified Status (2026-07-30)

This section supersedes older scaffold descriptions below. Two interchangeable
providers feed the retained `PoseStamped` on `/target_object_pose`: the
deterministic `fixed_pose_publisher`, and the marker pipeline
(`marker_pose_provider` detection/PnP -> `target_selector` selection, grasp
offset, tf2 transform to `world`). `MoveToObjectNode` consumes the interface
unchanged either way, sends a Cartesian goal for `panda_link8` to MoveIt, and
returns to the joint-space ready home. Both provider paths are
runtime-verified end to end; camera-frame accuracy is quantified in
`docs/objective3_evaluation.md`.

Objective 4.1 is also implemented as a separate Phase-0 simulated controller:
`assistive_interfaces` defines `AssistiveIntent`, `HandObservation`, and
`ViewControlCommand`, while `assistive_handoff` implements the five-state core
handoff flow plus two search states, simulated providers, freshness/timeout
checks, `ABORT`, and fail-closed release gating.
Historical results cover the 23-test Objective 1/2/3.1 baseline and the earlier
25-test Objective 4.1 controller suite. Seven ROS 2 packages now exist. On
2026-07-30 `markerless_object_perception` built and passed 44 tests;
`assistive_interfaces` and `target_selector` built, with 42 selector tests
passing in an isolated ROS domain. The current stored result summary is 227
tests with zero errors or failures and one optional external-asset skip.

Objective 4.2 has a separate `stereo_hand_observer` package implementing
synthetic and live rectified-image paths, `CameraInfo.P` stereo geometry,
a reusable MediaPipe Tasks 21-landmark hand detector plus representative-point
adapter, approximate synchronization, epipolar/reprojection rejection,
delivery-volume/N-frame gates, exact source stamps, and stream watchdog
invalidation. Its default package-local run passed 70 tests with one optional
external-asset smoke test skipped; that smoke test separately passed with an
official model and official single-hand image. Live dual-camera runtime and
bench-stereo validation are still pending.

Objective 3.2 is active software-first. `markerless_object_perception`
implements robust mask/aligned-XYZ localization, candidate construction, an
Ultralytics instance-segmentation/tracking adapter, and a laptop-camera demo.
`assistive_interfaces` defines the timestamped candidate contract, and
`target_selector` implements pure multi-track temporal stability. The live ROS
candidate publisher, real stereo point cloud, trained four-class model,
intent-driven target lock, and markerless `/target_object_pose` output remain
pending.

## Design Status

This document distinguishes the current implementation from the longer-term
architecture. The execution layer (Objectives 1–2), ArUco perception baseline
(Objective 3.1), and Phase-0 simulated handoff state machine (Objective 4.1)
exist today. Objective 4.2 is active and adds stereo-triangulated hand
keypoints; its live-camera/bench validation remains. Objective 4.3
Phase-0 bounded search with simulated view commands is implemented. Objective
3.2 now has a tested software foundation but still needs ROS/live-stereo and
selector integration. Objective 3.5 later adds STM32 discrete intent plus
proportional view control.

## Current Workspace Layout

```text
assistive_robot_ws/
├── docs/
├── src/
│   ├── object1_demo/          reaching coordinator + fixed-pose provider
│   ├── marker_pose_provider/  ArUco detection + camera-frame PnP (3.1)
│   ├── target_selector/       ArUco selection/TF + markerless stability core
│   ├── assistive_interfaces/  intent, hand, view, and candidate messages
│   ├── assistive_handoff/     simulated handoff + bounded view search (4.1/4.3)
│   ├── stereo_hand_observer/  stereo geometry/gates + live adapter (4.2)
│   └── markerless_object_perception/  mask/XYZ + YOLO adapter (3.2)
├── AGENTS.md
└── TODO.md
```

ROS 2 packages live under `src/`. Project-level planning and design documents
live at the workspace root.

## Current Verified Runtime — Objective 3.1

```text
marker_demo.launch.py                    (or: fixed_pose_publisher alone)
├── v4l2_camera driver
│   ├── /image_raw
│   └── /camera_info  (calibrated intrinsics from config/camera_info.yaml)
└── marker_node
    └── /detected_markers (PoseArray, camera frame, detections only)

extrinsics.launch.py
└── static TF world -> camera            (SIMULATION PLACEHOLDER extrinsic)

selector_node
├── subscribes /detected_markers
├── selects first pose, applies default grasp offset, transforms via tf2
└── /target_object_pose (PoseStamped, retained, latched once)

move_to_object
├── subscribes /target_object_pose
├── sends MoveGroup goals on /move_action
└── publishes /object1_target for RViz
```

`marker_node` publishes only when a marker exists; it does not publish an empty
heartbeat. Marker IDs are logged but discarded by `PoseArray`, so the current
selector cannot choose by ID and takes the first pose. It uses the default
offset, publishes one retained target, and then ignores later detections.

The Objective 1/2 coordinator waits for the first valid target pose, executes its existing
IDLE -> REACHING -> RETURNING -> DONE one-shot sequence, and ignores additional
target messages. These remain current regression-path facts. The separate
Objective 4.1 `assistive_handoff` package implements the Phase-0 handoff state
machine without replacing that verified one-shot node.

DDS `TRANSIENT_LOCAL` retention and the application one-shot latch are not
freshness. The selector/tracker will own N-frame stable acquisition and
last-seen tracking; the Objective 4.1 controller already owns the Phase-0
policy response to old targets and stale hand observations.

## Objective 1 Architecture

```text
fixed target-pose publisher
-> reaching coordinator
-> MoveIt 2 motion backend
-> simulated arm
-> RViz visualization
```

The first configured target is named `object1`. It is a deterministic test pose,
not the output of object recognition. The coordinator should request a reaching
trajectory and return the arm to a neutral or home position.

## Approved Architecture

```text
CURRENT / PRESERVED
fixed pose ---------------------------------------------> /target_object_pose
camera -> ArUco/PnP -> current target_selector --------> /target_object_pose
simulated target/intent/hand inputs --------------------> handoff controller
                                                          (Phase-0 state/actions)

OBJECTIVE 3.2 SOFTWARE FOUNDATION / IMPLEMENTED
image -> YOLO adapter -> class + confidence + mask + temporary track ID
mask + aligned XYZ -> robust candidates -> ObjectCandidateArray contract
candidate frames -> pure N-frame stability gate

PLANNED RUNTIME INTEGRATION
left/right RGB -> sync/rectify -> disparity/PointCloud2 --------+
left rectified RGB -> instance mask ----------------------------+-> /object_candidates
STM32 EMG -> USB/UART bridge -> /assistive_intent --------------+-> target selector
STM32 EMG -> USB/UART bridge -> /assistive_view_control --------+       |
                                                               /target_object_pose
stereo hand keypoints -> 3D /hand_observation ----------> handoff controller
                                                                     |
                                                             robot backend
```

Only one selected-pose source is active at a time. Each physical camera driver
is launched once in its own namespace; multiple consumers may reuse the
rectified streams. The preserved ArUco path may use the left camera alone.

Language / voice target selection is cut, not deferred — see `docs/proposal.md`
§5. It would occupy the same layer as biosignal intent and is functionally
redundant with it.

### Intent Layer

Objective 4.1 uses a simulated publisher on the future production interface.
Objective 3.5 later replaces that source with STM32 edge inference:

```text
3-channel ADC/DMA
-> approximately 20–450 Hz DSP
-> 200 ms windows / 50 ms hop (initial target)
-> MAV, RMS, zero crossing, waveform length
-> classical baseline or small quantized model
-> REST / NEXT_TARGET / CONFIRM / ABORT
-> calibrated direction + normalized activation envelope
-> USB CDC or UART packet
-> MVP Python/rclpy bridge -> /assistive_intent + /assistive_view_control
```

Firmware ownership is explicit: STM32CubeIDE C/C++ implements ADC/DMA,
DSP/features, inference, and the versioned packet producer. PC Python tools own
raw capture/replay, plots, training/quantization, and golden-vector comparison.
The first runtime bridge is Python/`rclpy` so protocol and diagnostics can be
iterated quickly. A later `rclcpp` receiver/parser with a fixed-size ring buffer
is an optional measured optimization, not an Objective 3.5 completion gate.

`REST` produces no event. Intent messages require source timestamp, command,
confidence, and sequence number; they are reliable and volatile, never retained
or replayed. `ViewControlCommand` is also volatile and contains
`LEFT`/`RIGHT`/`HOLD`, normalized activation, confidence, signal quality,
source time, and sequence. EMG does not publish `/target_object_pose`.

After per-session rest and comfortable-contraction calibration:

```text
activation = clip((envelope - rest) / (comfortable - rest), 0, 1)
target_angle = safe_center +/- configured_limit * activation
```

Activation changes target angle, not speed. The controller uses a fixed low
nominal speed plus acceleration/deceleration limits. Deadband, smoothing,
saturation, stable-window gating, and a stale-command watchdog suppress noise.
A newer valid command preempts the old search target after a smooth halt;
`ABORT` bypasses smoothing and always has priority.

### Stereo Sensing Foundation

Two ordinary USB RGB cameras form a passive stereo sensor only after they are
mounted in one rigid bracket and calibrated. They have no assumed hardware
trigger. The planned bench interface is:

```text
/stereo/left/image_raw       + /stereo/left/camera_info
/stereo/right/image_raw      + /stereo/right/camera_info
-> approximate-time pair with maximum skew
-> stereo rectification
-> /stereo/left/image_rect + /stereo/right/image_rect
-> /stereo/disparity + /stereo/points2
```

Calibration records each camera's intrinsics, the fixed left/right extrinsic,
rectification maps, and reprojection matrix. Acquisition is stop-and-look: the
robot or bench rig is stationary, vibration settles, and a short burst of
timestamp-checked pairs is processed. Excessive skew, invalid disparity, or
high reprojection error invalidates the observation.

Before Objective 5, the bracket may be fixed on a bench and has a static
planning-frame extrinsic. Objective 5 remounts it on one moving robot link,
requires stereo recalibration plus hand-eye calibration, and resolves
`base -> end_effector -> stereo_reference` at the image timestamp. Reusing the
old bench extrinsic or the latest TF is invalid.

### Perception and Selection

| Responsibility | Source / owner | Status |
| --- | --- | --- |
| deterministic selected pose | fixed pose publisher | implemented |
| geometric selected pose | ArUco + current selector | implemented (3.1) |
| semantic object candidates | instance mask + stereo points | pure builder/contract implemented; ROS/live stereo pending (3.2) |
| discrete intent | simulated input, then STM32 bridge | simulated source implemented (4.1); STM32 planned (3.5) |
| bounded search command | simulated input, then STM32 bridge | simulated contract/controller implemented (4.3); STM32 source planned (3.5) |
| target acquisition and final pose | generalized target selector | N-frame core implemented; ROS intent/lock/pose integration pending |

Objective 3.2 is bounded to closed-set instance segmentation plus
mask-filtered passive-stereo localization:

```text
approximately synchronized rectified left/right images
-> disparity + PointCloud2 in the stereo reference frame
left rectified image
-> class + confidence + instance mask + temporary track ID
instance mask ∩ valid stereo points
-> reject invalid disparity, outliers, and tabletop-plane points
-> robust metric object position
-> fixed/class-specific grasp offset, approach height, and orientation
-> timestamped /object_candidates
```

The 2026-07-30 software checkpoint implements the source-independent middle of
this flow without pretending that synthetic aligned XYZ is a camera result:

- `markerless_object_perception` converts model masks/tracks plus an aligned
  `HxWx3` XYZ array into robust candidate points and explicit rejection
  diagnostics. Its Ultralytics adapter and laptop-camera demo verify real 2D
  inference/tracking only.
- `assistive_interfaces/ObjectCandidate{,Array}` carries source time/frame,
  validity, pair skew, temporary track ID, class/localization confidence, and
  the robust 3D reference point. It does not carry a grasp pose or mask.
- `target_selector` has a pure gate for the frozen classes `bottle`, `cup`,
  `cell_phone`, and `medicine_box`. It rejects stale/non-increasing/cross-frame
  histories, long frame gaps, low confidence, and whole-window position span;
  stable outputs are ordered by track ID.

The next boundary is the ROS adapter/publisher for `/object_candidates` using
synthetic XYZ. Real `PointCloud2` alignment, class-specific grasp templates,
selected-track lock, last-seen/watchdog handling, intent cycling, TF at the
source stamp, and final pose publication remain planned.

The instance model runs on the host PC or edge-Linux computer. It supplies a
2D class and mask, not depth. Stereo correspondence supplies metric points only
where disparity is valid; the tabletop is a support and outlier constraint,
not the primary depth source. A point cloud does not automatically provide a
reliable grasp orientation, so the MVP keeps a fixed or class-specific grasp
template and does not claim full 6-DoF object-pose recovery.

Stop-and-look applies to target acquisition: motion stops, the bracket settles,
and only a fresh burst of pairs inside the configured skew/reprojection limits
may contribute candidates. Each candidate preserves the source-pair time,
stereo frame, class, confidence, temporary track identity, and localization
quality. An empty array or explicit invalid observation reports that no usable
candidate exists; silence is not treated as a fresh negative observation.

The generalized selector owns candidate identity, minimum confidence, bounded
position jitter, N-frame stability, selected-track lock, and last-seen time.
Its pure multi-track stability gate is implemented; the ROS subscriber,
selected-track lock/last-seen watchdog, intent combination, source-time TF, and
pose publisher are not. Once integrated, it combines stable candidates with
`/assistive_intent` and is the sole publisher of `/target_object_pose` for the
markerless path.
Candidate streams are volatile. `/target_object_pose` may remain reliable and
retained only because every consumer checks the preserved source timestamp
against a configured maximum age. Retention is not freshness.

The current `PoseArray` marker interface cannot prove the same marker ID across
frames because IDs are discarded. Objective 3.1 therefore keeps its verified
single-marker assumption unless a later compatibility adapter is added.

### Safety Observation

Hand-position observation is required to complete the vision-assisted Obj4
scope. Objective 4.1 consumes a simulated `/hand_observation`; its state
machine and fail-closed response are implemented without a camera model.
Objective 4.2 is active and reuses the shared stereo foundation:

```text
rectified left/right images
-> hand keypoints in both views
-> cross-view association and epipolar checks
-> triangulation + reprojection validation
-> robust 3D palm/hand point
-> configured 3D delivery-volume test
-> timestamped /hand_observation
```

The no-hardware portion is implemented in `stereo_hand_observer`: synthetic
and rectified-image inputs exercise detector association, triangulation,
epipolar/reprojection limits, the configured delivery volume, N-frame
stability, exact source stamping, explicit invalid results, and an unpaired
stream watchdog. Controller and image-topic refusal integration are verified
with deterministic injected detections. The actual MediaPipe Tasks frontend
also passed an official-model/official-single-hand-image smoke test and returned
all 21 landmarks. Completing Objective 4.2 still requires live dual-camera
runtime and a rigid calibrated fixed bench stereo pair, then measuring
timestamp skew and working-range 3D error. Eye-in-hand mounting and the
resulting stereo/hand-eye recalibration remain Objective 5.

The observation reports valid/no-hand status explicitly and carries source
time, frame, confidence, pair skew, reprojection quality, and the 3D point when
valid. The same valid hand-in-volume condition must remain stable for N frames.
Missing keypoints in either view, stale or excessive-skew pairs, failed
association, high reprojection error, low confidence, or instability all
invalidate the cue and block release. Hand instance segmentation is optional
future robustness work, not an Objective 4.2 dependency.

The observer owns measurement validity; the handoff controller owns the
fail-closed response. The current Phase-0 controller permits `READY -> RELEASE`
only with a fresh, stable observation plus the required user confirmation; the
full planned state path later names this transition
`HANDOFF_READY -> RELEASE`.
Stereo triangulation estimates approximate 3D hand position, but it is not
safety-rated separation monitoring. It does not replace collision checking,
force sensing, speed limits, an emergency stop, or a formal hardware safety
process. Objective 5 still delivers to a fixed zone rather than chasing a
moving hand.

### Reaching Coordinator

The implemented Objective 1/2 coordinator consumes a `PoseStamped`, runs one
reach/return sequence, and logs failures. That verified node remains the
Objective 1/2 regression path. Objective 4.1 adds a separate source-neutral
Phase-0 handoff controller while preserving the target-pose seam; its motion,
hold, and release actions are simulated rather than real MoveIt/gripper goals.

The current controller consumes target pose, intent, and hand observation. The
integrated controller evolves to consume five independent kinds of input:

```text
/target_object_pose + target status   selected target and freshness
/assistive_intent                     NEXT_TARGET / CONFIRM / ABORT events
/assistive_view_control               bounded search-angle command
/hand_observation                     3D delivery-volume cue and validity
robot-backend result/status           motion, cancellation, and held-object state
```

It rejects an over-age retained target before approach, owns state timeouts and
the response to stale target/hand/view inputs, and never treats message receipt
as proof of freshness. Goal construction remains backend-specific; the state
machine decides when a goal may be sent, cancelled, preempted, or followed by a
safe return. `ABORT` has global priority over motion smoothing and normal state
transitions.

Proportional view commands are not general teleoperation. They are accepted
only in `TARGET_SEARCH` or `HANDOFF_SEARCH`, control one designated viewing
joint/axis, and choose a bounded target angle rather than speed. A newer fresh
command smoothly preempts the old target; watchdog expiry holds position. The
controller must confirm that search motion has stopped before accepting a new
stop-and-look localization burst.

### Motion Backend

The robot backend sits behind a source-independent command abstraction:

```text
robot command abstraction
-> backend A: MoveIt 2 Panda simulation (Objective 1)
-> backend B: LeRobot SO-ARM101 real arm, planned/scripted (Objective 5)
-> backend C: learned ACT policy on SO-ARM101 (Objective 6, Phase 3)

Phase 3 training/evaluation environment
-> MuJoCo: repeated seeded rollouts for backend C
-> LeRobot SO-ARM101: real-arm subset validation
```

Backend A is the implemented Panda MoveIt path. It uses the verified
`moveit_msgs/action/MoveGroup` interface on `/move_action`; the planning group,
planning frame, end-effector frame, velocity/acceleration scaling, and
plan-only versus plan-and-execute mode remain explicit.

Backend B controls the real SO-ARM101 (5-DOF arm + gripper, 6 Feetech servos)
through LeRobot, optionally bridged from ROS 2. The 5-DOF arm cannot realize an
arbitrary 6-DoF end-effector pose, so the backend may use reachable
joint-space goals or constrained task-space commands while preserving the
upstream selected-target contract.

Objective 5 owns the physical eye-in-hand conversion. Both ordinary USB
cameras are fixed in one rigid bracket on one moving robot link; mounting is
followed by stereo recalibration and a measured
`end_effector -> stereo_reference` hand-eye transform. Every observation is
transformed through `base -> end_effector -> stereo_reference` at its source
timestamp. Reusing the bench extrinsic or looking up only the latest transform
is invalid.

Objective 5 also owns the deterministic stop-and-look retrieval sequence:

```text
PREGRASP -> stop and settle
-> REOBSERVE -> acquire a fresh valid stereo burst
-> REFINE -> compute and bound one target correction
-> GRASP -> verify held object
-> LIFT_CLEAR
```

Failed/stale reobservation, a correction outside configured bounds, exhausted
retry count, or failed grasp verification prevents `GRASP`/`LIFT_CLEAR` and
enters the defined retry, abort, or safe-return path. `HANDOFF_SEARCH` is
forbidden until `holding_object=true`; loaded search uses separate, tighter
angle and speed limits. Physical delivery remains to a fixed zone.

Backend C (Phase 3) is an optional learned ACT backend evaluated only after the
deterministic Objective 5 baseline is calibrated and measured. It does not own
stereo calibration, hand-eye calibration, or stop-and-look refinement. The
Phase 3 research studies a **human-intervention layer** on top of backend C — a
constrained EMG channel (abort / confirm / gate) that catches policy failures
and flags failure-adjacent states for correction-data collection. The
intervention channel is deliberately low-bandwidth (the same shared-autonomy
thesis as the intent layer); its cost is the object of study. See
`docs/proposal.md` Phase 3 and `TODO.md` P3.2.

MuJoCo is the selected Phase 3 simulation and learning-evaluation backend for
backend C. A thin adapter preserves the policy observation/action contract and
reproducible scenario seeds without changing the source-independent ROS 2
interfaces. MuJoCo must first reproduce joint limits, action conventions,
timing, and deterministic trajectory playback. It does not replace LeRobot
real-arm execution, MoveIt integration, calibration, or physical validation.

Calibrated geometry and `PREGRASP -> REOBSERVE -> REFINE -> GRASP` remain the
measured Objective 5 baseline. Backend C may later study bounded residuals or
failure recovery, but it must not be used to hide systematic stereo, hand-eye,
or robot-calibration bias.

### RViz Visualization

Objective 1 component. It should display the simulated arm, target pose, planned
trajectory, and reach-to-home execution.

### Safety-Aware Handoff State Machine

Objective 4 defines the source-independent control and safety contract. Its
simulation milestones exercise the same target, intent, view-control, and hand
interfaces that later drive Objective 5. The full nominal state path is:

```text
HOME/IDLE
├─ fresh stable target + CONFIRM -> APPROACH
└─ no fresh target -> TARGET_SEARCH -> target lock + CONFIRM -> APPROACH

APPROACH
-> PREGRASP -> REOBSERVE -> REFINE -> GRASP -> LIFT_CLEAR   (physical in Obj5)
-> HANDOFF_SEARCH -> stable hand -> HANDOFF_READY
-> RELEASE -> RETREAT -> RETURN_HOME
```

Objective 4.1 has implemented the core timeout, cancellation, failure,
target-age, confirmation, simulated hold/release, and return behavior.
Objective 4.2 is active; its live adapter can replace the simulated hand source
with a stereo-triangulated, quality-checked, N-frame-stable 3D observation while
preserving the fail-closed controller contract. Physical deployment and bench
measurement remain. Objective 4.3 implements the two bounded search states,
simulated view commands, and simulated view motion. Physical grasping,
held-object verification, loaded motion, and the
`PREGRASP -> REOBSERVE -> REFINE -> GRASP` subsequence remain Objective 5
responsibilities.

`TARGET_SEARCH` is allowed only before target lock with an empty/simulated
empty gripper. It accepts fresh `/assistive_view_control`, advances one
simulated view degree of freedom toward a bounded target angle at fixed low
nominal speed, and exits only after search motion is cancelled and fully
stopped and a new fresh target is acquired. The eventual selector/tracker, not
the search controller, owns N-frame target stability.

`HANDOFF_SEARCH` is a different loaded condition. It is allowed only after a
simulated or verified grasp/lift equivalent and requires
`holding_object=true`; Phase 0 sets this flag after its simulated approach
action. It uses independent, tighter angle and speed limits.
Stable hand acquisition cancels search, waits for a confirmed stop, and then
uses a new fresh stereo burst before entering `HANDOFF_READY`.

Only the search states accept proportional view commands. Activation selects
angle, not speed; configured relative travel is capped at 45 degrees and by
absolute joint/collision limits. Two search goals may never execute
concurrently. A newer command smoothly preempts the old goal, stale command
holds position, and search timeout follows the defined hold/return/failure
policy. `ABORT` bypasses smoothing, cancels active motion, and has global
priority.

`HANDOFF_READY -> RELEASE` additionally requires a fresh, confident,
N-frame-stable hand point inside the configured 3D delivery volume plus the
required `CONFIRM`. Missing or invalid hand input never permits release.
Objective 4 release is simulated; Objective 5 gripper release occurs only at a
fixed delivery zone under the real-hardware safety policy.

Isaac Lab / Isaac Sim RL is cut, not deferred. Conditional Phase 3 uses MuJoCo
for repeatable learning experiments and LeRobot for real-arm validation. See
`docs/proposal.md` §5.

## Interface Principles

- Keep ROS 2 packages under `src/`.
- Keep frame names and pose units explicit.
- Preserve the fixed-pose and Objective 3.1 ArUco paths as reproducible
  regression/fallback sources.
- Keep perception candidates, discrete intent, proportional view control,
  final selected pose, target status, and hand observation as separate
  interfaces. EMG never fabricates a Pose.
- Only the active selector/pose provider may publish `/target_object_pose`;
  preserved alternatives are selected by launch/configuration rather than
  competing on the same topic.
- Preserve camera/source timestamps through approximate pairing, stereo,
  candidate tracking, TF, and target publication. Retention is not freshness,
  and eye-in-hand transforms are resolved at the observation time.
- Treat two ordinary USB cameras as stereo only after rigid mounting,
  calibration/rectification, pair-skew checks, and measured working-range
  error. Do not call them RGB-D or hardware synchronized.
- Use stop-and-look acquisition for metric localization: halt, settle, reject
  old/excessive-skew pairs, then acquire a fresh stable burst.
- Keep candidate and observation streams volatile and explicit about empty or
  invalid data. Keep command streams volatile; do not replay old intent or view
  commands. A retained target is usable only while within its age contract.
- The selector/tracker owns candidate identity, confidence/jitter checks,
  N-frame target stability, lock, and last-seen time. The controller owns the
  state-machine response to stale/invalid inputs.
- Accept proportional view commands only in `TARGET_SEARCH` or
  `HANDOFF_SEARCH`. Map activation to one bounded target angle, not speed; use
  fixed low speed, acceleration limits, preemption, and a stale-command hold.
- Require `holding_object=true` and stricter loaded limits before
  `HANDOFF_SEARCH`. Two search goals must never execute concurrently.
- Give `ABORT` global priority over smoothing, search, and nominal transitions.
- Require a fresh, confident, N-frame-stable 3D hand cue for release, while
  documenting that it is not safety-rated separation monitoring.
- Keep physical handoff at a fixed delivery zone; reject unsupported or
  unsafe requests instead of guessing.
- Fix systematic stereo, hand-eye, and robot calibration errors before any
  learned residual or policy is evaluated.
- Reject unsupported requests instead of guessing.
- Do not execute motion after a planning failure.
- Log state transitions and failure causes.
- Keep target age, pair skew, reprojection quality, search angle/speed,
  correction magnitude/retries, hand stability, and delivery-distance limits
  configurable.
- Add tests as each behavioral layer is introduced.

## Planned Package Boundaries

Package names may change as implementation details become clear. The intended
responsibilities are:

| Responsibility | Current status |
| --- | --- |
| `object1_demo`: Objective 1 reaching orchestration | implemented (MoveIt plan-and-execute) |
| robot description and simulated arm setup | integrated (`moveit_resources_panda_*`) |
| MoveIt 2 configuration | integrated (`moveit_resources_panda_moveit_config`) |
| fixed target-pose publisher | implemented and runtime-verified |
| unified `/target_object_pose` interface | implemented and runtime-verified |
| ArUco pose baseline | implemented (`marker_pose_provider` + `target_selector`), accuracy quantified (3.1) |
| source-neutral handoff controller | implemented (Objectives 4.1/4.3 Phase-0; simulated actions plus target/intent/hand/view inputs and two bounded search states) |
| shared stereo acquisition/rectification | Objective 4.2 active; final validation uses two namespaced USB drivers, approximate sync, calibration, and stop-and-look |
| stereo hand-observation node | Objective 4.2 active; synthetic harness, live rectified-image/CameraInfo sync, reusable MediaPipe Tasks 21-landmark detector, representative-point adapter, quality/volume/stability gates, exact stamps, and watchdog implemented; real still-image smoke verified, live dual-camera and bench validation pending |
| bounded active-view search controller | implemented as Objective 4.3 Phase-0 simulated motion; physical view joint deferred to Objective 5 |
| simulated target/intent/hand providers | implemented (Objective 4.1) |
| simulated view provider | implemented (Objective 4.3) |
| markerless object perception | Objective 3.2 active; pure mask/aligned-XYZ localization, YOLO adapter, and mono smoke demo implemented; ROS/live stereo pending |
| generalized target selector/tracker | Objective 3.2/3.5 integration; multi-track N-frame gate implemented, intent/lock/status/pose wiring pending |
| shared ROS interfaces | `AssistiveIntent`, `HandObservation`, `ViewControlCommand`, `ObjectCandidate`, and `ObjectCandidateArray` implemented; target-status contract remains planned |
| STM32 EMG firmware | Objective 3.5 (planned STM32CubeIDE C/C++; ADC/DMA, DSP/features, inference, packet producer) |
| PC EMG tooling | Objective 3.5 (planned Python capture/plot/train/quantize/replay/golden vectors) |
| EMG USB/UART ROS bridge | Objective 3.5 MVP Python/`rclpy`; optional measured `rclcpp` receiver/parser/ring-buffer rewrite in Phase 3 |
| SO-ARM101 LeRobot backend (backend B) | Objective 5 (Phase 2; real-arm commands, cancellation, gripper/held-object status) |
| eye-in-hand calibration and deterministic visual refinement | Objective 5 (stereo remount/recalibration plus `PREGRASP -> REOBSERVE -> REFINE -> GRASP`) |
| learned ACT policy + EMG intervention layer (backend C) | Objective 6 (Phase 3, PhD research) |
| MuJoCo environment + thin policy adapter | Objective 6 (Phase 3; repeated seeded simulation and learning evaluation, not real-arm validation) |
| one measured rclcpp rewrite (reaching coordinator default; EMG bridge alternative), MuJoCo evaluation, LeRobot imitation learning | Phase 3 (conditional) |
| evaluation tooling | incremental scripts or package |

Avoid creating these packages until their milestone begins and their interfaces
are understood.
