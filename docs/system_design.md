# System Design

## Current Verified Status (2026-07-26)

This section supersedes older scaffold descriptions below. Two interchangeable
providers feed the retained `PoseStamped` on `/target_object_pose`: the
deterministic `fixed_pose_publisher`, and the marker pipeline
(`marker_pose_provider` detection/PnP -> `target_selector` selection, grasp
offset, tf2 transform to `world`). `MoveToObjectNode` consumes the interface
unchanged either way, sends a Cartesian goal for `panda_link8` to MoveIt, and
returns to the joint-space ready home. Both provider paths are
runtime-verified end to end; camera-frame accuracy is quantified in
`docs/objective3_evaluation.md`.

## Design Status

This document distinguishes the current implementation from the longer-term
architecture. The execution layer (Objectives 1–2) and ArUco perception
baseline (Objective 3.1) exist today. Objective 4 is active and is implemented
as two increments: Objective 4.1 uses the existing pose pipeline plus simulated
intent/hand observation, and Objective 4.2 adds camera hand-keypoint position.
Objective 3.2 then adds markerless perception, and Objective 3.5 adds STM32 EMG
intent.

## Current Workspace Layout

```text
assistive_robot_ws/
├── docs/
├── src/
│   ├── object1_demo/          reaching coordinator + fixed-pose provider
│   ├── marker_pose_provider/  ArUco detection + camera-frame PnP (3.1)
│   └── target_selector/       marker selection, grasp offset, TF (3.1)
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

The coordinator waits for the first valid target pose, executes its existing
IDLE -> REACHING -> RETURNING -> DONE one-shot sequence, and ignores additional
target messages. These are current implementation facts, not the future
Objective 4 state machine.

DDS `TRANSIENT_LOCAL` retention and the application one-shot latch are not
freshness. The selector/tracker will own N-frame stable acquisition and
last-seen tracking; the Objective 4 controller will own the policy response to
old targets or stale safety observations.

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

PLANNED
camera -> instance segmentation -> tabletop localizer -> /object_candidates
STM32 EMG -> USB/UART bridge --------------------------> /assistive_intent
                                                   |             |
                                                   +------v------+
                                                   target selector
                                                         |
                                              /target_object_pose
                                                         |
hand-keypoint monitor -> /hand_observation ------> handoff controller
                                                         |
                                                  MoveIt /move_action
```

Only one selected-pose source is active at a time. The camera driver should
also be launched once; multiple perception nodes may subscribe to the same
`/image_raw` and `/camera_info` streams.

Language / voice target selection is cut, not deferred — see `docs/proposal.md`
§5. It would occupy the same layer as biosignal intent and is functionally
redundant with it.

### Intent Layer

Objective 4 first uses a simulated publisher on the future production
interface. Objective 3.5 later replaces that source with STM32 edge inference:

```text
3-channel ADC/DMA
-> approximately 20–450 Hz DSP
-> 200 ms windows / 50 ms hop (initial target)
-> MAV, RMS, zero crossing, waveform length
-> classical baseline or small quantized model
-> REST / NEXT_TARGET / CONFIRM / ABORT
-> USB CDC or UART packet
-> ROS bridge -> /assistive_intent
```

`REST` produces no event. Intent messages require source timestamp, command,
confidence, and sequence number; they are reliable and volatile, never retained
or replayed. EMG does not publish `/target_object_pose`.

### Perception and Selection

| Responsibility | Source / owner | Status |
| --- | --- | --- |
| deterministic selected pose | fixed pose publisher | implemented |
| geometric selected pose | ArUco + current selector | implemented (3.1) |
| semantic object candidates | markerless perception | planned (3.2) |
| discrete intent | simulated input, then STM32 bridge | planned (4 / 3.5) |
| final selected pose | generalized target selector | planned integration |

Objective 3.2 is bounded to closed-set RGB instance segmentation and
tabletop-constrained localization:

```text
/image_raw + /camera_info
-> instance segmentation
-> class + confidence + mask + temporary instance/track ID
-> selected mask point
-> calibrated camera ray intersected with known tabletop plane
-> metric 3D position
-> fixed/class-specific approach height and orientation
-> timestamped /object_candidates
```

Instance segmentation itself does not produce depth or a graspable 6-DoF pose.
The monocular MVP assumes a known plane and configured object height/template.
RGB-D can later replace the plane assumption without changing downstream
selection. The vision model runs on the host PC or edge-Linux computer.

The generalized selector owns target identity, minimum confidence, bounded
pose jitter, N-frame stability, selected-track lock, and last-seen time. The
source observation timestamp must survive every transform. Candidate messages
are volatile; `/target_object_pose` may remain retained only because consumers
enforce maximum age.

The current `PoseArray` marker interface cannot prove the same marker ID across
frames because IDs are discarded. Objective 3.1 therefore keeps its verified
single-marker assumption unless a later compatibility adapter is added.

### Safety Observation

Hand-position observation is required to complete the vision-assisted Obj4
scope. Objective 4.1 first consumes a simulated `/hand_observation` so the state
machine and failure policy can be tested without a camera model. Objective 4.2
then supplies **2D hand position from hand keypoints**, not hand instance
segmentation. Valid keypoints entering a configured delivery ROI produce a
timestamped observation; the same hand must remain valid for N frames. Missing,
stale, unstable, or low-confidence input must not permit release.

This monocular cue is not safety-rated and does not estimate reliable 3D
human–robot separation. It does not replace collision checks, force sensing,
speed limits, an emergency stop, or a formal hardware safety process.

### Reaching Coordinator

The implemented Objective 1/2 coordinator consumes a `PoseStamped`, runs one
reach/return sequence, and logs failures. Objective 4 will add a source-neutral
handoff controller around or in place of that one-shot orchestration while
preserving the MoveIt action interface.

### Motion Backend

Objective 1 component. It should use MoveIt 2 with a supported simulated arm.
The robot description, planning group, planning frame, and end-effector frame
must be confirmed before implementation.

The backend should sit behind a robot-command abstraction so the same reaching
coordinator can target either platform:

```text
robot command abstraction
-> backend A: MoveIt 2 Panda simulation (Objective 1)
-> backend B: LeRobot SO-ARM101 real arm, planned/scripted (Objective 5)
-> backend C: learned ACT policy on SO-ARM101 (Objective 6, Phase 3)
```

Backend B controls a real SO-ARM101 (5-DOF arm + gripper, 6 Feetech servos)
through LeRobot, optionally bridged from a ROS 2 command node. The 5-DOF arm
cannot reach arbitrary 6-DOF poses, so joint-space goals are often more reliable
than full Cartesian pose goals.

Backend C (Phase 3) replaces the planner with a learned ACT policy. This is
where the two execution ceilings are compared: a planner's ceiling is fixed by
what can be specified, a learned policy's ceiling scales with data. The Phase 3
research studies a **human-intervention layer** on top of backend C — a
constrained EMG channel (abort / confirm / gate) that catches policy failures
and flags failure-adjacent states for correction-data collection. The
intervention channel is deliberately low-bandwidth (the same shared-autonomy
thesis as the intent layer); its cost is the object of study. See
`docs/proposal.md` Phase 3 and `TODO.md` P3.2.

### RViz Visualization

Objective 1 component. It should display the simulated arm, target pose, planned
trajectory, and reach-to-home execution.

### Safety-Aware Handoff State Machine

Objective 4 (Phase 0) component. Planned states:

```text
idle -> approach -> ready -> release -> return_home
```

`ready` is the handoff/delivery zone. Timeout, cancellation, failure, target
freshness, observation freshness, and user-confirmation transitions must be
explicit. During Objective 4, simulated `CONFIRM` and `ABORT` events exercise
the production intent contract. `CONFIRM` may begin approach from `idle` and
request simulated release from `ready`, but the transition is allowed only when
the hand observation is fresh, confident, and stable inside the delivery ROI.
`ABORT` cancels active behavior safely. Real gripper actuation remains
Objective 5 work.

Isaac Lab / Isaac Sim RL is cut, not deferred — LeRobot imitation learning
(Phase 3, conditional) covers the same ground on a real arm. See
`docs/proposal.md` §5.

## Interface Principles

- Keep ROS 2 packages under `src/`.
- Keep frame names and pose units explicit.
- Use a fixed-pose provider for reproducible tests.
- Keep perception candidates, user intent, final selected pose, and safety
  observations as separate interfaces.
- Preserve the source timestamp through localization and TF; retention is not
  freshness.
- Reject unsupported requests instead of guessing.
- Do not execute motion after a planning failure.
- Log state transitions and failure causes.
- Keep speed and handoff-distance limits configurable.
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
| delivery / handoff controller | Objective 4.1 (active; existing pose + simulated intent/hand input) |
| hand-keypoint position observation | Objective 4.2 (required for vision-assisted handoff) |
| markerless object perception | Objective 3.2 (planned host-side package) |
| generalized candidate/intent selector | Objective 3.2/3.5 integration (planned) |
| shared ROS interfaces | planned when message fields are frozen |
| STM32 EMG firmware | Objective 3.5 (planned, non-ROS firmware) |
| EMG USB/UART ROS bridge | Objective 3.5 (planned; intent only) |
| SO-ARM101 LeRobot backend (backend B) | Objective 5 (Phase 2) |
| learned ACT policy + EMG intervention layer (backend C) | Objective 6 (Phase 3, PhD research) |
| rclcpp reaching coordinator, LeRobot imitation learning | Phase 3 (conditional) |
| evaluation tooling | incremental scripts or package |

Avoid creating these packages until their milestone begins and their interfaces
are understood.
