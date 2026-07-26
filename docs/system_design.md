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
architecture. The execution layer (Objectives 1–2) and the perception layer
(Objective 3) exist today; the handoff state machine (Objective 4) and intent
layer (Objective 3.5) are the next increments.

## Current Workspace Layout

```text
assistive_robot_ws/
├── docs/
├── src/
│   ├── object1_demo/          reaching coordinator + fixed-pose provider
│   ├── marker_pose_provider/  ArUco detection + camera-frame PnP
│   └── target_selector/       selection, grasp offset, TF to world
├── AGENTS.md
└── TODO.md
```

ROS 2 packages live under `src/`. Project-level planning and design documents
live at the workspace root.

## Current Runtime

```text
marker_demo.launch.py                    (or: fixed_pose_publisher alone)
├── v4l2_camera driver
│   ├── /image_raw
│   └── /camera_info  (calibrated intrinsics from config/camera_info.yaml)
└── marker_node
    └── /detected_markers (PoseArray, camera frame, all detections)

extrinsics.launch.py
└── static TF world -> camera            (SIMULATION PLACEHOLDER extrinsic)

selector_node
├── subscribes /detected_markers
├── applies grasp offset (config/grasp_offsets.yaml), transforms via tf2
└── /target_object_pose (PoseStamped, retained, latched once)

move_to_object
├── subscribes /target_object_pose
├── sends MoveGroup goals on /move_action
└── publishes /object1_target for RViz
```

The coordinator waits for the first valid target pose, executes the existing
IDLE -> REACHING -> RETURNING -> DONE state machine, and ignores additional
target messages during that one-shot run. The selector likewise latches its
first detection — replacing that with N-stable-frames acquisition and a
staleness contract is Objective 4 work.

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

## Long-Term Architecture

Three layers, deliberately separated — each answers a different question.

```text
INTENT LAYER        biosignal provider          -> WHICH object, WHEN to act
                    (EMG)
                              |
                              v
PERCEPTION LAYER    marker localization         -> WHERE that object is now
                    (ArUco / AprilTag)
                              |
                              v
EXECUTION LAYER     /target_object_pose -> reaching coordinator -> motion
                    backend -> safety-aware handoff state machine
                              |
                              v
                    evaluation logger (layered metrics)
```

Language / voice target selection is cut, not deferred — see `docs/proposal.md`
§5. It would occupy the same layer as biosignal intent and is functionally
redundant with it.

### Intent Layer

Objective 3.5 (Phase 1). A biosignal provider decodes a few discrete intents
per second: **1** cycle target among detected candidates, **2** confirm/trigger
approach, **3** release/abort. EMG is the intent modality (sEMG hardware bought
in Phase 0); acquisition is new, but the feature-extraction and SVM pipeline
structure reuses the existing capstone's software. EOG is cut, not deferred —
see `docs/proposal.md` §5.

### Object-Pose Provider

The retained fixed target-pose provider is implemented. Later objectives
preserve that deterministic test path while adding new providers, all
implementing the same contract behind `/target_object_pose`:

| Provider | Role |
| --- | --- |
| fixed pose | deterministic regression test (done) |
| marker | perception — where the object is (Objective 3) |
| biosignal | intent — which object, when (Objective 3.5) |

### Reaching Coordinator

Implemented Objective 1/2 component. It consumes a `PoseStamped`, coordinates
a reach attempt and return-home action, and logs state transitions and failures.

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

`ready` is the handoff/delivery zone. Timeout, cancellation, failure, and
user-confirmation transitions must be explicit; `release` wires to intent 2
(confirm) and intent 3 (release/abort) once the intent layer exists (Phase 1).

Isaac Lab / Isaac Sim RL is cut, not deferred — LeRobot imitation learning
(Phase 3, conditional) covers the same ground on a real arm. See
`docs/proposal.md` §5.

## Interface Principles

- Keep ROS 2 packages under `src/`.
- Keep frame names and pose units explicit.
- Use a fixed-pose provider for reproducible tests.
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
| marker-based object-pose provider | implemented (`marker_pose_provider` + `target_selector`), accuracy quantified |
| delivery / handoff controller | Objective 4 (next) |
| EMG intent provider | Objective 3.5 (Phase 1, MVP completion point) |
| SO-ARM101 LeRobot backend (backend B) | Objective 5 (Phase 2) |
| learned ACT policy + EMG intervention layer (backend C) | Objective 6 (Phase 3, PhD research) |
| rclcpp reaching coordinator, LeRobot imitation learning | Phase 3 (conditional) |
| evaluation tooling | incremental scripts or package |

Avoid creating these packages until their milestone begins and their interfaces
are understood.
