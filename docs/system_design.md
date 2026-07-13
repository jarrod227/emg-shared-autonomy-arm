# System Design

## Current Verified Status (2026-07-13)

This section supersedes older scaffold descriptions below. The current MoveToObjectNode is a reaching coordinator: it sends a Cartesian object1 pose goal for panda_link8 in world to MoveIt, then sends a joint-space ready home goal. The fixed pose is configured inside the coordinator; a separate topic publisher is deferred to Objective 2. RViz execution is verified, while a dedicated visual target marker remains pending.


## Design Status

This document distinguishes the current ROS 2 scaffold, the Objective 1 target,
and the longer-term architecture. Only `src/object1_demo` exists today. The
remaining components should be added incrementally.

## Current Workspace Layout

```text
assistive_robot_ws/
├── docs/
├── src/
│   └── object1_demo/
├── AGENTS.md
└── TODO.md
```

ROS 2 packages live under `src/`. Project-level planning and design documents
live at the workspace root.

## Current Runtime

```text
object1_demo.launch.py
-> move_to_object executable
-> MoveToObjectNode
-> startup log
```

`MoveToObjectNode` is currently a scaffold. It does not yet define a pose,
command motion, or call a planning interface.

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

```text
request input
-> target selector
-> object-pose provider
-> reaching coordinator
-> motion backend
-> handoff state machine
-> evaluation logger
```

### Request Input and Target Selector

Roadmap component. Begin with a small, constrained vocabulary. A future
vision-language model can be evaluated only after the basic interface works.

### Object-Pose Provider

The fixed target-pose publisher is an Objective 1 deliverable. Later objectives
should preserve that deterministic test path while adding new providers:

```text
fixed-pose provider
-> marker-based provider
-> future perception provider
```

### Reaching Coordinator

Objective 1 component. It should coordinate a target pose, a reach attempt, and
a return-home action while logging failures clearly.

### Motion Backend

Objective 1 component. It should use MoveIt 2 with a supported simulated arm.
The robot description, planning group, planning frame, and end-effector frame
must be confirmed before implementation.

The backend should sit behind a robot-command abstraction so the same reaching
coordinator can target either platform:

```text
robot command abstraction
-> backend A: MoveIt 2 Panda simulation (Objective 1)
-> backend B: LeRobot SO-ARM101 real arm (Objective 5)
```

Backend B controls a real SO-ARM101 (5-DOF arm + gripper, 6 Feetech servos)
through LeRobot, optionally bridged from a ROS 2 command node. The 5-DOF arm
cannot reach arbitrary 6-DOF poses, so joint-space goals are often more reliable
than full Cartesian pose goals.

### RViz Visualization

Objective 1 component. It should display the simulated arm, target pose, planned
trajectory, and reach-to-home execution.

### Safety-Aware Handoff State Machine

Roadmap component. Add only after reaching is reliable. Planned states include:

```text
idle
-> approach
-> ready_for_handoff
-> handoff
-> return_home
```

Timeout, cancellation, failure, and user-confirmation transitions must be
explicit.

### Optional Isaac Lab Policy

Optional Phase 2 research component. Isaac Lab may be used to evaluate a narrow
safety-aware reaching policy after the classical ROS 2 baseline is stable.
Isaac Sim and Isaac Lab are not currently configured in this workspace.

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
| fixed target-pose publisher | Objective 1 node (pending) |
| unified `/target_object_pose` interface | Objective 2 |
| marker-based object-pose provider | Objective 3 |
| constrained target selector | Objective 3 |
| delivery / handoff controller | Objective 4 |
| SO-ARM101 LeRobot backend (backend B) | Objective 5 |
| LeRobot imitation-learning policy | Objective 6 (optional) |
| evaluation tooling | incremental scripts or package |

Avoid creating these packages until their milestone begins and their interfaces
are understood.
