# Biosignal-Driven Shared-Autonomy Assistive Manipulation

A ROS 2 workspace and STM32 firmware for steering an assistive robot arm from
three channels of surface EMG. The arm plans and executes; the wearer supplies
a few bits per second of intent — three discrete commands and a proportional
view-steering channel — and the boundary between them is where the work is.

Everything claimed below was measured, and the recordings it was measured from
are in `datasets/`. Where a result was withdrawn after measurement, the logs
say so.

## What runs today

| | |
| --- | --- |
| Discrete intent on the MCU | `REST` / `NEXT_TARGET` / `CONFIRM` / `ABORT` at 20 Hz, filter → features → Q18 LDA → activation threshold → event gate, entirely on an STM32F103C8T6 (40% flash, 58% RAM) |
| Proportional view control | wrist extension and ulnar deviation steer one bounded axis; contraction strength selects **rate** |
| Perception | ArUco baseline, and markerless instance segmentation with passive-stereo metric localization |
| Handoff | state machine with bounded active-view search, stale-input gating, and global `ABORT` |
| Arm motion | MoveIt on a simulated Panda, or a timer backend for tests; goal construction shared with the Objective 1 reaching path |

The whole chain runs end to end from one launch: EMG intent and steering →
candidate segmentation and stereo localization → target lock → handoff state
machine → MoveIt. Real-arm validation on an SO-ARM101 is the next objective
and has not started.

## Measured, on a wearer

From the closed-loop session of 2026-08-28 (`docs/objective35_classifier_log.md`):

- Contraction strength selects speed to within **1% of ideal** in four of five
  activation bands.
- **2.4%** of motion went against the gesture that commanded it.
- **Zero** false triggers and **zero** dropped packets in ten minutes of
  ordinary activity — typing, a cup, a phone, posture changes.
- Cross-donning classification, holding out one electrode application at a
  time: **93.3–95.1%** window accuracy over five donnings.
- A published intent reaches the state it causes in **2.4 ms** (median of 40
  cycles). The wearer feels that plus two parts measured separately: the event
  gate's 650 ms for `ABORT`, and a 26 ms median serial receipt. So an `ABORT`
  arrives in about **680 ms**, of which the gate is 96%.

## What does not work

Stated here rather than in a footnote, because both are properties of the
result and not open bugs.

- **Light contractions do not register at all.** The usable range starts at
  moderate effort. This is not the activation threshold: of the
  direction-gesture windows below it, 0 of 21 are recognised as the gesture,
  so lowering it admits windows the classifier calls rest. The minimum speed
  is set by `target_search_nominal_speed`, not by the threshold.
- **One subject, five donnings, three of them within two days.** The
  cross-donning numbers above do not carry to another wearer, and adding a
  fifth donning improved internal folds while *degrading* transfer to
  recordings from a fortnight earlier.

## Running the whole chain without hardware

```bash
ros2 launch assistive_handoff integrated_simulation.launch.py
# in another terminal, the wearer's inputs:
ros2 run assistive_handoff sim_intent_publisher      # n / c / a
```

Add `appears_after_sec:=8.0` to hold the object back and exercise the search
path: the controller sweeps, stops itself when the observation arrives, locks
on a second one after the stop, and one CONFIRM takes it to APPROACH.

Add `-p motion_backend:=moveit` to the controller to plan and execute on a
Panda instead of simulating the motion with timers, with the MoveIt stack
running separately.

## Layout

```
src/          ROS 2 packages: perception, handoff state machine, EMG bridge,
              and the MoveIt goal construction they share
firmware/     STM32 firmware, host tools, and the wire protocol
datasets/     every recording the numbers above come from
docs/         design, proposal, and the logs of how each result was reached
```

Start with `docs/system_design.md` for the architecture, and
`docs/objective35_classifier_log.md` for how the EMG side actually behaves —
it is written as a record of what failed and why, which is more useful than
the summary.

## Build and test

```bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash

python3 -m pytest src/                  # ROS packages
python3 -m pytest firmware/tools -q     # host tools and protocol mirrors
make -C firmware/test check              # C unit tests and golden vectors
```

`make check` regenerates the golden fixtures the host tools compare against.
They are not committed on purpose: a committed fixture lets the Python mirror
pass against bytes some earlier C produced.

## Running a session

The board must be flashed first (`firmware/README.md` covers the hardware).
Then, in order:

```bash
# 1. Calibrate this donning. Refuses to send a calibration it does not trust,
#    and reports what the deployed model calls each gesture's own trials.
python3 firmware/tools/emg_calibrate.py --port /dev/ttyACM0

# 2. Bridge the board onto ROS topics, with that calibration
ros2 run emg_intent_bridge emg_intent_bridge --ros-args \
  -p port:=/dev/ttyACM0 -p calibration_file:=datasets/emg_calibration/<file>.json

# 3. The handoff controller, with proportional steering enabled
ros2 run assistive_handoff handoff_controller --ros-args \
  -p proportional_search_available:=true
```

Read the calibration's own verdict before trusting a session. A donning can
pass its separation check and still be unusable — the classification report it
prints is what tells you, in ninety seconds rather than after a wasted run.

## Collecting training data

```bash
python3 firmware/tools/emg_guided_capture.py --donning A6
python3 firmware/tools/emg_train_lda.py datasets/emg --require-donning \
  --c-output firmware/src/emg_classifier_model.h
```

`--donning` is required, and folds are held out by donning rather than by
session. Holding out a session returns its own electrode application to the
training set, and the accuracy that comes out measures repeatability rather
than transfer — a model accepted that way identified `NEXT_TARGET` on 2% of
windows on one held-out donning.

## Status and plans

`TODO.md` is the roadmap and carries the current progress. `docs/proposal.md`
is the personal build plan behind it. Citations are tracked in
`docs/literature_ledger.md`, which records for each source the smallest claim
it actually supports and what it does not establish for this project.
