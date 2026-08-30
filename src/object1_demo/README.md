# object1_demo

ROS 2 Python package implementing the verified Objective 1/2 reaching baseline.

## Package role

The package provides a deterministic fixed-pose source and a reaching
coordinator. The coordinator consumes one retained
`geometry_msgs/PoseStamped` on `/target_object_pose`, sends MoveIt
`MoveGroup` goals on `/move_action`, and returns the Panda to its ready pose:

```text
receive fixed object1 reach target
-> plan and execute the reach
-> return home
```

The fixed provider and the Objective 3.1 ArUco pipeline have both driven this
coordinator without changing its target callback or MoveIt goal construction.
The current coordinator remains the one-shot
`IDLE -> REACHING -> RETURNING -> DONE` regression path. Objective 4.1 is
implemented separately in the `assistive_handoff` package as a completed
Phase-0 simulated handoff controller, so this package does not command a
gripper or perform a physical grasp.

Camera drivers, ArUco detection, future Objective 3.2 instance segmentation,
STM32 EMG intent, and Objective 4.2 stereo hand observations belong to other
packages. They must converge through the source-independent target, intent,
and hand-observation contracts rather than adding perception code here.

## Build

From the `assistive_robot_ws` directory in a ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch object1_demo object1_demo.launch.py
```

The launch file starts `fixed_pose_publisher` and `move_to_object`. For the
ArUco-driven path -- the frozen fallback, not the path under development --
the run sequence is in the module docstring of
[`selector_node.py`](../target_selector/target_selector/selector_node.py).
