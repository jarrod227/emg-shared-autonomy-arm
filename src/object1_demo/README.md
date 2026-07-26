# object1_demo

ROS 2 Python package implementing the verified Objective 1/2 reaching baseline.

## Package role

The package provides a deterministic fixed-pose source and a reaching
coordinator. The coordinator consumes one retained
`geometry_msgs/PoseStamped` on `/target_object_pose`, sends MoveIt
`MoveGroup` goals on `/move_action`, and returns the Panda to its ready pose:

```text
fixed object1 pose
-> pre-grasp pose
-> plan and execute
-> return home
```

The fixed provider and the Objective 3.1 ArUco pipeline have both driven this
coordinator without changing its target callback or MoveIt goal construction.
The current coordinator is still a one-shot
`IDLE -> REACHING -> RETURNING -> DONE` baseline; the Objective 4 handoff state
machine is not implemented yet.

Camera drivers, ArUco detection, future Objective 3.2 instance segmentation,
STM32 EMG intent, and future hand observations belong to other packages. They
must converge through the source-independent target/intent contracts rather
than adding perception code here.

## Build

From the `assistive_robot_ws` directory in a ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch object1_demo object1_demo.launch.py
```

The launch file starts `fixed_pose_publisher` and `move_to_object`. For the
ArUco-driven path, use the reproduction commands in
[`docs/project_context.md`](../../docs/project_context.md) instead.
