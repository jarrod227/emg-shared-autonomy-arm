# object1_demo

Minimal ROS 2 Python package for the first assistive robot milestone.

## Current milestone

Start the node and prepare the package for MoveIt 2 integration:

```text
fixed object1 pose
-> pre-grasp pose
-> plan and execute
-> return home
```

The first implementation intentionally excludes the camera, gripper, Gazebo,
language input, and handoff logic.

## Build

From the `assistive_robot_ws` directory in a ROS 2 environment:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch object1_demo object1_demo.launch.py
```
