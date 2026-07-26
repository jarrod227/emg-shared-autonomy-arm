"""Node-level integration tests for SelectorNode.

No camera needed: the test publishes the static world->camera transform and
fake /detected_markers itself, then checks what lands on /target_object_pose.
"""

import math
import time

import pytest
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from tf2_ros import StaticTransformBroadcaster

from target_selector.selector_node import SelectorNode, select_marker_pose


def make_pose_array(x, y, z, frame_id="camera"):
    msg = PoseArray()
    msg.header.frame_id = frame_id
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.w = 1.0
    msg.poses.append(pose)
    return msg


def spin_until(nodes, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


@pytest.fixture
def ros_graph():
    rclpy.init()
    selector = SelectorNode()
    helper = rclpy.create_node("test_helper")

    # Publish the extrinsic the selector expects: world -> camera,
    # z = 1.42, roll = pi (camera looking straight down).
    broadcaster = StaticTransformBroadcaster(helper)
    tf = TransformStamped()
    tf.header.frame_id = "world"
    tf.child_frame_id = "camera"
    tf.transform.translation.z = 1.42
    tf.transform.rotation.x = math.sin(math.pi / 2)  # roll = pi
    tf.transform.rotation.w = math.cos(math.pi / 2)
    broadcaster.sendTransform(tf)

    markers_pub = helper.create_publisher(PoseArray, "/detected_markers", 10)
    received = []
    helper.create_subscription(
        PoseStamped,
        "/target_object_pose",
        received.append,
        QoSProfile(depth=5, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    )

    yield selector, helper, markers_pub, received

    selector.destroy_node()
    helper.destroy_node()
    rclpy.shutdown()


def test_select_marker_pose_empty_returns_none():
    assert select_marker_pose(PoseArray()) is None


def test_detection_is_transformed_into_world(ros_graph):
    selector, helper, markers_pub, received = ros_graph
    # Let the selector's TF listener pick up the static transform first.
    spin_until([selector, helper], lambda: False, timeout=0.5)

    markers_pub.publish(make_pose_array(0.05, 0.02, 0.30))
    assert spin_until([selector, helper], lambda: received), (
        "no /target_object_pose within timeout"
    )

    target = received[0]
    assert target.header.frame_id == "world"
    # roll=pi, z=1.42: world = (x, -y, 1.42 - z); default grasp offset is
    # identity, so the marker pose passes through unchanged.
    assert target.pose.position.x == pytest.approx(0.05, abs=1e-6)
    assert target.pose.position.y == pytest.approx(-0.02, abs=1e-6)
    assert target.pose.position.z == pytest.approx(1.12, abs=1e-6)


def test_latches_first_detection(ros_graph):
    selector, helper, markers_pub, received = ros_graph
    spin_until([selector, helper], lambda: False, timeout=0.5)

    markers_pub.publish(make_pose_array(0.05, 0.02, 0.30))
    assert spin_until([selector, helper], lambda: received)

    # A different second detection must be ignored (latch-once contract).
    markers_pub.publish(make_pose_array(0.20, -0.10, 0.50))
    spin_until([selector, helper], lambda: False, timeout=1.0)
    assert len(received) == 1
    assert received[0].pose.position.x == pytest.approx(0.05, abs=1e-6)


def test_empty_pose_array_publishes_nothing(ros_graph):
    selector, helper, markers_pub, received = ros_graph
    spin_until([selector, helper], lambda: False, timeout=0.5)

    markers_pub.publish(PoseArray())
    spin_until([selector, helper], lambda: False, timeout=1.0)
    assert received == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
