import pytest
import rclpy

from object1_demo.fixed_pose_publisher import FixedPosePublisherNode


@pytest.fixture
def fixed_pose_publisher_node():
    rclpy.init()
    node = FixedPosePublisherNode()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_build_target_pose_uses_object1_parameters(fixed_pose_publisher_node):
    target_pose = fixed_pose_publisher_node.build_target_pose()

    assert target_pose.header.frame_id == "world"
    assert target_pose.pose.position.x == pytest.approx(0.106982)
    assert target_pose.pose.position.y == pytest.approx(0.0)
    assert target_pose.pose.position.z == pytest.approx(1.121022)
    assert target_pose.pose.orientation.x == pytest.approx(0.653269)
    assert target_pose.pose.orientation.y == pytest.approx(-0.270440)
    assert target_pose.pose.orientation.z == pytest.approx(0.653402)
    assert target_pose.pose.orientation.w == pytest.approx(-0.270496)
