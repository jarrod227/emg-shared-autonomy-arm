"""Publish simulated source-independent Objective 4.3 view commands."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from assistive_interfaces.msg import ViewControlCommand


class SimViewControlPublisher(Node):
    """Continuously publish a parameter-selected LEFT, RIGHT, or HOLD."""

    _DIRECTIONS = {
        "HOLD": ViewControlCommand.HOLD,
        "LEFT": ViewControlCommand.LEFT,
        "RIGHT": ViewControlCommand.RIGHT,
    }

    def __init__(self) -> None:
        super().__init__("sim_view_control_publisher")
        self.declare_parameter("direction", "RIGHT")
        self.declare_parameter("activation", 0.6)
        self.declare_parameter("confidence", 1.0)
        self.declare_parameter("signal_quality", 1.0)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("source_frame", "sim_view_control")

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(
                f"publish_rate_hz must be finite and > 0, got {rate_hz}"
            )
        self._read_command_values()

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            ViewControlCommand, "/assistive_view_control", qos
        )
        self._sequence = 0
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_command)
        self.get_logger().info(
            "publishing volatile simulated view commands on "
            "/assistive_view_control"
        )

    def _read_command_values(self) -> tuple[int, float, float, float]:
        direction_name = str(
            self.get_parameter("direction").value
        ).upper()
        if direction_name not in self._DIRECTIONS:
            raise ValueError(
                "direction must be HOLD, LEFT, or RIGHT, got "
                f"{direction_name}"
            )
        activation = self._unit_parameter("activation")
        confidence = self._unit_parameter("confidence")
        signal_quality = self._unit_parameter("signal_quality")
        direction = self._DIRECTIONS[direction_name]
        if direction == ViewControlCommand.HOLD:
            activation = 0.0
        return direction, activation, confidence, signal_quality

    def _unit_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        return value

    def _publish_command(self) -> None:
        try:
            direction, activation, confidence, quality = (
                self._read_command_values()
            )
        except ValueError as exc:
            self.get_logger().warning(f"view command not published: {exc}")
            return

        msg = ViewControlCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(
            self.get_parameter("source_frame").value
        )
        msg.direction = direction
        msg.activation = activation
        msg.confidence = confidence
        msg.signal_quality = quality
        msg.sequence = self._sequence
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        self._publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = SimViewControlPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
