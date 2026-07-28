"""Pure bounded-view policy and deterministic simulated motion.

The ROS controller owns state gating and command freshness. This module owns
only angle mapping and the Phase-0 single-axis motion model, so Objective 5 can
replace the simulator without changing the view-command contract.
"""

from dataclasses import dataclass
import math


MAX_RELATIVE_LIMIT_RAD = math.pi / 4.0
_EPSILON = 1.0e-9


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class SearchProfile:
    """Bounds and motion limits for unloaded or loaded search."""

    center_angle: float
    relative_limit: float
    min_angle: float
    max_angle: float
    nominal_speed: float
    acceleration: float
    deceleration: float

    def __post_init__(self) -> None:
        values = {
            name: _finite(getattr(self, name), name)
            for name in (
                "center_angle",
                "relative_limit",
                "min_angle",
                "max_angle",
                "nominal_speed",
                "acceleration",
                "deceleration",
            )
        }
        if not 0.0 < values["relative_limit"] <= MAX_RELATIVE_LIMIT_RAD:
            raise ValueError(
                "relative_limit must be in (0, pi/4], got "
                f"{values['relative_limit']}"
            )
        if values["min_angle"] >= values["max_angle"]:
            raise ValueError("min_angle must be less than max_angle")
        if not (
            values["min_angle"]
            <= values["center_angle"]
            <= values["max_angle"]
        ):
            raise ValueError("center_angle must be inside absolute angle bounds")
        for name in ("nominal_speed", "acceleration", "deceleration"):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be > 0, got {values[name]}")

    def target_for(self, direction: int, activation: float) -> float:
        """Map -1/+1 and normalized activation to a safe angle."""

        if direction not in (-1, 1):
            raise ValueError(f"direction must be -1 or 1, got {direction}")
        activation = _finite(activation, "activation")
        activation = min(1.0, max(0.0, activation))
        target = (
            self.center_angle
            + float(direction) * self.relative_limit * activation
        )
        return min(self.max_angle, max(self.min_angle, target))


class SimulatedViewMotion:
    """Single-axis simulator with smooth, serialized preemption.

    A replacement target remains pending while the old motion decelerates to
    zero. It becomes active only after the stop, so two simulated goals can
    never execute concurrently.
    """

    def __init__(
        self,
        profile: SearchProfile,
        *,
        initial_angle: float | None = None,
    ) -> None:
        self._profile = profile
        self._position = (
            profile.center_angle
            if initial_angle is None
            else _finite(initial_angle, "initial_angle")
        )
        if not profile.min_angle <= self._position <= profile.max_angle:
            raise ValueError("initial_angle must be inside absolute angle bounds")
        self._velocity = 0.0
        self._active_target: float | None = None
        self._pending_target: float | None = None

    @property
    def position(self) -> float:
        return self._position

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def active_target(self) -> float | None:
        return self._active_target

    @property
    def pending_target(self) -> float | None:
        return self._pending_target

    @property
    def moving(self) -> bool:
        return (
            abs(self._velocity) > _EPSILON
            or self._active_target is not None
            or self._pending_target is not None
        )

    @property
    def goal_count(self) -> int:
        """Number of goals executing now; pending work is not executing."""

        return int(self._active_target is not None)

    def configure(self, profile: SearchProfile) -> None:
        """Select unloaded/loaded limits after the axis has stopped."""

        if self.moving:
            raise RuntimeError("cannot reconfigure while view motion is active")
        if not profile.min_angle <= self._position <= profile.max_angle:
            raise ValueError(
                "current angle is outside the new absolute angle bounds"
            )
        self._profile = profile

    def request_target(self, target: float) -> float:
        """Request a bounded target, serializing any in-flight replacement."""

        target = _finite(target, "target")
        target = min(
            self._profile.max_angle,
            max(self._profile.min_angle, target),
        )
        if abs(self._velocity) <= _EPSILON:
            self._velocity = 0.0
            self._active_target = target
            self._pending_target = None
        elif self._active_target is not None:
            self._active_target = None
            self._pending_target = target
        else:
            self._pending_target = target
        return target

    def request_hold(self) -> None:
        """Smoothly decelerate and remain at the resulting angle."""

        self._active_target = None
        self._pending_target = None

    def emergency_stop(self) -> None:
        """Cancel immediately; used only for global ABORT/fault handling."""

        self._velocity = 0.0
        self._active_target = None
        self._pending_target = None

    def step(self, dt_sec: float) -> None:
        """Advance the deterministic motion model by dt_sec."""

        dt_sec = _finite(dt_sec, "dt_sec")
        if dt_sec <= 0.0:
            raise ValueError(f"dt_sec must be > 0, got {dt_sec}")

        if self._active_target is None:
            self._step_stopping(dt_sec)
        else:
            self._step_toward_target(dt_sec)

        if self._position <= self._profile.min_angle:
            self._position = self._profile.min_angle
            if self._velocity < 0.0:
                self._velocity = 0.0
        elif self._position >= self._profile.max_angle:
            self._position = self._profile.max_angle
            if self._velocity > 0.0:
                self._velocity = 0.0

    def _step_stopping(self, dt_sec: float) -> None:
        old_velocity = self._velocity
        self._velocity = self._move_toward(
            old_velocity,
            0.0,
            self._profile.deceleration * dt_sec,
        )
        self._position += 0.5 * (old_velocity + self._velocity) * dt_sec
        if abs(self._velocity) <= _EPSILON:
            self._velocity = 0.0
            if self._pending_target is not None:
                self._active_target = self._pending_target
                self._pending_target = None

    def _step_toward_target(self, dt_sec: float) -> None:
        target = self._active_target
        if target is None:
            return
        delta = target - self._position
        if abs(delta) <= _EPSILON:
            self._position = target
            self._velocity = 0.0
            self._active_target = None
            return

        direction = 1.0 if delta > 0.0 else -1.0
        old_velocity = self._velocity
        if old_velocity * direction < 0.0:
            new_velocity = self._move_toward(
                old_velocity,
                0.0,
                self._profile.deceleration * dt_sec,
            )
        else:
            stopping_distance = (
                old_velocity * old_velocity
                / (2.0 * self._profile.deceleration)
            )
            if stopping_distance >= abs(delta):
                desired_velocity = 0.0
                rate = self._profile.deceleration
            else:
                desired_velocity = direction * self._profile.nominal_speed
                rate = self._profile.acceleration
            new_velocity = self._move_toward(
                old_velocity,
                desired_velocity,
                rate * dt_sec,
            )

        new_position = (
            self._position
            + 0.5 * (old_velocity + new_velocity) * dt_sec
        )
        crossed_target = (
            direction > 0.0 and new_position >= target
        ) or (
            direction < 0.0 and new_position <= target
        )
        if crossed_target:
            self._position = target
            self._velocity = 0.0
            self._active_target = None
        else:
            self._position = new_position
            self._velocity = new_velocity

    @staticmethod
    def _move_toward(current: float, target: float, max_delta: float) -> float:
        if current < target:
            return min(target, current + max_delta)
        return max(target, current - max_delta)
