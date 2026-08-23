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

    def bounds(self) -> tuple[float, float]:
        """The reachable band: centre +/- relative_limit, inside the hard stops."""
        return (
            max(self.min_angle, self.center_angle - self.relative_limit),
            min(self.max_angle, self.center_angle + self.relative_limit),
        )

    def speed_for(self, direction: int, activation: float) -> float:
        """Map -1/+1 and normalized activation to a signed angular rate.

        This replaced an absolute map, ``center + relative_limit *
        activation``, which was measured to do the opposite of what the
        gesture says. Parked at 33 degrees, a 70% ulnar contraction asked for
        25 -- the angle 70% corresponds to -- so the arm retreated while the
        wearer was pushing it outward. Over one 90 s session, 8 of 44 pushes
        moved against their own gesture.

        A rate has the property the absolute map lacked: a light push is slow
        motion in the direction asked for, a hard push is fast motion in the
        same direction, and no push is no motion. Nothing the wearer can do
        makes the arm travel backwards.

        It costs the property that one effort level always names one angle.
        The same session showed that property to be the source of the
        confusion rather than a feature.

        The safety argument survives the change, but it moves: it was that the
        reachable set is exactly the band regardless of command history, which
        absolute mapping got by construction. Here the motion model gets it
        instead -- and it takes both halves, since goals inside the band are
        not sufficient on their own. See ``request_velocity`` for the goals
        and ``_clamp_into_band`` for the paths that have no goal at all.
        """

        if direction not in (-1, 1):
            raise ValueError(f"direction must be -1 or 1, got {direction}")
        activation = _finite(activation, "activation")
        activation = min(1.0, max(0.0, activation))
        return float(direction) * self.nominal_speed * activation


class DiscreteViewSweep:
    """One bounded step per event, reversing at the ends of the sweep band.

    The discrete input has exactly one gesture to spend. NEXT_TARGET says
    "another step" and carries no direction, so direction cannot come from the
    wearer and has to be a property of the sweep: it runs to one end of the
    band, turns around, and runs back. That is the whole reason this policy
    exists rather than the caller adding a step to the current angle.

    Decisions worth stating, because each one is arguable:

    - The band is ``SearchProfile.bounds()``, exactly the set the proportional
      path steps inside. Both input modes then sweep the same space, so
      switching between them cannot expose an angle the other mode considers
      unsafe.
    - Overshooting the band clamps to the edge instead of reversing early, so
      the edge is actually reachable when the band is not a whole number of
      steps. Reversal happens on the *next* event, once the edge is where the
      axis already is. Reversing early would leave a sliver at each end that
      no sequence of steps can visit.
    - Every call returns an angle different from the one passed in. A press
      that produces no motion reads as a dropped event to the wearer, who
      responds by pressing again, which is the opposite of what a bounded
      search wants.
    - The first step is positive regardless of where the axis starts. There is
      no wearer input to infer a preference from, and a fixed choice is easier
      to learn than one that depends on the starting angle.

    ``next_target`` anchors on the position it is given rather than on its own
    running total: the axis may have been clamped, preempted, or stopped
    short, and the sweep must continue from where the axis actually is.
    """

    def __init__(self, profile: SearchProfile, step_angle: float) -> None:
        if not isinstance(profile, SearchProfile):
            raise TypeError("profile must be a SearchProfile")
        step_angle = _finite(step_angle, "step_angle")
        if step_angle <= 0.0:
            raise ValueError(f"step_angle must be > 0, got {step_angle}")
        self._step_angle = step_angle
        # From the profile, not recomputed. This band and the one the
        # proportional path steps inside have to be the same set, and two
        # copies of the same two lines is how they stop being.
        self._lower, self._upper = profile.bounds()
        self._direction = 1

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def bounds(self) -> tuple[float, float]:
        return (self._lower, self._upper)

    def next_target(self, current_position: float) -> float:
        """Return the angle one step away, reversing at the band edges."""

        position = _finite(current_position, "current_position")
        for _ in range(2):
            bound = self._upper if self._direction > 0 else self._lower
            at_bound = abs(position - bound) <= _EPSILON
            if not at_bound:
                target = position + self._direction * self._step_angle
                if self._direction > 0:
                    target = min(target, self._upper)
                else:
                    target = max(target, self._lower)
                # A position outside the band clamps back onto the edge, which
                # is still real motion, so this cannot return the input.
                if abs(target - position) > _EPSILON:
                    return target
            self._direction = -self._direction
        # Both directions are blocked only if the band is a single point,
        # which SearchProfile's positive relative_limit already excludes.
        raise RuntimeError("sweep band admits no step in either direction")


class SimulatedViewMotion:
    """Single-axis simulator with smooth, serialized preemption.

    A replacement target remains pending while the old motion decelerates to
    zero. It becomes active only after the stop, so two simulated goals can
    never execute concurrently.

    Two ways to drive it, because the two input modes want different things:

    - ``request_target`` for the discrete sweep, which really is point to
      point: one gesture, one destination, decelerate to rest on arrival.
    - ``request_velocity`` for proportional search, which is a rate arriving
      at 20 Hz. Routing that through ``request_target`` was measured to make
      the axis crawl -- every command preempted the last one, so the axis
      spent its whole life in the deceleration phase and full effort produced
      0.059 rad/s against a 0.25 rad/s nominal, with activation 0.7 and 1.0
      indistinguishable. See ``request_velocity`` for how that is avoided.
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
        # Ceiling for the goal currently in flight. Point-to-point goals run
        # at the profile's nominal speed; a rate command lowers it, which is
        # the only thing that distinguishes the two modes.
        self._speed_limit = profile.nominal_speed

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
        self._speed_limit = profile.nominal_speed

    def request_target(self, target: float) -> float:
        """Request a bounded target, serializing any in-flight replacement."""

        target = _finite(target, "target")
        target = min(
            self._profile.max_angle,
            max(self._profile.min_angle, target),
        )
        self._speed_limit = self._profile.nominal_speed
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

    def request_velocity(self, velocity: float) -> float:
        """Command a signed angular rate; returns the rate actually adopted.

        Implemented as "head for the edge of the band, but no faster than
        this", so the trapezoid planner that already exists does the work and
        the axis decelerates to a stop exactly on the edge rather than being
        clamped against it. That keeps the axis inside the band while it has
        a goal; a reversal or a release takes the goal away, and
        ``_clamp_into_band`` is what bounds it then.

        The rule that matters for feel is that a command in the direction
        already being travelled updates only the speed ceiling and leaves the
        goal alone. It is the same goal at a new rate, not a second goal, so
        it must not trip the serialized-preemption stop -- doing so is what
        made a 20 Hz stream crawl. A reversal is a different goal and does
        take the stop-first path, which is what a reversal should do anyway.
        """

        velocity = _finite(velocity, "velocity")
        speed = min(abs(velocity), self._profile.nominal_speed)
        if speed <= _EPSILON:
            self.request_hold()
            return 0.0

        direction = 1.0 if velocity > 0.0 else -1.0
        lower, upper = self._profile.bounds()
        edge = upper if direction > 0.0 else lower
        # Outside the band, this permits only the direction that returns to
        # it: configure() can move the band under an axis that is parked.
        if (direction > 0.0 and self._position >= edge - _EPSILON) or (
            direction < 0.0 and self._position <= edge + _EPSILON
        ):
            self.request_hold()
            return 0.0

        in_flight = (
            self._active_target
            if self._active_target is not None
            else self._pending_target
        )
        if in_flight is None or abs(in_flight - edge) > _EPSILON:
            self.request_target(edge)
        self._speed_limit = speed
        return direction * speed

    def request_hold(self) -> None:
        """Smoothly decelerate and remain at the resulting angle."""

        self._active_target = None
        self._pending_target = None
        self._speed_limit = self._profile.nominal_speed

    def emergency_stop(self) -> None:
        """Cancel immediately; used only for global ABORT/fault handling."""

        self._velocity = 0.0
        self._active_target = None
        self._pending_target = None
        self._speed_limit = self._profile.nominal_speed

    def step(self, dt_sec: float) -> None:
        """Advance the deterministic motion model by dt_sec."""

        dt_sec = _finite(dt_sec, "dt_sec")
        if dt_sec <= 0.0:
            raise ValueError(f"dt_sec must be > 0, got {dt_sec}")

        start_position = self._position
        if self._active_target is None:
            self._step_stopping(dt_sec)
        else:
            self._step_toward_target(dt_sec)
        self._clamp_into_band(start_position)

    def _clamp_into_band(self, start_position: float) -> None:
        """Keep the band a hard bound, not merely where the goals are.

        The planner already stops on the edge when it has a goal there, so
        this only matters on the path that has no goal: a reversal or a hold
        drops the active target and hands the axis to the deceleration
        integrator, which used to be bounded by nothing but the absolute
        stops. A wearer flicking direction at full speed near the edge
        therefore coasted past it -- measured at +0.098 degrees, and the
        stopping distance at nominal speed allows 2.39.

        Small, but the claim being made about this axis is that its reachable
        set is exactly the band whatever sequence of commands arrives, and
        that claim was false as written until this existed.
        """

        lower, upper = self._profile.bounds()
        # An axis parked outside the band -- configure() can move the band out
        # from under it -- must not be yanked back in. It may only be stopped
        # from travelling further out.
        lower = max(min(lower, start_position), self._profile.min_angle)
        upper = min(max(upper, start_position), self._profile.max_angle)
        if self._position <= lower:
            self._position = lower
            if self._velocity < 0.0:
                self._velocity = 0.0
        elif self._position >= upper:
            self._position = upper
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
                desired_velocity = direction * min(
                    self._profile.nominal_speed, self._speed_limit
                )
                # Slowing down is deceleration even when it is not a stop:
                # easing off a proportional command should feel like braking,
                # not like a slow drift down to the new rate.
                rate = (
                    self._profile.acceleration
                    if abs(desired_velocity) > abs(old_velocity)
                    else self._profile.deceleration
                )
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
