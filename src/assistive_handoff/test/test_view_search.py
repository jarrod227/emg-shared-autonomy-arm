"""Focused unit tests for Objective 4.3 simulated view motion."""

import math

import pytest

from assistive_handoff.view_search import (
    MAX_RELATIVE_LIMIT_RAD,
    DiscreteViewSweep,
    SearchProfile,
    SimulatedViewMotion,
)


def make_profile(**overrides):
    values = {
        "center_angle": 0.0,
        "relative_limit": math.radians(40.0),
        "min_angle": math.radians(-50.0),
        "max_angle": math.radians(50.0),
        "nominal_speed": 1.0,
        "acceleration": 2.0,
        "deceleration": 2.0,
    }
    values.update(overrides)
    return SearchProfile(**values)


def advance_until_stopped(motion, timeout_steps=500, dt_sec=0.01):
    for _ in range(timeout_steps):
        motion.step(dt_sec)
        if not motion.moving:
            return
    raise AssertionError("simulated view motion did not stop")


def sweep_positions(sweep, start, count):
    positions = []
    position = start
    for _ in range(count):
        position = sweep.next_target(position)
        positions.append(position)
    return positions


def test_sweep_steps_out_to_the_band_edge_then_turns_around():
    # The band is center +/- relative_limit, the same set a rate stays in.
    sweep = DiscreteViewSweep(make_profile(relative_limit=0.6), 0.2)

    assert sweep.bounds == pytest.approx((-0.6, 0.6))
    assert sweep_positions(sweep, 0.0, 9) == pytest.approx(
        [0.2, 0.4, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6]
    )
    assert sweep.direction == -1


def test_a_band_that_is_not_a_whole_number_of_steps_still_reaches_its_edge():
    # Reversing as soon as a full step would overshoot leaves a sliver at each
    # end that no sequence of steps can visit.
    sweep = DiscreteViewSweep(make_profile(relative_limit=0.6), 0.25)

    positions = sweep_positions(sweep, 0.0, 4)

    assert positions == pytest.approx([0.25, 0.5, 0.6, 0.35])


def test_a_step_wider_than_the_band_alternates_between_the_two_edges():
    sweep = DiscreteViewSweep(make_profile(relative_limit=0.6), 2.0)

    assert sweep_positions(sweep, 0.0, 4) == pytest.approx(
        [0.6, -0.6, 0.6, -0.6]
    )


def test_a_position_outside_the_band_is_brought_back_onto_the_edge():
    # configure() can move the band under an axis that is already parked.
    sweep = DiscreteViewSweep(make_profile(relative_limit=0.6), 0.2)

    assert sweep_positions(sweep, 0.9, 2) == pytest.approx([0.6, 0.4])


def test_the_band_is_clipped_by_the_absolute_bounds():
    sweep = DiscreteViewSweep(
        make_profile(center_angle=0.5, relative_limit=0.6, max_angle=0.7),
        0.3,
    )

    assert sweep.bounds == pytest.approx((-0.1, 0.7))
    assert sweep_positions(sweep, 0.5, 4) == pytest.approx(
        [0.7, 0.4, 0.1, -0.1]
    )


def test_every_step_moves_the_axis():
    # A press that produces no motion reads as a dropped event, and the wearer
    # answers a dropped event by pressing again.
    for step in (0.05, 0.2, 0.37, 1.5):
        sweep = DiscreteViewSweep(make_profile(relative_limit=0.6), step)
        position = -1.0
        for _ in range(40):
            target = sweep.next_target(position)
            assert target != pytest.approx(position, abs=1e-9)
            position = target


def test_sweep_rejects_a_non_positive_or_non_finite_step():
    profile = make_profile()
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            DiscreteViewSweep(profile, bad)
    with pytest.raises(TypeError, match="SearchProfile"):
        DiscreteViewSweep(object(), 0.2)


def test_relative_limit_cannot_exceed_45_degrees():
    with pytest.raises(ValueError, match="pi/4"):
        make_profile(relative_limit=MAX_RELATIVE_LIMIT_RAD + 0.001)


def test_the_band_is_the_centre_plus_minus_the_limit_inside_the_hard_stops():
    profile = make_profile(
        center_angle=math.radians(20.0),
        relative_limit=math.radians(40.0),
        max_angle=math.radians(45.0),
    )

    lower, upper = profile.bounds()
    assert lower == pytest.approx(math.radians(-20.0))
    assert upper == pytest.approx(math.radians(45.0))


def test_effort_scales_the_rate_and_direction_chooses_its_sign():
    profile = make_profile(nominal_speed=0.25)

    assert profile.speed_for(1, 0.5) == pytest.approx(0.125)
    assert profile.speed_for(-1, 0.5) == pytest.approx(-0.125)
    assert profile.speed_for(1, 1.0) == pytest.approx(0.25)


def test_no_effort_is_no_motion():
    # The property the absolute map could not give: a push too weak to move
    # the arm leaves it where it is, instead of naming an angle behind it.
    profile = make_profile(nominal_speed=0.25)

    assert profile.speed_for(1, 0.0) == 0.0
    assert profile.speed_for(-1, 0.0) == 0.0


def test_activation_outside_the_unit_range_is_clamped_not_rejected():
    # The bridge normalizes against a session reference, so a contraction
    # stronger than the calibration one legitimately arrives above 1.0.
    profile = make_profile(nominal_speed=0.25)

    assert profile.speed_for(1, 2.0) == pytest.approx(0.25)
    assert profile.speed_for(1, -1.0) == pytest.approx(0.0)


def test_speed_rejects_a_direction_it_cannot_honour():
    profile = make_profile()

    with pytest.raises(ValueError, match="direction"):
        profile.speed_for(0, 1.0)
    with pytest.raises(ValueError, match="activation"):
        profile.speed_for(1, float("nan"))


def test_a_rate_command_travels_at_the_rate_it_asked_for():
    profile = make_profile(nominal_speed=0.4, acceleration=4.0)
    motion = SimulatedViewMotion(profile, initial_angle=0.0)

    motion.request_velocity(0.2)
    for _ in range(100):  # 1 s, of which the ramp is 0.05 s
        motion.step(0.01)

    assert motion.velocity == pytest.approx(0.2)
    assert motion.position == pytest.approx(0.2, abs=0.01)


def test_a_rate_command_stops_on_the_band_edge_without_crossing_it():
    # The whole safety argument. Absolute mapping bounded the reachable set by
    # naming only angles inside the band; a rate names no angle at all, so the
    # bound has to live here.
    profile = make_profile(relative_limit=0.6, nominal_speed=0.4)
    motion = SimulatedViewMotion(profile, initial_angle=0.0)

    for _ in range(1000):  # 10 s, far longer than crossing the band takes
        motion.request_velocity(profile.speed_for(1, 1.0))
        motion.step(0.01)
        assert motion.position <= 0.6 + 1e-9

    assert motion.position == pytest.approx(0.6)
    assert motion.velocity == pytest.approx(0.0)


def test_a_rate_command_outside_the_band_may_only_come_back():
    # configure() can move the band under an axis that is already parked.
    profile = make_profile(relative_limit=0.6, nominal_speed=0.4,
                           max_angle=1.0)
    motion = SimulatedViewMotion(profile, initial_angle=0.9)

    assert motion.request_velocity(profile.speed_for(1, 1.0)) == 0.0
    assert motion.request_velocity(profile.speed_for(-1, 1.0)) < 0.0


def test_easing_off_slows_the_axis_without_stopping_it():
    profile = make_profile(nominal_speed=0.4, acceleration=4.0,
                           deceleration=4.0)
    motion = SimulatedViewMotion(profile, initial_angle=0.0)

    motion.request_velocity(0.4)
    for _ in range(50):
        motion.step(0.01)
    assert motion.velocity == pytest.approx(0.4)

    motion.request_velocity(0.1)
    for _ in range(50):
        motion.step(0.01)
    assert motion.velocity == pytest.approx(0.1)


def test_a_reversal_passes_through_zero_before_moving_the_other_way():
    profile = make_profile(nominal_speed=0.4, acceleration=4.0,
                           deceleration=4.0)
    motion = SimulatedViewMotion(profile, initial_angle=0.0)

    motion.request_velocity(0.4)
    for _ in range(50):
        motion.step(0.01)

    motion.request_velocity(-0.4)
    velocities = []
    for _ in range(50):
        motion.step(0.01)
        velocities.append(motion.velocity)

    assert min(velocities) < 0.0, "never reversed"
    signs = [v > 0.0 for v in velocities]
    assert signs.index(False) > 0, "flipped sign without decelerating"
    assert all(
        abs(b - a) <= profile.deceleration * 0.01 + 1e-9
        for a, b in zip(velocities, velocities[1:])
    ), "velocity jumped rather than ramping"


def test_a_twenty_hertz_stream_moves_at_the_speed_the_effort_asked_for():
    """The regression this whole mechanism exists for.

    Proportional commands used to be routed through request_target, one
    small step ahead of the axis per command. Every command preempted the
    last, so the axis lived permanently in the deceleration phase: with the
    controller's shipped defaults, full effort produced 0.059 rad/s against a
    0.25 rad/s nominal, activation 0.7 and 1.0 were indistinguishable, and
    anything below half effort did not move the axis at all.

    Nothing in the node-level suite could see it, because those tests raise
    nominal_speed to 1.0 and acceleration to 4.0 to run quickly. So this
    checks the shipped defaults, and checks the shape of the curve rather
    than one point on it.
    """

    profile = make_profile(
        center_angle=0.0,
        relative_limit=math.pi / 4.0,
        min_angle=-math.pi / 3.0,
        max_angle=math.pi / 3.0,
        nominal_speed=0.25,
        acceleration=0.5,
        deceleration=0.75,
    )

    def average_speed(activation, seconds=3.0):
        motion = SimulatedViewMotion(profile, initial_angle=0.0)
        elapsed, next_command = 0.0, 0.0
        while elapsed < seconds:
            if elapsed >= next_command - 1e-9:
                motion.request_velocity(profile.speed_for(1, activation))
                next_command += 0.05  # 20 Hz, the EMG bridge's rate
            motion.step(0.02)  # 50 Hz, view_update_period_sec
            elapsed += 0.02
        return motion.position / seconds

    speeds = [average_speed(a) for a in (0.1, 0.3, 0.5, 0.7, 1.0)]

    assert all(b > a for a, b in zip(speeds, speeds[1:])), (
        f"effort does not choose speed: {speeds}"
    )
    for activation, measured in zip((0.1, 0.3, 0.5, 0.7, 1.0), speeds):
        ideal = 0.25 * activation
        # Only the acceleration ramp is missing; it costs more of a short,
        # slow run than of a fast one, so 90% is the floor across the range.
        assert 0.90 * ideal <= measured <= ideal + 1e-9, (
            f"activation {activation} gave {measured:.4f}, ideal {ideal:.4f}"
        )


def test_new_target_waits_for_smooth_stop_before_becoming_active():
    motion = SimulatedViewMotion(make_profile())
    motion.request_target(0.6)
    for _ in range(20):
        motion.step(0.01)
    assert motion.velocity > 0.0

    motion.request_target(-0.6)
    assert motion.active_target is None
    assert motion.pending_target == pytest.approx(-0.6)
    assert motion.goal_count == 0

    while motion.velocity > 0.0:
        previous_position = motion.position
        motion.step(0.01)
        assert motion.position >= previous_position

    assert motion.active_target == pytest.approx(-0.6)
    assert motion.pending_target is None
    assert motion.goal_count == 1


def test_only_latest_pending_target_survives_preemption():
    motion = SimulatedViewMotion(make_profile())
    motion.request_target(0.6)
    for _ in range(20):
        motion.step(0.01)

    motion.request_target(-0.6)
    motion.request_target(-0.2)

    assert motion.active_target is None
    assert motion.pending_target == pytest.approx(-0.2)
    assert motion.goal_count == 0


def test_hold_decelerates_then_stays_at_resulting_angle():
    motion = SimulatedViewMotion(make_profile())
    motion.request_target(0.6)
    for _ in range(20):
        motion.step(0.01)

    moving_position = motion.position
    motion.request_hold()
    advance_until_stopped(motion)

    assert motion.position > moving_position
    assert motion.velocity == pytest.approx(0.0)
    assert motion.active_target is None
    assert motion.pending_target is None


def test_emergency_stop_cancels_without_deceleration():
    motion = SimulatedViewMotion(make_profile())
    motion.request_target(0.6)
    for _ in range(20):
        motion.step(0.01)
    assert motion.moving

    stopped_at = motion.position
    motion.emergency_stop()
    motion.step(0.01)

    assert not motion.moving
    assert motion.position == pytest.approx(stopped_at)
