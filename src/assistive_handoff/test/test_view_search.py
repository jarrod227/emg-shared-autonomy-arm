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
    # The band is center +/- relative_limit, the same set target_for reaches.
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


def test_activation_maps_to_angle_and_obeys_absolute_bound():
    profile = make_profile(
        center_angle=math.radians(20.0),
        relative_limit=math.radians(40.0),
        max_angle=math.radians(45.0),
    )

    assert profile.target_for(-1, 0.5) == pytest.approx(0.0)
    assert profile.target_for(1, 1.0) == pytest.approx(math.radians(45.0))
    assert profile.target_for(1, 2.0) == pytest.approx(math.radians(45.0))


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
