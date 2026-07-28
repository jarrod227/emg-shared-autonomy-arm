"""Focused unit tests for Objective 4.3 simulated view motion."""

import math

import pytest

from assistive_handoff.view_search import (
    MAX_RELATIVE_LIMIT_RAD,
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
