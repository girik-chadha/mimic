"""kinematics.py: the taught-pose grid. No arm needed."""
import dataclasses

import pytest

from mimic import arm, config, kinematics
from mimic.kinematics import KinematicsError, PoseGrid, TaughtPoint


def linear_grid(step=4.0, placeholder=False):
    """A made-up taught grid where base = 40 + 3x and shoulder = 40 + 3y; elbow 120 (hover) / 100 (grab)."""
    points = []
    y = -2.0
    while y <= 26.0:
        x = -2.0
        while x <= 38.0:
            points.append(TaughtPoint(x, y, {"base": 40 + 3 * x, "shoulder": 40 + 3 * y, "elbow": 120},
                                      {"base": 40 + 3 * x, "shoulder": 40 + 3 * y, "elbow": 100}))
            x += step
        y += step
    return PoseGrid(points, grip_open=90, grip_closed=130,
                    camera_clear={"base": 20, "shoulder": 150, "elbow": 150}, placeholder=placeholder)


def wide_limits():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    return dataclasses.replace(limits, ranges_deg={j: (0, 180) for j in ("base", "shoulder", "elbow")},
                               grip_closed_limit_deg=140, placeholder=False, limits_checked_by_hand=True)


def test_exact_at_a_taught_point():
    grid = linear_grid()
    assert grid.to_angles(10.0, 6.0, "hover") == {"base": 70.0, "shoulder": 58.0, "elbow": 120.0}
    assert grid.to_angles(10.0, 6.0, "grab")["elbow"] == 100.0


def test_between_points_is_a_blend_of_the_neighbours():
    grid = linear_grid()
    angles = grid.to_angles(11.0, 7.5, "hover")
    assert 70 < angles["base"] < 82 and 58 < angles["shoulder"] < 70  # between the taught neighbours
    assert abs(angles["base"] - (40 + 3 * 11.0)) < 1.5               # and close to the true value


def test_a_little_outside_is_allowed_but_more_than_1_cm_is_refused():
    grid = linear_grid()
    grid.to_angles(38.5, 10.0, "grab")                       # 0.5 cm outside the hull: fine
    with pytest.raises(KinematicsError, match=r"\(39.5, 10.0\) cm is 1.5 cm outside the taught area"):
        grid.to_angles(39.5, 10.0, "grab")


def test_placeholder_grid_is_refused():
    with pytest.raises(KinematicsError, match="placeholder"):
        linear_grid(placeholder=True).to_angles(10, 6, "hover")


def test_needs_at_least_four_complete_points():
    grid = linear_grid()
    grid.points = grid.points[:3] + [TaughtPoint(30, 20, hover={"base": 1, "shoulder": 1, "elbow": 1})]
    with pytest.raises(KinematicsError, match="at least 4 points with both hover and grab; it has 3"):
        grid.to_angles(1, 1, "hover")


def test_needs_grip_values_and_camera_clear():
    grid = dataclasses.replace(linear_grid(), camera_clear=None)
    with pytest.raises(KinematicsError, match="camera_clear"):
        grid.check_ready()


def test_taught_values_must_be_inside_the_arm_limits():
    with pytest.raises(KinematicsError, match="grip_closed 150 is outside the grip range 90-140"):
        dataclasses.replace(linear_grid(), grip_closed=150).check_against(wide_limits())
    tight = dataclasses.replace(wide_limits(), ranges_deg={"base": (70, 110), "shoulder": (0, 180), "elbow": (0, 180)})
    with pytest.raises(KinematicsError, match="base .* is outside its range 70-110"):
        linear_grid().check_against(tight)


def test_save_and_load_round_trip(tmp_path):
    grid = linear_grid()
    kinematics.save_grid(grid, tmp_path / "grid.json")
    loaded = kinematics.load_grid(tmp_path / "grid.json")
    assert loaded == grid


def test_committed_grid_is_an_empty_placeholder():
    grid = kinematics.load_grid()
    assert grid.placeholder is True and grid.points == []
