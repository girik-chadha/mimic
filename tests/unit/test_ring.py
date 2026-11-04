"""ring.py: fitting the ring from base angles and tip positions, and using it. Made-up rings, no arm."""
import dataclasses
import math

import pytest

from mimic import config, ring
from mimic.ring import RingError

CENTRE, RADIUS = (18.5, 32.0), 14.0


def made_up(angles, scale=1.0, offset=-180.0, noise=None, centre=CENTRE, radius=RADIUS):
    """Calibration points on a known ring: theta = scale * phi + offset."""
    points = []
    for i, theta in enumerate(angles):
        phi = math.radians((theta - offset) / scale)
        x, y = centre[0] + radius * math.cos(phi), centre[1] + radius * math.sin(phi)
        if noise:
            x, y = x + noise[i][0], y + noise[i][1]
        points.append((theta, x, y))
    return points


def fitted(points, **changes) -> ring.Ring:
    result = ring.fit(points)
    return dataclasses.replace(ring.with_fit(ring.Ring(), result, points, "test"), **changes)


def test_five_points_give_back_the_ring_and_the_scale():
    result = ring.fit(made_up([80, 90, 100, 110, 120]))
    assert result.centre_cm == pytest.approx(CENTRE) and result.radius_cm == pytest.approx(RADIUS)
    assert result.scale == pytest.approx(1.0) and max(result.errors_cm) < 1e-9


def test_three_points_take_one_servo_degree_as_one_real_degree():
    result = ring.fit(made_up([80, 100, 120]))
    assert result.scale == 1.0 and result.radius_cm == pytest.approx(RADIUS)


def test_a_servo_that_turns_5_percent_short_is_fitted_from_5_points():
    result = ring.fit(made_up([80, 90, 100, 110, 120], scale=0.95, offset=-171))
    assert result.scale == pytest.approx(0.95, abs=0.003) and result.radius_cm == pytest.approx(RADIUS, abs=0.05)
    assert "fitted (clearly better than 1)" in result.scale_note


def test_with_only_3_points_a_short_servo_goes_unnoticed_as_a_wrong_radius():
    # Why ring_calibrate asks for 5 points: with 3, a servo that turns 15% short is absorbed into the radius.
    # On a 40-degree arc (base 80-120) the points still fit within 0.12 cm, but the radius comes out 16.4, not 14.
    points = made_up([80, 100, 120], scale=0.85, offset=-153)
    result = ring.fit(points)
    assert result.scale == 1.0 and max(result.errors_cm) < 0.15      # it looks fine...
    assert result.radius_cm == pytest.approx(16.4, abs=0.1)           # ...but the radius is wrong
    r = fitted(points)
    for theta, x, y in points:                                        # inside the arc it is still close
        assert math.dist(r.point_at(theta), (x, y)) < 0.15


def test_a_reversed_base_is_found_too():
    result = ring.fit(made_up([80, 90, 100, 110, 120], scale=-1.0, offset=0))
    assert result.scale == pytest.approx(-1.0) and result.centre_cm == pytest.approx(CENTRE)


def test_small_camera_errors_keep_the_scale_at_1_and_still_fit():
    noise = [(0.2, -0.1), (-0.15, 0.2), (0.1, 0.1), (-0.2, -0.15), (0.15, -0.2)]
    points = made_up([80, 90, 100, 110, 120], noise=noise)
    result = ring.fit(points)
    assert result.scale == 1.0 and "kept at 1" in result.scale_note
    assert max(result.errors_cm) < 0.5
    r = fitted(points)
    for theta in (80, 100, 120):     # inside the arc the claw goes where the true ring says, within 0.3 cm
        phi = math.radians(theta + 180)
        truth = (CENTRE[0] + RADIUS * math.cos(phi), CENTRE[1] + RADIUS * math.sin(phi))
        assert math.dist(r.point_at(theta), truth) < 0.3


@pytest.mark.parametrize("points, problem", [
    (made_up([80, 90]), "at least 3 calibration points"),
    (made_up([80, 85, 90, 95]), "cover only 15 degrees"),
    (made_up([80, 80, 100]), "same base angle"),
    (made_up([80, 90, 100, 110, 120], noise=[(0, 0), (0, 0), (1.2, 0), (0, 0), (0, 0)]), "base 100 is .* cm from"),
    (made_up([80, 90, 100, 110, 120], scale=0.7, offset=-126), "at the edge of 0.85-1.15"),
])
def test_bad_calibrations_are_refused(points, problem):
    with pytest.raises(RingError, match=problem):
        ring.fit(points)


def test_a_refused_fit_names_the_point_that_spoils_it_not_just_the_worst_error():
    points = made_up([80, 90, 100, 110, 120], noise=[(0, 0), (0, 0), (0, 1.5), (0, 0), (0, 0)])
    with pytest.raises(RingError, match="Without the point at base 100 the others fit within 0.00 cm"):
        ring.fit(points)


def test_base_angle_and_point_are_each_others_inverse():
    r = fitted(made_up([80, 90, 100, 110, 120]))
    for theta in (75, 95.5, 125):
        assert r.base_angle_for(*r.point_at(theta)) == pytest.approx(theta)


def test_a_point_more_than_1_cm_off_the_ring_is_refused():
    r = fitted(made_up([80, 90, 100, 110, 120]))
    x, y = r.point_at(100)
    out = ((x - CENTRE[0]) / RADIUS, (y - CENTRE[1]) / RADIUS)       # along the radius, outwards

    def moved(cm):
        return x + out[0] * cm, y + out[1] * cm

    assert r.base_angle_for(*moved(-0.8)) == pytest.approx(100)        # 0.8 cm inside: allowed, same angle
    with pytest.raises(RingError, match="1.2 cm inside the ring \\(allowed 1.0 cm\\)"):
        r.base_angle_for(*moved(-1.2))
    with pytest.raises(RingError, match="1.5 cm outside the ring"):
        r.base_angle_for(*moved(1.5))


def test_angles_that_wrap_past_half_a_turn_stay_near_the_calibration():
    # this ring's calibrated directions straddle phi = 180, where atan2 jumps from +180 to -180
    points = made_up([80, 90, 100, 110, 120], offset=-90)
    r = fitted(points)
    assert [round(r.angle_for(x, y), 6) for _, x, y in points] == [80, 90, 100, 110, 120]


def test_ready_needs_a_fit_slots_and_camera_clear():
    with pytest.raises(RingError, match="still a placeholder"):
        ring.Ring().check_ready()
    r = fitted(made_up([80, 90, 100, 110, 120]))
    with pytest.raises(RingError, match="no slots_deg"):
        r.check_ready()
    with pytest.raises(RingError, match="no camera_clear"):
        dataclasses.replace(r, slots_deg=[85, 100], camera_clear=None).check_ready()
    dataclasses.replace(r, slots_deg=[85, 100], camera_clear={"base": 90, "shoulder": 90}).check_ready()


def test_save_and_load_round_trip(tmp_path):
    r = fitted(made_up([80, 90, 100, 110, 120]), slots_deg=[85, 100, 115], camera_clear={"base": 90, "shoulder": 90})
    ring.save_ring(r, tmp_path / "ring.json")
    assert ring.load_ring(tmp_path / "ring.json") == r


def test_committed_ring_is_a_placeholder_with_girik_s_settings():
    r = ring.load_ring(config.ROOT / "calib" / "ring.json")
    assert r.placeholder is True and r.radius_cm is None and r.slots_deg == []
    assert r.camera_clear == {"base": 90, "shoulder": 90} and r.max_off_ring_cm == 1.0
    assert (r.shoulder_table, r.lift, r.approach) == (173, 20, 10)
    assert r.grip_hold == 20                 # found by hand, 4 Nov
    assert r.command_s == 0.01               # measured: 3.5 ms mean, 4.7 ms slowest
    assert r.stop_at is None and r.shoulder_stop == 176   # 3 degrees past the table angle


def test_the_shoulder_stop_follows_the_table_angle_unless_set():
    assert ring.Ring(shoulder_table=170).shoulder_stop == 173
    assert ring.Ring(shoulder_table=170, stop_at=172).shoulder_stop == 172
