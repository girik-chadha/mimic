"""calib.py tests with a made-up camera that sees the A3 sheet at an angle. No hardware, no network."""
import math

import cv2
import numpy as np
import pytest

from mimic import calib
from mimic.calib import CalibrationError

# The made-up camera: the A3 sheet (42 x 29.7 cm) shows up as this lopsided four-sided shape in a 1920x1080 photo.
SHEET_CM = [(0, 0), (42, 0), (42, 29.7), (0, 29.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
TRUE_CM_TO_PX = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(SHEET_PX))

CORNER_DOTS_CM = [(2, 2), (40, 2), (40, 27.7), (2, 27.7)]
CHECK_DOT_CM = (14, 20)  # a check point the map is tested against


def camera(pts_cm):
    """Where the made-up camera sees these table points, in pixels."""
    out = cv2.perspectiveTransform(np.float64(pts_cm).reshape(-1, 1, 2), TRUE_CM_TO_PX)
    return [tuple(p) for p in out.reshape(-1, 2)]


def corners_only():
    return calib.fit(camera(CORNER_DOTS_CM), CORNER_DOTS_CM)


def side_lengths(pts):
    return [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]


# ---------- mapping ----------

def test_round_trip_px_to_cm_to_px():
    c = corners_only()
    for x in range(400, 1600, 150):
        for y in range(200, 900, 150):
            back = c.cm_to_px(c.px_to_cm((x, y)))
            assert math.dist(back, (x, y)) < 0.01


def test_points_land_where_the_camera_saw_them():
    c = corners_only()
    for pt_cm in [(10, 10), (30, 5), (21, 25), (0, 0), (42, 29.7)]:
        assert math.dist(c.px_to_cm(camera([pt_cm])[0]), pt_cm) < 0.01


def test_tilted_sheet_measures_its_real_size():
    sheet_px = camera(SHEET_CM)
    px_sides = side_lengths(sheet_px)
    assert max(px_sides) / min(px_sides) > 1.4  # in pixels the sheet is visibly distorted...

    c = corners_only()  # ...the map is built from inset dots, not from the sheet corners themselves...
    sheet_cm = [c.px_to_cm(p) for p in sheet_px]
    for got, want in zip(side_lengths(sheet_cm), [42, 29.7, 42, 29.7]):
        assert got == pytest.approx(want, abs=0.01)  # ...yet every side measures its real length
    diagonal = math.hypot(42, 29.7)
    assert math.dist(sheet_cm[0], sheet_cm[2]) == pytest.approx(diagonal, abs=0.01)
    assert math.dist(sheet_cm[1], sheet_cm[3]) == pytest.approx(diagonal, abs=0.01)


def test_10_cm_square_is_10_cm_on_every_side():
    square_cm = [(25, 10), (35, 10), (35, 20), (25, 20)]
    c = corners_only()
    measured = [c.px_to_cm(p) for p in camera(square_cm)]
    for side in side_lengths(measured):
        assert side == pytest.approx(10, abs=0.01)


# ---------- check points ----------

def test_check_returns_the_error_in_cm():
    c = corners_only()
    seen_px = camera([(20, 15)])[0]
    assert c.check(seen_px, (20, 15)) < 0.01
    assert c.check(seen_px, (23, 19)) == pytest.approx(5.0, abs=0.01)  # 3-4-5 triangle


def test_four_points_report_about_zero():
    assert corners_only().fit_report() < 0.01


def test_five_points_all_right_pass_even_with_pixel_noise():
    dots = CORNER_DOTS_CM + [CHECK_DOT_CM]
    rng = np.random.default_rng(1)
    noisy_px = [(x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for x, y in camera(dots)]
    c = calib.fit(noisy_px, dots)
    assert c.fit_report() < calib.MAX_CHECK_ERROR_CM


def test_wrongly_measured_check_point_raises_with_the_numbers():
    dots_px = camera(CORNER_DOTS_CM + [CHECK_DOT_CM])
    measured_cm = CORNER_DOTS_CM + [(16, 20)]  # written down 2 cm too far right
    with pytest.raises(CalibrationError) as caught:
        calib.fit(dots_px, measured_cm)
    message = str(caught.value)
    assert "check point 5 is 2.00 cm off" in message
    assert "maps to (14.00, 20.00) cm but was measured at (16.00, 20.00) cm" in message


def test_check_limit_is_half_a_centimetre():
    dots_px = camera(CORNER_DOTS_CM + [CHECK_DOT_CM])
    calib.fit(dots_px, CORNER_DOTS_CM + [(14.4, 20)])  # 0.4 cm off: passes
    with pytest.raises(CalibrationError):
        calib.fit(dots_px, CORNER_DOTS_CM + [(14.6, 20)])  # 0.6 cm off: refused


def test_every_bad_check_point_is_named():
    dots_px = camera(CORNER_DOTS_CM + [CHECK_DOT_CM, (30, 8)])
    with pytest.raises(CalibrationError, match="check point 5 .* check point 6"):
        calib.fit(dots_px, CORNER_DOTS_CM + [(16, 20), (30, 10)])


# ---------- bad input ----------

def test_needs_at_least_4_points():
    with pytest.raises(ValueError, match="at least 4"):
        calib.fit(camera(CORNER_DOTS_CM[:3]), CORNER_DOTS_CM[:3])


def test_point_counts_must_match():
    with pytest.raises(ValueError, match="4 pixel points but 3 table points"):
        calib.fit(camera(CORNER_DOTS_CM), CORNER_DOTS_CM[:3])


def test_corner_points_in_a_line_raise():
    in_a_line = [(0, 0), (10, 0), (20, 0), (0, 20)]
    with pytest.raises(ValueError, match="straight line"):
        calib.fit(camera(in_a_line), in_a_line)


# ---------- saving ----------

def test_check_points_can_be_named():
    dots_px = camera(CORNER_DOTS_CM + [CHECK_DOT_CM])
    names = ["bottom_left", "bottom_right", "top_right", "top_left", "right_check"]
    with pytest.raises(CalibrationError, match="right_check is 2.00 cm off"):
        calib.fit(dots_px, CORNER_DOTS_CM + [(16, 20)], names=names)


def test_save_and_load_give_the_same_map(tmp_path):
    dots = CORNER_DOTS_CM + [CHECK_DOT_CM]
    c = calib.fit(camera(dots), dots)
    c.save(tmp_path / "camera.json")
    loaded = calib.Calib.load(tmp_path / "camera.json")
    assert np.allclose(loaded.H, c.H)
    assert loaded.table_pts_cm == c.table_pts_cm
    assert loaded.px_to_cm((800, 500)) == pytest.approx(c.px_to_cm((800, 500)))
