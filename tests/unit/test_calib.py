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

CORNER_STICKERS_CM = [(2, 2), (40, 2), (40, 27.7), (2, 27.7)]
CHECK_STICKER_CM = (14, 20)  # off-centre: not in line with any two corner stickers


def camera(pts_cm):
    """Where the made-up camera sees these table points, in pixels."""
    out = cv2.perspectiveTransform(np.float64(pts_cm).reshape(-1, 1, 2), TRUE_CM_TO_PX)
    return [tuple(p) for p in out.reshape(-1, 2)]


def corners_only():
    return calib.fit(camera(CORNER_STICKERS_CM), CORNER_STICKERS_CM)


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

    c = corners_only()  # ...the map is built from inset stickers, not from the sheet corners themselves...
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


# ---------- check stickers ----------

def test_check_returns_the_error_in_cm():
    c = corners_only()
    seen_px = camera([(20, 15)])[0]
    assert c.check(seen_px, (20, 15)) < 0.01
    assert c.check(seen_px, (23, 19)) == pytest.approx(5.0, abs=0.01)  # 3-4-5 triangle


def test_four_stickers_report_about_zero():
    assert corners_only().fit_report() < 0.01


def test_five_stickers_all_right_pass_even_with_pixel_noise():
    stickers = CORNER_STICKERS_CM + [CHECK_STICKER_CM]
    rng = np.random.default_rng(1)
    noisy_px = [(x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for x, y in camera(stickers)]
    c = calib.fit(noisy_px, stickers)
    assert c.fit_report() < calib.MAX_CHECK_ERROR_CM


def test_wrongly_measured_check_sticker_raises_with_the_numbers():
    stickers_px = camera(CORNER_STICKERS_CM + [CHECK_STICKER_CM])
    measured_cm = CORNER_STICKERS_CM + [(16, 20)]  # written down 2 cm too far right
    with pytest.raises(CalibrationError) as caught:
        calib.fit(stickers_px, measured_cm)
    message = str(caught.value)
    assert "check sticker 5 is 2.00 cm off" in message
    assert "maps to (14.00, 20.00) cm but was measured at (16.00, 20.00) cm" in message


def test_check_limit_is_half_a_centimetre():
    stickers_px = camera(CORNER_STICKERS_CM + [CHECK_STICKER_CM])
    calib.fit(stickers_px, CORNER_STICKERS_CM + [(14.4, 20)])  # 0.4 cm off: passes
    with pytest.raises(CalibrationError):
        calib.fit(stickers_px, CORNER_STICKERS_CM + [(14.6, 20)])  # 0.6 cm off: refused


def test_every_bad_check_sticker_is_named():
    stickers_px = camera(CORNER_STICKERS_CM + [CHECK_STICKER_CM, (30, 8)])
    with pytest.raises(CalibrationError, match="check sticker 5 .* check sticker 6"):
        calib.fit(stickers_px, CORNER_STICKERS_CM + [(16, 20), (30, 10)])


def test_two_maps_from_the_same_camera_agree():
    a = corners_only()
    b = calib.fit(camera(CORNER_STICKERS_CM), CORNER_STICKERS_CM)
    assert calib.disagreement_cm(a, b, (21, 14.85)) < 0.01


def test_bumped_camera_shows_as_disagreement():
    a = corners_only()
    bumped_px = [(x + 30, y + 10) for x, y in camera(CORNER_STICKERS_CM)]  # camera nudged ~30 px
    b = calib.fit(bumped_px, CORNER_STICKERS_CM)
    assert calib.disagreement_cm(a, b, (21, 14.85)) > 0.5


# ---------- bad input ----------

def test_needs_at_least_4_stickers():
    with pytest.raises(ValueError, match="at least 4"):
        calib.fit(camera(CORNER_STICKERS_CM[:3]), CORNER_STICKERS_CM[:3])


def test_point_counts_must_match():
    with pytest.raises(ValueError, match="4 pixel points but 3 table points"):
        calib.fit(camera(CORNER_STICKERS_CM), CORNER_STICKERS_CM[:3])


def test_corner_stickers_in_a_line_raise():
    in_a_line = [(0, 0), (10, 0), (20, 0), (0, 20)]
    with pytest.raises(ValueError, match="straight line"):
        calib.fit(camera(in_a_line), in_a_line)


# ---------- saving ----------

def test_save_and_load_give_the_same_map(tmp_path):
    stickers = CORNER_STICKERS_CM + [CHECK_STICKER_CM]
    c = calib.fit(camera(stickers), stickers)
    c.save(tmp_path / "camera.json")
    loaded = calib.Calib.load(tmp_path / "camera.json")
    assert np.allclose(loaded.H, c.H)
    assert loaded.table_pts_cm == c.table_pts_cm
    assert loaded.px_to_cm((800, 500)) == pytest.approx(c.px_to_cm((800, 500)))
