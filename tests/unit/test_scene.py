"""scene.py tests. SAM is faked: boxes come from a made-up tilted camera, so no network and no photos."""
import json
import math

import cv2
import numpy as np
import pytest

from mimic import config, sam, scene
from mimic.calib import Calib, CalibrationError
from mimic.sam import Box
from mimic.scene import HeightCorrection, Layout, SceneError

# Same made-up camera as test_calib.py: the sheet seen at an angle in a 1920x1080 photo.
SHEET_CM = [(-2, -2), (40, -2), (40, 27.7), (-2, 27.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
TRUE_CM_TO_PX = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(SHEET_PX))

DOTS_CM = {"bottom_left": (0, 0), "bottom_right": (37, 0), "top_right": (37, 24.7), "top_left": (0, 24.7),
           "centre_check": (18.5, 12.4), "right_check": (34.5, 12.4)}
DOT_PROMPT = "small coloured dot"
SLOTS_CM = [(32, 5), (32, 12.85), (32, 20.7)]
PHOTO = "photo.jpg"  # never opened: the fake SAM ignores it


def camera(pt_cm):
    out = cv2.perspectiveTransform(np.float64([pt_cm]).reshape(-1, 1, 2), TRUE_CM_TO_PX)
    return tuple(out.reshape(2))


def box_at(pt_cm, box_id=0):
    """A 40x40 px SAM box centred where the camera sees this table point."""
    x, y = camera(pt_cm)
    return Box(round(x) - 20, round(y) - 20, round(x) + 20, round(y) + 20, 1920, 1080, id=box_id)


def dot_box(px, size=20):
    """A SAM box of `size` pixels centred on this pixel (20 px = a dot in a 1920-wide photo)."""
    half = size / 2
    return Box(round(px[0] - half), round(px[1] - half), round(px[0] + half), round(px[1] + half), 1920, 1080)


def same_view(p):
    return p


def rotated_90(p):
    return (960 - (p[1] - 540), 540 + (p[0] - 960))


def rotated_180(p):
    return (1920 - p[0], 1080 - p[1])


def mirrored(p):
    return (1920 - p[0], p[1])


def dot_boxes(view=same_view, dots_cm=DOTS_CM):
    """The 6 dots as SAM would return them, in an order that has nothing to do with their roles."""
    pts = [view(camera(cm)) for cm in dots_cm.values()]
    return [dot_box(pts[i]) for i in (3, 0, 5, 1, 4, 2)]


def fake_sam(monkeypatch, boxes_by_name):
    asked = []

    def find(image_path, name, use_cache=True):
        asked.append(name)
        return boxes_by_name.get(name, [])

    monkeypatch.setattr(sam, "find", find)
    return asked


def make_layout(**changes):
    values = dict(dots_cm=DOTS_CM, dot_prompt=DOT_PROMPT, dot_size_frac=(0.007, 0.02), slots_cm=SLOTS_CM,
                  slot_tolerance_cm=3.0, sheet_min_cm=(-2, -2), sheet_max_cm=(40, 27.7), height=None)
    values.update(changes)
    return Layout(**values)


def calibrate_with(monkeypatch, boxes, layout=None, labelled_path=None):
    fake_sam(monkeypatch, {DOT_PROMPT: boxes})
    return scene.calibrate(PHOTO, layout or make_layout(), labelled_path)


def true_calib():
    return Calib(np.linalg.inv(TRUE_CM_TO_PX), [], [])


# ---------- layout.json ----------

def test_committed_layout_file_loads_and_is_sane():
    layout = scene.load_layout(config.LAYOUT_PATH)
    assert list(layout.dots_cm) == scene.ROLES
    assert layout.dot_prompt == "small coloured dot"
    assert len(layout.slots_cm) >= 3
    assert layout.height is None  # off by default
    for slot in layout.slots_cm:
        assert layout.sheet_min_cm[0] <= slot[0] <= layout.sheet_max_cm[0]
        assert layout.sheet_min_cm[1] <= slot[1] <= layout.sheet_max_cm[1]


def test_layout_with_height_enabled_but_empty_raises(tmp_path):
    text = config.LAYOUT_PATH.read_text().replace('"enabled": false', '"enabled": true')
    (tmp_path / "layout.json").write_text(text)
    with pytest.raises(SceneError, match="not all of its numbers"):
        scene.load_layout(tmp_path / "layout.json")


def test_slot_too_close_to_a_dot_is_refused(tmp_path):
    data = json.loads(config.LAYOUT_PATH.read_text())
    data["slots_cm"] = [[28.0, 5.0], [32.0, 12.4], [28.0, 20.7]]  # slot 2 is 2.5 cm from the right check dot
    (tmp_path / "layout.json").write_text(json.dumps(data))
    with pytest.raises(SceneError, match="Slot 2 is only 2.5 cm from the right_check dot"):
        scene.load_layout(tmp_path / "layout.json")


# ---------- every photo calibrates itself from its 6 dots ----------

def test_six_dots_get_the_right_roles(monkeypatch):
    c = calibrate_with(monkeypatch, dot_boxes())
    for seen_px, role in zip(c.pixel_pts, scene.ROLES):
        assert math.dist(seen_px, camera(DOTS_CM[role])) <= 1, role
    assert c.fit_report() < 0.05
    assert math.dist(c.px_to_cm(camera((20, 10))), (20, 10)) < 0.05


@pytest.mark.parametrize("view", [rotated_90, rotated_180])
def test_rotated_photo_still_reads_right(monkeypatch, view):
    c = calibrate_with(monkeypatch, dot_boxes(view))
    assert math.dist(c.px_to_cm(view(camera((10, 5)))), (10, 5)) < 0.05


def test_mirrored_photo_is_read_upside_down_known_limit(monkeypatch):
    # KNOWN LIMIT (docs/todo.md, blocker before Gate 3): both check dots lie on the sheet's middle line,
    # so a mirrored photo flipped top-to-bottom still puts them within 0.1 cm. It is NOT refused: it
    # calibrates with y upside down. Moving the check dots off the middle lines fixes this.
    c = calibrate_with(monkeypatch, dot_boxes(mirrored))
    assert math.dist(c.px_to_cm(mirrored(camera((10, 5)))), (10, 24.7 - 5)) < 0.05


def test_five_dots_stop(monkeypatch):
    with pytest.raises(SceneError, match="Found 5 dots, need exactly 6"):
        calibrate_with(monkeypatch, dot_boxes()[:5])


def test_seven_dots_stop(monkeypatch):
    extra = dot_box(camera((10, 20)))
    with pytest.raises(SceneError, match="Found 7 dots, need exactly 6"):
        calibrate_with(monkeypatch, dot_boxes() + [extra])


def test_tiny_speck_is_dropped_by_size(monkeypatch):
    speck = dot_box(camera((44, 10)), size=6)
    c = calibrate_with(monkeypatch, dot_boxes() + [speck])
    assert c.fit_report() < 0.05


def test_block_is_not_mistaken_for_a_dot(monkeypatch):
    block = dot_box(camera((8, 12)), size=60)
    c = calibrate_with(monkeypatch, dot_boxes() + [block])
    assert c.fit_report() < 0.05


def test_dot_sized_speck_off_the_sheet_stops(monkeypatch):
    speck = dot_box(camera((44, -4)), size=20)
    with pytest.raises(SceneError, match="Found 7 dots"):
        calibrate_with(monkeypatch, dot_boxes() + [speck])


def test_dots_dropped_for_size_are_listed(monkeypatch):
    block = dot_box(camera((8, 12)), size=60)
    with pytest.raises(SceneError, match=r"Dropped for size: .* 0\.031"):
        calibrate_with(monkeypatch, dot_boxes()[:5] + [block])


def test_wrong_layout_stops_naming_the_check_dot(monkeypatch):
    wrong = dict(DOTS_CM, right_check=(31.5, 12.4))  # layout.json says 3 cm left of where the dot really is
    with pytest.raises(CalibrationError, match="right_check 3.00 cm off"):
        calibrate_with(monkeypatch, dot_boxes(), make_layout(dots_cm=wrong))


def test_symmetric_check_dots_are_ambiguous(monkeypatch):
    symmetric = dict(DOTS_CM, centre_check=(10, 8), right_check=(27, 16.7))  # swap places under a half turn
    with pytest.raises(SceneError, match="Two different ways of naming the dots both fit"):
        calibrate_with(monkeypatch, dot_boxes(dots_cm=symmetric), make_layout(dots_cm=symmetric))


def test_check_dot_outside_the_corners_stops(monkeypatch):
    outside = dict(DOTS_CM, right_check=(39, 12.4))
    with pytest.raises(SceneError, match="5 dots are on the outside"):
        calibrate_with(monkeypatch, dot_boxes(dots_cm=outside), make_layout(dots_cm=outside))


def test_calibrate_refuses_a_placeholder_layout(monkeypatch):
    asked = fake_sam(monkeypatch, {DOT_PROMPT: dot_boxes()})
    with pytest.raises(SceneError, match="measure your sheet and set placeholder to false"):
        scene.calibrate(PHOTO, make_layout(placeholder=True))
    assert asked == []  # refused before spending anything on SAM


def test_labelled_photo_is_saved(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    cv2.imwrite(str(photo), np.full((1080, 1920, 3), 235, np.uint8))
    fake_sam(monkeypatch, {DOT_PROMPT: dot_boxes()})
    scene.calibrate(str(photo), make_layout(), tmp_path / "ok_dots.jpg")
    assert (tmp_path / "ok_dots.jpg").exists()


def test_labelled_photo_is_saved_when_it_stops_too(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    cv2.imwrite(str(photo), np.full((1080, 1920, 3), 235, np.uint8))
    fake_sam(monkeypatch, {DOT_PROMPT: dot_boxes()[:5]})
    with pytest.raises(SceneError, match="Labelled photo: .*bad_dots.jpg"):
        scene.calibrate(str(photo), make_layout(), tmp_path / "bad_dots.jpg")
    assert (tmp_path / "bad_dots.jpg").exists()


# ---------- positions ----------

def test_read_positions_in_cm(monkeypatch):
    fake_sam(monkeypatch, {"red block": [box_at((5, 4))], "blue block": [box_at((12, 20))]})
    got = scene.read_positions(PHOTO, ["red block", "blue block"], true_calib(), make_layout())
    assert math.dist(got["red block"], (5, 4)) < 0.05
    assert math.dist(got["blue block"], (12, 20)) < 0.05


def test_zero_boxes_raises(monkeypatch):
    fake_sam(monkeypatch, {"red block": []})
    with pytest.raises(SceneError, match="SAM found 0 boxes for 'red block'"):
        scene.read_positions(PHOTO, ["red block"], true_calib(), make_layout())


def test_two_boxes_raises(monkeypatch):
    fake_sam(monkeypatch, {"red block": [box_at((5, 4)), box_at((20, 10), box_id=1)]})
    with pytest.raises(SceneError, match="SAM found 2 boxes for 'red block'"):
        scene.read_positions(PHOTO, ["red block"], true_calib(), make_layout())


def test_object_off_the_sheet_raises(monkeypatch):
    fake_sam(monkeypatch, {"red block": [box_at((45, 10))]})
    with pytest.raises(SceneError, match=r"'red block' is at \(45.0, 10.0\) cm, which is off the sheet"):
        scene.read_positions(PHOTO, ["red block"], true_calib(), make_layout())


# ---------- height correction ----------

LENS = HeightCorrection(camera_height_cm=45.0, under_lens_cm=(19.0, 12.85), point_height_cm=1.25)


def test_height_correction_off_changes_nothing(monkeypatch):
    assert scene.correct_for_height((35.0, 3.0), None) == (35.0, 3.0)
    fake_sam(monkeypatch, {"red block": [box_at((35, 3))]})
    off = scene.read_positions(PHOTO, ["red block"], true_calib(), make_layout(height=None))
    assert off["red block"] == true_calib().px_to_cm(box_at((35, 3)).centre)


def test_height_correction_undoes_the_camera_ray():
    # Draw the ray from the lens through the raised point down to the table: that is where the flat map
    # puts it. The correction must bring it back to the point straight below.
    true_pt = (35.0, 3.0)
    lens = (*LENS.under_lens_cm, LENS.camera_height_cm)
    raised = (*true_pt, LENS.point_height_cm)
    s = lens[2] / (lens[2] - raised[2])
    seen_on_table = (lens[0] + (raised[0] - lens[0]) * s, lens[1] + (raised[1] - lens[1]) * s)
    assert math.dist(seen_on_table, true_pt) > 0.5  # without correction: more than half a cm out
    assert math.dist(scene.correct_for_height(seen_on_table, LENS), true_pt) < 1e-9


def test_height_correction_on_is_used_by_read_positions(monkeypatch):
    fake_sam(monkeypatch, {"red block": [box_at((35, 3))]})
    flat = true_calib().px_to_cm(box_at((35, 3)).centre)
    on = scene.read_positions(PHOTO, ["red block"], true_calib(), make_layout(height=LENS))
    assert on["red block"] == pytest.approx(scene.correct_for_height(flat, LENS))
    assert math.dist(on["red block"], flat) > 0.4  # it really moved


def test_height_correction_moves_points_the_right_amount():
    # 20 cm from the spot under the lens, 1.25 cm up, lens 45 cm up: pulled in by 20 * 1.25 / 45 = 0.556 cm.
    moved = scene.correct_for_height((39.0, 12.85), LENS)
    assert moved == pytest.approx((39.0 - 20 * 1.25 / 45, 12.85))
    assert scene.correct_for_height(LENS.under_lens_cm, LENS) == pytest.approx(LENS.under_lens_cm)


# ---------- slots and the target order ----------

def test_read_target_returns_slot_order(monkeypatch):
    fake_sam(monkeypatch, {"red block": [box_at((32.4, 20.5))],    # slot 3
                           "blue block": [box_at((31.8, 5.3))],    # slot 1
                           "green block": [box_at((32.2, 12.6))]})  # slot 2
    order = scene.read_target(PHOTO, ["red block", "blue block", "green block"], true_calib(), make_layout())
    assert order == ["blue block", "green block", "red block"]


def test_empty_slot_is_skipped():
    order = scene.assign_slots({"red block": (32, 20.7), "blue block": (32, 5)}, SLOTS_CM, 3.0)
    assert order == ["blue block", "red block"]


def test_object_not_in_a_slot_raises():
    with pytest.raises(SceneError, match=r"'red block' is 4.0 cm from the nearest slot \(slot 2\)"):
        scene.assign_slots({"red block": (28, 12.85)}, SLOTS_CM, 3.0)


def test_two_objects_in_one_slot_raise():
    with pytest.raises(SceneError, match="'red block' and 'blue block' are both in slot 2"):
        scene.assign_slots({"red block": (32, 12), "blue block": (32, 14)}, SLOTS_CM, 3.0)


def test_tolerance_over_half_the_slot_spacing_raises():
    # Slots are 7.85 cm apart, so the tolerance may be at most 3.925 cm.
    scene.assign_slots({"red block": (32, 5)}, SLOTS_CM, 3.9)
    with pytest.raises(SceneError, match="more than half the slot spacing"):
        scene.assign_slots({"red block": (32, 5)}, SLOTS_CM, 4.0)


# ---------- the two real photos (SAM replies saved 30 Oct) ----------

def real_dots(stem):
    """The committed layout (placeholder switched off in memory only) and a fake SAM giving the saved real reply."""
    reply = (config.RESPONSES_DIR / f"sam_photo_{stem}_small_coloured_dot.txt").read_text()
    photo = str(config.ROOT / "tests" / "fixtures" / "photos" / f"{stem}.jpg")
    layout = scene.load_layout(config.LAYOUT_PATH)
    layout.placeholder = False
    return photo, layout, {layout.dot_prompt: sam.parse_boxes(reply)}


def test_real_photo_6a_calibrates_itself(tmp_path, monkeypatch):
    # PROVISIONAL: layout.json's dot positions come from coin and paper sizes, not a ruler. Update these
    # expected numbers after the ruler measurement (the right check dot is expected to move to ~35.0 cm).
    photo, layout, replies = real_dots("dots_check_6a")
    fake_sam(monkeypatch, replies)
    c = scene.calibrate(photo, layout, tmp_path / "6a_dots.jpg")
    errors = dict(zip(scene.ROLES, (c.check(px, cm) for px, cm in zip(c.pixel_pts, c.table_pts_cm))))
    assert errors["right_check"] == pytest.approx(0.49, abs=0.01)
    assert errors["centre_check"] == pytest.approx(0.22, abs=0.01)
    assert c.pixel_pts[0] == (115.5, 791.5)  # the red dot, bottom left in the photo, is bottom_left
    assert c.pixel_pts[2] == (1066.5, 161.0)  # the green dot, top right in the photo, is top_right
    assert (tmp_path / "6a_dots.jpg").exists()


def test_real_photo_6b_stops_on_the_right_check_dot(tmp_path, monkeypatch):
    # PROVISIONAL: with the coin-and-paper numbers the right check dot lands 0.56 cm off, over the 0.5 cm
    # limit, so this photo is refused. Update this test after the ruler measurement (it should then pass).
    photo, layout, replies = real_dots("dots_check_6b")
    fake_sam(monkeypatch, replies)
    with pytest.raises(CalibrationError, match=r"right_check 0\.56 cm off"):
        scene.calibrate(photo, layout, tmp_path / "6b_dots.jpg")
    assert (tmp_path / "6b_dots.jpg").exists()


def test_real_replies_have_six_dot_sized_boxes_and_nothing_else():
    lo, hi = scene.load_layout(config.LAYOUT_PATH).dot_size_frac
    for stem in ["dots_check_6a", "dots_check_6b"]:
        boxes = sam.parse_boxes((config.RESPONSES_DIR / f"sam_photo_{stem}_small_coloured_dot.txt").read_text())
        sizes = [scene.dot_size(b) for b in boxes]
        assert len(boxes) == 6, stem
        assert all(lo <= s <= hi for s in sizes), (stem, sizes)
