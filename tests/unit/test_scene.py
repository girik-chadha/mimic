"""scene.py tests. SAM is faked: boxes come from a made-up tilted camera, so no network and no photos."""
import math

import cv2
import numpy as np
import pytest

from mimic import config, sam, scene
from mimic.calib import Calib
from mimic.sam import Box
from mimic.scene import HeightCorrection, Layout, SceneError

# Same made-up camera as test_calib.py: the sheet seen at an angle in a 1920x1080 photo.
SHEET_CM = [(-2, -2), (40, -2), (40, 27.7), (-2, 27.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
TRUE_CM_TO_PX = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(SHEET_PX))

STICKERS = ["pink sticker", "orange sticker", "purple sticker", "yellow sticker", "light blue sticker"]
STICKERS_CM = [(0, 0), (38, 0), (38, 25.7), (0, 25.7), (14, 18)]
SLOTS_CM = [(32, 5), (32, 12.85), (32, 20.7)]
PHOTO = "photo.jpg"  # never opened: the fake SAM ignores it


def camera(pt_cm):
    out = cv2.perspectiveTransform(np.float64([pt_cm]).reshape(-1, 1, 2), TRUE_CM_TO_PX)
    return tuple(out.reshape(2))


def box_at(pt_cm, box_id=0):
    """A 40x40 px SAM box centred where the camera sees this table point."""
    x, y = camera(pt_cm)
    return Box(round(x) - 20, round(y) - 20, round(x) + 20, round(y) + 20, 1920, 1080, id=box_id)


def fake_sam(monkeypatch, boxes_by_name):
    asked = []

    def find(image_path, name, use_cache=True):
        asked.append(name)
        return boxes_by_name.get(name, [])

    monkeypatch.setattr(sam, "find", find)
    return asked


def make_layout(**changes):
    values = dict(sticker_names=STICKERS, stickers_cm=STICKERS_CM, slots_cm=SLOTS_CM, slot_tolerance_cm=3.0,
                  sheet_min_cm=(-2, -2), sheet_max_cm=(40, 27.7), height=None)
    values.update(changes)
    return Layout(**values)


def true_calib():
    return Calib(np.linalg.inv(TRUE_CM_TO_PX), [], [])


# ---------- layout.json ----------

def test_committed_layout_file_loads_and_is_sane():
    layout = scene.load_layout(config.LAYOUT_PATH)
    assert len(layout.sticker_names) == len(layout.stickers_cm) >= 4
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


# ---------- stickers and calibration ----------

def test_find_stickers_returns_pixel_centres_in_order(monkeypatch):
    fake_sam(monkeypatch, {n: [box_at(cm)] for n, cm in zip(STICKERS, STICKERS_CM)})
    centres = scene.find_stickers(PHOTO, STICKERS)
    for centre, cm in zip(centres, STICKERS_CM):
        assert math.dist(centre, camera(cm)) <= 1  # boxes are rounded to whole pixels


def test_find_stickers_needs_exactly_one_box_each(monkeypatch):
    boxes = {n: [box_at(cm)] for n, cm in zip(STICKERS, STICKERS_CM)}
    boxes["purple sticker"] = []
    fake_sam(monkeypatch, boxes)
    with pytest.raises(SceneError, match="SAM found 0 boxes for 'purple sticker'"):
        scene.find_stickers(PHOTO, STICKERS)


def test_calibrate_refuses_a_placeholder_layout(monkeypatch):
    asked = fake_sam(monkeypatch, {n: [box_at(cm)] for n, cm in zip(STICKERS, STICKERS_CM)})
    with pytest.raises(SceneError, match="measure your sheet and set placeholder to false"):
        scene.calibrate(PHOTO, make_layout(placeholder=True))
    assert asked == []  # refused before spending anything on SAM


def test_calibrate_from_a_photo_of_the_stickers(monkeypatch):
    fake_sam(monkeypatch, {n: [box_at(cm)] for n, cm in zip(STICKERS, STICKERS_CM)})
    c = scene.calibrate(PHOTO, make_layout())
    assert math.dist(c.px_to_cm(camera((20, 10))), (20, 10)) < 0.05
    assert c.fit_report() < 0.05


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
