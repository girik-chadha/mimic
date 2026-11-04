"""scene.py tests. SAM is faked: boxes come from a made-up tilted camera, so no network and no photos."""
import json
import math

import cv2
import numpy as np
import pytest

from mimic import calib, config, sam, scene

REPLIES = config.ROOT / "tests" / "fixtures" / "meta_responses"
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
                  slot_tolerance_cm=3.0, sheet_min_cm=(-2, -2), sheet_max_cm=(40, 27.7), height=None,
                  dots_measured=True)
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
    # ruler-measured 3 Nov from the red dot's centre; slots, mess area and reach are still placeholders
    assert layout.dots_measured is True and layout.placeholder is True
    assert layout.dots_cm == {"bottom_left": (0.0, 0.0), "bottom_right": (36.0, -0.1), "top_right": (36.0, 24.9),
                              "top_left": (-0.2, 25.1), "centre_check": (18.3, 12.5), "right_check": (34.3, 12.4)}
    assert (layout.sheet_min_cm, layout.sheet_max_cm) == ((-2.5, -2.3), (38.5, 27.2))
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
    data["slots_cm"] = [[28.0, 5.0], [31.8, 12.4], [28.0, 20.7]]  # slot 2 is 2.5 cm from the right check dot
    (tmp_path / "layout.json").write_text(json.dumps(data))
    with pytest.raises(SceneError, match="Slot 2 is only 2.5 cm from the right_check dot"):
        scene.load_layout(tmp_path / "layout.json")


def test_mess_area_too_close_to_a_dot_is_refused(tmp_path):
    data = json.loads(config.LAYOUT_PATH.read_text())
    data["mess_area_cm"] = {"min": [5.0, 10.0], "max": [15.8, 20.0]}  # 2.5 cm from the centre check dot
    (tmp_path / "layout.json").write_text(json.dumps(data))
    with pytest.raises(SceneError, match="mess area is only 2.5 cm from the centre_check dot"):
        scene.load_layout(tmp_path / "layout.json")


def test_committed_layout_is_inside_the_reach():
    layout = scene.load_layout(config.LAYOUT_PATH)
    (x0, y0), (x1, y1) = layout.mess_area_cm
    for point in layout.slots_cm + [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        assert scene.reach_problem(point, layout.reach_cm) is None, point


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


# ---------- more than 6 dot-sized boxes: pick the 6 that fit ----------
# (Until 2 Nov, 7 dot-sized boxes always stopped the run. Now the best-fitting 6 are kept.)

CLAW_MARKER = (10, 20)     # a sticker on the claw tip, lying on the sheet
TABLE_SPECK = (44, -4)     # off the sheet
DOT_LIKE_BLOCK = (30, 6)   # a small block that SAM calls a dot


@pytest.mark.parametrize("extras", [[CLAW_MARKER], [CLAW_MARKER, TABLE_SPECK],
                                    [CLAW_MARKER, TABLE_SPECK, DOT_LIKE_BLOCK]])
def test_seven_eight_or_nine_candidates_find_the_right_six(monkeypatch, extras):
    boxes = dot_boxes() + [dot_box(camera(cm)) for cm in extras]
    c = calibrate_with(monkeypatch, boxes)
    for seen_px, role in zip(c.pixel_pts, scene.ROLES):
        assert math.dist(seen_px, camera(DOTS_CM[role])) <= 1, role
    assert c.fit_report() < 0.05


def test_extras_are_listed_as_leftovers():
    pts = [camera(cm) for cm in DOTS_CM.values()] + [camera(CLAW_MARKER), camera(TABLE_SPECK)]
    ordered, leftovers = scene.choose_dots(pts, make_layout())
    assert len(ordered) == 6
    assert sorted(leftovers) == sorted([camera(CLAW_MARKER), camera(TABLE_SPECK)])


def test_two_equally_good_sets_stop(monkeypatch):
    # a second dot-sized thing a few millimetres from the top-right dot: either could be that corner
    near_corner = (37.1, 24.7)
    with pytest.raises(SceneError, match="Two different sets of 6 dots fit the layout equally well"):
        calibrate_with(monkeypatch, dot_boxes() + [dot_box(camera(near_corner))])


def test_extras_with_no_good_six_stop(monkeypatch):
    wrong = dict(DOTS_CM, right_check=(31.5, 12.4))  # layout says 3 cm off from where the dot really is
    with pytest.raises(CalibrationError, match="Found 7 dot-sized boxes, but no 6 of them fit"):
        calibrate_with(monkeypatch, dot_boxes() + [dot_box(camera(CLAW_MARKER))], make_layout(dots_cm=wrong))


def test_more_than_twelve_candidates_stop(monkeypatch):
    extras = [dot_box(camera((5 + 2 * i, 18))) for i in range(7)]  # 13 in all
    with pytest.raises(SceneError, match="Found 13 dot-sized boxes, more than 12"):
        calibrate_with(monkeypatch, dot_boxes() + extras)


def test_tiny_speck_is_dropped_by_size(monkeypatch):
    speck = dot_box(camera((44, 10)), size=6)
    c = calibrate_with(monkeypatch, dot_boxes() + [speck])
    assert c.fit_report() < 0.05


def test_block_is_not_mistaken_for_a_dot(monkeypatch):
    block = dot_box(camera((8, 12)), size=60)
    c = calibrate_with(monkeypatch, dot_boxes() + [block])
    assert c.fit_report() < 0.05


def test_dot_sized_speck_off_the_sheet_is_left_out(monkeypatch):
    speck = dot_box(camera(TABLE_SPECK), size=20)
    c = calibrate_with(monkeypatch, dot_boxes() + [speck])
    assert c.fit_report() < 0.05


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


def test_calibrate_refuses_unmeasured_dots(monkeypatch):
    asked = fake_sam(monkeypatch, {DOT_PROMPT: dot_boxes()})
    with pytest.raises(SceneError, match="dot positions are not measured yet"):
        scene.calibrate(PHOTO, make_layout(placeholder=True, dots_measured=False))
    assert asked == []  # refused before spending anything on SAM


def test_calibrate_needs_only_the_dots_not_the_whole_layout(monkeypatch):
    fake_sam(monkeypatch, {DOT_PROMPT: dot_boxes()})
    scene.calibrate(PHOTO, make_layout(placeholder=True, dots_measured=True))   # slots still placeholders: fine


def test_the_arm_side_still_needs_the_whole_layout_real():
    with pytest.raises(SceneError, match="placeholder numbers"):
        scene.refuse_placeholder(make_layout(placeholder=True, dots_measured=True))
    with pytest.raises(SceneError, match="placeholder numbers"):
        scene.refuse_placeholder(make_layout(placeholder=False, dots_measured=False))
    scene.refuse_placeholder(make_layout(placeholder=False, dots_measured=True))


def test_layout_file_cannot_be_real_while_its_dots_are_not(tmp_path):
    data = json.loads(config.LAYOUT_PATH.read_text())
    data.update(placeholder=False, dots_measured=False)
    (tmp_path / "layout.json").write_text(json.dumps(data))
    with pytest.raises(SceneError, match="placeholder is false but dots_measured is not true"):
        scene.load_layout(tmp_path / "layout.json")
    del data["dots_measured"]
    data["placeholder"] = True
    (tmp_path / "layout.json").write_text(json.dumps(data))
    assert scene.load_layout(tmp_path / "layout.json").dots_measured is False   # missing means not measured


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


# ---------- the three real photos (SAM replies saved 30 Oct and 3 Nov), ruler-measured dots ----------

def real_dots(stem):
    """The committed layout, as it is (dots measured), and a fake SAM giving the saved real reply."""
    reply = (config.RESPONSES_DIR / f"sam_photo_{stem}_small_coloured_dot.txt").read_text()
    photo = str(config.ROOT / "tests" / "fixtures" / "photos" / f"{stem}.jpg")
    layout = scene.load_layout(config.LAYOUT_PATH)
    return photo, layout, {layout.dot_prompt: sam.parse_boxes(reply)}


def check_errors(c) -> dict:
    return {role: round(c.check(px, cm), 3) for role, px, cm in zip(scene.ROLES, c.pixel_pts, c.table_pts_cm)
            if role in scene.CHECKS}


# With the ruler numbers (3 Nov) all three photos calibrate. Errors in cm; the limit is 0.5.
REAL_PHOTO_CHECK_ERRORS = {
    "dots_check_6a": {"centre_check": 0.129, "right_check": 0.266},
    "dots_check_6b": {"centre_check": 0.153, "right_check": 0.186},
    "ivcam_arm":     {"centre_check": 0.202, "right_check": 0.189},
}


@pytest.mark.parametrize("stem", list(REAL_PHOTO_CHECK_ERRORS))
def test_real_photo_calibrates_with_the_ruler_measured_dots(stem, tmp_path, monkeypatch):
    photo, layout, replies = real_dots(stem)
    fake_sam(monkeypatch, replies)
    c = scene.calibrate(photo, layout, tmp_path / f"{stem}_dots.jpg")
    errors = check_errors(c)
    print(f"{stem}: check-dot errors {errors} cm (limit {calib.MAX_CHECK_ERROR_CM})")
    assert all(e <= calib.MAX_CHECK_ERROR_CM for e in errors.values())
    assert errors == pytest.approx(REAL_PHOTO_CHECK_ERRORS[stem], abs=0.01)
    assert (tmp_path / f"{stem}_dots.jpg").exists()


def test_real_photo_6a_roles_are_by_position():
    photo, layout, replies = real_dots("dots_check_6a")
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake_sam(monkeypatch, replies)
        c = scene.calibrate(photo, layout)
    assert c.pixel_pts[0] == (115.5, 791.5)  # the red dot, bottom left in the photo, is bottom_left
    assert c.pixel_pts[2] == (1066.5, 161.0)  # the green dot, top right in the photo, is top_right


def test_real_replies_have_six_dot_sized_boxes_and_nothing_else():
    lo, hi = scene.load_layout(config.LAYOUT_PATH).dot_size_frac
    for stem in ["dots_check_6a", "dots_check_6b"]:
        boxes = sam.parse_boxes((config.RESPONSES_DIR / f"sam_photo_{stem}_small_coloured_dot.txt").read_text())
        sizes = [scene.dot_size(b) for b in boxes]
        assert len(boxes) == 6, stem
        assert all(lo <= s <= hi for s in sizes), (stem, sizes)


def test_a_die_twice_the_size_of_the_dots_is_dropped_for_its_size(monkeypatch):
    # The fake camera's photos are 1920 px wide, so a die that is 38 px on a 1280-px frame (fraction 0.0297) is 57 px:
    # 1.9 times the 30-px dots. Sizes are judged against the other dots, so it is dropped (4 Nov).
    die_px = round(38 / 1280 * 1920)
    die = dot_box(camera((10, 20)), size=die_px)
    layout = make_layout(dot_size_frac=(0.005, 0.08))
    dots = [dot_box(camera(cm), size=30) for cm in DOTS_CM.values()]
    fake_sam(monkeypatch, {DOT_PROMPT: dots + [die]})
    centres, dropped = scene.find_dots(PHOTO, layout)
    assert len(centres) == 6 and dropped == [die]
    ordered, leftovers = scene.choose_dots(centres, layout)
    assert leftovers == []
    for seen_px, role in zip(ordered, scene.ROLES):
        assert math.dist(seen_px, camera(DOTS_CM[role])) <= 1, role


def test_real_ivcam_frame_reply_has_six_dot_sized_boxes():
    # Real reply (3 Nov) on an iVCam frame with the arm in place; it calibrates (see the test above).
    photo, layout, replies = real_dots("ivcam_arm")
    lo, hi = layout.dot_size_frac
    assert len(replies[layout.dot_prompt]) == 6
    assert all(lo <= scene.dot_size(b) <= hi for b in replies[layout.dot_prompt])


def test_a_photo_taken_closer_calibrates_with_the_fallback_prompt(monkeypatch):
    """A real phone photo (4 Nov, 1600 px wide, taken closer): "small coloured dot" found 4 of the 6 dots, one of them
    at 0.0312 of the width (over the old 0.030 limit); "small circle" found all 6, a speck and a hole in the arm's plate.
    The size test is relative to the other dots, and the layout's geometry picks the 6."""
    replies = {name: sam.parse_boxes((REPLIES / f"sam_photo_close_upload_{name.replace(' ', '_')}.txt").read_text())
               for name in ("small coloured dot", "small circle")}
    asked = []
    monkeypatch.setattr(sam, "find", lambda path, name, use_cache=True: asked.append(name) or replies[name])
    calib = scene.calibrate("close_upload.jpg", scene.load_layout())
    assert asked == ["small coloured dot", "small circle"]
    errors = [calib.check(px, cm) for px, cm in zip(calib.pixel_pts[4:], calib.table_pts_cm[4:])]
    assert max(errors) < 0.5
