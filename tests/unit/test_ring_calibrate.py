"""ring_calibrate.py with a fake arm, camera and SAM: the tip is where a made-up true ring puts it."""
import dataclasses
import importlib.util
import json
import math
import re
import shutil

import cv2
import numpy as np
import pytest

from mimic import arm, camera, config, meta_api, ring, sam, scene, session
from mimic.arm import Arm, FakeTransport
from mimic.sam import Box

spec = importlib.util.spec_from_file_location("ring_calibrate", config.ROOT / "scripts/ring_calibrate.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

REAL_CONNECT = session.connect   # kept before any test replaces it
W, H = 1280, 720
CENTRE, RADIUS = (18.5, 32.0), 14.0
LIMITS = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH),
                             ranges_deg={"base": (0, 180), "shoulder": (80, 180), "elbow": (30, 120)},
                             placeholder=False, limits_checked_by_hand=True)


def to_px(cm):
    return (100 + cm[0] * 28, 680 - cm[1] * 22)


def box_at(cm, size):
    x, y = to_px(cm)
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), W, H)


class FakeWorld:
    """The true ring: base theta puts the tip at phi = theta / scale + 180 degrees around CENTRE."""

    def __init__(self, monkeypatch, tmp_path, scale=1.0, table=180):
        self.scale, self.table = scale, table
        self.layout = scene.load_layout(config.LAYOUT_PATH)       # dots measured
        self.transport = FakeTransport(LIMITS)                     # disarmed, like the firmware
        self.arm = Arm(self.transport, LIMITS)
        self.ring_path = tmp_path / "ring.json"
        shutil.copy(config.ROOT / "calib" / "ring.json", self.ring_path)
        self.photos, self.budgets, self.asked = {}, [], []
        self.hide_dot_at, self.extra_marker_at, self.marker_shift = set(), set(), {}
        monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: LIMITS)
        monkeypatch.setattr(session, "connect", lambda dry_run, limits, echo=print: self.arm)
        monkeypatch.setattr(sam, "find", self.find)

    def tip(self, base):
        phi = math.radians(base / self.scale + 180)
        return (CENTRE[0] + RADIUS * math.cos(phi), CENTRE[1] + RADIUS * math.sin(phi))

    def capture(self, path):
        base, shoulder = self.transport.angles[0], self.transport.angles[1]
        assert shoulder == self.table, "the photo must be taken with the tip on the table"
        cv2.imwrite(str(path), np.full((H, W, 3), 230, np.uint8))
        self.photos[str(path)] = base
        return str(path)

    def find(self, image_path, name, use_cache=True):
        self.budgets.append(config.BUDGET_USD)
        base = self.photos[image_path]
        if name == self.layout.dot_prompt:
            dots = [box_at(cm, 25) for cm in self.layout.dots_cm.values()]
            return dots[:5] if base in self.hide_dot_at else dots
        if name in scene.DOT_FALLBACK_PROMPTS:
            return []   # a hidden dot stays hidden
        self.asked.append(name)
        x, y = self.tip(base)
        dx, dy = self.marker_shift.get(base, (0, 0))
        boxes = [box_at((x + dx, y + dy), 30)]
        return boxes * 2 if base in self.extra_marker_at else boxes

    def run(self, typed, argv=()):
        queue, said = list(typed), []
        code = rc.main(list(argv), read=lambda q: queue.pop(0), capture=self.capture, sleep=lambda s: None,
                       say=said.append, ring_path=self.ring_path, out_root=self.ring_path.parent / "runs")
        return code, said


ANGLES = ["80", "90", "100", "110", "120"]


def test_five_points_fit_the_true_ring_and_save_it(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    code, said = world.run(["yes"] + ANGLES + ["done", "yes"])
    assert code == 0
    saved = ring.load_ring(world.ring_path)
    assert saved.placeholder is False and len(saved.points) == 5 and saved.source == "camera"
    assert saved.centre_cm == pytest.approx(CENTRE, abs=0.1) and saved.radius_cm == pytest.approx(RADIUS, abs=0.1)
    for base in (80, 100, 120):
        assert math.dist(saved.point_at(base), world.tip(base)) < 0.1
    assert saved.camera_clear == {"base": 90, "shoulder": 90} and saved.max_off_ring_cm == 1.0   # kept
    assert any(line.startswith("Ring: centre") for line in said)
    assert set(world.budgets) == {0.05} and config.BUDGET_USD != 0.05           # capped, then put back
    assert world.transport.sent[-1] == "D" and any(line.startswith("Hold the arm") for line in said)


def test_the_table_angle_comes_from_ring_json_and_the_stop_is_3_past_it(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    code, said = world.run(["yes"] + ANGLES + ["done", "yes"])
    assert any(line.startswith("Table height: shoulder 180; the shoulder is never sent past 183. Marker: 'small yellow "
                               "rectangle'.") for line in said)
    assert set(world.asked) == {"small yellow rectangle"}             # marker_name from calib/ring.json
    moves = [[int(v) for v in line.split()[1:5]] for line in world.transport.sent if line.startswith("J")]
    assert max(m[1] for m in moves) == 180 and moves[0][1] == 170     # the approach starts 10 above the table


def test_table_shoulder_overrides_ring_json_and_is_saved_with_the_fit(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path, table=170)
    code, said = world.run(["yes"] + ANGLES + ["done", "yes"], argv=["--table-shoulder", "170"])
    assert code == 0
    assert any(line.startswith("Table height: shoulder 170; the shoulder is never sent past 173.") for line in said)
    moves = [[int(v) for v in line.split()[1:5]] for line in world.transport.sent if line.startswith("J")]
    assert max(m[1] for m in moves) == 170 and [m[1] for m in moves[:2]] == [160, 162]  # above: 170 - approach 10
    saved = ring.load_ring(world.ring_path)
    assert saved.shoulder_table == 170 and saved.stop_at is None and saved.grip_hold == 20   # the rest kept


def test_a_servo_that_turns_short_has_its_scale_fitted(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path, scale=0.9)
    code, said = world.run(["yes", "60", "75", "90", "105", "120", "done", "yes"])
    assert code == 0 and ring.load_ring(world.ring_path).scale == pytest.approx(0.9, abs=0.01)   # pixel rounding


def test_the_base_only_turns_with_the_claw_lifted_and_the_shoulder_never_passes_the_table(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    world.run(["yes"] + ANGLES + ["done", "yes"])
    moves = [[int(v) for v in line.split()[1:5]] for line in world.transport.sent if line.startswith("J")]
    assert max(m[1] for m in moves) == 180
    for before, after in zip(moves, moves[1:]):
        if before[0] != after[0]:
            assert before[1] <= 170 and after[1] <= 170      # turned only above the table (180 - approach 10)
    assert all(m[2] == 90 and m[3] == 90 for m in moves)      # elbow 90 (empty socket), claw open


def test_a_photo_with_a_hidden_dot_or_two_markers_is_skipped_and_another_angle_asked(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    world.hide_dot_at, world.extra_marker_at = {90}, {110}
    code, said = world.run(["yes", "80", "90", "95", "100", "110", "115", "120", "done", "yes"])
    assert code == 0 and [p["base_deg"] for p in ring.load_ring(world.ring_path).points] == [80, 95, 100, 115, 120]
    assert any("skipped: Found 5 dots" in line for line in said)
    assert any("skipped: SAM found 2 boxes for 'small yellow rectangle', 2 on the sheet (exactly 1 is needed)" in line
               for line in said)


def test_marker_name_override_is_used_and_saved_and_every_point_has_its_marker_photo(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    code, said = world.run(["yes"] + ANGLES + ["done", "yes"], argv=["--marker-name", "yellow tape"])
    assert code == 0 and set(world.asked) == {"yellow tape"}
    assert ring.load_ring(world.ring_path).marker_name == "yellow tape"
    run_dir = next((world.ring_path.parent / "runs").iterdir())
    assert len(list(run_dir.glob("*_marker.jpg"))) == 5


# ---------- choosing the marker among several boxes (the real iVCam frame, 4 Nov) ----------

def real_frame_px(cm):
    """A made-up camera in which, as on the real frame, (743, 182) is on the sheet and (578, 160) is just off it."""
    return (100 + cm[0] * 28, 715 - cm[1] * 20)


def frame_with(tmp_path, monkeypatch, marker_boxes):
    layout = scene.load_layout(config.LAYOUT_PATH)
    photo = tmp_path / "frame.jpg"
    cv2.imwrite(str(photo), np.full((H, W, 3), 230, np.uint8))

    def find(image_path, name, use_cache=True):
        if name in scene.DOT_FALLBACK_PROMPTS:
            return []
        if name == layout.dot_prompt:
            return [Box(*(round(v) for v in (x - 12, y - 12, x + 12, y + 12)), W, H)
                    for x, y in (real_frame_px(cm) for cm in layout.dots_cm.values())]
        return marker_boxes

    monkeypatch.setattr(sam, "find", find)
    return str(photo), layout


def box_around(px, w, h):
    return Box(px[0] - w // 2, px[1] - h // 2, px[0] + w // 2, px[1] + h // 2, W, H)


def test_the_marker_on_the_sheet_is_chosen_and_the_board_pins_are_dropped(tmp_path, monkeypatch):
    marker, pins = box_around((743, 182), 30, 20), box_around((578, 160), 40, 14)
    photo, layout = frame_with(tmp_path, monkeypatch, [marker, pins])
    said = []
    x, y = rc.measure_tip(photo, layout, tmp_path, "p1", "small yellow rectangle", said.append)
    assert (x, y) == pytest.approx(((743 - 100) / 28, (715 - 182) / 20), abs=0.05)    # the marker, on the sheet
    assert "(743, 182) px -> (23.0, 26.6) cm, on the sheet" in said[0]
    assert "(578, 160) px -> (17.1, 27.7) cm, off the sheet" in said[0] and "using the one on the sheet" in said[0]
    assert cv2.imread(str(tmp_path / "p1_marker.jpg")) is not None                     # the labelled photo


def test_no_box_on_the_sheet_or_two_on_it_refuse_the_point_and_list_every_box(tmp_path, monkeypatch):
    pins = box_around((578, 160), 40, 14)
    photo, layout = frame_with(tmp_path, monkeypatch, [pins])          # a lone box off the sheet is not the tip
    with pytest.raises(scene.SceneError, match=r"1 boxes for 'small yellow rectangle', 0 on the sheet .*"
                                               r"\(578, 160\) px -> \(17.1, 27.7\) cm, off the sheet"):
        rc.measure_tip(photo, layout, tmp_path, "p2", "small yellow rectangle", lambda s: None)
    two = [box_around((743, 182), 30, 20), box_around((700, 400), 30, 20)]
    photo, layout = frame_with(tmp_path, monkeypatch, two)
    with pytest.raises(scene.SceneError, match="2 boxes for 'small yellow rectangle', 2 on the sheet"):
        rc.measure_tip(photo, layout, tmp_path, "p3", "small yellow rectangle", lambda s: None)
    assert (tmp_path / "p2_marker.jpg").exists() and (tmp_path / "p3_marker.jpg").exists()   # saved to look at


def test_a_photo_still_too_bright_after_the_retake_refuses_that_point_only(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    real_capture = world.capture

    def capture(path):
        if world.transport.angles[0] == 90:
            raise camera.CameraError("The photo x is too bright (blown out): mean brightness 247.6, allowed 40-235, "
                                     "twice. Lock the exposure in iVCam")
        return real_capture(path)

    world.capture = capture
    code, said = world.run(["yes", "80", "90", "95", "100", "110", "120", "done", "yes"])
    assert code == 0 and [p["base_deg"] for p in ring.load_ring(world.ring_path).points] == [80, 95, 100, 110, 120]
    assert any("skipped: The photo x is too bright (blown out): mean brightness 247.6" in line for line in said)


def test_the_real_camera_reads_camera_frames_from_ring_json_or_the_option(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    grabs = []

    def fake_capture(path, frames=15, sleep=None, say=None, name=None):
        grabs.append((frames, name))
        return world.capture(path)

    monkeypatch.setattr(camera, "capture", fake_capture)
    cases = [([], None, (15, "e2eSoft iVCam")),                               # both from ring.json
             (["--camera-frames", "30"], None, (30, "e2eSoft iVCam")),
             ([], "Camo", (15, "Camo")),                                       # MIMIC_CAMERA over ring.json
             (["--camera-name", "Other Cam"], "Camo", (15, "Other Cam"))]      # the option over everything
    for argv, env, expected in cases:
        if env:
            monkeypatch.setenv("MIMIC_CAMERA", env)
        grabs.clear()
        world.transport.enabled = False
        queue = ["yes"] + ANGLES + ["done", "no"]
        said = []
        rc.main(argv, read=lambda q: queue.pop(0), sleep=lambda s: None, say=said.append,
                ring_path=world.ring_path, out_root=tmp_path / "runs")      # no capture given: the real camera
        assert set(grabs) == {expected} and len(grabs) == 5
        assert any(f"Camera: {expected[1]}, {expected[0]} frames per photo." in line for line in said)


def test_done_before_5_points_and_a_repeated_angle_are_refused(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    code, said = world.run(["yes", "80", "80", "done", "90", "100", "110", "120", "done", "yes"])
    assert code == 0
    assert "  base 80 is already measured: choose another angle" in said
    assert "  5 points are needed; there are 1." in said


def test_a_bad_fit_or_a_no_is_not_saved(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    before = world.ring_path.read_text()
    world.marker_shift = {100: (0.0, 1.5)}                    # one marker found 1.5 cm away from the tip
    code, said = world.run(["yes"] + ANGLES + ["done"])
    refusal = next(line for line in said if line.startswith("Not saved:"))
    assert code == 1 and "Without the point at base 100 the others fit" in refusal and "re-measure that one" in refusal
    assert world.ring_path.read_text() == before
    world.marker_shift = {}
    world.transport.enabled = False
    code, said = world.run(["yes"] + ANGLES + ["done", "no"])
    assert code == 1 and "Not saved." in said and world.ring_path.read_text() == before


def test_unmeasured_dots_or_a_locked_arm_are_refused_before_anything_opens(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path)
    monkeypatch.setattr(scene, "load_layout",
                        lambda path=config.LAYOUT_PATH: dataclasses.replace(world.layout, dots_measured=False))
    code, said = world.run([])
    assert code == 1 and said[0].startswith("STOPPED before opening anything: calib/layout.json's dot positions")
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: world.layout)
    monkeypatch.setattr(session, "connect", lambda dry_run, limits, echo=print: REAL_CONNECT(False, dataclasses.replace(
        limits, placeholder=True)))
    code, said = world.run([])
    assert code == 1 and "STOPPED before opening anything: calib/arm_limits.json still holds placeholder" in said[0]
