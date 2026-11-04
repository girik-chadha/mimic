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

from mimic import arm, config, meta_api, ring, sam, scene, session
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
        self.photos, self.budgets = {}, []
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
        assert name == "gripper marker"
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
    assert saved.placeholder is False and len(saved.points) == 5
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
    assert "Table height: shoulder 180; the shoulder is never sent past 183." in said
    moves = [[int(v) for v in line.split()[1:5]] for line in world.transport.sent if line.startswith("J")]
    assert max(m[1] for m in moves) == 180 and moves[0][1] == 170     # the approach starts 10 above the table


def test_table_shoulder_overrides_ring_json_and_is_saved_with_the_fit(monkeypatch, tmp_path):
    world = FakeWorld(monkeypatch, tmp_path, table=170)
    code, said = world.run(["yes"] + ANGLES + ["done", "yes"], argv=["--table-shoulder", "170"])
    assert code == 0
    assert "Table height: shoulder 170; the shoulder is never sent past 173." in said
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
    assert any("skipped: SAM found 2 boxes for 'gripper marker'" in line for line in said)


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
