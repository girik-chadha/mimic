"""ring_loop.run against a fake ring world: fake arm, camera and SAM. No hardware, no API calls.

The fake arm knows the ring: when the claw closes at table height it picks up the die under it, and when it opens
it drops it there, plus any miss the test asks for (degrees along the ring, cm along the radius).
"""
import dataclasses
import json
import math
import re

import cv2
import numpy as np
import pytest
from test_ring import fitted, made_up

from mimic import arm, brain, config, meta_api, ring_loop, sam, scene
from mimic.arm import Arm, FakeTransport
from mimic.sam import Box

W, H = 1280, 720
LIMITS = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH),
                             ranges_deg={"base": (0, 180), "shoulder": (80, 180), "elbow": (30, 120)},
                             grip_open_deg=90, grip_closed_limit_deg=3, grip_range_deg=(0, 140),
                             min_ms_per_degree=10, max_grip_hold_s=5, placeholder=False, limits_checked_by_hand=True)
RING = fitted(made_up([80, 90, 100, 110, 120]), slots_deg=[100, 125, 150], camera_clear={"base": 90, "shoulder": 90})
NAMES = ["green die", "blue die", "red die"]
DEMO_DEG = {"green die": 100, "blue die": 125, "red die": 150}     # in slots 1, 2, 3
MESS_DEG = {"green die": 40, "blue die": 60, "red die": 80}       # scattered on the ring, away from the slots


def to_px(cm):
    return (100 + cm[0] * 28, 680 - cm[1] * 22)


def box_at(cm, size):
    x, y = to_px(cm)
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), W, H)


def on_ring(base_deg, outward_cm=0.0, the_ring=RING):
    x, y = the_ring.point_at(base_deg)
    cx, cy = the_ring.centre_cm
    d = math.dist((x, y), (cx, cy))
    return (x + (x - cx) / d * outward_cm, y + (y - cy) / d * outward_cm)


class RingWorld(FakeTransport):
    def __init__(self, limits, the_ring):
        super().__init__(limits, enabled=True)
        self.ring = the_ring
        self.dice, self.held = {}, None
        self.miss_deg, self.slip_cm = {}, {}     # per die: where it lands relative to the claw

    def send(self, line):
        grip_before = self.angles[3]
        reply = super().send(line)
        if line.startswith("J") and reply == "OK" and self.angles[1] == self.ring.shoulder_table:
            base, grip = self.angles[0], self.angles[3]
            if grip_before == 90 and grip == self.ring.grip_hold:
                under = [n for n, p in self.dice.items() if p and math.dist(p, self.ring.point_at(base)) < 1.0]
                if under:
                    self.held, self.dice[under[0]] = under[0], None
            if grip_before == self.ring.grip_hold and grip == 90 and self.held:
                name = self.held
                self.dice[name] = on_ring(base + self.miss_deg.get(name, 0), self.slip_cm.get(name, 0), self.ring)
                self.held = None
        return reply


class Rig:
    def __init__(self, tmp_path, monkeypatch, the_ring=RING, limits=LIMITS):
        self.ring, self.limits = the_ring, limits
        self.world = RingWorld(limits, the_ring)
        self.arm = Arm(self.world, limits)
        self.layout = scene.load_layout(config.LAYOUT_PATH)   # dots measured; its slots are not used here
        self.photos, self.captured, self.said, self.prompts = {}, [], [], []
        self.mess_deg, self.mess_outward = dict(MESS_DEG), {}
        self.out_root = tmp_path / "runs"
        monkeypatch.setattr(sam, "find", self.find)
        monkeypatch.setattr(brain, "name_objects", lambda photo, attempts=None: list(NAMES))

    def capture(self, path):
        clear = self.ring.camera_clear
        assert self.world.angles[:2] == [clear["base"], clear["shoulder"]], "arm in the photo"
        cv2.imwrite(str(path), np.full((H, W, 3), 230, np.uint8))
        self.photos[str(path)] = dict(self.world.dice)
        self.captured.append(path.name)
        return str(path)

    def find(self, image_path, name, use_cache=True):
        if name == self.layout.dot_prompt:
            return [box_at(cm, 25) for cm in self.layout.dots_cm.values()]
        where = self.photos[image_path].get(name)
        return [box_at(where, 38)] if where else []

    def wait_for_user(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self.world.dice = {n: on_ring(d) for n, d in DEMO_DEG.items()}
        else:
            self.world.dice = {n: on_ring(d, self.mess_outward.get(n, 0)) for n, d in self.mess_deg.items()}

    def run(self):
        return ring_loop.run(self.arm, self.capture, self.layout, self.ring, self.limits, self.wait_for_user,
                             self.out_root, say=self.said.append, sleep=lambda s: None)

    def attempts(self):
        out_dir = next(self.out_root.iterdir())
        return [json.loads(line) for line in (out_dir / "attempts.jsonl").read_text().splitlines()]


def moves_sent(world):
    return [[int(v) for v in line.split()[1:5]] for line in world.sent if line.startswith("J")]


# ---------- a normal run ----------

def test_three_dice_are_carried_round_the_ring_into_their_slots(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    summary = rig.run()
    assert summary["success"] is True and summary["mode"] == "ring"
    assert [m["name"] for m in summary["moves"]] == NAMES and [a["result"] for a in rig.attempts()] == ["PASS"] * 3
    for name, slot_deg in DEMO_DEG.items():
        assert math.dist(rig.world.dice[name], RING.point_at(slot_deg)) < 0.3
    assert rig.captured == ["demo.jpg", "mess.jpg", "after_1.jpg", "after_2.jpg", "after_3.jpg"]
    moves = moves_sent(rig.world)
    assert max(m[1] for m in moves) == 173 and all(m[2] == 90 for m in moves)   # table height, elbow 90
    assert "H" not in rig.world.sent
    first = rig.attempts()[0]
    assert (first["pick_deg"], first["drop_deg"], first["slot_deg"]) == (40, 100, 100)


# ---------- misses ----------

def test_a_steady_angular_miss_shifts_the_next_drop_by_at_most_2_degrees(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.miss_deg = {"blue die": 8}           # blue always lands 8 degrees past where it is dropped
    summary = rig.run()
    blue = [a for a in rig.attempts() if a["name"] == "blue die"]
    assert [a["result"] for a in blue] == ["RETRY", "PASS"]
    assert blue[0]["angle_miss_deg"] == pytest.approx(8, abs=0.3)
    assert blue[0]["next_correction_deg"] == -2.0                    # clamped from -8
    assert blue[1]["drop_deg"] == 123 and blue[1]["pick_deg"] == 133  # aimed 2 short; picked from where it landed
    assert summary["success"] is True


def test_three_misses_log_FAIL_and_the_run_goes_on(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.miss_deg = {"red die": -15}
    summary = rig.run()
    red = [a for a in rig.attempts() if a["name"] == "red die"]
    assert [a["result"] for a in red] == ["RETRY", "RETRY", "FAIL"]
    assert [a["drop_deg"] for a in red] == [150, 152, 152]            # +2, then still +2 (the total is clamped)
    assert [m["result"] for m in summary["moves"]] == ["PASS", "PASS", "FAIL"] and summary["success"] is False


def test_a_miss_along_the_radius_cannot_be_fixed_by_the_base(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.slip_cm = {"blue die": 2.0}          # lands 2 cm outside the ring
    summary = rig.run()
    blue = [a for a in rig.attempts() if a["name"] == "blue die"]
    assert len(blue) == 1 and blue[0]["radial_miss_cm"] == pytest.approx(2.0, abs=0.1)
    assert any("along the radius, which a base correction cannot fix" in line for line in rig.said)
    assert any("cannot pick it up again" in line and "2.0 cm outside the ring" in line for line in rig.said)
    assert [m["result"] for m in summary["moves"]] == ["PASS", "FAIL", "PASS"]


# ---------- refused before anything moves ----------

def no_moves(rig):
    return not any(line.startswith("J") and int(line.split()[2]) > 90 for line in rig.world.sent)


@pytest.mark.parametrize("change, problem", [
    (lambda r, m: m.update({"blue die": 50}), "green die and blue die are 2.4 cm apart"),
    (lambda r, m: m.update({"red die": 92}), "red die is 1.9 cm from slot 1 \\(base 100\\)"),
    (lambda r, m: r.mess_outward.update({"green die": 1.5}), "green die: .* 1.5 cm outside the ring"),
])
def test_a_mess_that_does_not_fit_is_refused_with_the_numbers(tmp_path, monkeypatch, change, problem):
    rig = Rig(tmp_path, monkeypatch)
    change(rig, rig.mess_deg)
    with pytest.raises(ring_loop.RingLayoutError, match=problem):
        rig.run()
    assert no_moves(rig)


def test_a_pick_and_drop_too_long_for_the_grip_timer_is_refused_before_any_move(tmp_path, monkeypatch):
    slow = dataclasses.replace(RING, command_s=0.15)           # a slow bridge: every carry is over 4.5 s
    rig = Rig(tmp_path, monkeypatch, the_ring=slow)
    with pytest.raises(ring_loop.RingLayoutError, match="green die to slot 1: The claw would be closed for"):
        rig.run()
    assert no_moves(rig)


@pytest.mark.parametrize("the_ring, limits, problem", [
    (dataclasses.replace(RING, slots_deg=[100, 115, 130]), LIMITS,
     "Slots at base 100 and 115 are 3.7 cm apart on the ring; they need at least 5.5 cm \\(22.5 degrees"),
    (RING, dataclasses.replace(LIMITS, ranges_deg={"base": (80, 155), "shoulder": (80, 180), "elbow": (30, 120)}),
     "The ring has 18.3 cm of arc between base 80 and 155 .* need about 23.0 cm \\(94 degrees of base\\)"),
    (dataclasses.replace(RING, slots_deg=[100, 125, 190]), LIMITS, "Slot at base 190 is outside the base range 0-180"),
    (dataclasses.replace(RING, placeholder=True), LIMITS, "still a placeholder"),
])
def test_a_layout_that_does_not_fit_is_refused_before_anything_opens(tmp_path, monkeypatch, the_ring, limits, problem):
    rig = Rig(tmp_path, monkeypatch, the_ring=the_ring, limits=limits)
    with pytest.raises(Exception, match=problem):
        rig.run()
    assert rig.world.sent == [] and rig.captured == [] and not rig.out_root.exists()


def test_unmeasured_dots_are_refused_before_anything_opens(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.layout = dataclasses.replace(rig.layout, dots_measured=False)
    with pytest.raises(scene.SceneError, match="dot positions are not measured yet"):
        rig.run()
    assert rig.world.sent == []


def test_a_die_left_near_the_next_slot_stops_the_run(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.slip_cm = {"blue die": 2.0}
    rig.world.miss_deg = {"blue die": 18}          # blue fails 2 cm out, right next to slot 3 (base 150)
    with pytest.raises(ring_loop.RingLayoutError, match="blue die at .* from the slot at base 150 that red die is going"):
        rig.run()
    assert rig.world.sent[-1] == "S"
