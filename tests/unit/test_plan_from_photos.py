"""The Gate 1 script, run end to end offline: fake Spark, fake SAM, blank photos. No network."""
import importlib.util
import json
import math

import cv2
import numpy as np
import pytest

from mimic import calib, config, meta_api, sam, scene
from mimic.sam import Box

REAL_FIND, REAL_CALL, REAL_RESET = sam.find, meta_api.call, meta_api.reset


class NoNetwork:
    @property
    def responses(self):
        pytest.fail("A request would have left the laptop.")


spec = importlib.util.spec_from_file_location("plan_from_photos", config.ROOT / "scripts/plan_from_photos.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

SHEET_CM = [(-2, -2), (40, -2), (40, 27.7), (-2, 27.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
DOT_PROMPT = "small coloured dot"
DOTS_CM = [(0, 0), (37, 0), (37, 24.7), (0, 24.7), (18.5, 12.4), (34.5, 12.4)]  # as in calib/layout.json

# Demo: blocks sit in the slots (x = 28) in the order blue, green, red. Mess: scattered on the left.
DEMO_CM = {"red block": (28.3, 20.4), "blue block": (27.8, 5.2), "green block": (28.1, 13.1)}
MESS_CM = {"red block": (5, 4), "blue block": (12, 20), "green block": (8, 12.5)}
SPARK_REPLY = '["red block", "blue block", "green block"]'


def make_camera(shift_px=(0, 0)):
    px = [(x + shift_px[0], y + shift_px[1]) for x, y in SHEET_PX]
    H = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(px))

    def to_px(pt_cm):
        return tuple(cv2.perspectiveTransform(np.float64([pt_cm]).reshape(-1, 1, 2), H).reshape(2))
    return to_px


def box(to_px, pt_cm, size=40):
    x, y = to_px(pt_cm)
    half = size // 2
    return Box(round(x) - half, round(y) - half, round(x) + half, round(y) + half, 1920, 1080)


class Scene:
    """Two blank photos plus a fake SAM that knows where everything is in each of them."""

    def __init__(self, tmp_path, monkeypatch, mess_shift_px=(0, 0), mess_dots_cm=None):
        self.demo = str(tmp_path / "demo.jpg")
        self.mess = str(tmp_path / "mess.jpg")
        for path in (self.demo, self.mess):
            cv2.imwrite(path, np.full((1080, 1920, 3), 235, np.uint8))
        demo_cam, mess_cam = make_camera(), make_camera(mess_shift_px)
        mess_dots = mess_dots_cm or DOTS_CM
        # dots are 30 px across in a 1920-wide photo: inside layout.json's dot size range
        self.boxes = {self.demo: {DOT_PROMPT: [box(demo_cam, cm, size=30) for cm in DOTS_CM]},
                      self.mess: {DOT_PROMPT: [box(mess_cam, cm, size=30) for cm in mess_dots]}}
        self.boxes[self.demo].update({n: [box(demo_cam, cm)] for n, cm in DEMO_CM.items()})
        self.boxes[self.mess].update({n: [box(mess_cam, cm)] for n, cm in MESS_CM.items()})
        self.budget_seen = []
        monkeypatch.setattr(sam, "find", self.find)
        monkeypatch.setattr(meta_api, "call", self.spark)

        layout = json.loads(config.LAYOUT_PATH.read_text())
        layout["placeholder"] = False
        self.layout_path = tmp_path / "layout.json"
        self.layout_path.write_text(json.dumps(layout))
        self.out_root = tmp_path / "runs"

    def find(self, image_path, name, use_cache=True):
        self.budget_seen.append(config.BUDGET_USD)
        return self.boxes[image_path].get(name, [])

    def spark(self, model, content, use_cache=True):
        assert model == "muse-spark-1.3"
        return SPARK_REPLY

    def run(self):
        return script.run(self.demo, self.mess, self.layout_path, self.out_root)


def test_full_run_prints_the_plan_and_saves_the_run(tmp_path, monkeypatch, capsys):
    s = Scene(tmp_path, monkeypatch)
    result = s.run()
    out = capsys.readouterr().out

    assert result["target_order"] == ["blue block", "green block", "red block"]
    assert [(m["name"], m["slot"]) for m in result["moves"]] == [("blue block", 1), ("green block", 2), ("red block", 3)]
    for name, pt in MESS_CM.items():
        assert math.dist(result["positions_cm"][name], pt) < 0.1
    assert "Spark named: red block, blue block, green block" in out
    assert "Target order (slot 1 first): blue block, green block, red block" in out
    assert "1. blue block" in out and "-> slot 1 ( 28.0,   5.0)" in out
    assert "demo photo: check dots centre_check 0.0" in out
    assert "mess photo: check dots centre_check 0.0" in out

    out_dir = s.out_root / next(s.out_root.iterdir()).name
    assert f"labelled dots saved to {out_dir / 'demo_dots.jpg'}" in out
    for f in ["demo.jpg", "mess.jpg", "demo_annotated.jpg", "mess_annotated.jpg", "demo_dots.jpg",
              "mess_dots.jpg", "plan.json"]:
        assert (out_dir / f).exists(), f
    saved = json.loads((out_dir / "plan.json").read_text())
    assert saved["target_order"] == result["target_order"]
    assert len(saved["naming_attempts"]) == 3


def test_objects_are_drawn_in_the_colour_their_name_says():
    assert script.colour_for("red block", 0) == script.NAMED_COLOURS["red"]
    assert script.colour_for("light blue sticker", 0) == script.NAMED_COLOURS["blue"]
    assert script.colour_for("grey cube", 1) == script.OTHER_COLOURS[1]


def test_placeholder_layout_refuses_before_any_sam_call(tmp_path, monkeypatch):
    s = Scene(tmp_path, monkeypatch)
    s.layout_path.write_text(config.LAYOUT_PATH.read_text())  # the committed file: placeholder true
    with pytest.raises(scene.SceneError, match="measure your sheet and set placeholder to false"):
        s.run()
    assert s.budget_seen == []  # nothing was asked of SAM


def test_bad_check_dot_on_the_mess_photo_stops_the_run(tmp_path, monkeypatch, capsys):
    wrong = DOTS_CM[:5] + [(32.5, 12.4)]  # right check dot seen 2 cm away from where layout.json says
    s = Scene(tmp_path, monkeypatch, mess_dots_cm=wrong)
    with pytest.raises(calib.CalibrationError, match="right_check 2.00 cm off"):
        s.run()
    out = capsys.readouterr().out
    assert "demo photo: check dots centre_check 0.0" in out  # the demo photo was fine and was reported
    assert "mess photo: labelled dots saved to" in out       # the bad photo's dots can be checked by eye
    assert not list(s.out_root.glob("*/plan.json"))           # no plan saved


def test_camera_may_move_between_photos(tmp_path, monkeypatch, capsys):
    s = Scene(tmp_path, monkeypatch, mess_shift_px=(40, 15))
    result = s.run()
    out = capsys.readouterr().out
    assert "WARNING" not in out
    for name, pt in MESS_CM.items():  # each photo has its own map, so positions are still right
        assert math.dist(result["positions_cm"][name], pt) < 0.1


def test_run_is_capped_at_15_cents_and_the_cap_is_put_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 1.00)
    s = Scene(tmp_path, monkeypatch)
    s.run()
    assert set(s.budget_seen) == {0.15}  # every SAM call happened under the $0.15 cap
    assert config.BUDGET_USD == 1.00


def test_a_lower_budget_in_env_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 0.10)
    s = Scene(tmp_path, monkeypatch)
    s.run()
    assert set(s.budget_seen) == {0.10}


def test_cap_stops_the_run_through_the_real_budget_guard(tmp_path, monkeypatch):
    s = Scene(tmp_path, monkeypatch)
    monkeypatch.setattr(sam, "find", REAL_FIND)        # real SAM path this time...
    monkeypatch.setattr(meta_api, "call", REAL_CALL)   # ...through the real budget guard
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(meta_api, "_client", NoNetwork())

    def reset_then_pretend_15_cents_are_spent():
        REAL_RESET()
        monkeypatch.setattr(meta_api, "_spent_usd", 0.15)

    monkeypatch.setattr(meta_api, "reset", reset_then_pretend_15_cents_are_spent)
    with pytest.raises(meta_api.BudgetExceeded, match=r"Spent \$0.1500 of the \$0.15 budget"):
        s.run()
