"""loop.run end to end against a fake world: fake arm, fake camera, fake SAM. No hardware, no API calls.

The fake world uses the test pose grid (base = 40 + 3x, shoulder = 40 + 3y), so from the angles sent to
the arm it knows where the gripper is: when the grip closes at grab height it picks up the block there,
and when it opens the block lands there (plus any miss the test asks for). The fake camera remembers
where every block was when each photo was taken, and the fake SAM answers from that, through the same
made-up tilted camera as the other scene tests.
"""
import dataclasses
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_kinematics import linear_grid, wide_limits

from mimic import arm, config, loop, meta_api, sam, scene
from mimic.arm import Arm, ArmError, FakeTransport
from mimic.sam import Box

SHEET_CM = [(-2, -2), (40, -2), (40, 27.7), (-2, 27.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
CM_TO_PX = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(SHEET_PX))

DEMO = {"blue block": (20.1, 20.0), "green block": (26.2, 19.8), "red block": (31.9, 20.2)}  # slots 1, 2, 3
MESS = {"red block": (5.0, 4.0), "blue block": (12.0, 20.0), "green block": (8.0, 9.0)}
GRAB_ELBOW = 100


def to_px(pt_cm):
    return tuple(cv2.perspectiveTransform(np.float64([pt_cm]).reshape(-1, 1, 2), CM_TO_PX).reshape(2))


def box(pt_cm, size):
    x, y = to_px(pt_cm)
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), 1920, 1080)


class FakeWorld(FakeTransport):
    """The arm plus the blocks on the table."""

    def __init__(self, limits, grid):
        super().__init__(limits)
        self.grid = grid
        self.blocks = {}
        self.held = None
        self.release_misses = []      # (dx, dy) added to each release, in order
        self.lost_when_picked = set()

    def send(self, line):
        grip_before = self.angles[3]
        reply = super().send(line)
        if line.startswith("J") and reply == "OK":
            base, shoulder, elbow, grip = self.angles
            here = ((base - 40) / 3, (shoulder - 40) / 3)
            if grip == self.grid.grip_closed and grip_before != grip and elbow == GRAB_ELBOW:
                near = [n for n, p in self.blocks.items() if p and math.dist(p, here) < 2.0]
                if near:
                    self.blocks[near[0]] = None
                    self.held = None if near[0] in self.lost_when_picked else near[0]
            if grip == self.grid.grip_open and grip_before == self.grid.grip_closed and self.held:
                dx, dy = self.release_misses.pop(0) if self.release_misses else (0.0, 0.0)
                self.blocks[self.held] = (here[0] + dx, here[1] + dy)
                self.held = None
        return reply


class Rig:
    """Everything loop.run needs, faked."""

    def __init__(self, tmp_path, monkeypatch, transport_class=FakeWorld, limits=None):
        self.grid = linear_grid()
        self.limits = limits or wide_limits()
        self.world = transport_class(self.limits, self.grid)
        self.arm = Arm(self.world, self.limits)
        self.layout = dataclasses.replace(scene.load_layout(config.LAYOUT_PATH), placeholder=False)
        self.photos, self.captured, self.budget_seen, self.prompts = {}, [], [], []
        self.out_root = tmp_path / "runs"
        self.said = []
        self.fail_sam_at = None
        self.diagnoses = []          # Spark's replies to diagnose calls, in order (none left: an unusable reply)
        monkeypatch.setattr(sam, "find", self.find)
        monkeypatch.setattr(meta_api, "call", self.spark)

    def spark(self, model, content, use_cache=True):
        if isinstance(content, list) and len(content) == 3:   # text + two photos: a diagnose call
            return self.diagnoses.pop(0) if self.diagnoses else "no idea"
        return '["red block", "blue block", "green block"]'

    def capture(self, path):
        clear = self.grid.camera_clear
        assert self.world.angles[:3] == [clear["base"], clear["shoulder"], clear["elbow"]], "arm in the photo"
        cv2.imwrite(str(path), np.full((1080, 1920, 3), 235, np.uint8))
        self.photos[str(path)] = dict(self.world.blocks)
        self.captured.append(path.name)
        return str(path)

    def find(self, image_path, name, use_cache=True):
        self.budget_seen.append(config.BUDGET_USD)
        if self.fail_sam_at and image_path.endswith(self.fail_sam_at):
            raise meta_api.BudgetExceeded("Spent $0.1500 of the $0.15 budget.")
        if name == self.layout.dot_prompt:
            return [box(cm, 30) for cm in self.layout.dots_cm.values()]
        where = self.photos[image_path].get(name)
        return [box(where, 40)] if where else []

    def wait_for_user(self, prompt):
        self.prompts.append(prompt)
        self.world.blocks = dict(DEMO if len(self.prompts) == 1 else MESS)

    def run(self):
        return loop.run(self.arm, self.capture, self.layout, self.grid, self.limits, self.wait_for_user,
                        self.out_root, say=self.said.append, sleep=lambda s: None)

    def attempts(self):
        out_dir = next(self.out_root.iterdir())
        return [json.loads(line) for line in (out_dir / "attempts.jsonl").read_text().splitlines()]

    def summary(self):
        return json.loads((next(self.out_root.iterdir()) / "summary.json").read_text())


def test_normal_three_block_run(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    summary = rig.run()
    assert summary["success"] is True
    assert [m["name"] for m in summary["moves"]] == ["blue block", "green block", "red block"]
    assert [a["result"] for a in rig.attempts()] == ["PASS", "PASS", "PASS"]
    assert all(a["miss_cm"]["distance"] <= 1.5 for a in rig.attempts())
    assert rig.captured == ["demo.jpg", "mess.jpg", "after_1.jpg", "after_2.jpg", "after_3.jpg"]
    out_dir = next(rig.out_root.iterdir())
    for name in ["demo.jpg", "mess.jpg", "after_3.jpg", "attempts.jsonl", "summary.json"]:
        assert (out_dir / name).exists(), name
    assert rig.summary()["cost_usd"] == 0.0 and "cost_usd" in rig.summary()
    assert set(rig.budget_seen) == {0.30}                          # every SAM call under the $0.30 cap
    assert any("planned grip hold" in line for line in rig.said)
    for name, slot in zip(["blue block", "green block", "red block"], scene.load_layout().slots_cm):
        assert math.dist(rig.world.blocks[name], slot) < 1.5      # the world agrees: blocks are in their slots


def test_a_miss_is_corrected_on_the_retry_and_logged(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    # a steady offset (as from a slightly-off taught grid): this block lands 2.5 cm right every time
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    summary = rig.run()
    first, second = rig.attempts()[:2]
    assert first["result"] == "RETRY" and first["miss_cm"]["dx"] == pytest.approx(2.5, abs=0.6)
    assert first["next_correction_cm"][0] == pytest.approx(-2.0)   # aim the other way, at most 2 cm
    assert second["attempt"] == 2 and second["result"] == "PASS"
    assert second["aim_cm"][0] == pytest.approx(first["slot_cm"][0] - 2.0)
    assert second["from_cm"] == pytest.approx(first["measured_cm"])  # picked up from where it landed
    assert summary["success"] is True and summary["attempts"] == 4


def test_three_misses_log_FAIL_and_the_run_goes_on(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(5.0, 0.0)] * 3                    # too far off for a 2 cm correction to fix
    summary = rig.run()
    assert [a["result"] for a in rig.attempts()[:3]] == ["RETRY", "RETRY", "FAIL"]
    assert [a["aim_cm"][0] - a["slot_cm"][0] for a in rig.attempts()[:3]] == pytest.approx([0, -2, -2])
    assert [m["result"] for m in summary["moves"]] == ["FAIL", "PASS", "PASS"]
    assert summary["success"] is False


def test_lost_block_stops_with_a_clear_message(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.lost_when_picked = {"blue block"}                    # dropped somewhere out of sight
    with pytest.raises(scene.SceneError, match="SAM found 0 boxes for 'blue block' in .*after_1.jpg"):
        rig.run()
    assert "SAM found 0 boxes for 'blue block'" in rig.summary()["error"]
    assert rig.world.sent[-1] == "S"


def test_budget_stop_is_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 1.00)
    rig = Rig(tmp_path, monkeypatch)
    rig.fail_sam_at = "after_1.jpg"
    with pytest.raises(meta_api.BudgetExceeded):
        rig.run()
    assert "BudgetExceeded" in rig.summary()["error"]
    assert rig.world.sent[-1] == "S"
    assert rig.world.held is None                                  # nothing was held when it stopped
    assert config.BUDGET_USD == 1.00                               # the cap is put back afterwards


def test_real_transport_refused_while_locked_before_anything(tmp_path, monkeypatch):
    class RealLookingWorld(FakeWorld):
        is_real = True

    rig = Rig(tmp_path, monkeypatch, RealLookingWorld, limits=dataclasses.replace(wide_limits(), placeholder=True))
    with pytest.raises(ArmError, match="placeholder"):
        rig.run()
    assert rig.captured == [] and rig.budget_seen == [] and rig.world.sent == []
    assert not rig.out_root.exists()                               # refused before even making the run folder


def test_same_direction_misses_warn_that_the_arm_or_sheet_moved(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0), (2.5, 0.3), (2.5, 0.3)]  # moves 1 and 2 both off to the right
    rig.run()
    assert any("missed the same way" in line for line in rig.said)


# ---------- Spark's diagnosis of a miss ----------

def diagnosis(dx, dy, problem="missed_aim", explanation="It landed to the right of the slot.", seen=(2.5, 0.0)):
    return json.dumps({"problem": problem, "seen_offset_cm": {"dx_cm": seen[0], "dy_cm": seen[1]} if seen else None,
                       "fix": {"dx_cm": dx, "dy_cm": dy, "retry": True}, "explanation": explanation})


def test_sparks_fix_is_logged_and_printed_but_the_numeric_correction_decides(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_SPARK_FIX", False)
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = [diagnosis(-1.8, 0.0)]
    rig.run()
    first = rig.attempts()[0]
    assert first["correction_source"] == "numeric"
    assert first["next_correction_cm"] == pytest.approx([-2.0, 0.0], abs=0.6)   # numeric, not Spark's -1.8
    assert first["spark_fix"] == {"dx_cm": -1.8, "dy_cm": 0.0, "retry": True}
    assert first["spark_fix_note"].startswith("logged only")
    assert first["consistency"].startswith("agrees with the measurement")
    assert "  Spark: It landed to the right of the slot. (missed_aim)" in rig.said
    assert "  Spark's fix (-1.8, +0.0) cm: logged only (set MIMIC_USE_SPARK_FIX=1 to use it)" in rig.said
    assert any(line.startswith("  -> numeric correction") for line in rig.said)


def test_with_MIMIC_USE_SPARK_FIX_its_fix_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_SPARK_FIX", True)
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = [diagnosis(-1.8, 0.0)]
    rig.run()
    first = rig.attempts()[0]
    assert first["correction_source"] == "spark" and first["spark_fix_note"] == "used"
    assert first["next_correction_cm"] == pytest.approx([-1.8, 0.0])
    assert any(line.startswith("  -> Spark's fix") for line in rig.said)


def test_a_contradicting_diagnosis_is_flagged_in_the_log_and_terminal(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = [diagnosis(-1.0, 0.0, seen=(1.0, 0.0))]          # Spark "sees" 1 cm; the camera measured ~2.5
    rig.run()
    first = rig.attempts()[0]
    assert first["consistency"].startswith("contradicts measurement: Spark sees it at (+1.0, +0.0) cm")
    assert any(line.startswith("  !! Spark's position: contradicts measurement") for line in rig.said)


def test_at_most_two_diagnoses_per_run(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    calls = []
    real_spark = rig.spark

    def counting_spark(model, content, use_cache=True):
        if isinstance(content, list) and len(content) == 3:
            calls.append(1)
        return real_spark(model, content, use_cache)

    monkeypatch.setattr(meta_api, "call", counting_spark)
    rig.world.release_misses = [(5.0, 0.0)] * 3 + [(5.0, 0.0)] * 3   # two moves that keep missing: 4 retries
    rig.run()
    assert len(calls) == 2
    reasons = [a["fallback_reason"] for a in rig.attempts() if a["result"] == "RETRY"]
    assert reasons[-1] == "no usable diagnosis: skipped: at most 2 diagnoses per run"


def test_unusable_diagnosis_falls_back_to_the_numeric_correction(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = ["sorry, I cannot tell"]
    rig.run()
    first = rig.attempts()[0]
    assert first["correction_source"] == "numeric"
    assert first["fallback_reason"].startswith("no usable diagnosis")
    assert first["next_correction_cm"][0] == pytest.approx(-2.0)


# ---------- choose_correction: the code decides ----------

SLOT = (20.0, 20.0)
MOVE = SimpleNamespace(name="red block", slot=1)


def advice(dx, dy, seen=None, problem="missed_aim"):
    return lambda before, after, move, miss: {"problem": problem, "explanation": "x", "seen_offset_cm": seen,
                                               "fix": {"dx_cm": dx, "dy_cm": dy, "retry": True}}


def test_worked_example_block_landed_1_cm_right_so_a_fix_pointing_right_is_wrong():
    landed_right = (21.0, 20.0)
    correction, info = loop.choose_correction(SLOT, landed_right, (0.0, 0.0), "b", "a", MOVE, advice(+1.0, 0.0), True)
    assert info["correction_source"] == "numeric"
    assert "points the same way the block missed" in info["fallback_reason"]
    assert correction == pytest.approx((-1.0, 0.0))              # the numeric correction: aim 1 cm left
    correction, info = loop.choose_correction(SLOT, landed_right, (0.0, 0.0), "b", "a", MOVE, advice(-0.8, 0.0), True)
    assert info["correction_source"] == "spark" and correction == pytest.approx((-0.8, 0.0))


def test_out_of_range_fix_is_ignored_not_trimmed():
    correction, info = loop.choose_correction(SLOT, (21.0, 20.0), (0.0, 0.0), "b", "a", MOVE, advice(-3.5, 0.0), True)
    assert info["correction_source"] == "numeric" and "over 2 cm" in info["fallback_reason"]
    assert correction == pytest.approx((-1.0, 0.0))


def test_total_correction_stays_within_2_cm_whoever_suggests_it():
    # Spark: -1.5 so far plus a further -1.5 would be -3.0; limited to -2
    correction, info = loop.choose_correction(SLOT, (21.0, 20.0), (-1.5, 0.0), "b", "a", MOVE, advice(-1.5, 0.0), True)
    assert info["correction_source"] == "spark" and correction == pytest.approx((-2.0, 0.0))

    def unusable(*args):
        raise ValueError("no JSON")
    # numeric: -1.5 so far minus a 4 cm miss would be -5.5; limited to -2
    correction, info = loop.choose_correction(SLOT, (24.0, 20.0), (-1.5, 0.0), "b", "a", MOVE, unusable)
    assert info["correction_source"] == "numeric" and correction == pytest.approx((-2.0, 0.0))


def test_a_failing_diagnosis_never_stops_the_run():
    def broken(*args):
        raise RuntimeError("API timed out")
    correction, info = loop.choose_correction(SLOT, (21.0, 20.0), (0.0, 0.0), "b", "a", MOVE, broken)
    assert info["fallback_reason"] == "no usable diagnosis: API timed out"
    assert correction == pytest.approx((-1.0, 0.0))


# ---------- the consistency check ----------

def test_consistency_agrees_contradicts_or_cannot_tell():
    miss = (1.0, 0.0)
    assert loop.consistency({"problem": "missed_aim", "seen_offset_cm": (1.3, 0.2)}, miss)[0] is True
    agrees, words = loop.consistency({"problem": "missed_aim", "seen_offset_cm": (0.2, 0.0)}, miss)
    assert agrees is False and words.startswith("contradicts measurement") and "0.8 cm apart" in words
    agrees, words = loop.consistency({"problem": "not_found", "seen_offset_cm": None}, miss)
    assert agrees is False and "cannot see the block, but the camera measured it" in words
    assert loop.consistency({"problem": "slipped", "seen_offset_cm": None}, miss) == (None, "not checked: Spark gave no position")


def test_log_only_by_default():
    correction, info = loop.choose_correction(SLOT, (21.0, 20.0), (0.0, 0.0), "b", "a", MOVE,
                                              advice(-0.8, 0.0, seen=(1.0, 0.0)), False)
    assert info["correction_source"] == "numeric" and correction == pytest.approx((-1.0, 0.0))
    assert info["spark_fix"]["dx_cm"] == -0.8 and info["spark_fix_note"].startswith("logged only")


def test_when_enabled_a_contradicting_fix_is_not_used():
    correction, info = loop.choose_correction(SLOT, (21.0, 20.0), (0.0, 0.0), "b", "a", MOVE,
                                              advice(-0.8, 0.0, seen=(-1.0, 0.0)), True)
    assert info["correction_source"] == "numeric"
    assert info["spark_fix_note"].startswith("not used: contradicts measurement")

