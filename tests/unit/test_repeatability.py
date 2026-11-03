"""repeatability.py: spread, backlash and verdict; and a full fake session (arm, camera, SAM)."""
import dataclasses
import importlib.util
import math

import cv2
import numpy as np
import pytest
from test_kinematics import wide_limits

from mimic import arm, config, meta_api, sam, scene, session
from mimic.arm import Arm, FakeTransport
from mimic.sam import Box

spec = importlib.util.spec_from_file_location("repeatability", config.ROOT / "scripts/repeatability.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

SHEET_CM = [(-2, -2), (40, -2), (40, 27.7), (-2, 27.7)]
SHEET_PX = [(300, 950), (1650, 980), (1400, 150), (520, 130)]
CM_TO_PX = cv2.getPerspectiveTransform(np.float32(SHEET_CM), np.float32(SHEET_PX))


def to_px(pt):
    return tuple(cv2.perspectiveTransform(np.float64([pt]).reshape(-1, 1, 2), CM_TO_PX).reshape(2))


def box(pt, size=30):
    x, y = to_px(pt)
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), 1920, 1080)


def shots(points):
    return [{"shot": i + 1, "from": "+" if i % 2 == 0 else "-", "cm": p} for i, p in enumerate(points)]


def test_steady_arm_full_corrections_are_fine():
    numbers = rp.spread(shots([(10.0, 5.0), (10.1, 5.0), (10.0, 5.1), (10.1, 5.1)]))
    assert numbers["mean_cm"] == pytest.approx((10.05, 5.05))
    assert 2 * numbers["std_cm"] < 0.5
    assert "full corrections are fine" in rp.verdict(numbers["std_cm"])


def test_wobbly_arm_apply_only_part_of_the_miss():
    numbers = rp.spread(shots([(9.0, 5.0), (11.0, 5.0), (10.0, 4.0), (10.0, 6.0)]))
    assert 2 * numbers["std_cm"] > 1.0
    assert numbers["max_from_mean_cm"] == pytest.approx(1.0)
    assert "apply only part of the miss" in rp.verdict(numbers["std_cm"])


def test_in_between_is_borderline():
    assert "may overshoot now and then" in rp.verdict(0.35)


def test_backlash_is_the_gap_between_the_two_approaches_and_target_is_optional():
    numbers = rp.spread(shots([(10.3, 5.0), (10.0, 5.0), (10.3, 5.0), (10.0, 5.0)]), target=(9.0, 5.5))
    assert numbers["backlash_cm"] == pytest.approx(0.3)
    assert numbers["backlash_vector_cm"] == pytest.approx((0.3, 0.0))
    assert numbers["mean_offset_cm"] == pytest.approx((1.15, -0.5))
    said = []
    rp.report(numbers, said.append)
    text = "\n".join(said)
    assert "0.30 cm right" in text and "1.15 cm right, 0.50 cm towards you" in text
    without_target = []
    rp.report(rp.spread(shots([(1.0, 1.0), (1.0, 1.1)])), without_target.append)
    assert not any("mean offset" in line for line in without_target)


class FakeSession:
    """Fake arm, camera and SAM. The marker is dot-sized, so the dot finder sees 7 candidates."""

    def __init__(self, monkeypatch, markers, enabled=True):
        self.limits = wide_limits()
        self.transport = FakeTransport(self.limits, enabled=enabled)
        self.arm = Arm(self.transport, self.limits)
        self.layout = dataclasses.replace(scene.load_layout(config.LAYOUT_PATH), placeholder=False)
        self.markers = list(markers)
        self.photo_marker, self.budget_seen, self.cache_seen = {}, [], []
        monkeypatch.setattr(sam, "find", self.find)

    def capture(self, path):
        cv2.imwrite(str(path), np.full((1080, 1920, 3), 235, np.uint8))
        self.photo_marker[str(path)] = self.markers.pop(0)
        return str(path)

    def find(self, image_path, name, use_cache=True):
        self.budget_seen.append(config.BUDGET_USD)
        self.cache_seen.append(meta_api.cache_enabled)
        marker = box(self.photo_marker[image_path])
        if name == self.layout.dot_prompt:
            return [box(cm) for cm in self.layout.dots_cm.values()] + [marker]
        assert name == "gripper marker"
        return [marker]


def test_measure_alternates_the_approach_and_reads_the_marker_in_cm(monkeypatch, tmp_path):
    fake = FakeSession(monkeypatch, [(12.0, 8.0), (12.2, 8.1), (11.9, 8.0)])
    pose = {"base": 100, "shoulder": 80, "elbow": 95, "grip": 90}
    results = rp.measure(fake.arm, fake.capture, fake.layout, pose, 3, 5, tmp_path, say=lambda s: None,
                         sleep=lambda s: None)
    assert [r["from"] for r in results] == ["+", "-", "+"]
    for r, expected in zip(results, [(12.0, 8.0), (12.2, 8.1), (11.9, 8.0)]):
        assert math.dist(r["cm"], expected) < 0.05
    j = [line for line in fake.transport.sent if line.startswith("J")]
    assert j[0].startswith("J 105 85 100 90") and j[1].startswith("J 100 80 95 90")   # from +5
    assert j[2].startswith("J 95 75 90 90") and j[3].startswith("J 100 80 95 90")     # from -5
    assert fake.transport.sent.count("H") == 3                                        # home in between


def test_main_caps_cost_at_10_cents_and_turns_the_cache_off(monkeypatch, tmp_path):
    fake = FakeSession(monkeypatch, [(12.0, 8.0), (12.1, 8.0)])
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: fake.layout)
    monkeypatch.setattr(session, "connect", lambda dry_run, limits, echo=print: fake.arm)
    monkeypatch.setattr(config, "BUDGET_USD", 1.0)
    monkeypatch.setattr(meta_api, "cache_enabled", True)
    said = []
    code = rp.main(["--pose", "100", "80", "95", "90", "--shots", "2"], capture=fake.capture,
                   sleep=lambda s: None, say=said.append, out_root=tmp_path)
    assert code == 0
    assert set(fake.budget_seen) == {0.10} and set(fake.cache_seen) == {False}
    assert config.BUDGET_USD == 1.0 and meta_api.cache_enabled is True   # both put back
    assert any(line.startswith("Verdict:") for line in said)


def test_main_asks_before_enabling_and_a_no_moves_nothing(monkeypatch, tmp_path):
    fake = FakeSession(monkeypatch, [(12.0, 8.0), (12.1, 8.0)], enabled=False)   # disarmed, as at power-on
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: fake.layout)
    monkeypatch.setattr(session, "connect", lambda dry_run, limits, echo=print: fake.arm)
    said = []
    assert rp.main(["--pose", "100", "80", "95", "90", "--shots", "2"], capture=lambda p: pytest.fail("photo"),
                   sleep=lambda s: None, say=said.append, out_root=tmp_path, read=lambda q: "no") == 1
    assert fake.transport.sent == ["?"] and "Not enabled: nothing was moved." in said
    said.clear()
    assert rp.main(["--pose", "100", "80", "95", "90", "--shots", "2"], capture=fake.capture,
                   sleep=lambda s: None, say=said.append, out_root=tmp_path, read=lambda q: "yes") == 0
    assert "E" in fake.transport.sent and fake.transport.sent[-1] == "D"   # enabled first, disabled at the end
    assert any(line.startswith("Hold the arm") for line in said)


def test_measured_dots_are_enough_while_the_rest_of_the_layout_is_a_placeholder(monkeypatch, tmp_path):
    fake = FakeSession(monkeypatch, [(12.0, 8.0), (12.1, 8.0)])
    fake.layout = dataclasses.replace(fake.layout, placeholder=True)   # as committed: dots measured, slots not
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: fake.layout)
    monkeypatch.setattr(session, "connect", lambda dry_run, limits, echo=print: fake.arm)
    said = []
    assert rp.main(["--pose", "100", "80", "95", "90", "--shots", "2"], capture=fake.capture,
                   sleep=lambda s: None, say=said.append, out_root=tmp_path) == 0
    assert any(line.startswith("Verdict:") for line in said)


def test_unmeasured_dots_are_refused_before_anything_opens(monkeypatch, tmp_path):
    unmeasured = dataclasses.replace(scene.load_layout(config.LAYOUT_PATH), dots_measured=False)
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: unmeasured)
    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("arm connected"))
    said = []
    assert rp.main(["--pose", "100", "80", "95", "90"], capture=lambda p: pytest.fail("camera used"),
                   say=said.append, out_root=tmp_path) == 1
    assert "dot positions are not measured yet" in said[0]


def test_real_arm_refused_while_locked(monkeypatch, tmp_path):
    # The committed layout as it is (placeholder true, dots measured) now passes the layout check, so the arm's
    # lock is the only thing standing in the way, and it must still stop everything before the bridge opens.
    committed = scene.load_layout(config.LAYOUT_PATH)
    assert committed.placeholder and committed.dots_measured
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: committed)
    locked = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH), placeholder=True)
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: locked)
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    said = []
    assert rp.main(["--pose", "100", "80", "95", "90"], capture=lambda p: pytest.fail("camera used"),
                   say=said.append, out_root=tmp_path) == 1
    assert said[0].startswith("STOPPED before opening anything: calib/arm_limits.json")  # the arm's lock, not the layout
