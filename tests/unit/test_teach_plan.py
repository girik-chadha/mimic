"""teach_plan.py on the committed (placeholder) layout and on changed copies of it."""
import dataclasses
import importlib.util

import pytest

from mimic import config, scene

spec = importlib.util.spec_from_file_location("teach_plan", config.ROOT / "scripts/teach_plan.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

COMMITTED = scene.load_layout(config.LAYOUT_PATH)


def run(monkeypatch, layout=COMMITTED, argv=("--provisional",)):
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: layout)
    said = []
    return tp.main(list(argv), say=said.append), "\n".join(said)


def test_placeholder_layout_needs_provisional(monkeypatch):
    code, text = run(monkeypatch, argv=())
    assert code == 1 and "--provisional" in text


def test_slots_first_then_the_mess_centre_then_its_corners(monkeypatch):
    assert [label for label, _ in tp.plan_points(COMMITTED)] == [
        "slot 1", "slot 2", "slot 3", "mess centre",
        "mess bottom-left", "mess bottom-right", "mess top-right", "mess top-left"]
    code, text = run(monkeypatch)
    assert code == 0 and text.startswith("PROVISIONAL")
    assert "1. slot 1             ( 20.0,  20.0)   10.1 cm from the arm's base" in text
    assert "Each point needs 2 poses, hover and grab: 16 poses in all" in text
    assert "save hover 20 20   then   save grab 20 20" in text


def test_point_beyond_reach_is_refused(monkeypatch):
    short = dataclasses.replace(COMMITTED, reach_cm={"base": (18.5, 30.0), "min": 6.0, "max": 12.0})
    code, text = run(monkeypatch, short)
    assert code == 1
    assert "REFUSED: slot 3 at (32.0, 20.0) cm is 16.8 cm from the arm's base, 4.8 cm beyond its 12.0 cm reach" in text


def test_point_too_close_to_the_base_is_refused(monkeypatch):
    close = dataclasses.replace(COMMITTED, reach_cm={"base": (18.5, 30.0), "min": 11.0, "max": 20.0})
    code, text = run(monkeypatch, close)
    assert code == 1 and "REFUSED: slot 1" in text and "too close" in text


def test_measured_layout_needs_no_flag(monkeypatch):
    code, text = run(monkeypatch, dataclasses.replace(COMMITTED, placeholder=False), argv=())
    assert code == 0 and not text.startswith("PROVISIONAL")


def test_push_adds_the_points_behind_the_mess_area_and_beyond_the_last_slot(monkeypatch):
    code, text = run(monkeypatch, argv=("--provisional", "--push"))
    assert code == 0
    extra = tp.push_points(COMMITTED, tp.plan_points(COMMITTED))
    labels = [label for label, _ in extra]
    assert labels == ["push start behind mess bottom-left", "push start behind mess bottom-right",
                      "push start behind mess top-right", "push start behind mess top-left",
                      "push back beyond slot 3"]
    beyond_slot_3 = dict(extra)["push back beyond slot 3"]
    assert beyond_slot_3[0] == pytest.approx(35.0, abs=0.05)   # 3 cm past slot 3 (32, 20)
    assert "save hover 35.0 19.9" in text
    # with the placeholder reach, the bottom-left push start is too far from the base: listed, not taught
    assert "SKIPPED: push start behind mess bottom-left at (2.1, 16.4) cm is 21.3 cm from the arm's base" in text
    assert "grip_push is unknown until the grip test" in text


def test_without_push_no_push_points_are_listed(monkeypatch):
    code, text = run(monkeypatch)
    assert code == 0 and "Push mode" not in text and "push start" not in text
