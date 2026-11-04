"""Simulation mode (mimic/sim.py, mimic/sim_video.py, run_mimic.py --ring --sim) with fake SAM and Spark.

No hardware, no API calls, no ffmpeg (the encode step is faked). The dice sit about where they are in
tests/fixtures/photos/dice_sheet.jpg: red, black, blue from left to right across the middle of the sheet.
"""
import json
import math

import cv2
import numpy as np
import pytest

from mimic import arm, brain, camera, config, marks, ring_loop, sam, scene, sim, sim_video
from mimic import ring as ring_module
from mimic.sam import Box

W, H = 1280, 720
SEEN = {"red die": (11.9, 7.0), "black die": (17.9, 7.0), "blue die": (24.0, 7.2)}
PARK = {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}   # camera_clear in calib/ring.json


def to_px(cm):
    return (100 + cm[0] * 28, 680 - cm[1] * 22)


def box_at(cm, size):
    x, y = to_px(cm)
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), W, H)


class Fakes:
    """Fake SAM and Spark on a blank photo; a fake video maker that keeps the story."""

    def __init__(self, tmp_path, monkeypatch, seen=SEEN, names=None):
        self.photo = tmp_path / "photo.jpg"
        cv2.imwrite(str(self.photo), np.full((H, W, 3), 230, np.uint8))
        self.seen, self.said, self.stories = dict(seen), [], []
        self.out_root = tmp_path / "runs"
        layout = scene.load_layout()
        monkeypatch.setattr(sam, "find", lambda path, name, use_cache=True: (
            [box_at(cm, 25) for cm in layout.dots_cm.values()] if name == layout.dot_prompt
            else [box_at(self.seen[name], 38)] if name in self.seen else []))
        monkeypatch.setattr(brain, "name_objects", lambda path, attempts=None: list(names or self.seen))
        monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
        monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))

    def video(self, out_path, story, say=print):
        self.stories.append(story)
        out_path.write_bytes(b"fake video")

    def run(self, instruction=None, miss_deg=0.0):
        return sim.run(self.photo, instruction, miss_deg, self.out_root, say=self.said.append, make_video=self.video)

    def out_dir(self):
        return next(self.out_root.iterdir())

    def moves_json(self):
        return json.loads((self.out_dir() / "simulation_moves.json").read_text())

    def attempts(self):
        return [json.loads(line) for line in (self.out_dir() / "simulation_attempts.jsonl").read_text().splitlines()]


def the_ring():
    return sim.made_up_ring(ring_module.load_ring())


# ---------- a normal simulated run ----------

def test_the_photo_order_fills_slots_1_2_3_left_to_right(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run()
    assert summary["success"] is True and summary["simulation"] is True
    assert summary["target"] == ["red die", "black die", "blue die"]
    assert [(m["name"], m["slot"], m["result"]) for m in summary["moves"]] == [
        ("red die", 1, "PASS"), ("black die", 2, "PASS"), ("blue die", 3, "PASS")]
    final = fakes.stories[0].log[-1]["dice"]
    for name, slot_deg in zip(summary["target"], the_ring().slots_deg):
        assert math.dist(final[name], the_ring().point_at(slot_deg)) < 0.2   # each die in its slot
    assert any("placed on the made-up ring" in line for line in fakes.said)


def test_every_output_is_marked_simulation(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run()
    out = fakes.out_dir()
    assert out.name.endswith("_simulation")
    assert sorted(p.name for p in out.iterdir()) == [
        "simulation_attempts.jsonl", "simulation_moves.json", "simulation_photo.jpg", "simulation_photo_dots.jpg",
        "simulation_summary.json", "simulation_video.mp4"]
    assert json.loads((out / "simulation_summary.json").read_text())["simulation"] is True
    assert fakes.moves_json()["simulation"] is True and "SIMULATED EXECUTION" in fakes.moves_json()["what"]
    assert all(a["simulation"] is True for a in fakes.attempts()) and len(fakes.attempts()) == 3
    for picture in ("simulation_photo.jpg", "simulation_photo_dots.jpg"):
        assert marks.jpeg_comment(out / picture).startswith("SIMULATION")


def test_the_move_list_is_exactly_what_the_real_planner_sends(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run()
    sent = [c["send"] for c in fakes.moves_json()["commands"]]
    assert sent[:3] == ["E 90 150 90 90", "?", "J 90 90 90 90 1500"]   # enable at the start pose, park
    ring, limits = the_ring(), arm.load_limits()
    red = sim.place_on_ring(SEEN, ring)["red die"]
    steps, _, _ = ring_loop.pick_drop_plan(ring, limits, PARK, ring.angle_for(*red), ring.slots_deg[0])
    expected = ["J " + " ".join(str(s.angles[j]) for j in arm.JOINTS) + f" {s.ms}" for s in steps]
    assert sent[3:3 + len(expected)] == expected                         # the first carry, line for line
    assert sent[-1] == "D" and sent[-2].startswith("J 90 150 90 90")      # ramp back to the start pose, then off
    j_lines = [line for line in sent if line.startswith("J")]
    assert max(int(line.split()[2]) for line in j_lines) == ring.shoulder_table  # never past the table
    printed = [line for line in fakes.said if line.startswith('  {"t_s"')]
    assert len(printed) == len(sent) and json.loads(printed[0])["send"] == "E 90 150 90 90"


def test_a_typed_row_instruction_sets_the_order(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run("row blue, red, black")
    assert summary["target"] == ["blue die", "red die", "black die"]
    assert [m["result"] for m in summary["moves"]] == ["PASS"] * 3


@pytest.mark.parametrize("instruction, names, problem", [
    ("dance red, blue", ["red die"], 'An instruction is "row" or "stack"'),
    ("row purple", ["red die", "blue die"], "'purple' matches 0 of the dice Spark named"),
    ("row black blue", ["black die", "blue die"], "Separate the colours with commas"),
    ("row blue", ["blue die", "dark blue die"], "'blue' matches 2"),
    ("row red, red", ["red die"], "red die is named twice"),
])
def test_an_instruction_that_does_not_match_the_dice_is_refused(instruction, names, problem):
    with pytest.raises(sim.SimError, match=problem):
        sim.match_colours(sim.parse_instruction(instruction)[1], names)


def test_a_two_word_colour_matches_one_die():
    assert sim.match_colours(["dark blue", "blue die"], ["blue die", "dark blue die"]) == ["dark blue die", "blue die"]


def test_names_that_are_not_dice_are_left_out(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch, names=list(SEEN) + ["white cable"])
    summary = fakes.run()
    assert "white cable" not in summary["target"] and any("not dice, left out: white cable" in s for s in fakes.said)


# ---------- the made-up world ----------

def test_a_die_is_placed_on_the_ring_in_its_own_direction():
    ring = the_ring()
    placed = sim.place_on_ring({"die": (12.0, 2.0)}, ring)["die"]
    assert math.dist(placed, ring.centre_cm) == pytest.approx(ring.radius_cm)
    before = math.atan2(2.0 - ring.centre_cm[1], 12.0 - ring.centre_cm[0])
    assert math.atan2(placed[1] - ring.centre_cm[1], placed[0] - ring.centre_cm[0]) == pytest.approx(before)


def test_the_made_up_ring_keeps_the_pick_numbers_of_ring_json():
    real, made_up = ring_module.load_ring(), the_ring()
    assert (made_up.shoulder_table, made_up.grip_hold, made_up.lift, made_up.camera_clear) == (
        real.shoulder_table, real.grip_hold, real.lift, real.camera_clear)
    assert made_up.placeholder is False and "MADE UP" in made_up.note and made_up.slots_deg == [22, 38, 54]


def test_the_claw_picks_up_only_a_die_under_it(tmp_path, monkeypatch):
    ring, limits = the_ring(), arm.load_limits()
    world = sim.SimArm(limits, ring, {"red die": ring.point_at(70)}, say=lambda s: None)
    a = arm.Arm(world, limits)
    a.enable({"base": 90, "shoulder": 150, "elbow": 90, "grip": 90}, sleep=world.sleep)
    a.move({"base": 90, "shoulder": ring.shoulder_table, "elbow": 90, "grip": 90}, 2000)
    a.move({"base": 90, "shoulder": ring.shoulder_table, "elbow": 90, "grip": ring.grip_hold}, 1000)
    assert world.held is None and world.log[-1]["event"] == "closed on nothing"      # nothing at base 90
    a.move({"base": 90, "shoulder": ring.shoulder_table, "elbow": 90, "grip": 90}, 1000)
    a.move({"base": 70, "shoulder": ring.shoulder_table, "elbow": 90, "grip": 90}, 1000)
    a.move({"base": 70, "shoulder": ring.shoulder_table, "elbow": 90, "grip": ring.grip_hold}, 1000)
    assert world.held == "red die" and world.dice["red die"] is None
    a.move({"base": 70, "shoulder": 160, "elbow": 90, "grip": ring.grip_hold}, 1000)
    a.move({"base": 40, "shoulder": 160, "elbow": 90, "grip": ring.grip_hold}, 1000)
    a.move({"base": 40, "shoulder": ring.shoulder_table, "elbow": 90, "grip": ring.grip_hold}, 1000)
    a.move({"base": 40, "shoulder": ring.shoulder_table, "elbow": 90, "grip": 90}, 1000)
    assert world.held is None and math.dist(world.dice["red die"], ring.point_at(40)) < 1e-9


# ---------- an injected miss ----------

def test_an_injected_miss_shows_the_retry_and_is_labelled(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run(miss_deg=6)
    red = [a for a in fakes.attempts() if a["name"] == "red die"]
    assert [a["result"] for a in red] == ["RETRY", "PASS"]
    assert red[0]["angle_miss_deg"] == pytest.approx(6, abs=0.1) and red[0]["next_correction_deg"] == -2.0
    assert red[1]["drop_deg"] == 20                                        # slot 22, aimed 2 degrees short
    assert summary["success"] is True and summary["injected_miss"]["name"] == "red die"
    story = fakes.stories[0]
    t_miss = story.injected["t_s"]
    assert any("INJECTED TEST MISS" in text for text in sim_video.overlay_texts(story, t_miss + 0.5))
    assert not any("INJECTED" in text for text in sim_video.overlay_texts(story, t_miss - 0.5))
    assert not any("INJECTED" in text for text in sim_video.overlay_texts(story, red[1]["sim_time_s"] + 1.5))
    claw = [c.get("simulated_claw") for c in fakes.moves_json()["commands"] if c.get("simulated_claw")]
    assert claw[:2] == ["picked up red die", "INJECTED TEST MISS: red die put down +6 degrees from the claw"]
    assert "  INJECTED TEST MISS: red die put down +6 degrees from the claw (simulation only)" in fakes.said


# ---------- refused before anything moves ----------

def test_dice_too_close_on_the_made_up_ring_are_refused_before_any_move(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch, seen=dict(SEEN, **{"black die": (13.5, 7.0)}))
    with pytest.raises(ring_loop.RingLayoutError, match="red die and black die are .* apart"):
        fakes.run()
    out = fakes.out_dir()
    assert not (out / "simulation_moves.json").exists() and fakes.stories == []
    assert "RingLayoutError" in json.loads((out / "simulation_summary.json").read_text())["error"]


def test_stacking_is_refused_in_this_build(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    with pytest.raises(sim.SimError, match="Stacking is not built yet"):
        fakes.run("stack black, blue, red")


# ---------- the video ----------

def test_the_video_has_a_title_card_the_banner_on_every_frame_and_simulation_metadata(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run(miss_deg=6)
    story = fakes.stories[0]
    monkeypatch.setattr(sim_video, "FPS", 2)            # few frames: this test draws every one
    calls, frames = [], []

    def fake_ffmpeg(command, **kwargs):
        calls.append(command)
        frames_dir = out.parent / "simulation_frames"
        frames.extend(cv2.imread(str(p)) for p in sorted(frames_dir.iterdir()))
        out.write_bytes(b"mp4")
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    out = tmp_path / "simulation_video.mp4"
    sim_video.render(out, story, say=lambda s: None, run=fake_ffmpeg, to_windows=str)
    command = calls[0]
    assert command[command.index("-framerate") + 1] == "2"
    assert any(a.startswith("title=") and "SIMULATION" in a for a in command)
    assert any(a.startswith("comment=SIMULATION") for a in command)
    assert command[-1].endswith("simulation_video.mp4") and not (tmp_path / "simulation_frames").exists()
    title = round(sim_video.TITLE_S * 2)
    assert len(frames) == title + math.ceil((story.end_s + sim_video.END_S) * 2) + 1
    red = np.array(sim_video.RED)
    assert all((frame[20, 20] == red).all() for frame in frames)               # the banner on every frame
    assert (frames[0][H // 2, 5] == np.array((45, 40, 38))).all()               # the title card
    assert not (frames[title][H // 2, 5] == np.array((45, 40, 38))).all()       # then the run


def test_a_move_is_drawn_part_way_through():
    log = [{"t_s": 0.0, "send": "J 90 160 90 90 1000", "before": [50, 160, 90, 90], "after": [90, 160, 90, 90],
            "dice": {}, "held": None, "step": "swing", "carrying": ""}]
    assert sim_video.state_at(log, 0.5)["angles"][0] == pytest.approx(70)
    assert sim_video.state_at(log, 3.0)["angles"][0] == 90


def test_ffmpeg_failure_keeps_the_frames_and_says_why(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run()
    monkeypatch.setattr(sim_video, "FPS", 1)
    failed = lambda command, **k: type("Result", (), {"returncode": 1, "stderr": "Unknown encoder 'libx264'"})()
    with pytest.raises(sim_video.SimVideoError, match="Unknown encoder 'libx264'.*frames are in"):
        sim_video.render(tmp_path / "simulation_video.mp4", fakes.stories[0], say=lambda s: None, run=failed,
                         to_windows=str)
    assert any((tmp_path / "simulation_frames").iterdir())


def test_die_colours_come_from_the_name():
    assert sim_video.die_colour("red die") == sim_video.COLOURS["red"]
    assert sim_video.die_colour("dark blue die") == sim_video.COLOURS["blue"]
    assert sim_video.die_colour("mystery die") == (160, 160, 160)


# ---------- the real replies on the real photo ----------

REPLIES = config.ROOT / "tests/fixtures/meta_responses"


def test_the_real_replies_on_dice_sheet_fit_the_made_up_ring(tmp_path, monkeypatch):
    """SAM's and Spark's real replies on tests/fixtures/photos/dice_sheet.jpg (4 Nov, one real run, $0.0156)."""
    fakes = Fakes(tmp_path, monkeypatch)
    replies = {name: (REPLIES / f"sam_photo_dice_sheet_{name.replace(' ', '_')}.txt").read_text()
               for name in ("small coloured dot", "red die", "black die", "blue die")}
    monkeypatch.setattr(sam, "find", lambda path, name, use_cache=True: sam.parse_boxes(replies[name]))
    spark = (REPLIES / "spark_name_objects_dice_sheet.txt").read_text()
    monkeypatch.setattr(brain, "name_objects", lambda path, attempts=None: brain.parse_names(spark))
    summary = fakes.run()
    assert summary["target"] == ["red die", "black die", "blue die"] and summary["success"] is True
    assert any("red die: photo (11.7, 6.7) cm -> placed on the made-up ring at base 72" in s for s in fakes.said)
    assert [c["send"] for c in fakes.moves_json()["commands"]][-1] == "D"
