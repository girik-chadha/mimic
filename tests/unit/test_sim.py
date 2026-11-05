"""Simulation mode (mimic/sim.py, mimic/sim_video.py, run_mimic.py --ring --sim) with fake SAM and Spark.

No hardware, no API calls, no ffmpeg (the encode step is faked). The dice sit about where they are in
tests/fixtures/photos/dice_sheet.jpg: red, black, blue from left to right across the middle of the sheet.
"""
import dataclasses
import json
import math
import re

import cv2
import numpy as np
import pytest

from mimic import arm, brain, camera, config, marks, meta_api, ring_loop, sam, scene, sim, sim_video
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
        self.spark_calls = []
        monkeypatch.setattr(meta_api, "call", self.spark)
        monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
        monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))

    def spark(self, model, content, save_as=None, use_cache=True):
        text = content[0]["text"]
        goal = re.search(r"<goal>(.*?)</goal>", text, re.S).group(1)
        self.spark_calls.append(goal)
        if "you chose these search" in text:          # the one rephrase
            return json.dumps(RENAMES.get(goal, {}))
        return json.dumps(GOAL_REPLIES[goal])

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


def carries(kind, *pairs):
    return [{"object": o, "onto": onto} for o, onto in pairs]


LONG_GOAL = ("This is my esp32 robot arm and I want you to write hardware code to stack the black die on top of the blue "
             "die and then the red die on top of the black die in a three die stack")
ON_BLUE = {"objects": ["blue die", "black die", "red die"], "kind": "stack",
           "carries": carries("stack", ("black die", "blue die"), ("red die", "black die")),
           "setup": ["esp32 robot arm", "hardware code"], "task": "Stack black on blue, then red on black.",
           "reasoning": "Blue is the base, so it stays put. Black goes on it first, then red on top."}
GOAL_REPLIES = {   # what Spark would answer for each goal (the fakes look them up by the goal in the prompt)
    "row blue, red, black": {"objects": ["blue die", "red die", "black die"], "kind": "row",
                             "carries": carries("row", ("blue die", None), ("red die", None), ("black die", None))},
    "stack black, blue, red": {"objects": ["black die", "blue die", "red die"], "kind": "stack",
                               "carries": carries("stack", ("blue die", "black die"), ("red die", "blue die"))},
    "stack blue, black, red": ON_BLUE,
    "put black on blue, then red on black": ON_BLUE,
    LONG_GOAL: ON_BLUE,
    "paint the red die green": {"objects": ["red die"], "kind": "other", "carries": []},
    "put the green die on the blue die": {"objects": ["green die", "blue die"], "kind": "stack",
                                          "carries": carries("stack", ("green die", "blue die"))},
}
RENAMES = {"put the green die on the blue die": {"green die": "green cube"}}


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
        "simulation_attempts.jsonl", "simulation_check_1.jpg", "simulation_check_2.jpg", "simulation_check_3.jpg",
        "simulation_end.jpg", "simulation_moves.json", "simulation_photo.jpg", "simulation_photo_dots.jpg",
        "simulation_sam.jpg", "simulation_start.jpg", "simulation_summary.json", "simulation_video.mp4"]
    assert json.loads((out / "simulation_summary.json").read_text())["simulation"] is True
    assert fakes.moves_json()["simulation"] is True and "SIMULATED EXECUTION" in fakes.moves_json()["what"]
    assert all(a["simulation"] is True for a in fakes.attempts()) and len(fakes.attempts()) == 3
    for picture in out.glob("*.jpg"):
        assert marks.jpeg_comment(picture).startswith("SIMULATION")
    assert [a["photo"] for a in fakes.attempts()] == [f"simulation_check_{i}.jpg" for i in (1, 2, 3)]
    summary = json.loads((out / "simulation_summary.json").read_text())
    assert [p["claw_closed_s"] <= p["claw_limit_s"] == 4.5 for p in summary["plan"]] == [True] * 3
    assert set(summary["sam"]["check_cm"]) == {"centre_check", "right_check"} and len(summary["sam"]["boxes"]) == 3


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
    assert "[simulated]   INJECTED TEST MISS: red die put down +6 degrees from the claw (simulation only)" in fakes.said
    assert not any(line.lstrip().startswith(("lower to", "Hold the arm")) for line in fakes.said)   # all tagged
    assert any(line.startswith("[simulated] Hold the arm") for line in fakes.said)


# ---------- refused before anything moves ----------

def test_dice_too_close_on_the_made_up_ring_are_refused_before_any_move(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch, seen=dict(SEEN, **{"black die": (13.5, 7.0)}))
    with pytest.raises(ring_loop.RingLayoutError, match="red die and black die are .* apart"):
        fakes.run()
    out = fakes.out_dir()
    assert not (out / "simulation_moves.json").exists() and fakes.stories == []
    assert "RingLayoutError" in json.loads((out / "simulation_summary.json").read_text())["error"]


# ---------- stacking (simulation only) ----------

def test_a_stack_piles_every_die_on_slot_1_bottom_first(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run("stack black, blue, red")
    assert summary["stack"] is True and summary["success"] is True and "cannot stack" in summary["sim_only"]
    assert [(m["name"], m["slot"]) for m in summary["moves"]] == [("blue die", 1), ("red die", 1)]   # black stays
    final = fakes.stories[0].log[-1]
    spot = final["dice"]["black die"]
    assert all(math.dist(final["dice"][n], spot) < 0.3 for n in summary["target"])
    assert final["level"] == {"black die": 0, "blue die": 1, "red die": 2}
    assert sim_video._labels(final["dice"], final["level"]) == [(final["dice"]["black die"], "stack: black, blue, red")]
    claw = [c["simulated_claw"] for c in fakes.moves_json()["commands"] if "simulated_claw" in c]
    assert claw[-1] == "put down red die on top of blue die (stack level 3)"
    assert any("SIMULATION ONLY" in line for line in fakes.said)


def test_only_the_carries_onto_the_stack_are_marked_sim_only(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run("stack black, blue, red")
    moves = fakes.moves_json()
    assert "cannot stack" in moves["sim_only"]
    marked = {c["carrying"].split(" ->")[0] for c in moves["commands"] if c.get("sim_only")}
    assert marked == {"blue die", "red die"}                      # black, at the bottom, is an ordinary drop
    assert not any(c.get("sim_only") for c in moves["commands"] if not c["carrying"])   # enable, park, end
    # a carry onto the stack is the planner's ordinary pick and drop: it lowers to the table angle as usual
    lowered = [c["send"] for c in moves["commands"] if c["carrying"].startswith("red die") and c["step"] == "open"]
    assert lowered and all(int(line.split()[2]) == the_ring().shoulder_table for line in lowered)


def test_a_row_run_marks_nothing_sim_only(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run()
    assert fakes.moves_json()["sim_only"] is None and not any(c.get("sim_only") for c in fakes.moves_json()["commands"])


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


# ---------- the goal, read by code ----------












def test_a_pairs_goal_builds_the_pile_where_the_bottom_die_lies(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run("put black on blue, then red on black")
    assert summary["stack"] is True and summary["stack_in_place"] is True and summary["success"] is True
    assert summary["target"] == ["blue die", "black die", "red die"]
    assert [m["name"] for m in summary["moves"]] == ["black die", "red die"]        # blue is never carried
    log = fakes.stories[0].log
    assert log[-1]["dice"]["blue die"] == log[0]["dice"]["blue die"]                # where it was at the start
    assert not any(e["event"] == "picked up blue die" for e in log)
    assert log[-1]["level"] == {"blue die": 0, "black die": 1, "red die": 2}
    assert any("the pile is built where blue die lies (base 106); it stays put" in line for line in fakes.said)


def test_a_pile_whose_bottom_die_cannot_be_reached_goes_on_slot_1(tmp_path, monkeypatch):
    ring = the_ring()
    placed = sim.place_on_ring(SEEN, ring)
    limits = dataclasses.replace(arm.load_limits(), ranges_deg={"base": (0, 100), "shoulder": (80, 180),
                                                                "elbow": (30, 120)})
    with pytest.raises(ring_loop.RingLayoutError, match="blue die is at base 106, outside the base range 0-100"):
        sim.pile_in_place(placed, ["blue die", "black die", "red die"], ring, limits)
    fakes = Fakes(tmp_path, monkeypatch)
    monkeypatch.setattr(sim, "pile_in_place", lambda *a: (_ for _ in ()).throw(ring_loop.RingLayoutError("too far")))
    summary = fakes.run("put black on blue, then red on black")
    assert summary["stack_in_place"] is False and [m["name"] for m in summary["moves"]] == [
        "blue die", "black die", "red die"]
    assert any("building the pile on slot 1 instead" in line for line in fakes.said)



# ---------- Spark reads the goal with the photo; code checks every part ----------

def test_spark_reads_a_long_sentence_into_objects_and_moves(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    summary = fakes.run(LONG_GOAL)
    assert fakes.spark_calls == [LONG_GOAL]                               # one call: the goal and the photo together
    assert summary["target"] == ["blue die", "black die", "red die"] and summary["stack_in_place"] is True
    read = summary["goal_read"]
    assert read["by"] == "spark" and read["kind"] == "stack" and read["carries"] == [
        {"object": "black die", "onto": "blue die"}, {"object": "red die", "onto": "black die"}]
    assert [n["boxes"] for n in summary["naming"]] == [1, 1, 1]
    assert read["setup"] == ["esp32 robot arm", "hardware code"] and read["task"].startswith("Stack black on blue")
    assert read["reasoning"].startswith("Blue is the base")          # shown only; never used for a move


def test_the_display_only_fields_are_cleaned_and_never_needed():
    plan = brain.parse_goal_reply(json.dumps({"objects": ["red die", "blue die"], "kind": "row",
        "carries": [{"object": "red die", "onto": None}, {"object": "blue die", "onto": None}],
        "setup": ["robot arm", 7, "  esp32  "], "task": None}))
    assert plan["setup"] == ["robot arm", "esp32"] and plan["task"] == "" and plan["reasoning"] == ""


def test_a_goal_that_is_not_stacking_or_a_row_is_refused(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    with pytest.raises(brain.GoalError, match="Mimic can stack or line up the objects it finds. It can't do that yet."):
        fakes.run("paint the red die green")


def test_an_object_sam_cannot_find_gets_one_rephrase_then_a_plain_refusal(tmp_path, monkeypatch):
    fakes = Fakes(tmp_path, monkeypatch)
    with pytest.raises(brain.GoalError, match="^I couldn't find the green die in the photo.$"):
        fakes.run("put the green die on the blue die")
    assert len(fakes.spark_calls) == 2                                    # the read, and one rephrase


@pytest.mark.parametrize("reply, problem", [
    ({"objects": ["black die"], "kind": "stack", "carries": [{"object": "black die", "onto": "mug"}]},
     "not one of the objects it listed"),
    ({"objects": ["a b c d e"], "kind": "row", "carries": [{"object": "a b c d e", "onto": None}]}, "2 to 4 words"),
    ({"objects": ["red die", "blue die", "black die"], "kind": "stack",
      "carries": [{"object": "red die", "onto": "blue die"}, {"object": "black die", "onto": "blue die"}]},
     "must go onto the red die"),
    ({"objects": ["red die", "blue die"], "kind": "row", "carries": [{"object": "red die", "onto": "blue die"}]},
     "A row cannot put one object onto another"),
    ({"objects": ["red die"], "kind": "row", "carries": [{"object": "red die", "onto": None}] * 2},
     "carry the same object twice"),
])
def test_code_refuses_a_reply_that_does_not_make_sense(reply, problem):
    with pytest.raises(brain.GoalError, match=problem):
        brain.parse_goal_reply(json.dumps(reply))


def test_an_overlong_goal_is_refused_before_any_call():
    with pytest.raises(sim.SimError, match="please keep it under 600"):
        sim.check_goal("put black on blue " * 50)
