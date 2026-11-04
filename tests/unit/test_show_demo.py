"""show_demo.py end to end with a fake ffmpeg, SAM and Spark. A real (made-up) 10 fps video, no network."""
import dataclasses
import importlib.util
import json
import re
import shutil

import cv2
import numpy as np
import pytest

from mimic import brain, config, meta_api, sam, scene, video
from mimic.sam import Box

spec = importlib.util.spec_from_file_location("show_demo", config.ROOT / "scripts/show_demo.py")
show_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(show_demo)

W, H, FRAMES = 1280, 720, 60


def to_px(cm):
    """A simple made-up camera: 28 px per cm across, 22 px per cm up the sheet (y grows away from you)."""
    return (100 + cm[0] * 28, 680 - cm[1] * 22)


def box_at(px, size=40):
    return Box(round(px[0] - size / 2), round(px[1] - size / 2), round(px[0] + size / 2), round(px[1] + size / 2), W, H)


def slide(start, end, start_frame, stop_frame):
    """Pixel position per frame: at start until start_frame, sliding to end by stop_frame, then at end."""
    out = []
    for f in range(FRAMES):
        t = min(max((f - start_frame) / max(stop_frame - start_frame, 1), 0.0), 1.0)
        out.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return out


class FakeDemo:
    """Red goes to slot 1 last (settles at frame 30), blue to slot 2 first (frame 15), green sits in slot 3."""

    def __init__(self, monkeypatch, tmp_path):
        self.layout = dataclasses.replace(scene.load_layout(config.LAYOUT_PATH), placeholder=False)
        slot = [to_px(s) for s in self.layout.slots_cm]
        self.names = ["red block", "blue block", "green block"]
        self.px = {"red block": slide((300, 200), slot[0], 20, 30),
                   "blue block": slide((400, 250), slot[1], 5, 15),
                   "green block": [slot[2]] * FRAMES}
        self.lost_after = {}             # name -> last frame SAM sees it
        self.hidden_dot_frames = set()   # frames where a dot is covered
        self.photo_offset_px = {}        # name -> (dx, dy) added to its photo box only
        self.events, self.budgets = [], []
        self.video = tmp_path / "demo.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
        for _ in range(FRAMES):
            writer.write(np.full((H, W, 3), 230, np.uint8))
        writer.release()
        monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: self.layout)
        monkeypatch.setattr(video, "shrink", self.shrink)
        monkeypatch.setattr(sam, "find", self.find)
        monkeypatch.setattr(sam, "track", self.track)
        monkeypatch.setattr(brain, "name_objects", self.name_objects)

    def shrink(self, src, dst):
        self.events.append(("shrink",))
        shutil.copy(src, dst)
        return video.video_info(dst)

    def find(self, image_path, name, use_cache=True):
        if name in scene.DOT_FALLBACK_PROMPTS:
            return []   # a covered dot stays covered
        self.events.append(("find", name))
        self.budgets.append(config.BUDGET_USD)
        frame = int(re.search(r"frame_(\d+)\.jpg$", str(image_path)).group(1))
        if name == self.layout.dot_prompt:
            dots = [box_at(to_px(cm), 25) for cm in self.layout.dots_cm.values()]
            return dots[:5] if frame in self.hidden_dot_frames else dots
        x, y = self.px[name][frame]
        dx, dy = self.photo_offset_px.get(name, (0, 0))
        return [box_at((x + dx, y + dy))]

    def track(self, video_path, name, save_as=None):
        self.events.append(("track", name))
        self.budgets.append(config.BUDGET_USD)
        last = self.lost_after.get(name, FRAMES - 1)
        return {0: [(f, box_at(self.px[name][f])) for f in range(last + 1)]}, FRAMES

    def name_objects(self, image_path, attempts=None):
        self.events.append(("name", str(image_path)))
        return list(self.names)


def run(fake, tmp_path, typed="yes"):
    said = []

    def say(text=""):
        said.append(text)
        fake.events.append(("say", text))
    result = show_demo.run(str(fake.video), out_root=tmp_path / "runs", ask=lambda q: typed, say=say)
    return result, said


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BUDGET_USD", 1.0)
    return FakeDemo(monkeypatch, tmp_path)


def paid(events):
    return [e for e in events if e[0] in ("find", "track", "name")]


def test_the_target_comes_from_the_last_settled_frame_and_arrival_is_logged(fake, tmp_path):
    result, said = run(fake, tmp_path)
    assert result["chosen_frame"] == 59
    assert result["target_order"] == ["red block", "blue block", "green block"]
    assert result["arrival_order"] == ["green block", "blue block", "red block"]   # first to settle first
    assert result["arrival_frames"]["green block"] == 0
    saved = json.loads((tmp_path / "runs").glob("*/demo.json").__next__().read_text())
    assert saved["target_order"] == result["target_order"] and saved["arrival_order"] == result["arrival_order"]
    assert fake.events[fake.events.index(("name", saved["out_dir"] + "/frame_0059.jpg"))]  # named on the last frame
    assert [e for e in fake.events if e[0] == "track"] == [("track", n) for n in fake.names]  # one call per name


def test_the_warning_and_cost_come_before_anything_is_sent_and_the_cap_is_put_back(fake, tmp_path):
    run(fake, tmp_path)
    first_paid = fake.events.index(paid(fake.events)[0])
    before = [e[1] for e in fake.events[:first_paid] if e[0] == "say"]
    assert any(line.startswith("WARNING: the whole video goes to Meta") for line in before)
    assert any("could cost up to $0.10" in line for line in before)   # 3 blocks x 60 frames x $0.0002 + $0.06
    assert set(fake.budgets) == {0.30} and config.BUDGET_USD == 1.0


def test_anything_but_yes_sends_nothing(fake, tmp_path):
    with pytest.raises(show_demo.Cancelled, match="Nothing was sent"):
        run(fake, tmp_path, typed="y")
    assert paid(fake.events) == []


def test_main_says_cancelled(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    said = []
    assert show_demo.main([str(fake.video)], ask=lambda q: "no", say=said.append) == 1
    assert "Cancelled. Nothing was sent." in said


def test_a_video_that_could_cost_too_much_is_refused_before_asking(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(video, "shrink", lambda src, dst: video.VideoInfo(1000, 10.0, W, H))
    with pytest.raises(video.VideoError, match="could cost \\$0.66 .* Nothing was sent. Record at most 40 s"):
        show_demo.run(str(fake.video), out_root=tmp_path, ask=lambda q: pytest.fail("asked"), say=lambda s: None)
    assert paid(fake.events) == []


def test_unmeasured_dots_are_refused_before_shrinking(fake, tmp_path):
    fake.layout = dataclasses.replace(fake.layout, placeholder=True, dots_measured=False)
    with pytest.raises(scene.SceneError, match="dot positions are not measured yet"):
        run(fake, tmp_path)
    assert fake.events == []


def test_measured_dots_are_enough_while_the_rest_is_a_placeholder(fake, tmp_path):
    fake.layout = dataclasses.replace(fake.layout, placeholder=True)   # as committed: dots measured, slots not
    result, _ = run(fake, tmp_path)
    assert result["target_order"] == ["red block", "blue block", "green block"]


def test_a_last_frame_that_does_not_calibrate_stops_before_the_video_calls(fake, tmp_path):
    fake.hidden_dot_frames = {59}
    with pytest.raises(video.VideoError, match="last frame does not calibrate: Found 5 dots"):
        run(fake, tmp_path)
    assert not any(e[0] in ("track", "name") for e in fake.events)


def test_more_names_than_slots_stops_before_the_video_calls(fake, tmp_path):
    fake.names.append("yellow block")
    with pytest.raises(video.VideoError, match="named 4 objects but the sheet has 3 slots"):
        run(fake, tmp_path)
    assert not any(e[0] == "track" for e in fake.events)


def test_a_lost_track_stops(fake, tmp_path):
    fake.lost_after["blue block"] = 30
    with pytest.raises(video.VideoError, match="lost 'blue block' at 3.0 s"):
        run(fake, tmp_path)


def test_no_moment_when_everything_was_still_stops(fake, tmp_path):
    fake.px["red block"] = slide((300, 200), to_px(fake.layout.slots_cm[0]), 0, 59)  # moving the whole time
    with pytest.raises(video.VideoError, match="no moment where all 3 blocks had been still for 1 s"):
        run(fake, tmp_path)


def test_a_video_that_ends_before_the_last_block_settles_stops(fake, tmp_path):
    # Red sits in the mess area until frame 45, then moves to slot 1 as the video ends: the last moment everything
    # was still is before red moved, which is not the final arrangement.
    fake.px["red block"] = slide((300, 200), to_px(fake.layout.slots_cm[0]), 45, 59)
    with pytest.raises(video.VideoError, match="'red block' ends the video 1\\d.\\d cm from where it was in frame 4"):
        run(fake, tmp_path)


def test_a_covered_dot_in_the_latest_settled_frame_falls_back_one_second(fake, tmp_path):
    slot2 = to_px(fake.layout.slots_cm[1])
    for f in (57, 58, 59):  # blue is nudged 30 px at the very end, so the latest settled frame is 56
        fake.px["blue block"][f] = (slot2[0] + 30, slot2[1])
    fake.hidden_dot_frames = {56}
    result, _ = run(fake, tmp_path)
    assert result["chosen_frame"] == 46
    assert "frame 56 (5.6 s): Found 5 dots" in result["frames_not_used"][0]
    assert result["target_order"] == ["red block", "blue block", "green block"]


def test_no_settled_frame_with_all_6_dots_stops(fake, tmp_path):
    slot2 = to_px(fake.layout.slots_cm[1])
    for f in (57, 58, 59):
        fake.px["blue block"][f] = (slot2[0] + 30, slot2[1])
    fake.hidden_dot_frames = {56, 46}   # red is still from frame 38 on, so 56 and 46 are the only frames to try
    with pytest.raises(video.VideoError, match="No settled frame had all 6 dots: frame 56 .* \\| frame 46 [^|]*$"):
        run(fake, tmp_path)


def test_photo_and_track_that_disagree_stop(fake, tmp_path):
    fake.photo_offset_px["green block"] = (40, 0)   # 40 px is about 1.4 cm: a different object
    with pytest.raises(video.VideoError, match="'green block': .* 1.4 cm apart"):
        run(fake, tmp_path)


def test_main_reports_a_stop_and_the_cost(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    fake.lost_after["red block"] = 10
    said = []
    assert show_demo.main([str(fake.video)], ask=lambda q: "yes", say=said.append) == 1
    assert any(line.startswith("STOPPED: SAM lost 'red block'") for line in said)
    assert said[-1] == f"Cost so far: ${meta_api.cost_so_far():.4f}"
