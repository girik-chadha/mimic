"""sam.py tests, all on real SAM replies saved in tests/fixtures/meta_responses/. No network."""
import re

import pytest

from mimic import config, meta_api, sam
from mimic.sam import Box

REPLIES = config.ROOT / "tests/fixtures/meta_responses"
DESK = str(config.ROOT / "tests/fixtures/photos/desk.jpeg")


def saved(name):
    return (REPLIES / f"{name}.txt").read_text()


# ---------- the assumption the parser rests on ----------

def test_mask_text_never_contains_angle_brackets():
    replies = sorted(REPLIES.glob("*.txt"))
    assert len(replies) >= 6
    masks_checked = 0
    for path in replies:
        text = path.read_text()
        starts = [m.start() for m in re.finditer(r"<\|mask;", text)]
        for start in starts:
            end = text.index("|>", start)  # every saved mask is closed
            data = text[start + len("<|mask;"):end]
            assert "<" not in data and ">" not in data, f"{path.name} at {start}"
            masks_checked += 1
    assert masks_checked >= 300  # 317 video frames + 6 photo masks


# ---------- photos ----------

@pytest.mark.parametrize("name, expected", [
    ("sam_photo_desk_blue_lighter", Box(255, 363, 320, 583, 1280, 960, id=0)),
    ("sam_photo_desk_green_lighter", Box(602, 374, 677, 591, 1280, 960, id=0)),
    ("sam_photo_desk_black_rectangle", Box(848, 414, 969, 579, 1280, 960, id=0)),
])
def test_one_object_one_box(name, expected):
    assert sam.parse_boxes(saved(name)) == [expected]


def test_box_centre():
    assert Box(255, 363, 320, 583, 1280, 960).centre == (287.5, 473.0)


def test_no_match_is_an_empty_reply():
    assert saved("sam_photo_desk_red_cup") == ""
    assert sam.parse_boxes("") == []


def test_several_matches_keep_their_ids():
    boxes = sam.parse_boxes(saved("sam_photo_test_building"))
    assert [b.id for b in boxes] == [0, 1, 2]
    assert boxes[1] == Box(86, 749, 225, 823, 1920, 1080, id=1)
    assert all((b.w, b.h) == (1920, 1080) for b in boxes)


def test_photo_reply_cut_mid_mask_keeps_the_finished_boxes():
    text = saved("sam_photo_test_building")
    cut = text.index("<|mask;", text.index(",1<|box")) + 40  # inside the second mask
    assert [b.id for b in sam.parse_boxes(text[:cut])] == [0, 1]


def test_box_cut_in_half_is_ignored():
    text = saved("sam_photo_desk_blue_lighter")
    assert sam.parse_boxes(text[:text.index("y2=")]) == []


# ---------- video ----------

def test_video_one_track_through_all_317_frames():
    tracks = sam.parse_tracks(saved("sam_video_green_lighter_317f"))
    assert list(tracks) == [0]
    frames = [frame for frame, box in tracks[0]]
    assert frames == list(range(317))
    assert tracks[0][0][1] == Box(879, 249, 935, 448, 1280, 720, id=0)


def test_frame_with_nothing_seen_adds_nothing():
    text = "<0f>\n<1f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x=0;y=0;data=1,1,!|>\n<2f>\n"
    assert sam.parse_tracks(text) == {0: [(1, Box(1, 2, 3, 4, 10, 10, id=0))]}


def test_two_tracks_in_one_frame():
    text = ("<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=9;h=9|><|mask;data=!|>"
            ",1<|box;x1=5;y1=5;x2=6;y2=6;w=9;h=9|><|mask;data=!|>\n"
            "<1f>1<|box;x1=5;y1=5;x2=7;y2=7;w=9;h=9|><|mask;data=!|>\n")
    tracks = sam.parse_tracks(text)
    assert [f for f, _ in tracks[0]] == [0]
    assert [f for f, _ in tracks[1]] == [0, 1]


def test_video_reply_cut_mid_mask_does_not_crash():
    text = saved("sam_video_green_lighter_317f")
    tenth_mask = [m.start() for m in re.finditer(r"<\|mask;", text)][9]
    tracks = sam.parse_tracks(text[:tenth_mask + 50])
    assert [f for f, _ in tracks[0]] == list(range(10))  # frames 0-9; frame 9's box came before its mask


# ---------- find() ----------

def test_find_asks_sam_and_parses_the_reply(monkeypatch):
    sent = {}

    def fake_call(model, content, use_cache=True):
        sent.update(model=model, content=content, use_cache=use_cache)
        return saved("sam_photo_desk_blue_lighter")

    monkeypatch.setattr(meta_api, "call", fake_call)
    boxes = sam.find(DESK, "blue lighter")
    assert boxes == [Box(255, 363, 320, 583, 1280, 960, id=0)]
    assert sent["model"] == "sam-3.1"
    assert sent["content"][0] == {"type": "input_text", "text": "blue lighter"}
    assert sent["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert sent["use_cache"] is True
