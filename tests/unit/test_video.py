"""video.py: choosing the settled frame, arrival order, shrinking. The saved 317-frame reply plus made-up tracks."""
import subprocess

import cv2
import pytest

from mimic import config, sam, video
from mimic.sam import Box
from mimic.video import VideoError, VideoInfo

CLIP = config.ROOT / "tests/fixtures/clip.mp4"
REPLY = config.ROOT / "tests/fixtures/meta_responses/sam_video_green_lighter_317f.txt"
TEN_FPS = VideoInfo(frames=60, fps=10.0, width=1280, height=720)


def box_at(x, y, size=40):
    return Box(round(x - size / 2), round(y - size / 2), round(x + size / 2), round(y + size / 2), 1280, 720)


def path(points: dict) -> list:
    """{frame: (x, y)} -> [(frame, box), ...] as parse_tracks gives it."""
    return [(f, box_at(*points[f])) for f in sorted(points)]


def moves_then_stops(start, end, stop_frame, frames=60):
    """Slides from start to end over frames 0..stop_frame, then stays at end."""
    points = {}
    for f in range(frames):
        t = min(f / stop_frame, 1.0)
        points[f] = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
    return path(points)


def still(at, frames=60):
    return path({f: at for f in range(frames)})


# ---------- the real saved reply (a green lighter, handheld camera, 30 fps) ----------

def real_track():
    text = REPLY.read_text()
    info = video.video_info(CLIP)
    return video.check_track("green lighter", sam.parse_tracks(text), sam.frame_count(text), info), info


def test_real_clip_is_317_frames_at_30_fps_and_the_reply_matches_it():
    track, info = real_track()
    assert (info.frames, info.fps, info.width, info.height) == (317, 30.0, 1280, 720)
    assert len(track) == 317


def test_real_handheld_shake_still_counts_as_still():
    track, info = real_track()
    settled = video.settled_frames({"green lighter": track}, info)
    assert settled[0] == 316                      # the last frame qualifies
    assert len(settled) == 317 - 29               # every frame with a full second before it

def test_real_handheld_drift_over_the_whole_clip_shows_as_a_late_arrival():
    # The lighter never moved, but the handheld camera drifted 21 px over 10 s, so its first frames sit just
    # over 19.2 px from where it ends and it "arrives" at frame 9 (0.3 s). With the phone on a stand this drift
    # should not happen; arrival times are only logged, never used to plan (needs.md A3b).
    track, info = real_track()
    assert video.arrival_frames({"green lighter": track}, 316, info) == {"green lighter": 9}


def test_frames_to_try_step_back_one_second_at_a_time():
    track, info = real_track()
    settled = video.settled_frames({"green lighter": track}, info)
    assert video.frames_to_try(settled, info, 3) == [316, 286, 256]


def test_a_frame_is_saved_as_a_photo(tmp_path):
    out = video.extract_frame(CLIP, 316, tmp_path / "f.jpg")
    assert cv2.imread(out).shape == (720, 1280, 3)
    with pytest.raises(VideoError, match="ended before frame 317"):
        video.extract_frame(CLIP, 317, tmp_path / "g.jpg")


# ---------- made-up tracks, 10 fps ----------

def test_the_last_frame_with_everything_still_for_1_s_is_chosen_and_arrival_is_first_to_settle():
    tracks = {"red block": moves_then_stops((100, 600), (700, 300), stop_frame=40),
              "blue block": moves_then_stops((100, 400), (900, 300), stop_frame=20),
              "green block": still((500, 300))}
    settled = video.settled_frames(tracks, TEN_FPS)
    assert settled[0] == 59
    assert min(settled) == 48     # red is within 19.2 px of its end spot from frame 39, plus 9 more frames
    arrived = video.arrival_frames(tracks, 59, TEN_FPS)
    assert arrived["green block"] == 0
    assert 16 <= arrived["blue block"] <= 20 and 36 <= arrived["red block"] <= 40  # within 19 px of the end spot
    assert video.arrival_order(arrived, ["red block", "blue block", "green block"]) == \
        ["green block", "blue block", "red block"]


def test_nothing_still_for_a_whole_second_gives_no_frame():
    tracks = {"red block": moves_then_stops((100, 600), (700, 300), stop_frame=55)}  # only 5 still frames
    assert video.settled_frames(tracks, TEN_FPS) == []


def test_a_frame_missing_from_the_track_breaks_the_second():
    points = {f: (500, 300) for f in range(60) if f != 55}  # e.g. a hand over the block in frame 55
    settled = video.settled_frames({"red block": path(points)}, TEN_FPS)
    assert 59 not in settled and 54 in settled and 64 not in settled


def test_still_means_within_1_5_percent_of_the_width():
    shake = path({f: (500 + (15 if f % 2 else 0), 300) for f in range(60)})    # 15 px: under 19.2
    drift = path({f: (500 + (25 if f % 2 else 0), 300) for f in range(60)})    # 25 px: over
    assert video.settled_frames({"a": shake}, TEN_FPS)
    assert video.settled_frames({"a": drift}, TEN_FPS) == []


def test_arrival_ties_keep_the_given_order():
    assert video.arrival_order({"a": 0, "b": 0, "c": 3}, ["b", "a", "c"]) == ["b", "a", "c"]


# ---------- stops ----------

def test_a_block_sam_never_found_stops():
    with pytest.raises(VideoError, match="never found 'red block'"):
        video.check_track("red block", {}, 0, TEN_FPS)


def test_two_tracks_for_one_name_stop():
    with pytest.raises(VideoError, match="2 tracks .* lost the block and found it again"):
        video.check_track("red block", {0: still((5, 5), 30), 1: still((9, 9), 60)}, 60, TEN_FPS)


def test_a_lost_track_stops():
    lost = still((500, 300), frames=20)  # last seen in frame 19 of 60
    with pytest.raises(VideoError, match="lost 'red block' at 1.9 s .*frame 19 of 60"):
        video.check_track("red block", {0: lost}, 60, TEN_FPS)
    video.check_track("red block", {0: still((500, 300), frames=50)}, 60, TEN_FPS)  # missing only the last 1 s: ok


def test_a_reply_that_does_not_match_the_video_stops():
    with pytest.raises(VideoError, match="covers 59 frames but the video has 60"):
        video.check_track("red block", {0: still((5, 5))}, 59, TEN_FPS)
    small = [(f, Box(0, 0, 10, 10, 640, 360)) for f in range(60)]
    with pytest.raises(VideoError, match="640x360 picture"):
        video.check_track("red block", {0: small}, 60, TEN_FPS)


# ---------- cost ----------

def test_cost_estimate_and_the_longest_video_the_cap_allows():
    assert video.estimate_cost(80, 3) == pytest.approx(3 * 80 * 0.0002 + 0.06)     # an 8 s clip: $0.108
    assert video.estimate_cost(20, 3) == pytest.approx(3 * 50 * 0.0002 + 0.06)     # billed as 50 frames
    assert video.max_frames(0.30, 3) == 399                                        # about 40 s at 10 fps
    assert video.estimate_cost(video.max_frames(0.30, 3), 3) <= 0.30


# ---------- shrinking with Windows ffmpeg (faked here) ----------

def test_shrink_command_keeps_10_fps_at_most_720p_no_sound_no_metadata():
    landscape = video.shrink_command("in.mp4", "out.mp4", 1920, 1080)
    assert landscape[landscape.index("-vf") + 1] == "fps=10,scale=-2:720"
    assert "-an" in landscape and landscape[landscape.index("-map_metadata") + 1] == "-1"
    assert "libx264" in landscape and landscape[0] == config.FFMPEG_PATH
    portrait = video.shrink_command("in.mp4", "out.mp4", 1080, 1920)
    assert portrait[portrait.index("-vf") + 1] == "fps=10,scale=720:-2"
    small = video.shrink_command("in.mp4", "out.mp4", 1280, 720)
    assert small[small.index("-vf") + 1] == "fps=10"


def write_video(path, frames, fps, size=(1280, 720)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        writer.write(cv2.UMat(size[1], size[0], cv2.CV_8UC3, (200, 200, 200)).get())
    writer.release()


def fake_ffmpeg(fps=10, returncode=0, stderr=""):
    def run(command, **kwargs):
        if returncode == 0:
            write_video(command[-1], 20, fps)  # the fake "Windows path" is the real path here
        return subprocess.CompletedProcess(command, returncode, "", stderr)
    return run


def test_shrink_returns_the_new_video(tmp_path):
    info = video.shrink(CLIP, tmp_path / "sent.mp4", run=fake_ffmpeg(), to_windows=str)
    assert (info.frames, info.fps) == (20, 10.0)


def test_shrink_failures_are_clear(tmp_path):
    with pytest.raises(VideoError, match="exit 1\\): bad codec"):
        video.shrink(CLIP, tmp_path / "a.mp4", run=fake_ffmpeg(returncode=1, stderr="x\nbad codec"), to_windows=str)
    with pytest.raises(VideoError, match="at 30 fps, not 10 fps"):
        video.shrink(CLIP, tmp_path / "b.mp4", run=fake_ffmpeg(fps=30), to_windows=str)

    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 120)
    with pytest.raises(VideoError, match="more than 120 s"):
        video.shrink(CLIP, tmp_path / "c.mp4", run=slow, to_windows=str)
    with pytest.raises(VideoError, match="cannot be read as a video"):
        video.shrink(tmp_path / "missing.mp4", tmp_path / "d.mp4", run=fake_ffmpeg(), to_windows=str)
