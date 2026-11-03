"""The demo shown on video: which frame to read the arrangement from, and the order the blocks arrived in.

SAM follows each block through the video (one prompt per name, sam.track). From those tracks:
- settled_frames: frames where every block was seen in each of the last STILL_S seconds of frames and stayed
  within STILL_FRACTION of the image width of where it is in that frame, so nothing is still being moved;
- arrival_frames: each block arrived when it was first seen at its final spot after it was last seen elsewhere.

Before anything is sent, shrink() re-encodes the video with Windows ffmpeg: 10 frames a second, the shorter side
at most 720 px, no sound, no metadata (a phone video can carry its location). Fewer frames to pay for (an 8 s clip
is about 80), and no voices sent.
"""
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from mimic import camera, config

SEND_FPS = 10
MAX_SHORT_SIDE_PX = 720
STILL_S = 1.0             # a block must have been still this long (10 frames at 10 fps)
STILL_FRACTION = 0.015    # of the image width: 19 px at 1280 (about 0.8 cm on the sheet); the saved handheld clip
                          # shakes by at most 10.9 px (0.0085) within any second
LOST_AFTER_S = 1.0        # a track that ends this long before the video does has been lost
AGREE_CM = 1.0            # a block's photo box and its video box, same frame, must be this close
NAMING_AND_PHOTOS_USD = 0.06  # naming (Spark + SAM checks; retries capped at $0.03), up to 4 dot photos, 1 per block
FFMPEG_TIMEOUT_S = 120


class VideoError(ValueError):
    pass


@dataclass
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int

    @property
    def seconds(self) -> float:
        return self.frames / self.fps


def video_info(path) -> VideoInfo:
    """Frame count (the same count meta_api bills by), frame rate and size, read with OpenCV."""
    capture = cv2.VideoCapture(str(path))
    info = VideoInfo(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS),
                     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if info.frames <= 0 or info.fps <= 0:
        raise VideoError(f"{path} cannot be read as a video.")
    return info


def frames_for(info: VideoInfo, seconds: float) -> int:
    return round(seconds * info.fps)


# ---------- shrinking the video before it is sent ----------

def shrink_command(in_windows_path: str, out_windows_path: str, width: int, height: int) -> list[str]:
    scale = ""
    if min(width, height) > MAX_SHORT_SIDE_PX:  # -2: keep the shape, with an even number of pixels
        scale = f",scale=-2:{MAX_SHORT_SIDE_PX}" if width >= height else f",scale={MAX_SHORT_SIDE_PX}:-2"
    return [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y", "-i", in_windows_path,
            "-vf", f"fps={SEND_FPS}{scale}", "-an", "-map_metadata", "-1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_windows_path]


def shrink(in_path, out_path, run=subprocess.run, to_windows=camera.windows_path) -> VideoInfo:
    """Re-encode `in_path` into `out_path` (10 fps, at most 720p, no sound) and return the new video's info."""
    original = video_info(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # so an old file can never be mistaken for the new one
    command = shrink_command(to_windows(in_path), to_windows(out_path), original.width, original.height)
    try:
        result = run(command, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise VideoError(f"ffmpeg took more than {FFMPEG_TIMEOUT_S} s to shrink the video.")
    except FileNotFoundError:
        raise VideoError(f"ffmpeg not found at {config.FFMPEG_PATH}: set MIMIC_FFMPEG in .env.")
    if result.returncode != 0:
        last = (result.stderr or "").strip().splitlines()[-1:] or ["no message"]
        raise VideoError(f"ffmpeg could not shrink the video (exit {result.returncode}): {last[0]}")
    if not out_path.exists():
        raise VideoError("ffmpeg finished but wrote no video.")
    info = video_info(out_path)
    if round(info.fps) != SEND_FPS or min(info.width, info.height) > MAX_SHORT_SIDE_PX:
        raise VideoError(f"The shrunk video is {info.width}x{info.height} at {info.fps:g} fps, "
                         f"not {SEND_FPS} fps at most {MAX_SHORT_SIDE_PX}p.")
    return info


def estimate_cost(frames: int, blocks: int) -> float:
    """Most this video can cost: one SAM video call per block, plus naming and photos."""
    video = blocks * max(frames, config.SAM_MIN_VIDEO_FRAMES) * config.SAM_PER_VIDEO_FRAME
    return video + NAMING_AND_PHOTOS_USD


def max_frames(cap_usd: float, blocks: int) -> int:
    return int((cap_usd - NAMING_AND_PHOTOS_USD) / (blocks * config.SAM_PER_VIDEO_FRAME))


def extract_frame(video_path, frame: int, out_path) -> str:
    """Save frame number `frame` (0 = the first) as a JPEG, reading the video from the start."""
    capture = cv2.VideoCapture(str(video_path))
    for _ in range(frame):
        if not capture.grab():
            capture.release()
            raise VideoError(f"{video_path} ended before frame {frame}.")
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise VideoError(f"{video_path} ended before frame {frame}.")
    cv2.imwrite(str(out_path), image)
    return str(out_path)


# ---------- reading SAM's tracks ----------

def check_track(name: str, tracks: dict, reply_frames: int, info: VideoInfo) -> list:
    """The one track SAM gave for `name`, as [(frame, box), ...]. Stops if it is missing, doubled or lost."""
    if reply_frames and reply_frames != info.frames:
        raise VideoError(f"SAM's reply for {name!r} covers {reply_frames} frames but the video has {info.frames}: "
                         "its frame numbers cannot be matched to the video.")
    if not tracks:
        raise VideoError(f"SAM never found {name!r} in the video.")
    if len(tracks) > 1:
        raise VideoError(f"SAM gave {name!r} {len(tracks)} tracks (ids {sorted(tracks)}): it lost the block and "
                         "found it again, or saw two. Move the blocks more slowly and keep them in view.")
    seen = next(iter(tracks.values()))
    box = seen[0][1]
    if (box.w, box.h) != (info.width, info.height):
        raise VideoError(f"SAM's boxes for {name!r} are for a {box.w}x{box.h} picture, but the video is "
                         f"{info.width}x{info.height}.")
    last = seen[-1][0]
    if info.frames - 1 - last > frames_for(info, LOST_AFTER_S):
        raise VideoError(f"SAM lost {name!r} at {last / info.fps:.1f} s (last seen in frame {last} of "
                         f"{info.frames}) and never found it again.")
    return seen


def _still_at(by_frame: dict, frame: int, window: int, limit_px: float) -> bool:
    here = by_frame.get(frame)
    if here is None or frame - window + 1 < 0:
        return False
    for f in range(frame - window + 1, frame + 1):
        box = by_frame.get(f)
        if box is None or math.dist(box.centre, here.centre) > limit_px:
            return False
    return True


def settled_frames(tracks: dict, info: VideoInfo) -> list[int]:
    """Frames where every block has been still for STILL_S, latest first. tracks: {name: [(frame, box), ...]}."""
    window = frames_for(info, STILL_S)
    limit_px = STILL_FRACTION * info.width
    by_name = {name: dict(seen) for name, seen in tracks.items()}
    return [f for f in range(info.frames - 1, -1, -1)
            if all(_still_at(by_frame, f, window, limit_px) for by_frame in by_name.values())]


def frames_to_try(settled: list[int], info: VideoInfo, tries: int) -> list[int]:
    """The latest settled frame, then the latest one at least STILL_S earlier than the last pick, and so on."""
    gap = frames_for(info, STILL_S)
    picked = []
    for frame in settled:
        if not picked or picked[-1] - frame >= gap:
            picked.append(frame)
        if len(picked) == tries:
            break
    return picked


def arrival_frames(tracks: dict, frame: int, info: VideoInfo) -> dict[str, int]:
    """For each block: the first frame it was seen at its spot in `frame`, after it was last seen elsewhere."""
    limit_px = STILL_FRACTION * info.width
    arrived = {}
    for name, seen in tracks.items():
        final = dict(seen)[frame].centre
        away = [f for f, box in seen if f <= frame and math.dist(box.centre, final) > limit_px]
        last_away = max(away) if away else -1
        arrived[name] = next(f for f, _ in seen if f > last_away)
    return arrived


def arrival_order(arrived: dict[str, int], names: list[str]) -> list[str]:
    """First to settle first; blocks that arrived together keep the order of `names`."""
    return sorted(names, key=lambda name: (arrived[name], names.index(name)))
