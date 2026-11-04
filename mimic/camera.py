"""Take a photo with the phone camera, from WSL (needs.md A9).

The phone runs a webcam app (iVCam shows up on Windows as "e2eSoft iVCam", Camo as "Camo"). The device name comes
from --camera-name, then the MIMIC_CAMERA environment variable, then camera_name in calib/ring.json, then
"e2eSoft iVCam" (camera_name()). WSL cannot see webcams, so Windows ffmpeg.exe grabs the picture: it reads
`frames` frames (15 by default) and keeps overwriting the same file, so the last frame is what is saved. The first
frames are useless on both cameras: a 2-frame grab came out nearly white on iVCam (247.5) and black on Camo (0.0),
15-frame ones at 170.5 and 152.7 (needs.md A9).

After every photo the camera name, frame size and mean brightness (0-255) are printed. Outside 40-235 it is taken
once more after 1 s; if it is still outside, it is refused (CameraError) with the brightness and the camera name.
A photo is also refused if ffmpeg fails or takes over 10 s, if no file appears, or if it is smaller than 1280x720
(either way round: a portrait 720x1280 frame is fine).
"""
import os
import subprocess
import time
from pathlib import Path

import cv2

from mimic import config

TIMEOUT_S = 10
MIN_WIDTH, MIN_HEIGHT = 1280, 720
FRAMES = 15                       # read this many frames and keep the last: the first ones are blown out
TOO_DARK, TOO_BRIGHT = 40, 235    # mean grey level out of 255; a good photo of the sheet is about 150-170
RETAKE_WAIT_S = 1.0


class CameraError(RuntimeError):
    pass


def windows_path(path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def camera_name(option: str | None = None, from_ring: str | None = None) -> str:
    """The Windows webcam to read: --camera-name, then MIMIC_CAMERA, then ring.json's camera_name, then iVCam."""
    return option or os.environ.get("MIMIC_CAMERA") or from_ring or config.CAMERA_DEVICE


def ffmpeg_command(out_windows_path: str, frames: int = FRAMES, name: str | None = None) -> list[str]:
    return [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow", "-i", f"video={camera_name(name)}",
            "-frames:v", str(frames), "-update", "1", "-q:v", "2", out_windows_path]


def capture(path, run=subprocess.run, frames: int = FRAMES, sleep=time.sleep, say=print, name: str | None = None) -> str:
    """Save one photo to `path` (a WSL path) and return it, retaking once if its brightness is outside 40-235.
    `run` is replaceable for tests."""
    name = camera_name(name)
    for attempt in (1, 2):
        level, width, height = _grab(Path(path), run, frames, name)
        say(f"  photo {Path(path).name}: {name}, {width}x{height}, brightness {level:.1f}")
        if TOO_DARK <= level <= TOO_BRIGHT:
            return str(path)
        if attempt == 1:
            say(f"  brightness outside {TOO_DARK}-{TOO_BRIGHT}: taking it again in {RETAKE_WAIT_S:g} s")
            sleep(RETAKE_WAIT_S)
    what = "too bright (blown out)" if level > TOO_BRIGHT else "too dark"
    hint = "" if level > TOO_BRIGHT else f" Is the lens covered, or {name} not connected?"
    raise CameraError(f"The photo {Path(path).name} from {name} is {what}: mean brightness {level:.1f}, allowed "
                      f"{TOO_DARK}-{TOO_BRIGHT}, twice.{hint} Lock the exposure in the camera app (turn auto "
                      f"exposure off) or try more frames (--camera-frames; now {frames}).")


def _grab(path: Path, run, frames: int, name: str) -> tuple[float, int, int]:
    """One ffmpeg grab into `path`; returns the photo's mean brightness, width and height."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # so an old photo can never be mistaken for a new one
    try:
        result = run(ffmpeg_command(windows_path(path), frames, name), capture_output=True, text=True,
                     timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise CameraError(f"ffmpeg took more than {TIMEOUT_S} s: is {name} open on Windows and connected to the phone?")
    except FileNotFoundError:
        raise CameraError(f"ffmpeg not found at {config.FFMPEG_PATH}: set MIMIC_FFMPEG in .env.")
    if result.returncode != 0:
        last = (result.stderr or "").strip().splitlines()[-1:] or ["no message"]
        raise CameraError(f"ffmpeg failed (exit {result.returncode}): {last[0]}")
    if not path.exists():
        raise CameraError("ffmpeg finished but no photo was saved.")
    image = cv2.imread(str(path))
    if image is None:
        raise CameraError(f"The saved photo {path} cannot be read.")
    height, width = image.shape[:2]
    if max(width, height) < MIN_WIDTH or min(width, height) < MIN_HEIGHT:   # either way round
        raise CameraError(f"The photo from {name} is {width}x{height}, smaller than {MIN_WIDTH}x{MIN_HEIGHT}: "
                          "set the camera app to 1280x720 or more.")
    return float(image.mean()), width, height
