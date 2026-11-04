"""Take a photo with the phone camera, from WSL (needs.md A9).

The phone runs iVCam, which shows up on Windows as a webcam called "e2eSoft iVCam". WSL cannot see
webcams, so Windows ffmpeg.exe grabs the picture: it reads `frames` frames (15 by default) and keeps
overwriting the same file, so the last frame is what is saved. The first frames are blown out on this camera
(a 2-frame grab came out at brightness 247.5, a 15-frame one at 170.5: needs.md A9).

After every photo the mean brightness (0-255) is printed. Outside 40-235 it is taken once more after 1 s; if
it is still outside, it is refused (CameraError) with the brightness and a hint to lock iVCam's exposure.
A photo is also refused if ffmpeg fails or takes over 10 s, if no file appears, or if it is smaller than 1280x720.
"""
import subprocess
import time
from pathlib import Path

import cv2

from mimic import config

TIMEOUT_S = 10
MIN_WIDTH, MIN_HEIGHT = 1280, 720
FRAMES = 15                       # read this many frames and keep the last: the first ones are blown out
TOO_DARK, TOO_BRIGHT = 40, 235    # mean grey level out of 255; a good photo of the sheet is about 170
RETAKE_WAIT_S = 1.0


class CameraError(RuntimeError):
    pass


def windows_path(path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def ffmpeg_command(out_windows_path: str, frames: int = FRAMES) -> list[str]:
    return [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow", "-i", f"video={config.CAMERA_DEVICE}",
            "-frames:v", str(frames), "-update", "1", "-q:v", "2", out_windows_path]


def capture(path, run=subprocess.run, frames: int = FRAMES, sleep=time.sleep, say=print) -> str:
    """Save one photo to `path` (a WSL path) and return it, retaking once if its brightness is outside 40-235.
    `run` is replaceable for tests."""
    for attempt in (1, 2):
        level = _grab(Path(path), run, frames)
        say(f"  photo {Path(path).name}: brightness {level:.1f}")
        if TOO_DARK <= level <= TOO_BRIGHT:
            return str(path)
        if attempt == 1:
            say(f"  brightness outside {TOO_DARK}-{TOO_BRIGHT}: taking it again in {RETAKE_WAIT_S:g} s")
            sleep(RETAKE_WAIT_S)
    what = "too bright (blown out)" if level > TOO_BRIGHT else "too dark"
    hint = "" if level > TOO_BRIGHT else " Is the lens covered, or iVCam not connected?"
    raise CameraError(f"The photo {Path(path).name} is {what}: mean brightness {level:.1f}, allowed "
                      f"{TOO_DARK}-{TOO_BRIGHT}, twice.{hint} Lock the exposure in iVCam (turn auto exposure off) "
                      f"or try more frames (--camera-frames; now {frames}).")


def _grab(path: Path, run, frames: int) -> float:
    """One ffmpeg grab into `path`; returns the photo's mean brightness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # so an old photo can never be mistaken for a new one
    try:
        result = run(ffmpeg_command(windows_path(path), frames), capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise CameraError(f"ffmpeg took more than {TIMEOUT_S} s: is iVCam open on Windows and connected to the phone?")
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
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise CameraError(f"The photo is {width}x{height}, smaller than {MIN_WIDTH}x{MIN_HEIGHT}: "
                          "set iVCam to 1280x720 or more.")
    return float(image.mean())
