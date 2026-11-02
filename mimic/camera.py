"""Take a photo with the phone camera, from WSL (needs.md A9).

The phone runs iVCam, which shows up on Windows as a webcam called "e2eSoft iVCam". WSL cannot see
webcams, so Windows ffmpeg.exe grabs the picture: it reads 15 frames (giving the camera a moment to
settle its exposure) and keeps overwriting the same file, so the last frame is what is saved.

A photo is refused (CameraError) if ffmpeg fails or takes over 10 s, if no file appears, if the picture
is nearly black (lens covered, phone not connected), or if it is smaller than 1280x720.
"""
import subprocess
from pathlib import Path

import cv2

from mimic import config

TIMEOUT_S = 10
MIN_WIDTH, MIN_HEIGHT = 1280, 720
MIN_BRIGHTNESS = 20   # average grey level out of 255; a real photo of the white sheet is about 140


class CameraError(RuntimeError):
    pass


def windows_path(path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def ffmpeg_command(out_windows_path: str) -> list[str]:
    return [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow", "-i", f"video={config.CAMERA_DEVICE}",
            "-frames:v", "15", "-update", "1", "-q:v", "2", out_windows_path]


def capture(path, run=subprocess.run) -> str:
    """Save one photo to `path` (a WSL path) and return it. `run` is replaceable for tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # so an old photo can never be mistaken for a new one
    try:
        result = run(ffmpeg_command(windows_path(path)), capture_output=True, text=True, timeout=TIMEOUT_S)
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
    if image.mean() < MIN_BRIGHTNESS:
        raise CameraError(f"The photo is nearly black (average brightness {image.mean():.0f} of 255): "
                          "is the lens covered, or iVCam not connected?")
    return str(path)
