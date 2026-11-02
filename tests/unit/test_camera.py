"""camera.py with a fake ffmpeg: no camera, no Windows programs run."""
import subprocess

import cv2
import numpy as np
import pytest

from mimic import camera, config
from mimic.camera import CameraError


def fake_ffmpeg(writes=None, returncode=0, stderr="", raises=None):
    """Pretends to be ffmpeg: optionally writes `writes` (an image array) to the photo path."""
    calls = []

    def run(command, capture_output, text, timeout):
        calls.append((command, timeout))
        if raises:
            raise raises
        if writes is not None:
            cv2.imwrite(run.target, writes)
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    run.calls = calls
    return run


def grab(tmp_path, run):
    path = tmp_path / "photo.jpg"
    run.target = str(path)
    return camera.capture(path, run=run)


def sheet(width=1280, height=720, grey=140):
    return np.full((height, width, 3), grey, np.uint8)


def test_good_photo_is_returned(tmp_path):
    run = fake_ffmpeg(writes=sheet())
    assert grab(tmp_path, run) == str(tmp_path / "photo.jpg")


def test_ffmpeg_is_asked_for_15_frames_from_ivcam_into_a_windows_path(tmp_path):
    run = fake_ffmpeg(writes=sheet())
    grab(tmp_path, run)
    command, timeout = run.calls[0]
    assert command[0] == config.FFMPEG_PATH
    assert command[command.index("-f") + 1] == "dshow"
    assert command[command.index("-i") + 1] == "video=e2eSoft iVCam"
    assert command[command.index("-frames:v") + 1] == "15"
    assert command[command.index("-update") + 1] == "1"
    assert command[command.index("-q:v") + 1] == "2"
    assert command[-1].startswith("\\\\wsl") and command[-1].endswith("photo.jpg")
    assert timeout == 10


def test_ffmpeg_failure_is_a_camera_error(tmp_path):
    run = fake_ffmpeg(returncode=1, stderr="Could not find video device with name [e2eSoft iVCam]\n")
    with pytest.raises(CameraError, match="exit 1.*Could not find video device"):
        grab(tmp_path, run)


def test_timeout_is_a_camera_error(tmp_path):
    run = fake_ffmpeg(raises=subprocess.TimeoutExpired("ffmpeg", 10))
    with pytest.raises(CameraError, match="more than 10 s: is iVCam open"):
        grab(tmp_path, run)


def test_missing_ffmpeg_is_a_camera_error(tmp_path):
    run = fake_ffmpeg(raises=FileNotFoundError())
    with pytest.raises(CameraError, match="ffmpeg not found .* MIMIC_FFMPEG"):
        grab(tmp_path, run)


def test_no_file_is_a_camera_error_and_an_old_photo_never_counts(tmp_path):
    cv2.imwrite(str(tmp_path / "photo.jpg"), sheet())  # left over from before
    with pytest.raises(CameraError, match="no photo was saved"):
        grab(tmp_path, fake_ffmpeg(writes=None))


def test_nearly_black_photo_is_a_camera_error(tmp_path):
    with pytest.raises(CameraError, match="nearly black"):
        grab(tmp_path, fake_ffmpeg(writes=sheet(grey=5)))


def test_small_photo_is_a_camera_error(tmp_path):
    with pytest.raises(CameraError, match="640x480, smaller than 1280x720"):
        grab(tmp_path, fake_ffmpeg(writes=sheet(640, 480)))
