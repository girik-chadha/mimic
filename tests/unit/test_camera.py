"""camera.py with a fake ffmpeg: no camera, no Windows programs run."""
import subprocess

import cv2
import numpy as np
import pytest

from mimic import camera, config
from mimic.camera import CameraError


def fake_ffmpeg(writes=None, returncode=0, stderr="", raises=None):
    """Pretends to be ffmpeg: optionally writes `writes` (an image array, or a list of them, one per grab) to the
    photo path."""
    calls = []
    frames = list(writes) if isinstance(writes, list) else None

    def run(command, capture_output, text, timeout):
        calls.append((command, timeout))
        if raises:
            raise raises
        image = frames.pop(0) if frames is not None else writes
        if image is not None:
            cv2.imwrite(run.target, image)
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    run.calls = calls
    return run


def grab(tmp_path, run, said=None, waits=None, frames=camera.FRAMES):
    path = tmp_path / "photo.jpg"
    run.target = str(path)
    return camera.capture(path, run=run, frames=frames, say=(said if said is not None else []).append,
                          sleep=(waits if waits is not None else []).append)


def sheet(width=1280, height=720, grey=170):
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
    with pytest.raises(CameraError, match="more than 10 s: is e2eSoft iVCam open"):
        grab(tmp_path, run)


def test_missing_ffmpeg_is_a_camera_error(tmp_path):
    run = fake_ffmpeg(raises=FileNotFoundError())
    with pytest.raises(CameraError, match="ffmpeg not found .* MIMIC_FFMPEG"):
        grab(tmp_path, run)


def test_no_file_is_a_camera_error_and_an_old_photo_never_counts(tmp_path):
    cv2.imwrite(str(tmp_path / "photo.jpg"), sheet())  # left over from before
    with pytest.raises(CameraError, match="no photo was saved"):
        grab(tmp_path, fake_ffmpeg(writes=None))


def test_a_normal_photo_is_kept_first_time_and_its_brightness_printed(tmp_path):
    said, waits = [], []
    run = fake_ffmpeg(writes=sheet(grey=170))
    grab(tmp_path, run, said, waits)
    assert said == ["  photo photo.jpg: e2eSoft iVCam, 1280x720, brightness 170.0"] and waits == []
    assert len(run.calls) == 1


def test_a_blown_out_photo_is_taken_again_after_1_s(tmp_path):
    said, waits = [], []
    run = fake_ffmpeg(writes=[sheet(grey=248), sheet(grey=170)])      # the first grab blown out, like iVCam's
    assert grab(tmp_path, run, said, waits) == str(tmp_path / "photo.jpg")
    assert waits == [1.0] and len(run.calls) == 2
    assert said[0] == "  photo photo.jpg: e2eSoft iVCam, 1280x720, brightness 248.0"
    assert said[-1] == "  photo photo.jpg: e2eSoft iVCam, 1280x720, brightness 170.0"


def test_too_bright_twice_is_refused_with_the_brightness_and_the_exposure_hint(tmp_path):
    run = fake_ffmpeg(writes=[sheet(grey=248), sheet(grey=248)])
    with pytest.raises(CameraError, match=r"from e2eSoft iVCam is too bright \(blown out\): mean brightness 248.0, "
                                          r"allowed 40-235, twice\. Lock the exposure in the camera app"):
        grab(tmp_path, run)
    assert len(run.calls) == 2


def test_too_dark_twice_is_refused_with_the_brightness_and_both_hints(tmp_path):
    run = fake_ffmpeg(writes=[sheet(grey=20), sheet(grey=25)])
    with pytest.raises(CameraError, match=r"from e2eSoft iVCam is too dark: mean brightness 25.0, allowed 40-235, "
                                          r"twice\. Is the lens covered, or e2eSoft iVCam not connected\? Lock the"):
        grab(tmp_path, run)


def grab_from(tmp_path, run, name, said):
    path = tmp_path / "photo.jpg"
    run.target = str(path)
    return camera.capture(path, run=run, say=said.append, sleep=lambda s: None, name=name)


def test_camo_s_black_first_frame_is_retaken_and_the_camera_named_in_every_line(tmp_path):
    said = []
    run = fake_ffmpeg(writes=[sheet(1920, 1080, grey=0), sheet(1920, 1080, grey=153)])   # Camo's 2-frame black
    grab_from(tmp_path, run, "Camo", said)
    assert said[0] == "  photo photo.jpg: Camo, 1920x1080, brightness 0.0"
    assert said[-1] == "  photo photo.jpg: Camo, 1920x1080, brightness 153.0"
    assert run.calls[0][0][run.calls[0][0].index("-i") + 1] == "video=Camo"


def test_a_refused_photo_names_the_camera(tmp_path):
    run = fake_ffmpeg(writes=[sheet(grey=0), sheet(grey=0)])
    with pytest.raises(CameraError, match="from Camo is too dark: mean brightness 0.0, .* or Camo not connected"):
        grab_from(tmp_path, run, "Camo", [])


def test_the_camera_name_comes_from_the_option_then_mimic_camera_then_ring_json_then_ivcam(monkeypatch):
    assert camera.camera_name() == "e2eSoft iVCam"
    assert camera.camera_name(None, "Camo") == "Camo"                       # ring.json
    monkeypatch.setenv("MIMIC_CAMERA", "Camo")
    assert camera.camera_name() == "Camo" and camera.camera_name(None, "e2eSoft iVCam") == "Camo"   # env over ring
    assert camera.camera_name("Other Cam", "e2eSoft iVCam") == "Other Cam"  # the option over everything
    command = camera.ffmpeg_command("C:\\x.jpg")
    assert command[command.index("-i") + 1] == "video=Camo"


def test_a_portrait_frame_is_fine_and_a_small_one_is_refused_without_naming_ivcam(tmp_path):
    grab(tmp_path, fake_ffmpeg(writes=sheet(720, 1280)))                    # 720x1280: 1280x720 the other way round
    with pytest.raises(CameraError, match="from Camo is 640x480, smaller than 1280x720: set the camera app"):
        grab_from(tmp_path, fake_ffmpeg(writes=sheet(640, 480)), "Camo", [])


def test_the_edges_of_the_window_are_fine(tmp_path):
    for grey in (40, 235):
        run = fake_ffmpeg(writes=sheet(grey=grey))
        grab(tmp_path, run)
        assert len(run.calls) == 1


def test_the_frame_count_goes_to_ffmpeg_and_the_default_is_15(tmp_path):
    assert camera.ffmpeg_command("C:\\x.jpg")[camera.ffmpeg_command("C:\\x.jpg").index("-frames:v") + 1] == "15"
    run = fake_ffmpeg(writes=sheet())
    grab(tmp_path, run, frames=30)
    command = run.calls[0][0]
    assert command[command.index("-frames:v") + 1] == "30" and "-update" in command


def test_small_photo_is_a_camera_error(tmp_path):
    with pytest.raises(CameraError, match="640x480, smaller than 1280x720"):
        grab(tmp_path, fake_ffmpeg(writes=sheet(640, 480)))
