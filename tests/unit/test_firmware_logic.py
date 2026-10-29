"""Compile the firmware's safety logic (arm_logic.h) with g++ and run its C++ tests. No board needed."""
import shutil
import subprocess

import pytest

from mimic import config

GXX = shutil.which("g++")
FIRMWARE_DIR = config.ROOT / "firmware" / "mimic_arm"
TEST_CPP = config.ROOT / "tests" / "unit" / "firmware_logic_test.cpp"

pytestmark = pytest.mark.skipif(GXX is None, reason="g++ is not installed")


def build_and_run(tmp_path, *defines):
    exe = tmp_path / "firmware_logic_test"
    built = subprocess.run([GXX, "-std=c++17", "-Wall", "-Wextra", "-Werror", *defines, "-I", str(FIRMWARE_DIR),
                            str(TEST_CPP), "-o", str(exe)], capture_output=True, text=True)
    if built.returncode != 0:
        pytest.fail("arm_logic.h does not compile:\n" + built.stderr)
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    print(result.stdout)
    return result


def test_placeholder_limits_refuse_every_move(tmp_path):
    result = build_and_run(tmp_path)  # arm_logic.h exactly as committed
    assert "placeholder=1" in result.stdout
    assert result.returncode == 0, result.stdout


def test_measured_but_not_checked_by_hand_still_refuses(tmp_path):
    result = build_and_run(tmp_path, "-DLIMITS_ARE_PLACEHOLDER=0")
    assert "ok   J refused" in result.stdout
    assert result.returncode == 0, result.stdout


def test_moves_clamps_ramps_and_grip_timer(tmp_path):
    result = build_and_run(tmp_path, "-DLIMITS_ARE_PLACEHOLDER=0", "-DLIMITS_CHECKED_BY_HAND=1")
    assert "ok   grip never closes past its limit" in result.stdout
    assert result.returncode == 0, result.stdout
