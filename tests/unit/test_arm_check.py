"""arm_check.py: the read-only first test with the real arm, run here against fakes."""
import importlib.util
import sys

import pytest

from mimic import arm, config
from mimic.arm import ArmError, FakeTransport, SerialBridgeTransport

spec = importlib.util.spec_from_file_location("arm_check", config.ROOT / "scripts/arm_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

LIMITS = arm.load_limits(config.ARM_LIMITS_PATH)


def run(transport=None, answer="yes", argv=()):
    said, opened = [], []

    def make_transport():
        opened.append(True)
        return transport

    def ask(question):
        assert question == "Batteries disconnected and switch OFF? Type yes: "
        return answer

    code = check.run(list(argv), ask, make_transport, said.append)
    return code, "\n".join(said), opened


def test_locked_firmware_passes_with_exactly_four_lines():
    t = FakeTransport(LIMITS, locked=True)
    code, said, _ = run(t)
    assert code == 0
    assert t.sent == ["?", "S", "J 90 90 90 90 800", "H"]
    assert "PASS" in said


def test_nothing_is_opened_without_yes():
    code, said, opened = run(FakeTransport(LIMITS, locked=True), answer="no")
    assert code == 1 and opened == []
    assert "the port was not opened" in said


def test_takes_no_arguments():
    code, said, opened = run(FakeTransport(LIMITS, locked=True), argv=["--port", "COM7"])
    assert code == 1 and opened == []


def test_status_without_L_means_not_locked():
    t = FakeTransport(LIMITS, locked=False)
    code, said, _ = run(t)
    assert code == 2
    assert t.sent == ["?"]  # stopped before any J or H
    assert "NOT locked" in said


def test_accepted_J_is_stopped_at_once():
    class LockBrokenForJ(FakeTransport):
        def send(self, line):
            reply = super().send(line)
            return "OK" if line.startswith("J") else reply

    t = LockBrokenForJ(LIMITS, locked=True)
    code, said, _ = run(t)
    assert code == 2
    assert t.sent == ["?", "S", "J 90 90 90 90 800", "S"]  # S straight after the accepted J, H never sent
    assert "FIRMWARE IS NOT LOCKED" in said


def test_first_question_timing_out_suggests_pressing_EN():
    class Silent(FakeTransport):
        def send(self, line):
            raise ArmError("The arm did not reply to '?' (the serial bridge timed out).")

    code, said, _ = run(Silent(LIMITS))
    assert code == 1
    assert "Press the EN button on the ESP32 once, wait 2 seconds" in said


def test_end_to_end_through_the_real_bridge_script(tmp_path):
    command = [sys.executable, "-u", str(config.ROOT / "scripts/serial_bridge.py"), "--fake", "locked",
               "--reply-timeout", "0.3"]
    code, said, _ = run(SerialBridgeTransport(command))
    assert code == 0, said
