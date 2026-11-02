"""The Windows serial bridge, tested in WSL without hardware: its fake locked arm stands in for COM7."""
import importlib.util
import io
import sys
import time

import pytest

from mimic import config
from mimic.arm import ArmError, SerialBridgeTransport

SCRIPT = config.ROOT / "scripts" / "serial_bridge.py"
spec = importlib.util.spec_from_file_location("serial_bridge", SCRIPT)
bridge_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_script)

FOUR_LINES = ["?", "S", "J 90 90 90 90 800", "H"]
LOCKED_REPLIES = ["A 90 90 90 90 L", "OK", "ERR limits not measured", "ERR limits not measured"]


def fake_bridge(tmp_path, mode="locked", reply_timeout=0.3):
    """The real bridge script, run by WSL's Python, talking to its fake arm instead of COM7."""
    log = tmp_path / "sent_to_arm.txt"
    command = [sys.executable, "-u", str(SCRIPT), "--fake", mode, "--reply-timeout", str(reply_timeout),
               "--fake-log", str(log)]
    return command, log


# ---------- small pieces ----------

def test_reply_lines_are_recognised():
    assert all(bridge_script.is_reply(r) for r in LOCKED_REPLIES)
    assert not bridge_script.is_reply("rst:0x1 (POWERON_RESET),boot:0x13")
    assert not bridge_script.is_reply("E (512) some log message")


PORTS = [("COM5", None, None), ("COM6", None, None), ("COM7", 0x10C4, 0xEA60)]  # what this laptop shows


def test_auto_finds_the_cp210x_and_skips_bluetooth_ports():
    assert bridge_script.choose_port("auto", PORTS) == "COM7"
    assert bridge_script.choose_port("COM7", PORTS) == "COM7"


@pytest.mark.parametrize("wanted, ports, problem", [
    ("COM5", PORTS, "COM5 is not the arm's CP210x port"),
    ("auto", PORTS[:2], "found 0 CP210x ports"),
    ("auto", PORTS + [("COM9", 0x10C4, 0xEA60)], "found 2 CP210x ports"),
])
def test_wrong_ports_are_refused(wanted, ports, problem):
    with pytest.raises(RuntimeError, match=problem):
        bridge_script.choose_port(wanted, ports)


# ---------- the bridge loop, in-process ----------

def run_bridge(port, lines, reply_timeout=0.3):
    out, err = io.StringIO(), io.StringIO()
    bridge_script.bridge(port, [line + "\n" for line in lines], out, err, reply_timeout)
    return out.getvalue().splitlines(), err.getvalue()


def test_bridge_answers_each_line_and_sends_S_at_the_end(tmp_path):
    log = tmp_path / "log.txt"
    out, err = run_bridge(bridge_script.FakePort("locked", log), FOUR_LINES)
    assert out == ["READY"] + LOCKED_REPLIES
    assert "ignored: rst:0x1" in err and "ignored: E (512)" in err  # boot text and log lines are skipped
    assert log.read_text().splitlines() == FOUR_LINES + ["S"]


def test_silent_arm_gives_TIMEOUT_quickly_and_still_gets_S(tmp_path):
    log = tmp_path / "log.txt"
    start = time.monotonic()
    out, _ = run_bridge(bridge_script.FakePort("silent", log), ["?"], reply_timeout=0.2)
    assert out == ["READY", "TIMEOUT"]
    assert time.monotonic() - start < 1.5
    assert log.read_text().splitlines() == ["?", "S"]


def test_half_lines_are_joined():
    class SlowPort:
        chunks = [b"A 90 ", b"90 90 90 L\n"]

        def write(self, data):
            pass

        def readline(self):
            return self.chunks.pop(0) if self.chunks else b""

    out, _ = run_bridge(SlowPort(), ["?"])
    assert out[1] == "A 90 90 90 90 L"


def test_S_is_sent_even_if_the_bridge_fails(tmp_path):
    log = tmp_path / "log.txt"

    def lines():
        yield "?\n"
        raise OSError("pipe broke")

    with pytest.raises(OSError):
        bridge_script.bridge(bridge_script.FakePort("locked", log), lines(), io.StringIO(), io.StringIO(), 0.3)
    assert log.read_text().splitlines() == ["?", "S"]


# ---------- end to end: SerialBridgeTransport <-> the real script over pipes ----------

def test_transport_talks_to_the_bridge_and_closing_sends_S(tmp_path):
    command, log = fake_bridge(tmp_path)
    t = SerialBridgeTransport(command)
    assert [t.send(line) for line in FOUR_LINES] == LOCKED_REPLIES
    t.close()
    assert t.process.returncode == 0
    assert log.read_text().splitlines() == FOUR_LINES + ["S"]


def test_missing_reply_raises_instead_of_hanging(tmp_path):
    command, _ = fake_bridge(tmp_path, mode="silent", reply_timeout=0.3)
    t = SerialBridgeTransport(command, reply_timeout_s=2.0)
    start = time.monotonic()
    with pytest.raises(ArmError, match="did not reply to '\\?' \\(the serial bridge timed out\\)"):
        t.send("?")
    assert time.monotonic() - start < 1.5
    with pytest.raises(ArmError, match="not usable any more"):
        t.send("S")
    t.close()


def test_stuck_bridge_raises_instead_of_hanging():
    # a bridge that says READY and then never answers anything (it exits when its input closes)
    stuck = [sys.executable, "-c", "import sys; print('READY', flush=True); [None for _ in sys.stdin]"]
    t = SerialBridgeTransport(stuck, reply_timeout_s=0.5)
    start = time.monotonic()
    with pytest.raises(ArmError, match="No answer from the serial bridge within 0.5 s .* it may be stuck"):
        t.send("?")
    assert time.monotonic() - start < 2
    t.close()


def test_port_that_cannot_open_raises_with_the_bridge_message(tmp_path):
    command, _ = fake_bridge(tmp_path, mode="broken")
    with pytest.raises(ArmError, match="could not start: could not open port \\(fake\\)"):
        SerialBridgeTransport(command)


def test_bridge_that_exits_raises():
    gone = [sys.executable, "-c", "print('READY', flush=True)"]
    t = SerialBridgeTransport(gone)
    with pytest.raises(ArmError, match="stopped|not usable"):
        t.send("?")
    t.close()


def test_ctrl_c_makes_the_bridge_send_S_and_exit(tmp_path):
    import signal
    import subprocess
    command, log = fake_bridge(tmp_path)
    bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert bridge.stdout.readline().strip() == "READY"
    bridge.stdin.write("?\n")
    bridge.stdin.flush()
    assert bridge.stdout.readline().strip() == "A 90 90 90 90 L"
    bridge.send_signal(signal.SIGINT)  # what Ctrl+C does
    assert bridge.wait(timeout=5) == 130
    assert log.read_text().splitlines() == ["?", "S"]
