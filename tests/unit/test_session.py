"""session.py: "s" at any prompt and Ctrl+C stop the arm (S) and close the bridge."""
import dataclasses

import pytest

from mimic import arm, config, session
from mimic.arm import Arm, ArmError, FakeTransport
from mimic.session import StopRequested

LIMITS = arm.load_limits(config.ARM_LIMITS_PATH)
LOCKED = dataclasses.replace(LIMITS, placeholder=True, limits_checked_by_hand=False)


class ClosingFake(FakeTransport):
    def __init__(self, limits):
        super().__init__(limits)
        self.closed = False

    def close(self):
        self.closed = True


def fake_arm():
    t = ClosingFake(LIMITS)
    return Arm(t, LIMITS), t


def test_s_at_any_prompt_means_stop():
    with pytest.raises(StopRequested):
        session.ask("anything? ", read=lambda q: " S ")
    assert session.ask("anything? ", read=lambda q: " Y ") == "y"


@pytest.mark.parametrize("stop, code", [(StopRequested(), 130), (KeyboardInterrupt(), 130), (ValueError("x"), 1)])
def test_stop_ctrl_c_or_error_sends_S_and_closes(stop, code):
    a, t = fake_arm()
    said = []

    def body():
        raise stop

    assert session.guarded(a, body, said.append) == code
    assert t.sent[-1] == "S" and t.closed


def test_normal_end_closes_without_S():
    a, t = fake_arm()
    assert session.guarded(a, lambda: None, lambda s: None) == 0
    assert "S" not in t.sent and t.closed


def test_command_loop_quit_and_s():
    handled = []
    session.command_loop(handled.append, ["base +5\n", "quit\n", "never\n"], say=lambda s: None)
    assert handled == ["base +5\n"]
    with pytest.raises(StopRequested):
        session.command_loop(handled.append, ["s\n"], say=lambda s: None)


def test_connect_refuses_a_locked_real_arm_before_the_bridge(monkeypatch):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    with pytest.raises(ArmError, match="placeholder"):
        session.connect(False, LOCKED)
    assert not session.connect(True, LIMITS, echo=None).transport.is_real
