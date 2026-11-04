"""session.py: "s", Ctrl+C and errors send S then D; a normal end disables with a warning; enabling asks first."""
import dataclasses

import pytest

from mimic import arm, config, session
from mimic.arm import Arm, ArmError, FakeTransport
from mimic.session import StopRequested

LIMITS = arm.load_limits(config.ARM_LIMITS_PATH)
LOCKED = dataclasses.replace(LIMITS, placeholder=True, limits_checked_by_hand=False)


class ClosingFake(FakeTransport):
    def __init__(self, limits, enabled=True):
        super().__init__(limits, enabled=enabled)
        self.closed = False

    def close(self):
        self.closed = True


def fake_arm(enabled=True):
    t = ClosingFake(LIMITS, enabled=enabled)
    return Arm(t, LIMITS), t


def test_s_at_any_prompt_means_stop():
    with pytest.raises(StopRequested):
        session.ask("anything? ", read=lambda q: " S ")
    assert session.ask("anything? ", read=lambda q: " Y ") == "y"


@pytest.mark.parametrize("stop, code", [(StopRequested(), 130), (KeyboardInterrupt(), 130), (ValueError("x"), 1)])
def test_stop_ctrl_c_or_error_sends_S_then_D_and_closes(stop, code):
    a, t = fake_arm()
    said = []

    def body():
        raise stop

    assert session.guarded(a, body, said.append, sleep=lambda s: pytest.fail("waited")) == code
    assert t.sent[-2:] == ["S", "D"] and not t.enabled and t.closed
    assert "S and D sent, the servos are off" in "".join(said)


def test_an_error_is_printed_before_S_and_D_are_sent():
    a, t = fake_arm()
    events = []
    real_send = t.send
    t.send = lambda line: (events.append(("sent", line)), real_send(line))[1]

    def body():
        raise ValueError("SAM found 0 boxes")

    session.guarded(a, body, lambda text: events.append(("said", text)))
    first_said = events.index(("said", "\nSTOPPED: ValueError: SAM found 0 boxes"))
    assert first_said < events.index(("sent", "S")) < events.index(("sent", "D"))


def test_normal_end_says_hold_the_arm_waits_then_disables():
    a, t = fake_arm()
    said, waits = [], []
    assert session.guarded(a, lambda: None, said.append, sleep=waits.append) == 0
    assert said[0] == "Hold the arm: it goes limp in 2 s (disabling)." and waits == [2.0]
    assert t.sent == ["?", "D"] and not t.enabled and t.closed      # no S: nothing was moving


def test_normal_end_with_the_servos_already_off_sends_nothing_more():
    a, t = fake_arm(enabled=False)
    said = []
    assert session.guarded(a, lambda: None, said.append, sleep=lambda s: pytest.fail("waited")) == 0
    assert t.sent == ["?"] and said == [] and t.closed


def test_enable_asks_first_and_only_yes_switches_the_servos_on():
    a, t = fake_arm(enabled=False)
    said = []
    pose = session.confirm_and_enable(a, read=lambda q: "yes", say=said.append, sleep=lambda s: None)
    assert pose == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
    assert t.sent == ["?", "E", "?"] and t.enabled
    assert "one at a time, 0.4 s apart" in said[0] and "base 90, shoulder 90, elbow 90, grip 90" in said[0]

    b, u = fake_arm(enabled=False)
    with pytest.raises(session.Declined, match="nothing was moved"):
        session.confirm_and_enable(b, read=lambda q: "y", say=lambda s: None, sleep=lambda s: None)
    assert u.sent == ["?"] and not u.enabled
    with pytest.raises(StopRequested):
        session.confirm_and_enable(b, read=lambda q: "s", say=lambda s: None, sleep=lambda s: None)


def test_enable_with_angles_and_an_arm_that_is_already_on():
    a, t = fake_arm(enabled=False)
    pose = {"base": 100, "shoulder": 85, "elbow": 70, "grip": 90}
    assert session.confirm_and_enable(a, read=lambda q: "yes", say=lambda s: None, sleep=lambda s: None,
                                      angles=pose) == pose
    assert "E 100 85 70 90" in t.sent
    said = []
    assert session.confirm_and_enable(a, read=lambda q: pytest.fail("asked"), say=said.append) == pose
    assert said == ["The arm is already enabled: its servos are on."]


def test_a_declined_enable_ends_the_script_without_moving():
    a, t = fake_arm(enabled=False)
    said = []

    def body():
        session.confirm_and_enable(a, read=lambda q: "no", say=said.append, sleep=lambda s: None)
        pytest.fail("carried on")

    assert session.guarded(a, body, said.append) == 1
    assert said[-1] == "Not enabled: nothing was moved." and t.sent == ["?"] and t.closed


class FakeTerminal:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_typed_ahead_input_is_thrown_away_only_on_a_real_terminal(monkeypatch):
    flushed = []
    monkeypatch.setattr(session.termios, "tcflush", lambda stream, how: flushed.append((stream, how)))
    terminal = FakeTerminal(tty=True)
    session.discard_typed_ahead(terminal)
    assert flushed == [(terminal, session.termios.TCIFLUSH)]
    session.discard_typed_ahead(FakeTerminal(tty=False))   # a pipe or a test: nothing to throw away
    assert len(flushed) == 1


def test_ask_throws_away_typed_ahead_input_before_reading_the_answer():
    events = []
    answer = session.ask("smooth? ", read=lambda q: (events.append("read"), "n")[1],
                         flush=lambda: events.append("flush"))
    assert answer == "n" and events == ["flush", "read"]   # an early "n" is gone before the question is read


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
