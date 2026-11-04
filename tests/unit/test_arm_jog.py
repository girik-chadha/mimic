"""arm_jog.py tests: typed commands -> exact lines sent to a fake arm."""
import dataclasses
import importlib.util
import json

import pytest

from mimic import arm, config
from mimic.arm import Arm, ArmError, FakeTransport

spec = importlib.util.spec_from_file_location("arm_jog", config.ROOT / "scripts/arm_jog.py")
jog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jog)


@pytest.fixture
def fake():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    transport = FakeTransport(limits, enabled=True)
    return Arm(transport, limits), transport


def test_one_joint_at_a_time(fake):
    a, t = fake
    assert jog.handle(a, "base +5") == "base now 95"
    assert jog.handle(a, "elbow -3") == "elbow now 87"
    assert t.sent == ["?", "J 95 90 90 90 400", "J 95 90 87 90 400"]


def test_step_time_respects_the_speed_limit(fake):
    a, t = fake
    jog.handle(a, "shoulder +20")
    assert t.sent[-1] == "J 90 110 90 90 400"
    jog.handle(a, "elbow -40")
    assert t.sent[-1] == "J 90 110 50 90 400"  # 40 degrees * 10 ms = 400 ms


def test_home_is_a_slow_move_to_the_start_pose_not_H(fake, tmp_path, monkeypatch):
    path = tmp_path / "start_pose.json"
    path.write_text(json.dumps({"base": 90, "shoulder": 120, "elbow": 90, "grip": 90}))
    monkeypatch.setattr(arm, "START_POSE_PATH", path)
    a, t = fake
    assert jog.handle(a, "home") == "home: to the start pose (base 90, shoulder 120, elbow 90, grip 90) over 1000 ms"
    assert t.sent == ["?", "J 90 120 90 90 1000"] and "H" not in t.sent     # 30 degrees x 30 ms, at least 1 s


def test_grip_home_status_stop(fake):
    a, t = fake
    assert jog.handle(a, "home").startswith("home: to the start pose")
    assert jog.handle(a, "grip close").startswith("refused: grip_hold_deg is not known yet")  # never the limit
    assert jog.handle(a, "grip open") == "grip open"
    assert jog.handle(a, "status") == ("{'base': 90, 'shoulder': 150, 'elbow': 90, 'grip': 90, 'locked': False, "
                                       "'armed': True}")
    assert jog.handle(a, "stop") == "stopped"
    assert t.sent == ["?", "J 90 150 90 90 1800", "J 90 150 90 90 500", "?", "S"]   # 60 degrees x 30 ms home


def test_grip_close_goes_to_grip_hold_once_it_is_known():
    limits = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH), grip_hold_deg=45)
    t = FakeTransport(limits, enabled=True)
    a = Arm(t, limits)
    assert jog.handle(a, "grip close") == "grip close"
    assert t.sent[-1] == "J 90 90 90 45 500"                    # a lower angle closes this claw
    assert jog.handle(a, "grip -43").startswith("refused: grip 2 degrees is outside its range 3-140")


def test_a_grip_step_onto_the_closed_limit_warns(fake):
    a, t = fake
    reply = jog.handle(a, "grip -87")                           # 90 -> 3, the closed limit
    assert reply.startswith("grip now 3\nWARNING: the grip is at its closed limit (3, PROVISIONAL)")
    assert jog.handle(a, "grip +2") == "grip now 5"             # off the limit: no warning


def test_step_past_a_limit_is_refused_and_not_sent(fake):
    a, t = fake
    jog.handle(a, "home")                                       # the start pose: shoulder 150
    reply = jog.handle(a, "shoulder -75")
    assert reply == "refused: shoulder 75 degrees is outside its range 80-180."
    assert t.sent == ["?", "J 90 150 90 90 1800"]


@pytest.mark.parametrize("typed", ["", "wrist +5", "base 5", "base +five", "fly"])
def test_unknown_commands_show_help_and_send_nothing(fake, typed):
    a, t = fake
    assert jog.handle(a, typed) == jog.HELP
    assert t.sent == []


def test_real_mode_refuses_placeholder_limits_before_opening_the_bridge(monkeypatch):
    def must_not_open(*args, **kwargs):
        pytest.fail("the serial bridge was started while the limits are locked")
    monkeypatch.setattr(arm, "SerialBridgeTransport", must_not_open)
    limits = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH), placeholder=True)
    with pytest.raises(ArmError, match="placeholder numbers"):
        jog.make_arm(dry_run=False, limits=limits)


def test_jog_on_a_locked_firmware_refuses_steps():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    t = FakeTransport(limits, locked=True)
    reply = jog.handle(Arm(t, limits), "base +5")
    assert reply.startswith("refused: The arm's firmware is locked")
    assert t.sent == ["?"]


def test_enable_and_disable(monkeypatch):
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    t = FakeTransport(limits)                      # disarmed, like the firmware at power-on
    a = Arm(t, limits)
    monkeypatch.setattr(arm.time, "sleep", lambda s: None)
    assert jog.handle(a, "base +5").startswith("refused: The arm is not enabled")
    assert jog.handle(a, "enable") == "enabled, servos on at base 90, shoulder 150, elbow 90, grip 90"
    assert jog.handle(a, "grip -5") == "grip now 85"            # below the open stop: the range is 40-140
    assert jog.handle(a, "disable") == "disabled: the servos are off (limp)"
    assert jog.handle(a, "enable 100 85 70 90") == "enabled, servos on at base 100, shoulder 85, elbow 70, grip 90"
    assert jog.handle(a, "enable 100 85") == jog.HELP
    assert [line for line in t.sent if line[0] in "EDJ"] == ["E 90 150 90 90", "J 90 150 90 85 400", "D",
                                                             "E 100 85 70 90"]


def test_boot_says_why_the_board_last_started(fake):
    a, t = fake
    assert jog.handle(a, "boot") == "last start: power-on, 12 s ago (switched on, the reset button, or an upload)"
    t.boot_reason = "brownout"
    assert "brownout" in jog.handle(a, "boot") and "check the batteries" in jog.handle(a, "boot")
    assert t.sent == ["B", "B", "B"]


def test_boot_on_old_firmware_is_refused_politely(fake):
    a, t = fake
    t.send = lambda line: "ERR bad command"
    assert jog.handle(a, "boot").startswith("refused: The arm's firmware has no B command")


def test_dry_run_works_with_placeholder_limits():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    a = jog.make_arm(dry_run=True, limits=limits)
    assert not a.transport.is_real
