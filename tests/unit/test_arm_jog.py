"""arm_jog.py tests: typed commands -> exact lines sent to a fake arm."""
import importlib.util

import pytest

from mimic import arm, config
from mimic.arm import Arm, ArmError, FakeTransport

spec = importlib.util.spec_from_file_location("arm_jog", config.ROOT / "scripts/arm_jog.py")
jog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jog)


@pytest.fixture
def fake():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    transport = FakeTransport(limits)
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
    jog.handle(a, "shoulder -40")
    assert t.sent[-1] == "J 90 70 90 90 400"  # 40 degrees * 10 ms = 400 ms


def test_grip_home_status_stop(fake):
    a, t = fake
    assert jog.handle(a, "home") == "home"
    assert jog.handle(a, "grip close") == "grip close"
    assert jog.handle(a, "grip open") == "grip open"
    assert jog.handle(a, "status") == "{'base': 90, 'shoulder': 90, 'elbow': 90, 'grip': 90}"
    assert jog.handle(a, "stop") == "stopped"
    assert t.sent == ["H", "J 90 90 90 120 500", "J 90 90 90 90 500", "?", "S"]


def test_step_past_a_limit_is_refused_and_not_sent(fake):
    a, t = fake
    jog.handle(a, "home")
    reply = jog.handle(a, "base +25")
    assert reply == "refused: base 115 degrees is outside its range 70-110."
    assert t.sent == ["H"]


@pytest.mark.parametrize("typed", ["", "wrist +5", "base 5", "base +five", "fly"])
def test_unknown_commands_show_help_and_send_nothing(fake, typed):
    a, t = fake
    assert jog.handle(a, typed) == jog.HELP
    assert t.sent == []


def test_real_mode_refuses_placeholder_limits():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    with pytest.raises(ArmError, match="placeholder numbers"):
        jog.make_arm(dry_run=False, limits=limits)


def test_dry_run_works_with_placeholder_limits():
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    a = jog.make_arm(dry_run=True, limits=limits)
    assert not a.transport.is_real
