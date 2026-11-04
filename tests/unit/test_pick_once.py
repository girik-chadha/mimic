"""pick_once.py with the fake arm: the planned moves, the 4.5 s grip rule, refusals, the enable question, stops."""
import dataclasses
import importlib.util

import pytest

from mimic import arm, config, session
from mimic.arm import FakeTransport

spec = importlib.util.spec_from_file_location("pick_once", config.ROOT / "scripts/pick_once.py")
po = importlib.util.module_from_spec(spec)
spec.loader.exec_module(po)

REAL_LOAD_LIMITS = arm.load_limits
# Pinned here so these tests do not change when calib/arm_limits.json does: the 4 Nov numbers.
LIMITS = dataclasses.replace(REAL_LOAD_LIMITS(config.ARM_LIMITS_PATH),
                             ranges_deg={"base": (0, 180), "shoulder": (80, 180), "elbow": (30, 120)},
                             grip_open_deg=90, grip_closed_limit_deg=3, grip_range_deg=(0, 140),
                             min_ms_per_degree=10, max_grip_hold_s=5, placeholder=False, limits_checked_by_hand=True)
START = {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
DEFAULTS = dict(shoulder_pick=173, grip_hold=23, lift=20, approach=10, stop_at=175)


def run(monkeypatch, argv, typed="yes", sleep=lambda s: None):
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: LIMITS)
    transports, prompts, said = [], [], []
    real_connect = session.connect

    def connect(dry_run, limits, echo=print):
        a = real_connect(dry_run, limits, echo=None)
        transports.append(a.transport)
        return a

    monkeypatch.setattr(session, "connect", connect)

    def read(question):
        prompts.append(question)
        return typed

    code = po.main(list(argv) + ["--dry-run"], read=read, sleep=sleep, say=said.append)
    return code, (transports[0].sent if transports else []), prompts, said


def j_lines(sent):
    return [[int(v) for v in line.split()[1:6]] for line in sent if line.startswith("J")]


# ---------- the moves ----------

def test_the_default_pick_and_drop_sends_exactly_the_planned_moves_and_never_H(monkeypatch):
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"])
    assert code == 0
    moves = j_lines(sent)
    assert [m[:4] for m in moves[:7]] == [[100, 150, 90, 90], [100, 163, 90, 90], [100, 165, 90, 90],
                                          [100, 167, 90, 90], [100, 169, 90, 90], [100, 171, 90, 90],
                                          [100, 173, 90, 90]]                      # open, above, down in 2s
    assert moves[7][:4] == [100, 173, 90, 23] and moves[8][:4] == [100, 153, 90, 23]   # close, lift 20
    assert moves[9][:4] == [120, 153, 90, 23]                                          # swing to the drop
    assert [m[1] for m in moves[10:20]] == list(range(155, 174, 2))                    # down in 2-degree steps
    assert moves[20][:4] == [120, 173, 90, 90] and moves[21][:4] == [120, 153, 90, 90]  # open, lift away
    assert moves[22] == [90, 150, 90, 90, 1000]                  # then a slow ramp back to the start pose (at least 1 s)
    assert len(moves) == 23
    assert all(m[2] == 90 for m in moves)                       # the dead elbow: always 90 (an empty socket)
    assert max(m[1] for m in moves) == 173                      # never past the pick angle, so never past 175
    assert "H" not in sent and sent[-1] == "D"                   # no H; disabled at the end
    assert any(line.startswith("Hold the arm") for line in said)


def test_the_shoulder_moves_at_25_ms_per_degree_and_the_base_and_claw_at_the_limits_speed(monkeypatch):
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"])
    moves = j_lines(sent)
    assert moves[1][4] == 13 * 25            # shoulder 150 (the start pose) -> 163
    assert moves[2][4] == 2 * 25             # each 2-degree lowering step
    assert moves[7][4] == 67 * 10            # close 90 -> 23
    assert moves[9][4] == 20 * 10            # swing 100 -> 120


def test_the_claw_closed_time_runs_from_closing_until_the_open_command():
    steps, hold_s, _ = po.plan(LIMITS, START, base_pick=100, base_drop=120, **DEFAULTS)
    # close 0.67 s, lift 0.5, swing 0.2, 10 steps of 0.05: each + 0.1 s wait + 0.05 s command; + the open command
    assert hold_s == pytest.approx((0.67 + 0.15) + (0.5 + 0.15) + (0.2 + 0.15) + 10 * (0.05 + 0.15) + 0.05)
    assert hold_s == pytest.approx(3.87)
    _, without_bridge, _ = po.plan(LIMITS, START, base_pick=100, base_drop=120, command_s=0.0, **DEFAULTS)
    assert without_bridge == pytest.approx(3.87 - 14 * 0.05)    # 13 holding moves + the open command


def test_the_time_is_printed_before_the_enable_question(monkeypatch):
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"], typed="no")
    assert said[0].startswith("Planned: 22 moves. Claw closed for 3.87 s") and "limit 4.5 s" in said[0]


# ---------- the 4.5 s rule ----------

def test_too_long_a_hold_is_refused_before_anything_is_sent(monkeypatch):
    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("arm connected"))
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: LIMITS)
    said = []
    code = po.main(["--base-pick", "0", "--base-drop", "180", "--dry-run"], read=lambda q: pytest.fail("asked"),
                   say=said.append)
    assert code == 1
    assert said[0] == ("STOPPED: The claw would be closed for 5.47 s, over the 4.5 s allowed (max_grip_hold_s 5 s "
                       "minus 0.5 s): a 180-degree swing is too far. Nothing was moved.")


def test_an_80_degree_swing_fits_and_a_90_degree_one_does_not():
    po.plan(LIMITS, START, base_pick=40, base_drop=120, **DEFAULTS)
    with pytest.raises(po.PickError, match="a 90-degree swing is too far"):
        po.plan(LIMITS, START, base_pick=30, base_drop=120, **DEFAULTS)


def test_a_slower_bridge_counts_against_the_limit():
    po.plan(LIMITS, START, base_pick=100, base_drop=120, command_s=0.05, **DEFAULTS)
    with pytest.raises(po.PickError, match="over the 4.5 s allowed"):
        po.plan(LIMITS, START, base_pick=100, base_drop=120, command_s=0.1, **DEFAULTS)   # 14 x 0.05 s more


# ---------- refused before anything moves ----------

@pytest.mark.parametrize("argv, problem", [
    (["--base-pick", "100", "--base-drop", "120", "--shoulder-pick", "176"], "--shoulder-pick 176 is past --stop-at 175"),
    (["--base-pick", "100", "--base-drop", "120", "--shoulder-pick", "181", "--stop-at", "182"],
     "shoulder 181 is outside its range 80-180"),
    (["--base-pick", "100", "--base-drop", "120", "--lift", "100"], "shoulder 73 is outside its range 80-180"),
    (["--base-pick", "100", "--base-drop", "120", "--grip-hold", "3"], "--grip-hold 3 must close the claw"),
    (["--base-pick", "100", "--base-drop", "120", "--grip-hold", "2"], "--grip-hold 2 must close the claw"),
    (["--base-pick", "100", "--base-drop", "120", "--grip-hold", "95"], "--grip-hold 95 must close the claw"),
    (["--base-pick", "100", "--base-drop", "120", "--approach", "0"], "--lift and --approach must be above 0"),
    (["--base-pick", "190", "--base-drop", "120"], "base 190 is outside its range 0-180"),
    (["--base-pick", "100"], "--base-pick and --base-drop are required"),
])
def test_bad_options_are_refused_before_anything_is_sent(monkeypatch, argv, problem):
    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("arm connected"))
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: LIMITS)
    said = []
    assert po.main(argv + ["--dry-run"], read=lambda q: pytest.fail("asked"), say=said.append) == 1
    assert problem in said[0]


def test_a_base_outside_the_approved_range_warns_but_still_runs(monkeypatch):
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "70", "--base-drop", "100"])
    assert code == 0
    assert said[0] == ("WARNING: base 70 is outside the range you approved (80-120, needs.md B4); "
                       "calib/arm_limits.json still allows 0-180.")
    assert not any("base 100 is outside" in line for line in said)


# ---------- enabling, stopping ----------

def test_the_enable_question_says_how_to_pose_the_arm_and_a_no_moves_nothing(monkeypatch):
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"], typed="no")
    assert code == 1 and sent == ["?"]
    assert "pose the arm like the start pose: upper arm vertical, forearm horizontal, claw open" in prompts[0]
    assert "Not enabled: nothing was moved." in said


def test_enable_and_the_first_move_start_from_the_start_pose(monkeypatch, tmp_path):
    path = tmp_path / "start_pose.json"
    path.write_text('{"base": 90, "shoulder": 120, "elbow": 90, "grip": 90}')
    monkeypatch.setattr(arm, "START_POSE_PATH", path)
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"])
    assert code == 0 and "E 90 120 90 90" in sent
    moves = j_lines(sent)
    assert moves[0] == [100, 120, 90, 90, 10 * 10]              # the base turns; the shoulder stays at 120
    assert moves[1] == [100, 163, 90, 90, 43 * 25]              # 120 -> 163, at 25 ms per degree
    assert moves[-1][:4] == [90, 120, 90, 90]                    # and it ends back at the start pose


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), session.StopRequested()])
def test_a_stop_partway_sends_S_then_D(monkeypatch, stop):
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 5:     # during the lowering steps
            raise stop

    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--base-drop", "120"], sleep=sleep)
    assert code == 130 and sent[-2:] == ["S", "D"]
    assert not any(line.startswith("J") and line.split()[4] == "23" for line in sent)   # never got to close


def test_pick_only_lifts_holds_2_s_and_opens_in_the_air(monkeypatch):
    waits = []
    code, sent, prompts, said = run(monkeypatch, ["--base-pick", "100", "--pick-only"], sleep=waits.append)
    assert code == 0
    moves = j_lines(sent)
    assert moves[-4][:4] == [100, 173, 90, 23] and moves[-3][:4] == [100, 153, 90, 23]   # close, lift
    assert moves[-2][:4] == [100, 153, 90, 90]                                           # open, still lifted
    assert moves[-1][:4] == [90, 150, 90, 90]                                            # back to the start pose
    assert all(m[0] == 100 for m in moves[1:-1])                                         # no swing
    assert 2.0 in waits and sent[-1] == "D"
    steps, hold_s, _ = po.plan(LIMITS, START, base_pick=100, base_drop=100, pick_only=True, **DEFAULTS)
    assert hold_s == pytest.approx(0.82 + 0.65 + 2.0 + 0.05)


def test_measure_latency_times_20_status_calls_and_moves_nothing(monkeypatch):
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: LIMITS)
    transports = []
    real_connect = session.connect
    monkeypatch.setattr(session, "connect",
                        lambda dry_run, limits, echo=None: transports.append(real_connect(dry_run, limits, None))
                        or transports[-1])
    ticks = iter(range(100))
    said = []
    code = po.main(["--measure-latency", "--dry-run"], say=said.append, clock=lambda: next(ticks) * 0.013)
    assert code == 0
    sent = transports[0].transport.sent
    assert sent.count("?") == 21 and not any(line[0] in "EJH" for line in sent)   # 20 timed, 1 at the end
    assert "20 status calls: mean 13.0 ms, slowest 13.0 ms. Use --command-s 0.02" in said[0]
