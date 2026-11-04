"""first_moves.py with the fake arm and typed-in answers. calib/arm_limits.json is never changed."""
import dataclasses
import importlib.util
import json

import pytest

from mimic import arm, config, session
from mimic.arm import Arm, FakeTransport

REAL_LOAD_LIMITS = arm.load_limits  # kept before any test patches arm.load_limits

spec = importlib.util.spec_from_file_location("first_moves", config.ROOT / "scripts/first_moves.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def limits_for_test(shoulder=(80, 140), elbow=(30, 150)):
    """Base wide open (so only --max-delta caps it), shoulder with only 10 degrees below home."""
    return dataclasses.replace(REAL_LOAD_LIMITS(config.ARM_LIMITS_PATH),
                               ranges_deg={"base": (0, 180), "shoulder": shoulder, "elbow": elbow})


def answers(*typed):
    queue = list(typed)
    return lambda question: queue.pop(0)


def j_lines(transport):
    return [[int(v) for v in line.split()[1:5]] for line in transport.sent if line.startswith("J")]  # angles only


def test_steps_in_one_direction_stop_at_the_range_edge_or_max_delta():
    assert fm.deltas(90, 180, 45) == [5, 10, 15, 20, 25, 30, 35, 40, 45]
    assert fm.deltas(90, 80, 45) == [5, 10]          # the shoulder: only 10 degrees below home
    assert fm.deltas(90, 78, 45) == [5, 10, 12]      # a last, smaller step right up to the edge
    assert fm.deltas(90, 90, 45) == []


def test_each_direction_stops_at_its_first_n():
    limits = limits_for_test()
    t = FakeTransport(limits, enabled=True)
    a = Arm(t, limits)
    approved = fm.explore(a, limits, 45, answers(
        "y", "y", "y", "n", "n",      # base: +5 y, -5 y, +10 y, -10 n, +15 n
        "y", "y", "y", "y", "n",      # shoulder: +5, -5, +10, -10 (the edge), +15 n
        "n", "n"),                    # elbow: +5 n, -5 n
        say=lambda s: None, sleep=lambda s: None)
    assert approved == {"base": {"plus": 10, "minus": 5}, "shoulder": {"plus": 10, "minus": 10},
                        "elbow": {"plus": 0, "minus": 0}}
    moves = j_lines(t)
    assert min(m[1] for m in moves) == 80                      # never below the shoulder's range
    assert all(m == [90, 90, 90, 90] for m in moves[2::2])    # back home after every test move


def test_moves_are_slow():
    limits = limits_for_test()
    t = FakeTransport(limits, enabled=True)
    fm.explore(Arm(t, limits), limits, 45, answers(*["n"] * 6), say=lambda s: None, sleep=lambda s: None)
    for line in t.sent:
        if line.startswith("J"):
            assert int(line.split()[-1]) >= fm.MIN_MOVE_MS


def test_base_is_capped_by_max_delta():
    limits = limits_for_test(shoulder=(85, 95), elbow=(85, 95))
    t = FakeTransport(limits, enabled=True)
    fm.explore(Arm(t, limits), limits, 45, answers(*["y"] * 22), say=lambda s: None, sleep=lambda s: None)
    bases = [m[0] for m in j_lines(t)]
    assert max(bases) == 135 and min(bases) == 45


def test_proposal_is_the_approved_range_minus_5():
    approved = {"base": {"plus": 10, "minus": 5}, "shoulder": {"plus": 10, "minus": 10},
                "elbow": {"plus": 0, "minus": 0}}
    data = fm.proposal(config.ARM_LIMITS_PATH, limits_for_test(), approved)
    assert data["ranges_deg"] == {"base": [90, 95], "shoulder": [85, 95], "elbow": [90, 90]}
    assert data["placeholder"] is False and data["limits_checked_by_hand"] is True
    assert data["approved_deg"] == approved and "NOT ACTIVE" in data["note"]


def run_main(monkeypatch, tmp_path, typed, argv=("--dry-run",)):
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: limits_for_test((85, 95), (85, 95)))
    said = []
    out = tmp_path / "proposed.json"
    code = fm.main(list(argv), read=answers(*typed), sleep=lambda s: None, say=said.append, proposed_path=out)
    return code, out, said


def test_dry_run_writes_the_proposal_and_never_arm_limits(monkeypatch, tmp_path):
    before = config.ARM_LIMITS_PATH.read_bytes()
    code, out, said = run_main(monkeypatch, tmp_path, ["yes"] + ["y"] * 18 + ["n"] * 4)   # yes: enable first
    assert code == 0 and out.exists()
    assert json.loads(out.read_text())["ranges_deg"]["base"] == [50, 130]
    assert config.ARM_LIMITS_PATH.read_bytes() == before


def test_s_stops_with_S_and_writes_nothing(monkeypatch, tmp_path):
    sent = []
    real_connect = session.connect

    def connect(dry_run, limits, echo=print):
        a = real_connect(dry_run, limits, echo=None)
        sent.append(a.transport)
        return a

    monkeypatch.setattr(session, "connect", connect)
    code, out, said = run_main(monkeypatch, tmp_path, ["yes", "y", "s"])
    assert code == 130 and not out.exists()
    assert sent[0].sent[-2:] == ["S", "D"]   # stop, then switch the servos off


def test_it_enables_first_and_a_no_moves_nothing(monkeypatch, tmp_path):
    sent = []
    real_connect = session.connect

    def connect(dry_run, limits, echo=print):
        a = real_connect(dry_run, limits, echo=None)
        sent.append(a.transport)
        return a

    monkeypatch.setattr(session, "connect", connect)
    code, out, said = run_main(monkeypatch, tmp_path, ["no"])
    assert code == 1 and not out.exists()
    assert sent[0].sent == ["?"] and "Not enabled: nothing was moved." in said
    assert any("one at a time, 0.4 s apart" in line for line in said)


def test_real_arm_refused_while_locked(monkeypatch, tmp_path):
    locked = dataclasses.replace(REAL_LOAD_LIMITS(config.ARM_LIMITS_PATH), placeholder=True)
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: locked)
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    said = []
    assert fm.main([], read=answers(), sleep=lambda s: None, say=said.append, proposed_path=tmp_path / "p.json") == 1
    assert "placeholder" in said[0]


# ---------- saved after every joint, merged, resumable (4 Nov) ----------

BASE_ANSWERS = ["y"] * 18          # base: +5 -5 ... +45 -45, all fine (max_delta 45)
SMALL_JOINT = ["y", "y"]           # shoulder or elbow at 85-95: +5, -5


def saved(out):
    return json.loads(out.read_text())


def test_a_stop_during_the_shoulder_keeps_the_finished_base(monkeypatch, tmp_path):
    code, out, said = run_main(monkeypatch, tmp_path, ["yes"] + BASE_ANSWERS + ["s"])
    assert code == 130
    data = saved(out)
    assert data["joints_done"] == ["base"] and data["approved_deg"] == {"base": {"plus": 45, "minus": 45}}
    assert data["limits_checked_by_hand"] is False and "INCOMPLETE: missing shoulder, elbow" in data["note"]
    assert any("shoulder not finished: approved +0 / -0 so far (NOT saved)" in line for line in said)


def test_joints_runs_only_those_and_merges_keeping_the_rest(monkeypatch, tmp_path):
    out = tmp_path / "proposed.json"
    out.write_text(json.dumps(fm.proposal(config.ARM_LIMITS_PATH, limits_for_test((85, 95), (85, 95)),
                                          {"base": {"plus": 35, "minus": 15}}, {"base": "run 2026-11-04 09:30"})))
    code, out, said = run_main(monkeypatch, tmp_path, ["yes"] + SMALL_JOINT * 2,
                               argv=("--dry-run", "--joints", "shoulder", "elbow"))
    assert code == 0
    data = saved(out)
    assert data["joints_done"] == ["base", "shoulder", "elbow"] and data["limits_checked_by_hand"] is True
    assert data["approved_deg"]["base"] == {"plus": 35, "minus": 15}                  # kept
    assert data["approved_from"]["base"] == "run 2026-11-04 09:30"
    assert data["ranges_deg"]["base"] == [80, 120] and "COMPLETE" in data["note"]
    assert any(line.startswith("Keeping base from") and "+35 / -15" in line for line in said)


def test_base_seed_is_saved_before_anything_moves_and_the_base_never_moves(monkeypatch, tmp_path):
    out = tmp_path / "proposed.json"
    seen_at_connect = []
    real_connect = session.connect

    def connect(dry_run, limits, echo=print):
        seen_at_connect.append(saved(out)["approved_deg"])
        a = real_connect(dry_run, limits, echo=None)
        sent.append(a.transport)
        return a

    sent = []
    monkeypatch.setattr(session, "connect", connect)
    code, out, said = run_main(monkeypatch, tmp_path, ["yes"] + SMALL_JOINT * 2,
                               argv=("--dry-run", "--joints", "shoulder", "elbow", "--base-minus", "15",
                                     "--base-plus", "35"))
    assert code == 0
    assert seen_at_connect == [{"base": {"plus": 35, "minus": 15}}]          # on disk before the arm opened
    assert all(m[0] == 90 for m in j_lines(sent[0]))                          # the base never moved
    data = saved(out)
    assert data["ranges_deg"]["base"] == [80, 120] and data["limits_checked_by_hand"] is True
    assert data["approved_from"]["base"].startswith("typed in (--base-minus/--base-plus)")


@pytest.mark.parametrize("argv, problem", [
    (("--base-minus", "15"), "Give both --base-minus and --base-plus"),
    (("--base-minus", "15", "--base-plus", "35"), "leave base out of --joints"),
    (("--joints", "shoulder", "--base-minus", "15", "--base-plus", "50"), "--base-plus 50 must be between 0 and 45"),
    (("--joints", "shoulder", "--base-minus", "-5", "--base-plus", "35"), "--base-minus -5 must be between 0 and 45"),
    (("--joints", "shoulder", "shoulder"), "--joints lists a joint twice"),
])
def test_bad_options_are_refused_before_anything_is_written_or_moved(monkeypatch, tmp_path, argv, problem):
    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("arm connected"))
    code, out, said = run_main(monkeypatch, tmp_path, [], argv=("--dry-run",) + argv)
    assert code == 1 and problem in said[0] and not out.exists()


def test_a_full_run_replaces_an_older_entry_for_the_same_joint(monkeypatch, tmp_path):
    out = tmp_path / "proposed.json"
    out.write_text(json.dumps(fm.proposal(config.ARM_LIMITS_PATH, limits_for_test((85, 95), (85, 95)),
                                          {"base": {"plus": 10, "minus": 10}}, {"base": "run 2026-11-03 18:00"})))
    code, out, said = run_main(monkeypatch, tmp_path, ["yes"] + BASE_ANSWERS + SMALL_JOINT * 2)
    assert code == 0 and saved(out)["approved_deg"]["base"] == {"plus": 45, "minus": 45}
    assert saved(out)["approved_from"]["base"].startswith("run ")
    assert not any(line.startswith("Keeping base") for line in said)


def test_an_unreadable_proposal_stops_before_anything_moves(monkeypatch, tmp_path):
    (tmp_path / "proposed.json").write_text("{not json")
    monkeypatch.setattr(session, "connect", lambda *a, **k: pytest.fail("arm connected"))
    code, out, said = run_main(monkeypatch, tmp_path, [])
    assert code == 1 and "cannot be read" in said[0] and "fix or delete it" in said[0]


def test_an_incomplete_proposal_loads_but_is_not_checked_by_hand(tmp_path):
    data = fm.proposal(config.ARM_LIMITS_PATH, limits_for_test(), {"base": {"plus": 35, "minus": 15}})
    assert data["ranges_deg"]["shoulder"] == list(limits_for_test().ranges_deg["shoulder"])   # active range kept
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))
    limits = arm.load_limits(path)
    with pytest.raises(arm.ArmError, match="not been checked by hand"):
        arm.refuse_unchecked_limits(limits)   # so no real move can use it
