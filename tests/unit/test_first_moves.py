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
    t = FakeTransport(limits)
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
    t = FakeTransport(limits)
    fm.explore(Arm(t, limits), limits, 45, answers(*["n"] * 6), say=lambda s: None, sleep=lambda s: None)
    for line in t.sent:
        if line.startswith("J"):
            assert int(line.split()[-1]) >= fm.MIN_MOVE_MS


def test_base_is_capped_by_max_delta():
    limits = limits_for_test(shoulder=(85, 95), elbow=(85, 95))
    t = FakeTransport(limits)
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
    code, out, said = run_main(monkeypatch, tmp_path, ["y"] * 18 + ["n"] * 4)
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
    code, out, said = run_main(monkeypatch, tmp_path, ["y", "s"])
    assert code == 130 and not out.exists()
    assert sent[0].sent[-1] == "S"


def test_real_arm_refused_while_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    said = []
    assert fm.main([], read=answers(), sleep=lambda s: None, say=said.append, proposed_path=tmp_path / "p.json") == 1
    assert "placeholder" in said[0]
