"""plan.py tests. Pure Python: no network, no hardware."""
import logging

import pytest

from mimic.plan import Move, extra_objects, plan_moves

SLOTS = [(30.0, 5.0), (30.0, 10.0), (30.0, 15.0)]
# Listed in a different order from the target, to prove the plan follows the target, not this dict.
MESS = {"green block": (8.0, 12.5), "red block": (5.0, 4.0), "blue block": (12.0, 20.0)}


def test_moves_follow_the_target_order_exactly():
    moves = plan_moves(MESS, ["red block", "blue block", "green block"], SLOTS)
    assert moves == [
        Move("red block", (5.0, 4.0), (30.0, 5.0), slot=1),
        Move("blue block", (12.0, 20.0), (30.0, 10.0), slot=2),
        Move("green block", (8.0, 12.5), (30.0, 15.0), slot=3),
    ]


def test_a_different_target_gives_a_different_order():
    moves = plan_moves(MESS, ["green block", "red block", "blue block"], SLOTS)
    assert [(m.name, m.slot) for m in moves] == [("green block", 1), ("red block", 2), ("blue block", 3)]


def test_fewer_names_than_slots_uses_the_first_slots():
    moves = plan_moves(MESS, ["blue block", "red block"], SLOTS)
    assert moves == [
        Move("blue block", (12.0, 20.0), (30.0, 5.0), slot=1),
        Move("red block", (5.0, 4.0), (30.0, 10.0), slot=2),
    ]


def test_extra_objects_are_left_alone_and_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="mimic.plan"):
        moves = plan_moves(MESS, ["red block", "blue block"], SLOTS)
    assert "green block" not in [m.name for m in moves]
    assert extra_objects(MESS, ["red block", "blue block"]) == ["green block"]
    assert "green block" in caplog.text


def test_no_warning_when_there_are_no_extras(caplog):
    with caplog.at_level(logging.WARNING, logger="mimic.plan"):
        plan_moves(MESS, ["red block", "blue block", "green block"], SLOTS)
    assert caplog.text == ""


# ---------- errors ----------

def test_empty_target_raises():
    with pytest.raises(ValueError, match="empty"):
        plan_moves(MESS, [], SLOTS)


def test_missing_name_raises():
    with pytest.raises(ValueError, match=r"Not found on the table: \['yellow block'\]"):
        plan_moves(MESS, ["red block", "yellow block"], SLOTS)


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match=r"more than once in the target: \['red block'\]"):
        plan_moves(MESS, ["red block", "blue block", "red block"], SLOTS)


def test_more_names_than_slots_raises():
    with pytest.raises(ValueError, match="3 objects to place but only 2 slots"):
        plan_moves(MESS, ["red block", "blue block", "green block"], SLOTS[:2])


def test_two_targets_closer_than_1_cm_raise():
    close = {"red block": (5.0, 4.0), "orange block": (5.6, 4.0), "blue block": (12.0, 20.0)}
    with pytest.raises(ValueError, match="'red block' and 'orange block' are only 0.60 cm apart"):
        plan_moves(close, ["red block", "orange block", "blue block"], SLOTS)


def test_targets_just_over_1_cm_apart_are_fine():
    near = {"red block": (5.0, 4.0), "orange block": (6.1, 4.0)}
    assert len(plan_moves(near, ["red block", "orange block"], SLOTS)) == 2


def test_an_extra_object_close_to_a_target_is_not_an_error():
    crowded = dict(MESS, **{"pencil": (5.2, 4.0)})  # not in the target, so not checked
    assert len(plan_moves(crowded, ["red block", "blue block"], SLOTS)) == 2


def test_two_slots_at_the_same_point_raise():
    with pytest.raises(ValueError, match=r"Slots 1 and 3 are the same point \(30.0, 5.0\)"):
        plan_moves(MESS, ["red block"], [(30.0, 5.0), (30.0, 10.0), (30.0, 5.0)])
