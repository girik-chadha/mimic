"""ring_fit_manual.py: ruler-measured points on a made-up ring, fitted and saved. No arm, camera or SAM."""
import dataclasses
import importlib.util
import math
import shutil

import pytest
from test_ring import CENTRE, RADIUS, made_up

from mimic import config, ring

spec = importlib.util.spec_from_file_location("ring_fit_manual", config.ROOT / "scripts/ring_fit_manual.py")
rfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfm)


@pytest.fixture
def ring_path(tmp_path):
    """A copy of the committed ring.json, with slots and camera_clear set, as they would be by hand."""
    path = tmp_path / "ring.json"
    shutil.copy(config.ROOT / "calib" / "ring.json", path)
    ring.save_ring(dataclasses.replace(ring.load_ring(path), slots_deg=[95, 120, 145],
                                       camera_clear={"base": 90, "shoulder": 90}), path)
    return path


def argv_for(points):
    argv = []
    for base, x, y in points:
        argv += ["--point", str(base), f"{x:.3f}", f"{y:.3f}"]
    return argv


def test_five_ruler_points_fit_the_ring_and_save_it_as_manual(ring_path):
    said = []
    assert rfm.main(argv_for(made_up([80, 90, 100, 110, 120])), say=said.append, ring_path=ring_path) == 0
    saved = ring.load_ring(ring_path)
    assert saved.source == "manual" and saved.placeholder is False and len(saved.points) == 5
    assert saved.centre_cm == pytest.approx(CENTRE, abs=0.01) and saved.radius_cm == pytest.approx(RADIUS, abs=0.01)
    assert saved.slots_deg == [95, 120, 145] and saved.camera_clear == {"base": 90, "shoulder": 90}   # kept
    assert saved.shoulder_table == 180 and saved.grip_hold == 20                                       # kept
    assert said[0].startswith("Ring: centre (18.50, 32.00) cm, radius 14.00 cm")
    assert said[1].startswith("  base 80: (") and said[-1] == f"Saved {ring_path} (source manual)."
    assert [p["base_deg"] for p in saved.points] == [80, 90, 100, 110, 120]                          # whole numbers


def test_fewer_than_5_points_are_refused_and_nothing_is_saved(ring_path):
    before = ring_path.read_text()
    said = []
    assert rfm.main(argv_for(made_up([80, 100, 120, 140])), say=said.append, ring_path=ring_path) == 1
    assert said == ["STOPPED: 5 --point values are needed; there are 4. Nothing was saved."]
    assert ring_path.read_text() == before


def test_a_bad_point_refuses_the_fit_names_it_and_saves_nothing(ring_path):
    before = ring_path.read_text()
    points = made_up([80, 90, 100, 110, 120], noise=[(0, 0), (0, 0), (0, 1.5), (0, 0), (0, 0)])
    said = []
    assert rfm.main(argv_for(points), say=said.append, ring_path=ring_path) == 1
    assert said[0].startswith("STOPPED, nothing was saved: The point at base")
    assert "Without the point at base 100 the others fit" in said[0]
    assert ring_path.read_text() == before


def test_a_second_fit_says_what_it_replaces(ring_path):
    rfm.main(argv_for(made_up([80, 90, 100, 110, 120])), say=lambda s: None, ring_path=ring_path)
    said = []
    assert rfm.main(argv_for(made_up([70, 85, 100, 115, 130])), say=said.append, ring_path=ring_path) == 0
    assert any(line.startswith("Replacing the fit") and "it came from: manual" in line for line in said)
    saved = ring.load_ring(ring_path)
    assert math.dist(saved.point_at(130), made_up([130])[0][1:]) < 0.01
