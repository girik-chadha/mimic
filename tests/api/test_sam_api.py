"""Real SAM call. Costs $0.0025. Run on purpose only: pytest -m api"""
import pytest

from mimic import config, sam

DESK = str(config.ROOT / "tests/fixtures/photos/desk.jpeg")


@pytest.mark.api
def test_find_blue_lighter_on_desk():
    boxes = sam.find(DESK, "blue lighter", use_cache=False)
    assert len(boxes) == 1
    x, y = boxes[0].centre
    assert abs(x - 287.5) < 20 and abs(y - 473) < 20
