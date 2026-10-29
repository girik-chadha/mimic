"""Real Spark call on desk.jpeg. Costs about $0.005. Run on purpose only: pytest -m api"""
import pytest

from mimic import brain, config, meta_api

DESK = str(config.ROOT / "tests/fixtures/photos/desk.jpeg")


@pytest.mark.api
def test_spark_names_three_objects_on_desk():
    reply = meta_api.call(config.SPARK_MODEL, brain.name_objects_content(DESK), use_cache=False)
    names = brain.parse_names(reply)
    assert len(names) == 3
