"""meta_api tests. A fake client stands in for Meta, so nothing here touches the network."""
import base64
import json
import traceback
from types import SimpleNamespace

import cv2
import httpx2
import numpy as np
import openai
import pytest

from mimic import config, meta_api

PHOTO = str(config.ROOT / "tests/fixtures/photos/desk.jpeg")


class FakeClient:
    """Pretends to be the OpenAI client. Hands out the replies (or errors) it was given, in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.responses = self

    def create(self, model, input):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(text, usage=None):
    return SimpleNamespace(output_text=text, usage=usage)


def http_error(error_class, status, message="fake"):
    request = httpx2.Request("POST", "https://api.meta.ai/v1/responses")
    return error_class(message, response=httpx2.Response(status, request=request), body=None)


def photo_content(prompt="blue lighter"):
    return [{"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": meta_api.image_url(PHOTO)}]


def make_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), i, np.uint8))
    writer.release()
    return str(path)


def no_network(**kwargs):
    pytest.fail("A unit test tried to create a real Meta client. Use a FakeClient.")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets an empty cache, a fresh cost counter, no real client and no real waiting."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESPONSES_DIR", tmp_path / "responses")
    monkeypatch.setattr(config, "BUDGET_USD", 1.00)
    monkeypatch.setattr(meta_api, "_client", None)
    monkeypatch.setattr(meta_api.openai, "OpenAI", no_network)
    monkeypatch.setattr(meta_api.time, "sleep", lambda seconds: None)
    meta_api.reset()


def use(monkeypatch, client):
    monkeypatch.setattr(meta_api, "_client", client)
    return client


# ---------- data URLs ----------

def test_image_url_is_base64_jpeg():
    url = meta_api.image_url(PHOTO)
    assert url.startswith("data:image/jpeg;base64,")
    with open(PHOTO, "rb") as f:
        assert base64.b64decode(url.split(",", 1)[1]) == f.read()


def test_video_url_is_mp4(tmp_path):
    assert meta_api.video_url(make_video(tmp_path / "v.mp4", 3)).startswith("data:video/mp4;base64,")


# ---------- cache ----------

def test_second_identical_call_comes_from_cache(monkeypatch):
    client = use(monkeypatch, FakeClient(reply("<|box;...|>")))
    first = meta_api.call("sam-3.1", photo_content())
    second = meta_api.call("sam-3.1", photo_content())
    assert first == second == "<|box;...|>"
    assert client.calls == 1
    assert meta_api.cost_so_far() == pytest.approx(0.0025)  # the cached answer was free


def test_different_prompt_is_not_a_cache_hit(monkeypatch):
    client = use(monkeypatch, FakeClient(reply("a"), reply("b")))
    assert meta_api.call("sam-3.1", photo_content("blue lighter")) == "a"
    assert meta_api.call("sam-3.1", photo_content("green lighter")) == "b"
    assert client.calls == 2


def test_cache_holds_no_key_and_no_photo(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "secret-key-do-not-store")
    use(monkeypatch, FakeClient(reply("ok")))
    content = photo_content()
    meta_api.call("sam-3.1", content)
    for path in config.CACHE_DIR.iterdir():
        stored = path.read_text()
        assert "secret-key-do-not-store" not in stored
        assert content[1]["image_url"][30:100] not in stored


def test_use_cache_false_skips_reading_and_writing(monkeypatch):
    client = use(monkeypatch, FakeClient(reply("a"), reply("b")))
    assert meta_api.call("sam-3.1", photo_content(), use_cache=False) == "a"
    assert meta_api.call("sam-3.1", photo_content(), use_cache=False) == "b"
    assert client.calls == 2
    assert meta_api.cost_so_far() == pytest.approx(0.005)  # both calls really paid for
    assert not list(config.CACHE_DIR.glob("*.json"))      # nothing written to the cache


# ---------- cost ----------

def test_photo_costs_quarter_of_a_cent(monkeypatch):
    use(monkeypatch, FakeClient(reply("ok")))
    meta_api.call("sam-3.1", photo_content())
    assert meta_api.cost_so_far() == pytest.approx(0.0025)


def test_short_video_is_billed_as_50_frames(tmp_path):
    content = [{"type": "input_video", "video_url": meta_api.video_url(make_video(tmp_path / "v.mp4", 10))}]
    assert meta_api.sam_cost(content) == pytest.approx(50 * 0.0002)


def test_long_video_is_billed_per_frame(tmp_path):
    content = [{"type": "input_video", "video_url": meta_api.video_url(make_video(tmp_path / "v.mp4", 60))}]
    assert meta_api.sam_cost(content) == pytest.approx(60 * 0.0002)


def test_spark_cost_uses_token_counts_when_given(monkeypatch):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200)
    use(monkeypatch, FakeClient(reply('["red block"]', usage)))
    meta_api.call("muse-spark-1.3", "name the objects")
    assert meta_api.cost_so_far() == pytest.approx(1000 * 1.25e-6 + 200 * 4.25e-6)


def test_spend_log_records_token_counts_or_none(monkeypatch):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200)
    use(monkeypatch, FakeClient(reply("a", usage), reply("b")))
    meta_api.call("muse-spark-1.3", "first")
    meta_api.call("muse-spark-1.3", "second")
    lines = [json.loads(line) for line in (config.CACHE_DIR / "spend_log.jsonl").read_text().splitlines()]
    assert lines[0]["usage"] == {"input_tokens": 1000, "output_tokens": 200}
    assert lines[0]["estimated"] is False
    assert lines[1]["usage"] is None
    assert lines[1]["estimated"] is True


def test_spark_cost_is_estimated_without_token_counts():
    cost, estimated = meta_api.spark_cost("x" * 400, "y" * 40, usage=None)
    assert estimated
    assert cost == pytest.approx(100 * 1.25e-6 + 10 * 4.25e-6)


def test_reset_starts_a_new_run_from_zero(monkeypatch):
    use(monkeypatch, FakeClient(reply("a"), reply("b"), reply("c")))
    meta_api.call("sam-3.1", photo_content("one"))
    meta_api.call("sam-3.1", photo_content("two"))
    assert meta_api.cost_so_far() == pytest.approx(0.005)
    meta_api.reset()
    assert meta_api.cost_so_far() == 0
    meta_api.call("sam-3.1", photo_content("three"))
    assert meta_api.cost_so_far() == pytest.approx(0.0025)


# ---------- budget guard ----------

def test_budget_stops_the_next_call(monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 0.004)
    client = use(monkeypatch, FakeClient(reply("a"), reply("b"), reply("c")))
    meta_api.call("sam-3.1", photo_content("one"))   # spent 0.0025
    meta_api.call("sam-3.1", photo_content("two"))   # spent 0.0050, now over budget
    with pytest.raises(meta_api.BudgetExceeded):
        meta_api.call("sam-3.1", photo_content("three"))
    assert client.calls == 2  # the third request never left the laptop


def test_cached_answer_still_works_over_budget(monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 0.001)
    use(monkeypatch, FakeClient(reply("a")))
    meta_api.call("sam-3.1", photo_content())        # spent 0.0025, over budget
    assert meta_api.call("sam-3.1", photo_content()) == "a"


# ---------- retries ----------

def test_retries_after_rate_limit(monkeypatch):
    waits = []
    monkeypatch.setattr(meta_api.time, "sleep", waits.append)
    client = use(monkeypatch, FakeClient(http_error(openai.RateLimitError, 429),
                                         http_error(openai.InternalServerError, 503),
                                         reply("ok")))
    assert meta_api.call("sam-3.1", photo_content()) == "ok"
    assert client.calls == 3
    assert waits == [1, 2]


def test_gives_up_after_three_retries(monkeypatch):
    client = use(monkeypatch, FakeClient(*[http_error(openai.RateLimitError, 429)] * 4))
    with pytest.raises(openai.RateLimitError):
        meta_api.call("sam-3.1", photo_content())
    assert client.calls == 4
    assert meta_api.cost_so_far() == 0  # failed calls cost nothing


def test_bad_request_is_not_retried(monkeypatch):
    client = use(monkeypatch, FakeClient(http_error(openai.BadRequestError, 400)))
    with pytest.raises(openai.BadRequestError):
        meta_api.call("sam-3.1", photo_content())
    assert client.calls == 1


# ---------- model safety ----------

def test_contributor_model_is_refused(monkeypatch):
    client = use(monkeypatch, FakeClient())
    with pytest.raises(ValueError, match="Never use -contributor"):
        meta_api.call("muse-spark-1.3-contributor", "hello")
    assert client.calls == 0


def test_unknown_model_is_refused(monkeypatch):
    use(monkeypatch, FakeClient())
    with pytest.raises(ValueError, match="Unknown model"):
        meta_api.call("muse-image-1.0", "hello")


# ---------- the API key never shows up in an error ----------

KEY = "test-key-abc123-never-show-me"


def test_api_key_is_scrubbed_from_errors(monkeypatch):
    monkeypatch.setenv("META_API_KEY", KEY)
    use(monkeypatch, FakeClient(http_error(openai.AuthenticationError, 401,
                                           f"Incorrect API key provided: {KEY}")))
    with pytest.raises(openai.AuthenticationError) as caught:
        meta_api.call("sam-3.1", photo_content())
    shown = "".join(traceback.format_exception(caught.value))
    assert KEY not in shown
    assert "Incorrect API key provided: ***" in shown


def test_api_key_is_scrubbed_from_the_error_that_caused_it(monkeypatch):
    monkeypatch.setenv("META_API_KEY", KEY)
    try:
        raise ValueError(f"header was Bearer {KEY}")
    except ValueError as inner:
        outer = RuntimeError("request failed")
        outer.__cause__ = inner
    use(monkeypatch, FakeClient(outer))
    with pytest.raises(RuntimeError) as caught:
        meta_api.call("sam-3.1", photo_content())
    shown = "".join(traceback.format_exception(caught.value))
    assert KEY not in shown
    assert "Bearer ***" in shown


def test_client_has_60_second_timeout_and_no_sdk_retries(monkeypatch):
    monkeypatch.setenv("META_API_KEY", KEY)
    settings = {}
    monkeypatch.setattr(meta_api.openai, "OpenAI", lambda **kwargs: settings.update(kwargs))
    meta_api._get_client()
    assert settings["timeout"] == 60
    assert settings["max_retries"] == 0
    assert settings["base_url"] == "https://api.meta.ai/v1"


# ---------- saving real replies as fixtures ----------

def test_unit_tests_cannot_reach_the_network():
    with pytest.raises(pytest.fail.Exception):
        meta_api.call("sam-3.1", photo_content())


def test_save_as_writes_fixture(monkeypatch):
    use(monkeypatch, FakeClient(reply("<|box;x1=1|>")))
    meta_api.call("sam-3.1", photo_content(), save_as="sam_photo_test")
    assert (config.RESPONSES_DIR / "sam_photo_test.txt").read_text() == "<|box;x1=1|>"
