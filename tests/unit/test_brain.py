"""brain.py tests. Spark's real reply is saved in tests/fixtures/meta_responses/. No network."""
import pytest

from mimic import brain, config, meta_api, sam

REPLIES = config.ROOT / "tests/fixtures/meta_responses"
DESK = str(config.ROOT / "tests/fixtures/photos/desk.jpeg")


def saved(name):
    return (REPLIES / f"{name}.txt").read_text()


def test_real_spark_reply_gives_three_names():
    assert brain.parse_names(saved("spark_name_objects_desk")) == ["blue lighter", "green lighter", "black box"]


def test_prompt_asks_for_lowercase_colour_and_type_one_per_object():
    prompt = brain.NAME_OBJECTS_PROMPT
    assert "colour + object type" in prompt
    assert "lowercase" in prompt
    assert "exactly one name per object" in prompt
    assert "ONLY a JSON list" in prompt


@pytest.mark.parametrize("reply", [
    '```json\n["red block", "blue block"]\n```',
    'Here are the objects: ["red block", "blue block"]. Hope that helps!',
    '["Red Block", "  blue   block "]',
])
def test_list_is_found_and_cleaned(reply):
    assert brain.parse_names(reply) == ["red block", "blue block"]


@pytest.mark.parametrize("reply, problem", [
    ("I see a red block and a blue block.", "No JSON list"),
    ('["red block", "blue block"', "No JSON list"),
    ("[red block, blue block]", "not valid JSON"),
    ("[]", "non-empty list"),
    ('["red block", 3]', "non-empty list"),
    ('["red block", "red block"]', "same name"),
    ('["red block", "Red  Block"]', "same name"),
    ('["block"]', "2 or 3 words"),
    ('["small shiny dark red block"]', "2 or 3 words"),
])
def test_unusable_replies_are_rejected(reply, problem):
    with pytest.raises(ValueError, match=problem):
        brain.parse_names(reply)


# ---------- name_objects: Spark proposes, SAM checks, Spark renames ----------

# Which saved SAM reply the fake SAM gives for each name: 1, 0 or 3 boxes.
SAM_REPLY_FOR = {
    "blue lighter": "sam_photo_desk_blue_lighter",        # 1 box
    "green lighter": "sam_photo_desk_green_lighter",      # 1 box
    "black rectangle": "sam_photo_desk_black_rectangle",  # 1 box
    "black box": "sam_photo_desk_black_box",              # 0 boxes
    "black tile": "sam_photo_desk_red_cup",               # 0 boxes
    "dark tile": "sam_photo_desk_red_cup",                # 0 boxes
    "grey building": "sam_photo_test_building",           # 3 boxes
}


class FakeMeta:
    """Plays Spark (canned replies, in order) and SAM (saved replies, looked up by name)."""

    def __init__(self, monkeypatch, *spark_replies):
        self.spark_replies = list(spark_replies)
        self.spark_prompts = []
        self.sam_names = []
        monkeypatch.setattr(meta_api, "call", self.spark)
        monkeypatch.setattr(sam, "find", self.find)

    def spark(self, model, content, use_cache=True):
        assert model == "muse-spark-1.3"
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")  # photo sent every time
        self.spark_prompts.append(content[0]["text"])
        return self.spark_replies.pop(0)

    def find(self, image_path, name, use_cache=True):
        self.sam_names.append(name)
        return sam.parse_boxes(saved(SAM_REPLY_FOR[name]))


def test_names_pass_first_time(monkeypatch):
    fake = FakeMeta(monkeypatch, '["blue lighter", "green lighter", "black rectangle"]')
    attempts = []
    assert brain.name_objects(DESK, attempts) == ["blue lighter", "green lighter", "black rectangle"]
    assert fake.spark_prompts == [brain.NAME_OBJECTS_PROMPT]  # no retry
    assert attempts == [{"round": 0, "name": "blue lighter", "boxes": 1},
                        {"round": 0, "name": "green lighter", "boxes": 1},
                        {"round": 0, "name": "black rectangle", "boxes": 1}]


def test_name_fails_then_passes(monkeypatch):
    fake = FakeMeta(monkeypatch, saved("spark_name_objects_desk"),  # the real reply: "black box" finds nothing
                    '{"black box": "black rectangle"}')
    attempts = []
    assert brain.name_objects(DESK, attempts) == ["blue lighter", "green lighter", "black rectangle"]
    retry = fake.spark_prompts[1]
    assert '- "black box": 0 matches' in retry
    assert retry.count('\n- "') == 1  # only the failed name is listed; names that passed are not
    assert fake.sam_names == ["blue lighter", "green lighter", "black box", "black rectangle"]
    assert attempts[-2:] == [{"round": 0, "name": "black box", "boxes": 0},
                             {"round": 1, "name": "black rectangle", "boxes": 1}]


def test_gives_up_after_two_retries_and_says_what_was_tried(monkeypatch):
    fake = FakeMeta(monkeypatch, saved("spark_name_objects_desk"),
                    '{"black box": "black tile"}',
                    '{"black tile": "dark tile"}')
    attempts = []
    with pytest.raises(brain.NamingFailed) as caught:
        brain.name_objects(DESK, attempts)
    message = str(caught.value)
    assert "object 3 (first named 'black box')" in message
    assert "'black box' (0 boxes), 'black tile' (0 boxes), 'dark tile' (0 boxes)" in message
    assert len(fake.spark_prompts) == 3  # first ask + 2 retries, no more
    assert [a["round"] for a in attempts] == [0, 0, 0, 1, 2]
    assert '"black box", "black tile"' in fake.spark_prompts[2]  # earlier tries listed as used


def test_name_with_several_boxes_is_retried(monkeypatch):
    fake = FakeMeta(monkeypatch, '["blue lighter", "grey building"]',
                    '{"grey building": "green lighter"}')
    assert brain.name_objects(DESK) == ["blue lighter", "green lighter"]
    assert '- "grey building": 3 matches' in fake.spark_prompts[1]


def test_retry_message_lists_every_failed_name(monkeypatch):
    fake = FakeMeta(monkeypatch, '["black box", "grey building", "blue lighter"]',
                    '{"black box": "black rectangle", "grey building": "green lighter"}')
    assert brain.name_objects(DESK) == ["black rectangle", "green lighter", "blue lighter"]
    assert '- "black box": 0 matches\n- "grey building": 3 matches' in fake.spark_prompts[1]


def test_retries_stop_at_the_naming_budget(monkeypatch):
    fake = FakeMeta(monkeypatch, saved("spark_name_objects_desk"), '{"black box": "black tile"}')
    costs = iter([0.010, 0.010, 0.045])  # before retries; check before retry 1; check before retry 2
    monkeypatch.setattr(meta_api, "cost_so_far", lambda: next(costs))
    with pytest.raises(meta_api.BudgetExceeded, match="Renaming objects has cost"):
        brain.name_objects(DESK)
    assert len(fake.spark_prompts) == 2  # the second retry was never sent


def test_rename_that_clashes_with_another_name_is_rejected(monkeypatch):
    FakeMeta(monkeypatch, saved("spark_name_objects_desk"), '{"black box": "blue lighter"}')
    with pytest.raises(ValueError, match="same name"):
        brain.name_objects(DESK)


def test_real_spark_rename_reply():
    reply = saved("spark_rename_desk_round1")
    assert brain.parse_renames(reply, ["black box"]) == {"black box": "black rectangle"}


def test_rename_reply_must_cover_every_failed_name():
    with pytest.raises(ValueError, match="no new name"):
        brain.parse_renames('{"something else": "red block"}', ["black box"])
