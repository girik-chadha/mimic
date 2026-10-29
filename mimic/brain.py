"""Muse Spark: the judgement layer. It looks at photos and answers in words.

name_objects(photo) -> ["red block", "blue block", ...]  names SAM can search for.
    Spark proposes names, SAM checks each one finds exactly 1 object, and any name that
    fails goes back to Spark with the match counts, up to 2 times.
diagnose(...) comes in Phase 4.
"""
import json
import logging
import re

from mimic import config, meta_api, sam

log = logging.getLogger(__name__)

MAX_RETRIES = 2              # new names Spark may give for one object after the first
NAMING_RETRY_BUDGET_USD = 0.03  # retry rounds stop once they have cost this much in one name_objects call

NAME_OBJECTS_PROMPT = """Look at this photo of objects on a table.
List every separate object that a robot arm could pick up.
Ignore the table, any paper sheet, drawn lines, numbers, and small calibration stickers.

Name each object as "colour + object type", for example "red block" or "blue lighter":
- all lowercase
- the main colour first, then what the object is (2 or 3 words in total)
- exactly one name per object
- every name different

Reply with ONLY a JSON list of strings and no other text, like this:
["red block", "blue lighter"]"""

RENAME_PROMPT = """Look at this photo of objects on a table again.
You named the objects: {names}.
An object finder searched the photo for each name. A good name matches exactly 1 object.
These names did not:
{failures}

Give a new name for each of those objects. Same rules as before: "colour + object type", all lowercase,
2 or 3 words, and different from every other name. Do not reuse any name already tried: {tried}.

Reply with ONLY a JSON object that maps each failed name to its new name, and no other text, like this:
{{"black box": "black tile"}}"""


class NamingFailed(RuntimeError):
    pass


# ---------- reading Spark's replies ----------

def _clean(names) -> list[str]:
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Expected a non-empty list of names, got: {names!r}")
    names = [" ".join(n.lower().split()) for n in names]  # lowercase, single spaces
    for n in names:
        if not 2 <= len(n.split()) <= 3:
            raise ValueError(f"{n!r} is not a 'colour + object type' name of 2 or 3 words")
    if len(set(names)) != len(names):
        raise ValueError(f"Two objects got the same name, so SAM could not tell them apart: {names}")
    return names


def _json_in(reply: str, pattern: str, what: str):
    found = re.search(pattern, reply, re.S)  # tolerates ```json fences or a sentence around it
    if not found:
        raise ValueError(f"No JSON {what} in Spark's reply: {reply[:200]!r}")
    try:
        return json.loads(found.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"Spark's {what} is not valid JSON: {found.group(0)[:200]!r}") from error


def parse_names(reply: str) -> list[str]:
    """Pull the JSON list out of Spark's reply and check every name is usable as a SAM prompt."""
    return _clean(_json_in(reply, r"\[.*\]", "list"))


def parse_renames(reply: str, failed: list[str]) -> dict[str, str]:
    """Pull the {old name: new name} object out of Spark's reply. Every failed name must get a new one."""
    renames = _json_in(reply, r"\{.*\}", "object")
    if not isinstance(renames, dict):
        raise ValueError(f"Expected a JSON object of renames, got: {renames!r}")
    renames = {" ".join(str(k).lower().split()): v for k, v in renames.items()}
    missing = [name for name in failed if name not in renames]
    if missing:
        raise ValueError(f"Spark gave no new name for {missing}: {reply[:200]!r}")
    new_names = _clean([renames[name] for name in failed])
    return dict(zip(failed, new_names))


# ---------- asking Spark ----------

def _with_photo(text: str, image_path: str) -> list:
    return [{"type": "input_text", "text": text},
            {"type": "input_image", "image_url": meta_api.image_url(image_path)}]


def name_objects_content(image_path: str) -> list:
    return _with_photo(NAME_OBJECTS_PROMPT, image_path)


def rename_content(image_path: str, names: list[str], failures: dict[str, int], tried: list[str]) -> list:
    failure_lines = "\n".join(f'- "{name}": {count} matches' for name, count in failures.items())
    text = RENAME_PROMPT.format(names=json.dumps(names), failures=failure_lines, tried=json.dumps(tried))
    return _with_photo(text, image_path)


def _check_retry_budget(cost_before_retries: float) -> None:
    spent = meta_api.cost_so_far() - cost_before_retries
    if spent >= NAMING_RETRY_BUDGET_USD:
        raise meta_api.BudgetExceeded(
            f"Renaming objects has cost ${spent:.4f}, over its ${NAMING_RETRY_BUDGET_USD:.2f} cap.")


def name_objects(image_path: str, attempts: list | None = None) -> list[str]:
    """Spark looks at the demo photo and returns one name per object that SAM finds exactly once.

    Every SAM check is appended to `attempts` (if given) as {"round", "name", "boxes"},
    so the run log can show how each name was chosen.
    """
    attempts = [] if attempts is None else attempts
    names = parse_names(meta_api.call(config.SPARK_MODEL, name_objects_content(image_path)))
    history = [[] for _ in names]  # per object: [(name, box count), ...]
    to_check = list(range(len(names)))
    cost_before_retries = None

    for round_no in range(MAX_RETRIES + 1):
        if round_no > 0:
            if cost_before_retries is None:
                cost_before_retries = meta_api.cost_so_far()
            _check_retry_budget(cost_before_retries)
            failures = {names[i]: history[i][-1][1] for i in to_check}
            tried = [name for h in history for name, _ in h]
            reply = meta_api.call(config.SPARK_MODEL, rename_content(image_path, names, failures, tried))
            renames = parse_renames(reply, list(failures))
            names = [renames.get(name, name) if i in to_check else name for i, name in enumerate(names)]
            _clean(names)  # a new name must not clash with another object's name

        for i in to_check:
            boxes = len(sam.find(image_path, names[i]))
            history[i].append((names[i], boxes))
            attempts.append({"round": round_no, "name": names[i], "boxes": boxes})
            log.info("naming round %d: %r -> %d box(es)", round_no, names[i], boxes)

        to_check = [i for i in to_check if history[i][-1][1] != 1]
        if not to_check:
            return names

    problems = []
    for i in to_check:
        tries = ", ".join(f"{name!r} ({count} boxes)" for name, count in history[i])
        problems.append(f"object {i + 1} (first named {history[i][0][0]!r}): tried {tries}")
    raise NamingFailed("No name that SAM finds exactly once after "
                       f"{MAX_RETRIES} retries. " + "; ".join(problems))
