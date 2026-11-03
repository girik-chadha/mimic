"""Muse Spark: the judgement layer. It looks at photos and answers in words.

name_objects(photo) -> ["red block", "blue block", ...]  names SAM can search for.
    Spark proposes names, SAM checks each one finds exactly 1 object, and any name that
    fails goes back to Spark with the match counts, up to 2 times.
diagnose(before, after, move, miss) -> {"problem", "fix", "explanation"}
    Spark looks at the photos before and after a missed placement and says what went wrong.
    Its fix is advice only: loop.choose_correction checks it and decides.
"""
import json
import logging
import math
import re

from mimic import config, meta_api, sam

log = logging.getLogger(__name__)

MAX_RETRIES = 2              # new names Spark may give for one object after the first
NAMING_RETRY_BUDGET_USD = 0.03  # retry rounds stop once they have cost this much in one name_objects call

NAME_OBJECTS_PROMPT = """Look at this photo of objects on a table.
The objects are small painted dice, so name each one 'colour die' (for example 'red die').
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
{{"red cup": "red mug"}}"""


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


# ---------- diagnosing a missed placement ----------

PROBLEMS = ("slipped", "dropped", "knocked", "missed_aim", "not_found", "other")
DIAGNOSE_RESERVE_USD = 0.03   # a two-photo Spark call cost $0.0207 (needs.md A5c); never let diagnosis hit the run's cap
EXPLANATION_MAX_CHARS = 300

DIAGNOSE_PROMPT = """You are checking a small robot arm that puts blocks into slots drawn on a sheet of paper.
Photo 1 was taken before the move, photo 2 after it.
The robot tried to put the {name} into slot {slot}. The camera measured where it landed: {distance:.1f} cm from the
slot centre, {dx:+.1f} cm in x and {dy:+.1f} cm in y.

Directions on the sheet: +x points from the bottom-left dot towards the bottom-right dot (usually to the right in
the photo); +y points from the bottom-left dot towards the top-left dot (usually up in the photo).

What went wrong? Choose one problem:
- slipped: it was put down in the right place but slid after the claw let go
- dropped: it fell from the claw before reaching the slot
- knocked: the arm or the block bumped it, or another block, out of place
- missed_aim: the arm put it down in the wrong place
- not_found: you cannot see the {name} in photo 2
- other: anything else
If you cannot see the {name} in photo 2, answer not_found. Do not repeat the measured miss.

Say where YOU see the {name} in photo 2, relative to the centre of slot {slot}, in cm (seen_offset_cm), judged
from the photo itself; use null if you cannot see it.

Then suggest a fix: how far to move the robot's aim for the next try, in cm. Example: if the block landed 1 cm too
far in +x, a fix of dx_cm -1.0 corrects it; a fix in +x would make it worse.
The fix is advice only. The robot's own code checks it, never moves the aim more than 2 cm, and decides whether to
use it.

Reply with ONLY a JSON object and no other text, like this:
{{"problem": "missed_aim", "seen_offset_cm": {{"dx_cm": 1.0, "dy_cm": 0.0}},
 "fix": {{"dx_cm": -1.0, "dy_cm": 0.0, "retry": true}},
 "explanation": "One or two short sentences in plain English for the person watching."}}"""


class DiagnosisError(ValueError):
    pass


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DiagnosisError(f"{what} is not a number: {value!r}")
    return float(value)


def parse_diagnosis(reply: str) -> dict:
    """Spark's reply -> {"problem", "seen_offset_cm", "fix": {"dx_cm", "dy_cm", "retry"}, "explanation"},
    or DiagnosisError. seen_offset_cm is (dx, dy) or None (not given, or Spark cannot see the block)."""
    try:
        data = _json_in(reply, r"\{.*\}", "object")
    except ValueError as error:
        raise DiagnosisError(str(error)) from error
    if not isinstance(data, dict):
        raise DiagnosisError(f"Expected a JSON object, got: {data!r}")
    problem = str(data.get("problem", "")).strip().lower()
    if problem not in PROBLEMS:
        raise DiagnosisError(f"Unknown problem {problem!r}; expected one of {PROBLEMS}")
    fix = data.get("fix")
    if not isinstance(fix, dict) or not isinstance(fix.get("retry"), bool):
        raise DiagnosisError(f"The fix needs dx_cm, dy_cm and retry (true/false): {fix!r}")
    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise DiagnosisError("No explanation in the reply.")
    seen = data.get("seen_offset_cm")
    if seen is not None:
        if not isinstance(seen, dict):
            raise DiagnosisError(f"seen_offset_cm must be an object with dx_cm and dy_cm, or null: {seen!r}")
        seen = (_number(seen.get("dx_cm"), "seen dx_cm"), _number(seen.get("dy_cm"), "seen dy_cm"))
    return {"problem": problem, "seen_offset_cm": seen,
            "fix": {"dx_cm": _number(fix.get("dx_cm"), "dx_cm"), "dy_cm": _number(fix.get("dy_cm"), "dy_cm"),
                    "retry": fix["retry"]},
            "explanation": " ".join(explanation.split())[:EXPLANATION_MAX_CHARS]}


def diagnose_content(before_path: str, after_path: str, move, error_cm) -> list:
    text = DIAGNOSE_PROMPT.format(name=move.name, slot=move.slot, distance=math.hypot(*error_cm),
                                  dx=error_cm[0], dy=error_cm[1])
    return [{"type": "input_text", "text": text},
            {"type": "input_image", "image_url": meta_api.image_url(before_path)},
            {"type": "input_image", "image_url": meta_api.image_url(after_path)}]


def diagnose(before_path: str, after_path: str, move, error_cm) -> dict:
    """One Spark call with both photos and the measured miss. Raises DiagnosisError if the run's cost cap
    is too close to afford it, or if the reply is not a usable diagnosis."""
    spent, cap = meta_api.cost_so_far(), config.BUDGET_USD
    if spent + DIAGNOSE_RESERVE_USD > cap:
        raise DiagnosisError(f"skipped: the run's cost cap is close (${spent:.3f} of ${cap:.2f} spent)")
    reply = meta_api.call(config.SPARK_MODEL, diagnose_content(before_path, after_path, move, error_cm))
    return parse_diagnosis(reply)

