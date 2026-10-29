"""The one door to Meta's models.

call(model, content) sends a request and returns the reply text. On the way it:
- answers from the disk cache if we asked exactly this before (free, works offline),
- refuses to start a call once the run's budget is used up,
- retries when the server is busy (429) or broken (5xx),
- adds the cost of the call to the counter and to .cache/spend_log.jsonl,
- replaces the API key with *** in any error before raising it.
"""
import argparse
import base64
import hashlib
import json
import mimetypes
import os
import tempfile
import time

import cv2
import openai

from mimic import config


class BudgetExceeded(RuntimeError):
    pass


RETRY_WAITS_S = [1, 2, 4]  # wait this long before retry 1, 2 and 3
SPARK_TOKENS_PER_IMAGE_GUESS = 1000  # only used if the reply has no token counts (needs.md B10)

_client = None
_spent_usd = 0.0


# ---------- turning files into something we can send ----------

def _data_url(path: str, default_mime: str) -> str:
    mime = mimetypes.guess_type(path)[0] or default_mime
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def image_url(path: str) -> str:
    return _data_url(path, "image/jpeg")


def video_url(path: str) -> str:
    return _data_url(path, "video/mp4")


# ---------- money ----------

def cost_so_far() -> float:
    """Dollars spent on paid calls since the last reset() (cache hits are free)."""
    return _spent_usd


def reset() -> None:
    """Start a new run from $0. loop.run() calls this first."""
    global _spent_usd
    _spent_usd = 0.0


def _video_frames(data_url: str) -> int:
    # OpenCV only reads videos from files, so write a temporary copy and count its frames.
    video_bytes = base64.b64decode(data_url.split(",", 1)[1])
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(video_bytes)
        f.flush()
        return int(cv2.VideoCapture(f.name).get(cv2.CAP_PROP_FRAME_COUNT))


def sam_cost(content: list) -> float:
    cost = 0.0
    for part in content:
        if part["type"] == "input_image":
            cost += config.SAM_PER_IMAGE
        elif part["type"] == "input_video":
            frames = max(_video_frames(part["video_url"]), config.SAM_MIN_VIDEO_FRAMES)
            cost += frames * config.SAM_PER_VIDEO_FRAME
    return cost


def spark_cost(content, reply_text: str, usage) -> tuple[float, bool]:
    """Returns (cost, estimated). Uses the token counts in the reply when Meta sends them."""
    if usage is not None and getattr(usage, "input_tokens", None) is not None:
        tokens_in, tokens_out, estimated = usage.input_tokens, usage.output_tokens, False
    else:
        # Rough estimate: about 4 characters per token, plus a guess for each image.
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        chars = sum(len(p["text"]) for p in parts if p["type"] == "input_text")
        images = sum(1 for p in parts if p["type"] == "input_image")
        tokens_in = chars // 4 + images * SPARK_TOKENS_PER_IMAGE_GUESS
        tokens_out = len(reply_text) // 4
        estimated = True
    cost = tokens_in * config.SPARK_PER_INPUT_TOKEN + tokens_out * config.SPARK_PER_OUTPUT_TOKEN
    return cost, estimated


def _log_spend(model: str, cost: float, estimated: bool) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
            "cost_usd": round(cost, 6), "estimated": estimated, "run_total_usd": round(_spent_usd, 6)}
    with open(config.CACHE_DIR / "spend_log.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


# ---------- cache ----------

def _cache_key(model: str, content) -> str:
    # Replace each (huge) data URL by its hash, so the key depends on the file's bytes, not its name.
    def shrink(value):
        if isinstance(value, str) and value.startswith("data:"):
            return "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        if isinstance(value, list):
            return [shrink(v) for v in value]
        if isinstance(value, dict):
            return {k: shrink(v) for k, v in value.items()}
        return value
    text = json.dumps({"model": model, "content": shrink(content)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- the call ----------

def _get_client():
    global _client
    if _client is None:
        # max_retries=0: our own retry loop below is the only one, so it is visible and tested.
        _client = openai.OpenAI(api_key=config.api_key(), base_url=config.BASE_URL,
                                timeout=config.REQUEST_TIMEOUT_S, max_retries=0)
    return _client


def _worth_retrying(error: Exception) -> bool:
    if isinstance(error, openai.APIConnectionError):  # includes timeouts
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def _scrub(error: Exception) -> Exception:
    """Replace the API key with *** in the error's text, and in any error that caused it."""
    key = os.environ.get("META_API_KEY")
    current = error
    for _ in range(10):  # follow the "caused by" chain, at most 10 deep
        if current is None or not key:
            break
        current.args = tuple(a.replace(key, "***") if isinstance(a, str) else a for a in current.args)
        if isinstance(getattr(current, "message", None), str):
            current.message = current.message.replace(key, "***")
        current = current.__cause__ or current.__context__
    return error


def _send(model: str, content):
    request = content if isinstance(content, str) else [{"role": "user", "content": content}]
    for attempt in range(len(RETRY_WAITS_S) + 1):
        try:
            return _get_client().responses.create(model=model, input=request)
        except Exception as error:
            if attempt == len(RETRY_WAITS_S) or not _worth_retrying(error):
                raise _scrub(error)
            time.sleep(RETRY_WAITS_S[attempt])


def call(model: str, content, save_as: str | None = None, use_cache: bool = True) -> str:
    """Send content (a text string, or a list of input_text / input_image / input_video parts).

    use_cache=False neither reads nor writes the cache, so the benchmark sees real costs and times.
    """
    global _spent_usd
    if model.endswith("-contributor"):
        raise ValueError("Never use -contributor models: Meta may train on what we send.")
    if model not in (config.SAM_MODEL, config.SPARK_MODEL):
        raise ValueError(f"Unknown model {model!r}: add it and its price to config.py first.")

    cache_file = config.CACHE_DIR / f"{_cache_key(model, content)}.json"
    if use_cache and cache_file.exists():
        text = json.loads(cache_file.read_text())["text"]
    else:
        if _spent_usd >= config.BUDGET_USD:
            raise BudgetExceeded(f"Spent ${_spent_usd:.4f} of the ${config.BUDGET_USD:.2f} budget.")
        resp = _send(model, content)
        text = resp.output_text
        if model == config.SAM_MODEL:
            cost, estimated = sam_cost(content), False
        else:
            cost, estimated = spark_cost(content, text, getattr(resp, "usage", None))
        _spent_usd += cost
        _log_spend(model, cost, estimated)
        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": model, "text": text, "cost_usd": cost}))

    if save_as:
        config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESPONSES_DIR / f"{save_as}.txt").write_text(text)
    return text


# ---------- command line: one call, optionally saved as a test fixture ----------

def main():
    parser = argparse.ArgumentParser(description="Send one prompt (and optionally a photo or video) to a Meta model.")
    parser.add_argument("model", help="sam-3.1 or muse-spark-1.3")
    parser.add_argument("prompt")
    parser.add_argument("file", nargs="?", help="a photo or an .mp4 video")
    parser.add_argument("--save", metavar="NAME", help="also save the reply to tests/fixtures/meta_responses/NAME.txt")
    args = parser.parse_args()

    if args.file is None:
        content = args.prompt
    elif args.file.lower().endswith(".mp4"):
        content = [{"type": "input_text", "text": args.prompt},
                   {"type": "input_video", "video_url": video_url(args.file)}]
    else:
        content = [{"type": "input_text", "text": args.prompt},
                   {"type": "input_image", "image_url": image_url(args.file)}]

    text = call(args.model, content, save_as=args.save)
    print(text if len(text) <= 1500 else text[:1500] + f"\n... [{len(text)} characters in total]")
    print(f"\ncost this run: ${cost_so_far():.4f}")


if __name__ == "__main__":
    main()
