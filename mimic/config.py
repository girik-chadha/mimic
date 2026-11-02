"""All settings in one place: models, prices, budget, folders. Reads .env from the repo root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
REQUEST_TIMEOUT_S = 60  # give up on a single request after this long

# Prices in US dollars, from Meta's pricing page (docs/needs.md A4).
SAM_PER_IMAGE = 2.50 / 1000
SAM_PER_VIDEO_FRAME = 0.20 / 1000
SAM_MIN_VIDEO_FRAMES = 50
SPARK_PER_INPUT_TOKEN = 1.25 / 1_000_000
SPARK_PER_OUTPUT_TOKEN = 4.25 / 1_000_000

# Hard cap per run, enforced in code (Meta's spend limit only sends a notification).
BUDGET_USD = float(os.environ.get("MIMIC_BUDGET_USD", "1.00"))

CACHE_DIR = ROOT / ".cache"
RESPONSES_DIR = ROOT / "tests" / "fixtures" / "meta_responses"
RUNS_DIR = ROOT / "runs"
LAYOUT_PATH = ROOT / "calib" / "layout.json"  # stickers, slots, sheet size
ARM_LIMITS_PATH = ROOT / "calib" / "arm_limits.json"  # pins, joint ranges, grip limit

# The phone camera (needs.md A9): iVCam on Windows, read by Windows ffmpeg (installed with winget).
# The winget folder name contains the version, so it can be overridden in .env after an upgrade.
FFMPEG_PATH = os.environ.get("MIMIC_FFMPEG") or (
    "/mnt/c/Users/girik/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe")
CAMERA_DEVICE = "e2eSoft iVCam"

# Spark's fix for a missed placement is logged only, unless MIMIC_USE_SPARK_FIX=1 (needs.md A5c).
USE_SPARK_FIX = os.environ.get("MIMIC_USE_SPARK_FIX") == "1"


def api_key() -> str:
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("META_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key
