from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_imports():
    import mimic
    assert mimic.__doc__


def test_saved_video_reply_is_in_fixtures():
    reply = ROOT / "tests/fixtures/meta_responses/sam_video_317f.txt"
    assert reply.exists()


def test_env_file_is_gitignored():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert ".env" in ignored
