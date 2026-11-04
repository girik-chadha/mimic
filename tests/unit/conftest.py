"""Shared by every unit test: nothing from the shell running the tests may change them."""
import pytest


@pytest.fixture(autouse=True)
def no_camera_from_the_shell(monkeypatch):
    """MIMIC_CAMERA (for example "Camo") picks the webcam on the real setup; the tests must not depend on it."""
    monkeypatch.delenv("MIMIC_CAMERA", raising=False)
