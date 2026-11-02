"""arm_lock.py flips the four lock values together. Tested on copies, never on the real files."""
import importlib.util
import shutil

import pytest

from mimic import config

spec = importlib.util.spec_from_file_location("arm_lock", config.ROOT / "scripts/arm_lock.py")
arm_lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm_lock)

HEADER = config.ROOT / "firmware" / "mimic_arm" / "arm_logic.h"


@pytest.fixture
def copies(tmp_path):
    limits, header = tmp_path / "arm_limits.json", tmp_path / "arm_logic.h"
    shutil.copy(config.ARM_LIMITS_PATH, limits)
    shutil.copy(HEADER, header)
    return limits, header


def test_unlock_then_lock_changes_only_the_four_values(copies):
    limits, header = copies
    before = (limits.read_text(), header.read_text())
    code = arm_lock.main(["unlock"], ask=lambda q: "yes", say=lambda s: None, limits_path=limits, header_path=header)
    assert code == 0
    assert '"placeholder": false' in limits.read_text()
    assert '"limits_checked_by_hand": true' in limits.read_text()
    assert "#define LIMITS_ARE_PLACEHOLDER 0" in header.read_text()
    assert "#define LIMITS_CHECKED_BY_HAND 1" in header.read_text()
    arm_lock.main(["lock"], say=lambda s: None, limits_path=limits, header_path=header)
    assert (limits.read_text(), header.read_text()) == before  # back exactly as it was


def test_unlock_needs_yes(copies):
    limits, header = copies
    before = (limits.read_text(), header.read_text())
    code = arm_lock.main(["unlock"], ask=lambda q: "y", say=lambda s: None, limits_path=limits, header_path=header)
    assert code == 1
    assert (limits.read_text(), header.read_text()) == before


def test_nothing_is_written_if_a_file_looks_wrong(copies):
    limits, header = copies
    header.write_text(header.read_text().replace("#define LIMITS_CHECKED_BY_HAND 0", "// removed"))
    before_limits = limits.read_text()
    with pytest.raises(RuntimeError, match="Nothing was changed"):
        arm_lock.set_lock(False, limits, header)
    assert limits.read_text() == before_limits  # the JSON was not half-changed


def test_unknown_argument_shows_usage(copies):
    limits, header = copies
    said = []
    assert arm_lock.main(["open"], say=said.append, limits_path=limits, header_path=header) == 1
    assert "Usage" in said[0]
