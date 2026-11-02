"""benchmark.py with made-up run summaries: no arm, no camera, no API calls."""
import importlib.util

import pytest

from mimic import arm, camera, config, meta_api

spec = importlib.util.spec_from_file_location("benchmark", config.ROOT / "scripts/benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def run_summary(success, attempts, max_error, cost, results=("PASS", "PASS", "PASS")):
    return {"success": success, "attempts": attempts, "max_error_cm": max_error, "cost_usd": cost, "error": None,
            "moves": [{"name": f"b{i}", "result": r, "final_error_cm": max_error} for i, r in enumerate(results)]}


def test_numbers_from_three_runs():
    numbers = bench.summarise([run_summary(True, 3, 0.8, 0.04), run_summary(True, 4, 1.2, 0.05),
                               run_summary(False, 7, 3.1, 0.07, ("PASS", "FAIL", "PASS"))])
    assert numbers["runs"] == 3 and numbers["successful_runs"] == 2
    assert numbers["success_rate"] == pytest.approx(2 / 3)
    assert numbers["attempts_per_block"] == pytest.approx(14 / 9)
    assert numbers["worst_error_cm"] == 3.1
    assert numbers["cost_per_run_usd"] == [0.04, 0.05, 0.07]
    assert numbers["average_cost_usd"] == pytest.approx(0.16 / 3)


def test_report_prints_all_four_numbers():
    said = []
    bench.report(bench.summarise([run_summary(True, 3, 0.8, 0.04), run_summary(False, 5, 2.0, 0.06)]), said.append)
    text = "\n".join(said)
    assert "success rate 50%" in text
    assert "Average attempts per block: 1.33" in text
    assert "Worst final placement error: 2.00 cm" in text
    assert "Cost per run: $0.0400, $0.0600  (average $0.0500)" in text


def test_waits_for_the_blocks_to_be_reset_between_runs():
    prompts = []
    runs = iter([run_summary(True, 3, 0.5, 0.04)] * 3)
    summaries = bench.benchmark(3, lambda: next(runs), lambda p: prompts.append(p) or "", say=lambda s: None)
    assert len(summaries) == 3
    assert prompts == ["Run 2 of 3 next. Reset the blocks, then press Enter (q to stop): ",
                       "Run 3 of 3 next. Reset the blocks, then press Enter (q to stop): "]


def test_q_stops_early():
    summaries = bench.benchmark(5, lambda: run_summary(True, 3, 0.5, 0.04), lambda p: "q", say=lambda s: None)
    assert len(summaries) == 1


def test_a_run_that_stops_with_an_error_ends_the_benchmark():
    def run_once():
        raise RuntimeError("SAM found 0 boxes for 'blue block'")
    summaries = bench.benchmark(5, run_once, lambda p: "", say=lambda s: None)
    assert len(summaries) == 1 and summaries[0]["success"] is False
    assert "SAM found 0 boxes" in bench.summarise(summaries)["stopped_with_error"][0]


def test_cache_is_switched_off_and_nothing_opens_while_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(meta_api, "cache_enabled", True)
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))
    assert bench.main([]) == 1  # the committed layout, grid and limits are still placeholders
    assert meta_api.cache_enabled is False
    assert "STOPPED before opening anything" in capsys.readouterr().out
