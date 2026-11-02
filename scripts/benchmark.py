"""Run the full loop several times and report the numbers for the README and the video.

    python scripts/benchmark.py            5 runs
    python scripts/benchmark.py --runs 10

The cache is switched off, so every SAM and Spark call is real and the cost per run is honest.
Between runs it waits for you to reset the blocks (type q to stop early). A run that stops with an
error ends the benchmark: something needs looking at before the next run.
Prints success rate, average attempts per block, worst error and cost per run, and saves them to
runs/benchmark_<time>.json. Each run's own log is in runs/<time>/ as usual.
"""
import argparse
import json
import sys
import time

from mimic import arm as arm_module
from mimic import camera, config, kinematics, loop, meta_api, scene
from mimic.arm import Arm, ArmError


def summarise(summaries: list[dict]) -> dict:
    """The benchmark numbers from the runs' summary.json contents."""
    moves = [m for s in summaries for m in s.get("moves", [])]
    finished = [s for s in summaries if not s.get("error")]
    errors = [s["max_error_cm"] for s in finished if s.get("max_error_cm") is not None]
    return {
        "runs": len(summaries),
        "successful_runs": sum(1 for s in summaries if s.get("success")),
        "success_rate": sum(1 for s in summaries if s.get("success")) / len(summaries) if summaries else 0.0,
        "blocks": len(moves),
        "attempts_per_block": sum(s.get("attempts", 0) for s in summaries) / len(moves) if moves else None,
        "worst_error_cm": max(errors) if errors else None,
        "cost_per_run_usd": [s.get("cost_usd", 0.0) for s in summaries],
        "average_cost_usd": sum(s.get("cost_usd", 0.0) for s in summaries) / len(summaries) if summaries else 0.0,
        "stopped_with_error": [s["error"] for s in summaries if s.get("error")],
    }


def report(numbers: dict, say) -> None:
    say(f"Runs: {numbers['runs']}  ({numbers['successful_runs']} fully successful, "
        f"success rate {numbers['success_rate']:.0%})")
    if numbers["attempts_per_block"] is not None:
        say(f"Average attempts per block: {numbers['attempts_per_block']:.2f}  ({numbers['blocks']} blocks)")
    if numbers["worst_error_cm"] is not None:
        say(f"Worst final placement error: {numbers['worst_error_cm']:.2f} cm")
    say("Cost per run: " + ", ".join(f"${c:.4f}" for c in numbers["cost_per_run_usd"])
        + f"  (average ${numbers['average_cost_usd']:.4f})")
    for error in numbers["stopped_with_error"]:
        say(f"Stopped with an error: {error}")


def benchmark(runs: int, run_once, wait_for_user, say=print) -> list[dict]:
    summaries = []
    for n in range(1, runs + 1):
        if n > 1:
            answer = wait_for_user(f"Run {n} of {runs} next. Reset the blocks, then press Enter (q to stop): ")
            if answer.strip().lower() == "q":
                break
        say(f"=== Run {n} of {runs}")
        try:
            summary = run_once()
        except Exception as error:  # the run's own summary.json already has the details
            summaries.append({"success": False, "moves": [], "attempts": 0, "max_error_cm": None,
                              "cost_usd": round(meta_api.cost_so_far(), 4), "error": f"{type(error).__name__}: {error}"})
            say(f"Run {n} stopped: {type(error).__name__}: {error}. Ending the benchmark.")
            break
        summaries.append(summary)
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full loop several times and report the numbers.")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)
    meta_api.cache_enabled = False  # every call real, so the costs are honest
    try:
        layout, grid, limits = loop.load_and_check()  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, kinematics.KinematicsError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)
    try:
        summaries = benchmark(args.runs, lambda: loop.run(arm, camera.capture, layout, grid, limits), input)
    finally:
        arm.transport.close()  # the bridge sends S as it closes
    numbers = summarise(summaries)
    print()
    report(numbers, print)
    out = config.RUNS_DIR / f"benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"numbers": numbers, "runs": summaries}, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
