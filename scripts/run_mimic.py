"""The full run with the real arm and camera: watch your demo, then rebuild it from the mess.

    python scripts/run_mimic.py
    python scripts/run_mimic.py --push     push mode (opt-in): slide each block with the claw at grip_push
    python scripts/run_mimic.py --ring     ring mode (elbow taped): dice on the claw's ring, calib/ring.json
    --camera-frames N                      frames per photo grab, keeping the last (ring mode: camera_frames in
                                           ring.json; otherwise 15)
    --camera-name NAME                     the Windows webcam, e.g. Camo (else MIMIC_CAMERA, then, in ring mode,
                                           camera_name in ring.json, then "e2eSoft iVCam")

    python scripts/run_mimic.py --ring --sim --photo tests/fixtures/photos/dice_sheet.jpg
    --instruction "row black, blue, red"   simulation only: the target typed (default: the photo order, left to right)
    --sim-miss-deg 6                       simulation only: put the first die down 6 degrees off once (shown as an
                                           INJECTED TEST MISS), so the camera check and retry show

Simulation mode (mimic/sim.py): real SAM and Spark calls on one real photo and the real ring-mode planner, but the
moves go to a simulated arm on a MADE-UP ring. Nothing is uploaded and no serial port or camera is opened. Outputs go
to runs/<time>_simulation/ (simulation_video.mp4, simulation_moves.json, ...), every name marked "simulation"; then
scripts/report.py writes simulation_report.html and simulation_summary.png there.
    --instruction "stack black, blue, red" piles the dice on slot 1 (black at the bottom): SIMULATION ONLY, and refused
                                           without --sim, because the real arm cannot stack with the elbow taped.

Needs: the sheet measured (calib/layout.json), the pose grid taught (scripts/teach.py), the arm unlocked
(needs.md C5), the phone webcam app (iVCam or Camo) running on Windows. Everything is checked before the serial port is even opened.
Logs go to runs/<time>/ (photos, attempts.jsonl, summary.json).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

from mimic import arm as arm_module
from mimic import brain, camera, kinematics, loop, meta_api, ring, ring_loop, scene, session, sim, sim_video
from mimic.arm import Arm, ArmError


def main(argv=None, read=input) -> int:
    parser = argparse.ArgumentParser(description="The full run with the real arm and camera.")
    parser.add_argument("--push", action="store_true",
                        help="push mode: push each block into its slot instead of picking it up (needs grip_push)")
    parser.add_argument("--ring", action="store_true",
                        help="ring mode (elbow taped): pick and drop on the claw's ring (needs calib/ring.json)")
    parser.add_argument("--camera-frames", type=int, help="frames per photo grab, keeping the last")
    parser.add_argument("--camera-name", help="the Windows webcam, e.g. Camo")
    parser.add_argument("--sim", action="store_true",
                        help="with --ring: simulated arm on a made-up ring; real SAM and Spark on --photo")
    parser.add_argument("--photo", help="--sim: the real photo SAM and Spark look at")
    parser.add_argument("--instruction", help='--sim: the target typed, e.g. "row black, blue, red"')
    parser.add_argument("--sim-miss-deg", type=float, default=0.0,
                        help="--sim: put the first die down this many degrees off once (an INJECTED TEST MISS)")
    args = parser.parse_args(argv)
    if args.ring and args.push:
        print("STOPPED: --ring and --push cannot be used together.")
        return 1
    problem = _sim_option_problem(args)
    if problem:
        print(f"STOPPED: {problem}")
        return 1
    if args.sim:
        return _main_sim(args.photo, args.instruction, args.sim_miss_deg)
    if args.ring:
        return _main_ring(read, args.camera_frames, args.camera_name)
    try:
        layout, grid, limits = loop.load_and_check(push=args.push)  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, kinematics.KinematicsError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)

    def body():
        session.confirm_and_enable(arm, read)  # the board starts disarmed
        frames, name = args.camera_frames or camera.FRAMES, camera.camera_name(args.camera_name)
        summary = loop.run(arm, lambda path: camera.capture(path, frames=frames, name=name), layout, grid, limits,
                           push=args.push)
        print(f"{'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: "
              f"{summary['attempts']} attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}")
        return 0 if summary["success"] else 2

    # s, Ctrl+C or an error: S then D; a normal end: "hold the arm", then D. The bridge closes either way.
    return session.guarded(arm, body)


def _sim_option_problem(args) -> str | None:
    """The simulation's options only make sense together; a stack can never go to the real arm."""
    if args.instruction and args.instruction.strip().lower().startswith("stack") and not args.sim:
        return ("stacking is simulation-only: with the elbow taped, lifting the claw also moves it along the "
                "radius, so the real arm cannot put a die on top of another. Add --ring --sim.")
    if not args.sim:
        extra = [o for o, v in (("--photo", args.photo), ("--instruction", args.instruction),
                                ("--sim-miss-deg", args.sim_miss_deg)) if v]
        return f"{', '.join(extra)} only work with --ring --sim." if extra else None
    if not args.ring:
        return "--sim runs ring mode: use --ring --sim."
    if not args.photo:
        return "--sim needs --photo, the real photo SAM and Spark look at."
    return None


def _main_sim(photo, instruction, miss_deg) -> int:
    """Simulation mode: no serial port, no camera, no upload. Real SAM and Spark calls on the photo."""
    try:
        summary = sim.run(photo, instruction, miss_deg)
    except (sim.SimError, sim_video.SimVideoError, scene.SceneError, ring.RingError, ring_loop.RingLayoutError,
            ArmError, brain.NamingFailed, meta_api.BudgetExceeded, ValueError) as error:
        print(f"STOPPED (simulation): {error}")
        return 1
    try:
        _write_report(summary["out_dir"])
    except Exception as error:
        print(f"The report could not be written: {type(error).__name__}: {error}")
        return 1
    placed = sum(1 for m in summary["moves"] if m["result"] == "PASS")
    print(f"SIMULATION {'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: {placed} of "
          f"{len(summary['moves'])} dice {'stacked' if summary.get('stack') else 'placed'}, {summary['attempts']} "
          f"attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}. Outputs: {summary['out_dir']}")
    return 0 if summary["success"] else 2


def _write_report(out_dir) -> None:
    """scripts/report.py on the simulation's folder (simulation_report.html, simulation_summary.png)."""
    spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parent / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    if report.main([str(out_dir)]) != 0:
        raise RuntimeError(f"report.py found no simulation_summary.json in {out_dir}")


def _main_ring(read, camera_frames=None, camera_name=None) -> int:
    try:
        layout, the_ring, limits = ring_loop.load_and_check()  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, ring.RingError, ring_loop.RingLayoutError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)

    def body():
        session.confirm_and_enable(arm, read)  # the board starts disarmed; enable starts at the start pose
        frames = camera_frames or the_ring.camera_frames
        name = camera.camera_name(camera_name, the_ring.camera_name)
        summary = ring_loop.run(arm, lambda path: camera.capture(path, frames=frames, name=name), layout, the_ring,
                                limits)
        print(f"{'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: "
              f"{summary['attempts']} attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}")
        return 0 if summary["success"] else 2

    return session.guarded(arm, body)


if __name__ == "__main__":
    sys.exit(main())
