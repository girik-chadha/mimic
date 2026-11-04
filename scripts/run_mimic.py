"""The full run with the real arm and camera: watch your demo, then rebuild it from the mess.

    python scripts/run_mimic.py
    python scripts/run_mimic.py --push     push mode (opt-in): slide each block with the claw at grip_push
    python scripts/run_mimic.py --ring     ring mode (elbow taped): dice on the claw's ring, calib/ring.json

Needs: the sheet measured (calib/layout.json), the pose grid taught (scripts/teach.py), the arm unlocked
(needs.md C5), iVCam running on Windows. Everything is checked before the serial port is even opened.
Logs go to runs/<time>/ (photos, attempts.jsonl, summary.json).
"""
import argparse
import sys

from mimic import arm as arm_module
from mimic import camera, kinematics, loop, ring, ring_loop, scene, session
from mimic.arm import Arm, ArmError
from mimic.pickdrop import ENABLE_NOTE


def main(argv=None, read=input) -> int:
    parser = argparse.ArgumentParser(description="The full run with the real arm and camera.")
    parser.add_argument("--push", action="store_true",
                        help="push mode: push each block into its slot instead of picking it up (needs grip_push)")
    parser.add_argument("--ring", action="store_true",
                        help="ring mode (elbow taped): pick and drop on the claw's ring (needs calib/ring.json)")
    args = parser.parse_args(argv)
    if args.ring and args.push:
        print("STOPPED: --ring and --push cannot be used together.")
        return 1
    if args.ring:
        return _main_ring(read)
    try:
        layout, grid, limits = loop.load_and_check(push=args.push)  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, kinematics.KinematicsError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)

    def body():
        session.confirm_and_enable(arm, read)  # the board starts disarmed
        summary = loop.run(arm, camera.capture, layout, grid, limits, push=args.push)
        print(f"{'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: "
              f"{summary['attempts']} attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}")
        return 0 if summary["success"] else 2

    # s, Ctrl+C or an error: S then D; a normal end: "hold the arm", then D. The bridge closes either way.
    return session.guarded(arm, body)


def _main_ring(read) -> int:
    try:
        layout, the_ring, limits = ring_loop.load_and_check()  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, ring.RingError, ring_loop.RingLayoutError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)

    def body():
        session.confirm_and_enable(arm, read, note=ENABLE_NOTE)  # the board starts disarmed
        summary = ring_loop.run(arm, camera.capture, layout, the_ring, limits)
        print(f"{'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: "
              f"{summary['attempts']} attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}")
        return 0 if summary["success"] else 2

    return session.guarded(arm, body)


if __name__ == "__main__":
    sys.exit(main())
