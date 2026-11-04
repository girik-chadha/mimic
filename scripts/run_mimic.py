"""The full run with the real arm and camera: watch your demo, then rebuild it from the mess.

    python scripts/run_mimic.py
    python scripts/run_mimic.py --push     push mode (opt-in): slide each block with the claw at grip_push
    python scripts/run_mimic.py --ring     ring mode (elbow taped): dice on the claw's ring, calib/ring.json
    --camera-frames N                      frames per photo grab, keeping the last (ring mode: camera_frames in
                                           ring.json; otherwise 15)
    --camera-name NAME                     the Windows webcam, e.g. Camo (else MIMIC_CAMERA, then, in ring mode,
                                           camera_name in ring.json, then "e2eSoft iVCam")

Needs: the sheet measured (calib/layout.json), the pose grid taught (scripts/teach.py), the arm unlocked
(needs.md C5), the phone webcam app (iVCam or Camo) running on Windows. Everything is checked before the serial port is even opened.
Logs go to runs/<time>/ (photos, attempts.jsonl, summary.json).
"""
import argparse
import sys

from mimic import arm as arm_module
from mimic import camera, kinematics, loop, ring, ring_loop, scene, session
from mimic.arm import Arm, ArmError


def main(argv=None, read=input) -> int:
    parser = argparse.ArgumentParser(description="The full run with the real arm and camera.")
    parser.add_argument("--push", action="store_true",
                        help="push mode: push each block into its slot instead of picking it up (needs grip_push)")
    parser.add_argument("--ring", action="store_true",
                        help="ring mode (elbow taped): pick and drop on the claw's ring (needs calib/ring.json)")
    parser.add_argument("--camera-frames", type=int, help="frames per photo grab, keeping the last")
    parser.add_argument("--camera-name", help="the Windows webcam, e.g. Camo")
    args = parser.parse_args(argv)
    if args.ring and args.push:
        print("STOPPED: --ring and --push cannot be used together.")
        return 1
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
