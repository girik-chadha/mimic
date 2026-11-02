"""The full run with the real arm and camera: watch your demo, then rebuild it from the mess.

    python scripts/run_mimic.py

Needs: the sheet measured (calib/layout.json), the pose grid taught (scripts/teach.py), the arm unlocked
(needs.md C5), iVCam running on Windows. Everything is checked before the serial port is even opened.
Logs go to runs/<time>/ (photos, attempts.jsonl, summary.json).
"""
import sys

from mimic import arm as arm_module
from mimic import camera, kinematics, loop, scene
from mimic.arm import Arm, ArmError


def main() -> int:
    try:
        layout = scene.load_layout()
        grid = kinematics.load_grid()
        limits = arm_module.load_limits()
        scene.refuse_placeholder(layout)
        grid.check_ready()
        grid.check_against(limits)
        arm_module.refuse_unchecked_limits(limits)  # before the bridge opens the serial port
    except (ArmError, scene.SceneError, kinematics.KinematicsError) as error:
        print(f"STOPPED before opening anything: {error}")
        return 1
    arm = Arm(arm_module.SerialBridgeTransport(), limits)
    try:
        summary = loop.run(arm, camera.capture, layout, grid, limits)
        print(f"{'SUCCESS' if summary['success'] else 'FINISHED WITH FAILURES'}: "
              f"{summary['attempts']} attempts, worst error {summary['max_error_cm']} cm, ${summary['cost_usd']}")
        return 0 if summary["success"] else 2
    except Exception as error:
        print(f"STOPPED: {type(error).__name__}: {error}")
        return 1
    finally:
        arm.transport.close()  # the bridge sends S as it closes


if __name__ == "__main__":
    sys.exit(main())
