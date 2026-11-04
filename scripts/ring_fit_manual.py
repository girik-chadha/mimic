"""Fit the ring from ruler measurements, with the elbow taped (needs.md A10d): no arm, no camera, no SAM.

    python scripts/ring_fit_manual.py --point 80 16.1 18.2 --point 90 18.5 18.0 --point 100 20.9 18.2 \\
        --point 110 23.3 18.9 --point 120 25.5 19.9

Each --point is a base angle and where the claw put a die's centre at that angle, in sheet cm from the red dot's
centre (bottom_left dot), measured with a ruler. At least 5 points, spread across the base range you will use.
The fit is mimic/ring.py's, exactly as in ring_calibrate.py: centre, radius and base scale; any point more than
0.5 cm from the fitted ring refuses it, naming the point that spoils it if leaving one out lets the others fit.
Saved to calib/ring.json with source "manual" and placeholder false; slots_deg, camera_clear and the pick numbers
already there are kept. A ruler measures the die where it sits, so there is no camera height parallax.
"""
import argparse
import sys
import time

from mimic import ring

POINTS_NEEDED = 5   # the same as ring_calibrate.py


def main(argv=None, say=print, ring_path=ring.RING_PATH) -> int:
    parser = argparse.ArgumentParser(description="Fit the ring from ruler-measured points (base angle, x, y in cm).")
    parser.add_argument("--point", nargs=3, type=float, action="append", default=[], metavar=("BASE", "X", "Y"),
                        help="a base angle and the die's centre in sheet cm from the red dot (repeat, 5 or more)")
    args = parser.parse_args(argv)
    points = [(int(b) if b.is_integer() else b, x, y) for b, x, y in args.point]
    if len(points) < POINTS_NEEDED:
        say(f"STOPPED: {POINTS_NEEDED} --point values are needed; there are {len(points)}. Nothing was saved.")
        return 1
    try:
        the_ring = ring.load_ring(ring_path)
        result = ring.fit(points)
    except ring.RingError as error:
        say(f"STOPPED, nothing was saved: {error}")
        return 1
    for line in ring.describe(result, points):
        say(line)
    if not the_ring.placeholder:
        say(f"Replacing the fit in {ring_path} (it came from: {the_ring.source or 'an earlier run'}).")
    note = (f"FITTED by scripts/ring_fit_manual.py on {time.strftime('%Y-%m-%d %H:%M')} from {len(points)} "
            "ruler-measured points (base angle, die centre in sheet cm from the red dot). slots_deg and camera_clear "
            "are set by hand. See mimic/ring.py and needs.md A10d.")
    ring.save_ring(ring.with_fit(the_ring, result, points, note, source="manual"), ring_path)
    say(f"Saved {ring_path} (source manual).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
