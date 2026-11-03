"""The demo, shown on video: film yourself putting the blocks into the slots, get the target order out.

    python scripts/show_demo.py demo.mp4

1. Shrinks the video with Windows ffmpeg (10 frames a second, at most 720p, no sound, no metadata), then refuses
   it before anything is sent if it could cost more than $0.30.
2. Warns that the video goes to Meta (hands, faces, anything in view) and asks you to type yes.
3. The video's last frame is calibrated from the 6 dots and Spark names the blocks on it, so end the video with
   the blocks in their slots and your hands out of view for a second.
4. One SAM video call per name: SAM follows each block through the video.
5. Picks the last frame where every block has been still for 1 s (10 frames) and all 6 dots are found (up to 3
   frames are tried, 1 s apart), and reads the target order there, as plan_from_photos.py does with a photo.
   Stops if a block ends the video more than the slot tolerance from where it was in that frame (the video ended
   before the last block settled).
6. Logs the order the blocks arrived in, first to settle first. It is only logged: the row task uses slot order.

Saves the shrunk video, the frames used, their labelled dots and demo.json to runs/<date-time>/.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from mimic import brain, calib, config, meta_api, sam, scene, video

RUN_BUDGET_USD = 0.30   # hard cap for one run of this script
FRAMES_TO_TRY = 3       # settled frames tried for the 6 dots, each 1 s before the last

WARNING = """WARNING: the whole video goes to Meta (SAM 3.1), once per block. Hands, faces and anything else in
view are sent. While we use free credits, Meta may use what we send to improve its products (needs.md A1).
Keep faces and anything private out of the video. Sound and metadata (such as location) were removed."""


class Cancelled(RuntimeError):
    pass


def check_dot_errors(c: calib.Calib) -> dict[str, float]:
    return {role: round(c.check(px, cm), 3) for role, px, cm in zip(scene.ROLES, c.pixel_pts, c.table_pts_cm)
            if role in scene.CHECKS}


def run(video_path: str, layout_path=config.LAYOUT_PATH, out_root=config.RUNS_DIR, ask=input, say=print,
        save_replies: bool = False) -> dict:
    layout = scene.load_layout(layout_path)
    scene.refuse_placeholder(layout)  # before anything is shrunk or sent
    out_dir = Path(out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_path = out_dir / "demo_sent.mp4"
    info = video.shrink(video_path, sent_path)  # local: Windows ffmpeg, nothing sent yet
    blocks = len(layout.slots_cm)
    estimate = video.estimate_cost(info.frames, blocks)
    say(f"Video to send: {info.frames} frames, {info.seconds:.1f} s at {info.fps:g} fps, {info.width}x{info.height}")
    if estimate > RUN_BUDGET_USD:
        longest = video.max_frames(RUN_BUDGET_USD, blocks) / info.fps
        raise video.VideoError(f"This video could cost ${estimate:.2f} with {blocks} blocks, over the "
                               f"${RUN_BUDGET_USD:.2f} cap. Nothing was sent. Record at most {longest:.0f} s.")
    say(WARNING)
    say(f"It could cost up to ${estimate:.2f} (cap ${RUN_BUDGET_USD:.2f}).")
    if ask("Type yes to send it: ").strip().lower() != "yes":
        raise Cancelled("Nothing was sent.")

    meta_api.reset()
    budget_before = config.BUDGET_USD
    config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)
    calibrations = {}

    def calibrate_frame(frame: int) -> calib.Calib:
        if frame not in calibrations:
            path = video.extract_frame(sent_path, frame, out_dir / f"frame_{frame:04d}.jpg")
            calibrations[frame] = scene.calibrate(path, layout, out_dir / f"frame_{frame:04d}_dots.jpg")
        return calibrations[frame]

    def frame_path(frame: int) -> str:
        return str(out_dir / f"frame_{frame:04d}.jpg")

    try:
        # The last frame first: cheap, and it stops a hidden dot or wrong layout before the video calls.
        last = info.frames - 1
        try:
            calibrate_frame(last)
        except (scene.SceneError, calib.CalibrationError) as error:
            raise video.VideoError(f"The video's last frame does not calibrate: {error} End the video with all 6 "
                                   "dots in view and your hands away.") from None
        attempts = []
        names = brain.name_objects(frame_path(last), attempts)
        say("Spark named (on the last frame): " + ", ".join(names))
        if len(names) > blocks:
            raise video.VideoError(f"Spark named {len(names)} objects but the sheet has {blocks} slots.")

        tracks = {}
        for name in names:
            save_as = f"sam_video_demo_{name.replace(' ', '_')}_{info.frames}f" if save_replies else None
            found, reply_frames = sam.track(str(sent_path), name, save_as=save_as)
            tracks[name] = video.check_track(name, found, reply_frames, info)
            say(f"  {name}: seen in {len(tracks[name])} of {info.frames} frames")

        settled = video.settled_frames(tracks, info)
        if not settled:
            raise video.VideoError(f"There is no moment where all {len(names)} blocks had been still for "
                                   f"{video.STILL_S:g} s. Hold still, hands away, for at least 1 s at the end.")
        problems, chosen = [], None
        for frame in video.frames_to_try(settled, info, FRAMES_TO_TRY):
            try:
                demo_map = calibrate_frame(frame)
                chosen = frame
                break
            except (scene.SceneError, calib.CalibrationError) as error:
                problems.append(f"frame {frame} ({frame / info.fps:.1f} s): {error}")
        if chosen is None:
            raise video.VideoError("No settled frame had all 6 dots: " + " | ".join(problems))
        say(f"Reading frame {chosen} ({chosen / info.fps:.1f} s): every block still for {video.STILL_S:g} s")

        positions = scene.read_positions(frame_path(chosen), names, demo_map, layout)
        for name in names:  # the photo of that frame and the video track must be the same object
            track_box = dict(tracks[name])[chosen]
            track_cm = scene.correct_for_height(demo_map.px_to_cm(track_box.centre), layout.height)
            apart = math.dist(track_cm, positions[name])
            if apart > video.AGREE_CM:
                raise video.VideoError(f"{name!r}: SAM's photo of frame {chosen} and its video track are "
                                       f"{apart:.1f} cm apart: it may have followed a different object.")
        end_map = calibrations[last]
        for name in names:  # the frame read must show the final arrangement: nothing may move after it
            end_box = tracks[name][-1][1]
            end_cm = scene.correct_for_height(end_map.px_to_cm(end_box.centre), layout.height)
            moved = math.dist(end_cm, positions[name])
            if moved > layout.slot_tolerance_cm:
                raise video.VideoError(f"{name!r} ends the video {moved:.1f} cm from where it was in frame {chosen}, "
                                       "the last moment everything was still: the blocks moved after it. Hold still, "
                                       "hands away, for at least 1 s at the end.")
        target = scene.assign_slots(positions, layout.slots_cm, layout.slot_tolerance_cm)
        say("Target order (slot 1 first): " + ", ".join(target))

        arrived = video.arrival_frames(tracks, chosen, info)
        order = video.arrival_order(arrived, target)
        say("Arrived (first to settle first): " + ", ".join(
            f"{name} {arrived[name] / info.fps:.1f} s" for name in order))

        result = {
            "video": str(video_path), "sent": {"frames": info.frames, "fps": info.fps, "width": info.width,
                                               "height": info.height},
            "names": names, "naming_attempts": attempts,
            "frames_seen": {name: len(seen) for name, seen in tracks.items()},
            "chosen_frame": chosen, "chosen_time_s": round(chosen / info.fps, 2), "frames_not_used": problems,
            "check_dot_errors_cm": check_dot_errors(demo_map),
            "positions_cm": positions, "target_order": target,
            "arrival_frames": arrived, "arrival_order": order,
            "cost_usd": meta_api.cost_so_far(), "out_dir": str(out_dir),
        }
        (out_dir / "demo.json").write_text(json.dumps(result, indent=2))
        say(f"Cost this run: ${meta_api.cost_so_far():.4f} (cap ${config.BUDGET_USD:.2f})")
        say(f"Saved: {out_dir}")
        return result
    finally:
        config.BUDGET_USD = budget_before


def main(argv=None, ask=input, say=print) -> int:
    parser = argparse.ArgumentParser(description="A demo video in, the target order out.")
    parser.add_argument("video", help="video of you putting the blocks into the slots (ends with hands away)")
    parser.add_argument("--layout", default=str(config.LAYOUT_PATH), help="sheet layout (default calib/layout.json)")
    parser.add_argument("--save-replies", action="store_true",
                        help="also save SAM's video replies in tests/fixtures/meta_responses/")
    args = parser.parse_args(argv)
    try:
        run(args.video, args.layout, ask=ask, say=say, save_replies=args.save_replies)
        return 0
    except Cancelled as error:
        say(f"Cancelled. {error}")
        return 1
    except (video.VideoError, scene.SceneError, calib.CalibrationError, brain.NamingFailed,
            meta_api.BudgetExceeded, meta_api.VideoTimedOut) as error:
        say(f"STOPPED: {error}")
        say(f"Cost so far: ${meta_api.cost_so_far():.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
