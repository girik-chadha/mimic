# Mimic: verified facts, open questions, and setup

Everything in Part A was tested on Wed 28 Oct 2026 with real API calls from this laptop, or read from a source named next to it. Items in Part B are **unverified** and each has a test that settles it. When something in Part B is settled, move it to Part A with the date and the result.

---

## Part A: Verified

### A1. Meta Model API basics (tested 28 Oct)

- Base URL `https://api.meta.ai/v1`, called with the OpenAI Python SDK: `OpenAI(api_key=..., base_url="https://api.meta.ai/v1")`, then `client.responses.create(model=..., input=...)`. The reply text is `resp.output_text`.
- `client.models.list()` works. Models visible to our key:
  `muse-wonder-1.0`, `sam-3.1`, `muse-spark-1.3`, `muse-spark-1.2`, `muse-spark-1.1`, `muse-spark-1.3-contributor`, `muse-spark-1.2-contributor`, `muse-voice-transcribe-1.0`, `muse-image-1.0`.
- **Never use the `-contributor` models.** Contributor mode lets Meta train on what we send.
- The account runs on $150 of free credits. The billing page says that while free credits are in use, content may be used for product improvement. So: keep faces, people and anything private out of every photo and video we send.

### A2. SAM 3.1 on a photo (tested 28 Oct)

Request:

```python
client.responses.create(model="sam-3.1", input=[{"role": "user", "content": [
    {"type": "input_text", "text": "blue lighter"},
    {"type": "input_image", "image_url": "data:image/jpeg;base64,..."},
]}])
```

- One match per object looks like `<|box;x1=86;y1=749;x2=225;y2=823;w=1920;h=1080|>` followed by `<|mask;...|>`. Box values are pixels; `w`, `h` are the image size. Matches are separated by `,N` where N is the match id.
- The mask is a long encoded outline. We ignore it for now; the box centre is enough to locate a block. Meta's `meta-sam-parser` package can decode masks if we ever need outlines.
- **One short name per object works.** On a photo of a blue lighter, a green lighter and a black rectangle, the prompts `"blue lighter"`, `"green lighter"`, `"black rectangle"` each returned exactly 1 match at the right place. See `desk.jpeg`.
- A generic prompt (`"building"`) returns every match in the image.
- **Reply format, checked on 5 saved replies (29 Oct), corrects the line above about `,N`:** a photo reply has
  the same shape as a video reply. It starts with `<0f>` (frame 0); each match is its id followed by the box and
  mask, the first with no comma: `<0f>0<|box;...|><|mask;...|>,1<|box;...|><|mask;...|>,2<|box;...`.
  The example box above (`x1=86;y1=749`) is match 1 of the "building" reply on `test.jpg`, not the lighter.
- **No match is an empty reply** (0 characters): `"red cup"` on `desk.jpeg` (29 Oct).
- **Mask text uses 86 characters and never `<`, `>`, `|`, `,`, `;`, `"` or `\`** (all 6 saved replies, 323 masks,
  29 Oct). `sam.py` relies on there being no `<` or `>`; `tests/unit/test_sam.py` re-checks every saved reply.
- `desk.jpeg` is 1280x960. Boxes on it (29 Oct): blue lighter (255,363)-(320,583), green lighter
  (602,374)-(677,591), black rectangle (848,414)-(969,579). Fixtures in `tests/fixtures/meta_responses/`.

### A2b. SAM 3.1 finds the calibration dots (tested 29-30 Oct)

- Colour names do not work for small dots: SAM cannot tell red from orange or blue from purple (Girik, 29 Oct).
- **One prompt finds them all:** `"small coloured dot"` returned exactly 5 boxes on `dots_check.jpg` (Girik, 29 Oct)
  and exactly 6 on each of `dots_check_6a.jpg` and `dots_check_6b.jpg` (30 Oct), every box a dot, no specks.
  `"dot"` returned 7, including two table specks outside the sheet (Girik, 29 Oct).
- Dot boxes are 19-25 px (longer side) in a 1280-wide photo, i.e. 0.0148-0.0195 of the width, in both photos.
  A table speck was 15 px (0.0117). `layout.json` keeps 0.012-0.026.
- With the provisional dot positions, the right check dot lands at x = 34.98 cm (6a) and 35.06 cm (6b), against
  34.5 in layout.json: 6a calibrates (0.49 cm), 6b is refused (0.56 cm). Both photos agree, which points at the
  provisional number; the ruler measurement replaces it.
- **Real iVCam frame with the arm in place (3 Nov, `ivcam_arm.jpg`, 1280x720):** "small coloured dot" gave exactly
  the 6 dots, boxes 25-33 px (0.0195-0.0258; the orange check dot is biggest because of its white paper halo).
  The arm at the top edge hid no dot. Roles came out right, but the right check dot lands at x = 35.07 cm and the
  centre one at 18.67, so the photo is refused (right_check 0.57 cm). Three photos now agree: right check dot at
  34.98 / 35.06 / 35.07, centre at 18.71 / 18.68 / 18.67. Fixture `sam_photo_ivcam_arm_small_coloured_dot.txt`.
- **dot_size_frac raised to 0.012-0.030 (3 Nov):** 0.026 left the orange dot 0.2 px of margin. A 19 mm die
  (about 38 px, 0.030) is now kept as a candidate, and the robust dot finding leaves it out because no set of 6
  that includes it fits the layout.

### A3. SAM 3.1 on a video (tested 28 Oct)

Request: same as A2 but `{"type": "input_video", "video_url": "data:video/mp4;base64,..."}`. A 3.2 MB, 317-frame, 1280x720 MP4 sent as a data URL worked.

- Output per frame: `<12f>0<|box;...|><|mask;...|>`. `<12f>` = frame 12. The number before `<|box` is the **track id**, stable across frames.
- One prompt per object again: `"blue bottle"`, `"black rectangle"`, `"green lighter"` each gave 1 track seen in 315-317 of 317 frames. Camera shake moved the centres by 11-21 px.
- **Not tested yet:** whether a track keeps its id when the object is moved or covered by a hand. Only matters if we use video (stacking). Photos are enough for the row task.

### A3b. Video facts checked locally, without a call (3 Nov)

- **SAM's frame numbers are the video's frames:** the 317-frame reply has exactly 317 `<Nf>` markers (0-316) for
  the 317 frames OpenCV counts in `clip.mp4` (30 fps, 1280x720). `show_demo.py` stops if the counts ever differ.
- **Camera shake in that handheld clip:** at most 10.9 px (0.0085 of the width) within any one second, and 21 px
  over the whole 10.6 s. So the "still" limit is 1.5% of the width (19 px at 1280, about 0.8 cm on the sheet). The
  slow drift makes the lighter, which never moved, "arrive" at frame 9 (0.3 s): arrival times are approximate
  with a handheld camera, which is why they are only logged.
- **Windows ffmpeg 9.0.2 (with libx264) shrinks videos as planned:** `clip.mp4` (317 frames, 3.2 MB) became 106
  frames at 10 fps, 548 KB, in about 2 s; made-up 1920x1080 and 1080x1920 videos came out 1280x720 and 720x1280.
  Sound and metadata are dropped.

### A4. Pricing and limits

From Meta's pricing page (read 27 Oct) and third-party docs as marked.

- SAM 3.1: $2.50 per 1,000 images; $0.20 per 1,000 video frames, minimum 50 frames billed per video request. So one photo costs $0.0025 and our 317-frame clip about $0.06.
- Muse Spark 1.3 standard: $1.25 per 1M input tokens, $4.25 per 1M output tokens.
- Hosted SAM: up to 15,000 frames and 16 tracked objects per frame (third-party: LearnOpenCV). Text prompts only on the hosted API (third-party: KuCoin news).
- The spend-limit setting only sends a notification. It does not stop charges. Our code must enforce its own budget.

### A5. Muse Spark 1.3, text in, JSON out (tested 28 Oct)

- `client.responses.create(model="muse-spark-1.3", input="...")` returned bare JSON when told "Reply with ONLY a JSON list". It parsed first time.
- **Lesson:** asked to plan moves, it returned one "insert" move that a real arm cannot do. Prompts must define what an action is allowed to be, and exact planning stays in Python.

### A5b. Muse Spark 1.3 sees images (B1 and B10, settled 29 Oct)

- **B1: yes.** Spark accepts `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` in the same
  content list as SAM. On `desk.jpeg`, with `brain.NAME_OBJECTS_PROMPT`, it replied with bare JSON:
  `["blue lighter", "green lighter", "black box"]`. Saved as `spark_name_objects_desk.txt`.
- **B10: yes, `resp.usage` exists** with `input_tokens` and `output_tokens`. That call: 1,661 in, 704 out, $0.005068.
  704 output tokens for a 45-character answer: hidden reasoning tokens are billed, so never estimate Spark
  output cost from the visible text.
- **Spark's names are not always SAM's names.** SAM found exactly 1 box for "blue lighter" and "green lighter",
  but 0 for "black box" (the object SAM finds as "black rectangle"). Saved as `sam_photo_desk_black_box.txt`.
- **Renaming works.** `brain.name_objects` sends any name that does not give exactly 1 SAM box back to Spark
  with the match counts (up to 2 retries). On `desk.jpeg`, retry 1 turned "black box" into "black rectangle"
  (1 box). That retry cost $0.0091 (1,692 tokens in, 1,644 out). Saved as `spark_rename_desk_round1.txt`.
- **(3 Nov) The naming prompt now has one line of scene context:** "The objects are small painted dice, so name
  each one 'colour die' (for example 'red die')." The rename loop and the exactly-1-box rule are unchanged. The two
  saved Spark replies above were recorded with the old prompt on `desk.jpeg` (lighters, not dice): they still test
  reading the reply and the rename loop, but no real reply to the new prompt has been recorded yet. That needs a
  photo of the dice on the sheet (B14).

### A5c. Spark with two photos (B11, settled 2 Nov)

- **Yes:** Muse Spark 1.3 accepts two `input_image` parts in one call (`brain.diagnose` sends before and after) and
  replied with JSON that `parse_diagnosis` accepts. Saved as `spark_diagnose_two_photos_no_block.txt`.
- **Cost: $0.0207 for one call** (3,457 tokens in, 3,859 out; mostly hidden reasoning), about double the $0.01
  estimate. Six diagnoses would be about $0.12 of a run's $0.15 cap.
- **Warning, it did not look properly:** the test photos (`dots_check_6a`/`6b`) contain no block at all, but Spark
  answered `missed_aim`, "The red block is sitting about 1 cm to the right of the center slot", with a fix of
  -1.0 cm: it repeated the measured miss given in the prompt instead of saying `not_found`. Its fix passed the
  code's checks (size and direction), which cannot tell whether the explanation is true. Treat Spark's explanations
  as unverified until tested on real before/after photos with a block in them.
- **Prompt fixed (2 Nov, second call, $0.0194, 3,546 in / 3,517 out):** added "If you cannot see the {block} in
  photo 2, answer not_found. Do not repeat the measured miss." and a `seen_offset_cm` field (where Spark itself sees
  the block relative to the slot, or null). On the same no-block photos Spark now answered `not_found`,
  `seen_offset_cm: null`, fix 0/0 with `retry: false`: "I cannot see a red block in the second photo, so there is
  nothing to measure." Saved as `spark_diagnose_two_photos_no_block_v2.txt`. Still to test: real photos WITH a block.
- **Decisions (2 Nov):** Spark's fix is **logged and printed only**; the numeric correction always decides, unless
  `MIMIC_USE_SPARK_FIX=1` in `.env` (then it must also pass the size, direction and consistency checks). A
  **consistency check** marks a diagnosis "contradicts measurement" (log, terminal, report) when Spark's
  `seen_offset_cm` is more than 0.5 cm from the camera's measured miss, or when Spark says not_found although the
  camera measured the block. Limits: reserve $0.03 per diagnosis, **at most 2 diagnoses per run**, per-run cap
  **$0.30** for full runs (`loop.RUN_BUDGET_USD`; the budget guard still stops a run beyond it).
  Limit of the check: it catches disagreement, not copying. Spark is still told the measured miss, so it could
  copy it into `seen_offset_cm` and "agree". Leaving the miss out of the prompt would make the check independent.

### A6. The laptop

- Microsoft Surface Laptop 7, Snapdragon X Plus (ARM64), Windows 11 Home, 16 GB RAM.
- Development happens in WSL2 Ubuntu (ARM64) in `~/mimic`, Python venv `.venv`.
- USB devices and webcams are **not** visible inside WSL by default. Anything that talks to USB hardware runs from Windows, or goes over the network.

### A7. The arm (from the listing and customer reviews, 28 Oct)

KEYESTUDIO ESP32 Smart Robot Arm Kit, 4 DOF, joystick, ordered 28 Oct, expected Fri 30 Oct.

- ESP32 board, programmable with the Arduino toolchain: board "ESP32 Dev Module" (`esp32:esp32:esp32`) and the `ESP32Servo` library (a French reviewer, Windows, ESP32Servo 1.2.1).
- **USB does not power the arm.** It needs 2 lithium cells, 6 AA, or a 9 V battery. We bought 8 rechargeable NiMH AA (1.2 V each, 7.2 V for 6).
- A reviewer's gripper servo overheated on 9 V, ran cooler on 6 V, and stopped overheating once a cable stopped pulling on it.
- Uploading worked on Windows and failed on Linux for one reviewer. It needs the Silicon Labs CP210x USB driver.
- If an upload says "Failed to connect", hold the BOOT button during the upload.
- Assembly has very small parts and takes an evening. One review says a part was missing.
- Wi-Fi and Bluetooth are on the board. The listing says it can be controlled from a computer on the same Wi-Fi.

### A7b. Arm toolchain and power (settled 2 Nov; B5)

- **B5: yes.** `arduino-cli` and the CP210x driver work on Windows 11 ARM64: Girik compiled and uploaded from
  PowerShell. From WSL, the Windows `arduino-cli.exe` compiles our sketch:
  `"/mnt/c/Program Files/Arduino CLI/arduino-cli.exe" compile --fqbn esp32:esp32:esp32 "$(wslpath -w firmware/mimic_arm)"`
  (run it from a folder under `/mnt/c`, or Windows `cmd.exe` prints harmless "UNC paths are not supported" lines).
- **Pinned versions:** arduino-cli 1.5.1, ESP32 core `esp32:esp32` **2.0.12**, `ESP32Servo` **1.2.1**, board
  `esp32:esp32:esp32` (ESP32 Dev Module). Newer versions exist (core 3.3.12, ESP32Servo 3.2.1); do not upgrade
  during the event: the 2.0.12 + 1.2.1 pair is the one that works. (Core 3.x changed the PWM
  functions ESP32Servo uses, per the ESP32 core 3.0 migration notes; not tested here.)
- `firmware/mimic_arm` compiles with these versions (2 Nov). **Correction (2 Nov, later):** our own files
  (`mimic_arm.ino`, `arm_logic.h`) give no warnings even with `--warnings all` on a clean build, but the
  ESP32Servo 1.2.1 library itself has an unused variable (`TAG`) that `--warnings all` turns into an error.
  Compile and upload with the default warning level (no `--warnings` flag). The earlier "no warnings in the
  library" was wrong: those builds reused library files compiled with default warnings.
  Locked build: 289,449 bytes (22%) of flash, 21,872 bytes (6%) of RAM.
- **The servo ports need battery power.** USB powers the ESP32 board, but USB alone does not move the servos.
- **Pins (from the kit's sample code, untested on our arm):** base GPIO 16, shoulder 17, elbow 2, grip 4.
  In `calib/arm_limits.json` and `arm_logic.h` (a unit test keeps them equal). GPIO 2 is also the board's LED
  and a boot-mode pin on many ESP32 boards; the kit uses it, but if uploads start failing with the servos
  connected, that is the first thing to check.

### A9. The camera (B2, decided 2 Nov)

- **iPhone + iVCam.** The phone runs iVCam; on Windows it is a webcam called `e2eSoft iVCam`. WSL cannot see
  webcams, so `mimic/camera.py` runs Windows ffmpeg (winget `Gyan.FFmpeg` 9.0.2; full path in `config.py`, override
  with `MIMIC_FFMPEG` in `.env`, because the winget folder name contains the version):
  `ffmpeg.exe -f dshow -i video="e2eSoft iVCam" -frames:v 15 -update 1 -q:v 2 <photo as a Windows path>`
  (15 frames so the exposure settles; the last one is kept). 10 s timeout.
- A photo is refused (`CameraError`) if ffmpeg fails, no file appears, it is nearly black (average brightness
  under 20/255), or it is smaller than 1280x720. Girik's test photo through iVCam (`ivcam_sheet.jpg`, not
  committed) is 1280x720 with average brightness 140, so iVCam must stay at 1280x720 or more.
- **Not yet done (needs the camera):** the B2 test from the plan, ten sharp overhead photos in a row in under 3 s each.
- Never commit camera frames with people in them.

### A10. Laptop-to-arm link (B3, settled 2 Nov)

- **WSL talks to the arm through a Windows-side bridge** over USB serial: `scripts/serial_bridge.py` runs on Windows
  Python 3.14.6 with pyserial 3.5, started from WSL by `mimic.arm.SerialBridgeTransport`. It finds the CP210x
  (COM7, USB 10C4:EA60) by its USB ID, sets DTR/RTS low before opening, waits 2 s and discards boot text, time-limits
  every reply and sends S when it closes. (Option 1, Wi-Fi HTTP, was not needed.)
- **Read-only test on the real board passed (2 Nov),** locked firmware uploaded from PowerShell, batteries
  disconnected, USB only, `python scripts/arm_check.py`: `?` -> `A 90 90 90 90 L`, `S` -> `OK`,
  `J 90 90 90 90 800` -> `ERR limits not measured`, `H` -> `ERR limits not measured`.
  So the bridge, the serial link, the protocol and the firmware's lock all work on the real hardware.
  No servo has been driven yet.

### A10b. Why the board last started: the B command (3 Nov, firmware not uploaded yet)

- `B` replies `B <reason> <seconds since start-up>`, e.g. `B brownout 42`. Reasons: power-on, brownout, software
  reset, watchdog, other. Read-only, so it works while locked, like `?` and `S`. `arm.boot_reason()` in Python,
  `boot` in `arm_jog.py`.
- The words come from the ESP32's own reset code (`esp_reset_reason()`); the codes were read from core 2.0.12's
  `esp_system.h` (POWERON 1, SW 3, INT_WDT 5, TASK_WDT 6, WDT 7, BROWNOUT 9), and `mimic_arm.ino` checks each
  number with a `static_assert` when it compiles. A crash (panic) is "other".
- **It only shows ESP32 resets, not servo supply sag.** If the servo battery dips but the ESP32 keeps running (for
  example because USB powers it), `B` still says the old reason with a growing number of seconds. A brownout shows
  only when the ESP32's own supply dropped far enough to restart it.
- On the ESP32, the EN (reset) button and an upload also show as power-on.
- The seconds tell a fresh restart from an old one: the bridge opens COM7 with DTR and RTS low so that opening the
  port does not reset the board, but that is untested (B12). If every `boot` straight after opening shows only 2-3
  seconds, opening the port does reset the board, and then a brownout would be hidden by the next power-on.

### A8. Repo

- Public: https://github.com/girik-chadha/mimic, first commit 28 Oct 2026.
- `.env` (holds `META_API_KEY`) is gitignored and has never been committed.

---

## Part B: Open questions (settle each with its test)

| # | Question | Test that settles it | When |
|---|---|---|---|
| B4 | Arm joint ranges, link lengths, gripper opening width; confirm the pins (A7b has the kit's pins, untested). **3 Nov: the kit's own ranges, from its sample code and untested on this arm, are in `calib/arm_limits.json` and `arm_logic.h` for the first powered session: base 0-180, shoulder 80-180, elbow 30-120, grip open 90 / closed limit 140, home 90 for every joint.** The grip closed limit was raised from 120 to 140 the same day (the kit's own limit is 150): a 19 mm die needs the jaws closer than 120 would allow. How far the jaws move per degree is not measured yet, so `grip_test.py`'s 2-degree steps find the angle that holds. | `scripts/first_moves.py` (needs.md C7) measures the real ranges; link lengths and gripper opening with a ruler. | First powered session |
| B6 | How precise is the arm at our table distances? | `scripts/repeatability.py --pose B S E G`: the same pose 10 times from alternating sides, the camera measures the claw-tip marker; spread (mean, standard deviation, max), backlash, verdict against the 1.5 cm tolerance. | First powered session |
| B7 | Does the Baseten Muse Spark Open endpoint use the OpenAI-compatible format? Model id given by the organisers: `meta-models/muse-spark-open-1026`. | One text call through the OpenAI SDK with Baseten's base URL. | Mon |
| B8 | Does Muse Code bill to our credits or need its own plan? | Check the Muse Code page on dev.meta.ai. Only matters if we want to use it. | Any time |
| B9 | Exact judging weights and bonus prize criteria. | Logged-in hackathon details page and Discord. | Any time |
| B13 | The demo video, on the first real `show_demo.py` run: does SAM accept the shrunk video (10 fps H.264, made by ffmpeg, not a phone)? How long does a video call take (our timeout for video is 300 s; the 28 Oct call was never timed)? Does a track keep its id while a hand moves the block (A3's open question; if not, `show_demo.py` stops with "2 tracks")? | `python scripts/show_demo.py demo.mp4 --save-replies` on an 8-10 s video of the real sheet, after the dots are ruler-measured (`layout.json` must not be a placeholder). About $0.10-0.13 for 3 blocks. | After the dots are measured |
| B14 | Does Spark name the dice "colour die" with the new prompt line, and does SAM find exactly 1 box for each such name (two dice of similar colours could share one)? | One real `brain.name_objects` call on a photo of the dice on the sheet; save Spark's reply and each SAM reply as fixtures. About $0.01-0.02. | When the dice photo is taken |
| B12 | Does opening COM7 reset the board? The bridge sets DTR and RTS low before opening so that it should not. If it does, every session restarts the arm (each servo jumps to home again) and a brownout is hidden by the next power-on. | After the upload (C7 step 4): `arm_jog.py`, type `boot` at once, `quit`, wait a minute, repeat. If the seconds are always 2-3, opening resets the board. | First powered session |

---

## Part C: Setup

### C1. Accounts and keys

- Meta Model API key in `~/mimic/.env` as `META_API_KEY=...` (done).
- GitHub `girik-chadha`, logged in with `gh` (done).
- Baseten account and credits (Mon, only for Spark Open).

### C2. Environment variables (`.env`, never committed)

```
META_API_KEY=...
MIMIC_BUDGET_USD=1.00        # hard cap per run, enforced in code
MIMIC_ARM_URL=http://<esp32-ip>   # once B3 is settled
BASETEN_API_KEY=...          # Mon, optional
```

### C3. Software

- WSL Python: `openai`, `python-dotenv`, `numpy`, `opencv-python-headless`, `pytest`, `requests`, later `fastapi` + `uvicorn` for the web page.
- Windows: Arduino CLI (or Arduino IDE 2), ESP32 core (board manager URL `https://espressif.github.io/arduino-esp32/package_esp32_index.json`), `ESP32Servo` library, Silicon Labs CP210x driver, and the camera software from B2.

### C4. Physical setup

- **Arm power-on while locked (2 Nov):** the firmware never attaches a servo and never writes an angle, so the
  servos get no signal at all. Only `?` (replies `A 90 90 90 90 L`, the firmware's own numbers, not a real pose)
  and `S` work; `J` and `H` reply `ERR limits not measured`.
- **Arm power-on once unlocked (29 Oct):** hobby servos cannot report their position, so when the arm is powered
  the firmware sends every joint to the home pose and they jump there. The home pose must therefore be a safe pose:
  clear of the table, the blocks and the camera. Always send `H` before switching off, so the next jump is small.

- White table, bright even light, no hard shadows.
- An A3 sheet taped to the table as the work area, with 6 dots drawn on it (changed 30 Oct): 4 corner dots and
  2 check dots, plus 3-4 numbered target slots. Each photo finds all 6 dots ("small coloured dot"), works out which
  is which by position, builds its own map from the corners and checks it with the 2 check dots (0.5 cm limit).
  Only the 4 corners must never have 3 in a straight line. Limit: a wrong *corner* position only partly shows at
  the check dots, so Gate 1 also compares computed cm against a ruler.
- **Known limit until the check dots move (30 Oct):** both check dots are on the sheet's middle line, so a photo
  flipped top-to-bottom (a mirror image) fits within ~0.1 cm and would be read upside down. Only rotations are
  tried for now. Blocker in todo.md before Gate 3.
- **Keep every block at least 4 cm from every dot:** a covered dot stops the run (each photo needs all 6).
- The camera looks down from 40-50 cm. It may move between photos: every photo calibrates itself.
- Blocks: solid, 2-3 cm, clearly different colours, none white or pale. Buy them after measuring the gripper opening (B4).
- Nothing personal, no faces, in the camera's view.

### C7. First powered session (scripts written 2 Nov against fakes; checklist revised 3 Nov)

In every script, **s** at any prompt or **Ctrl+C** sends S (stop) and exits. If anything buzzes, strains or
smells: switch the batteries off.

**Before power** (steps 1-3 prepared 3 Nov: kit ranges in both files, grip closed limit 140, unlocked, staggered
start-up, firmware compiled and copied to `C:\Users\girik\mimic_firmware\mimic_arm`; not uploaded yet)
1. Write the kit's own joint ranges (B4) into **both** `calib/arm_limits.json` and
   `firmware/mimic_arm/arm_logic.h`, home 90 for every joint. `.venv/bin/pytest` (the sync test must pass).
2. `python scripts/arm_lock.py unlock` (type yes). For this session the hand check is earned by `first_moves.py`
   itself (step 10). Update `test_committed_limits_are_unlocked...` in the same commit.
3. Upload from PowerShell with the **batteries disconnected** (USB only), needs.md C5. If the upload fails with
   the servos plugged in, unplug the **elbow** servo's signal wire and try again: the elbow is on GPIO 2, which is
   also a boot pin on the ESP32 (A7b).
4. **Check what is on the board (still no batteries):** `python scripts/arm_jog.py`, type `status`, expect
   `{'base': 90, 'shoulder': 90, 'elbow': 90, 'grip': 90, 'locked': False}` (the board answers `A 90 90 90 90`,
   no L), then `boot`, expect `last start: power-on, ...` (the upload restarted the board), then `quit`. Type nothing
   else: with no battery power nothing would move, but that is not the point. If `boot` is refused with "no B
   command", the old build is still on the board. (`scripts/arm_check.py` is only for the locked firmware: on this
   board it would report NOT locked.)

**Power on**
5. Clear the area: nothing within the arm's reach, cables free, sheet and blocks out of the way.
6. **Pose the arm by hand first:** with everything switched off, gently move each joint to about 90 degrees (the
   home pose) and leave the claw empty. At power-on each servo jumps straight to 90 degrees, unramped (it cannot
   report where it was), so the closer it already is, the smaller the jump.
7. **Unplug the USB cable**, then connect the batteries **with a hand on the switch**, then plug the USB cable back
   in. Why: the firmware switches the servos on one at a time, 400 ms apart (base, shoulder, elbow, claw), but only
   when the board starts. If the board is already running on USB when the batteries are connected, all four jump
   at once. If the board does not start on battery power alone (not tested yet), plugging USB in starts it with the
   batteries already there, which still gives the one-at-a-time start. If the arm twitches over and over, the
   battery is dipping and resetting the board: switch off.
8. `python scripts/arm_jog.py`, `status` again, then `boot`: `power-on` with a few seconds is the start you just
   did. **Whenever the arm twitches, jumps back to home, or stops answering, type `boot` straight away:**
   `brownout` (or a small number of seconds you did not expect) means the ESP32 restarted because its supply
   dropped. Switch off and check the batteries. Then `quit`.

**Ranges**
9. `python scripts/first_moves.py`: base, then shoulder, then elbow; +5, -5, +10, -10 ... from home, slowly, never
   past the range in arm_limits.json and never more than 45 degrees from home (`--max-delta` to widen; so the base
   stays within 45-135). Answer y/n after each move; each direction stops at its first n. The shoulder has only 10
   degrees below home (range 80-180).
   **Write down which joint and direction lifts the claw** (for example "shoulder -" or "elbow +"): `grip_test.py`
   needs it as `--lift`, and teaching needs to know which way is up.
10. It writes `calib/arm_limits_proposed.json`: each direction's largest approved step minus 5 degrees. It never
    edits `calib/arm_limits.json`. **A completed first_moves session is the "checked by hand" step:** the proposal
    has `limits_checked_by_hand: true`. Review it, copy the ranges into **both** files, run the tests, switch the
    batteries off, upload (step 3), then power on again (steps 6-8).

**Grip and repeatability**
11. `python scripts/grip_test.py --lift "<the lift you wrote down, e.g. shoulder -4>"` (one or more joints, 10 degrees
    in total at most). At home with the claw open, place the cube while nothing is sent, press Enter. It closes in
    2-degree steps (opening again between steps, because of the 5 s grip timer) and asks "did it hold?"; at the
    first y it lifts, holds 2 s, lowers and opens. The angle that held goes into the proposal as `grip_hold_deg`.
    Use this rather than `grip close` in `arm_jog.py`, which goes straight to the 140 limit.
    **Push mode only (opt-in): `grip_push` is unknown until this test.** With the claw empty, in
    `scripts/teach.py` use `set grip <angle>` in 2-degree steps from the angle that held until the jaw tips just
    touch (the firmware opens a closed claw after 5 s; the next `set` closes it again). `grip_push` is that angle
    minus 2 degrees: write it into `calib/pose_grid.json` as `"grip_push"`. Never use the 140 limit: a stalled
    claw overheats. Until it is set, push mode refuses to run.
12. Put the gripper marker on the claw tip. `python scripts/repeatability.py --pose B S E G` (a pose that hides no
    dot; `--target X Y` from a ruler is optional). Cost cap $0.10. Verdict: 2 x standard deviation under 0.5 cm, full
    corrections are fine; over 1 cm, apply only part of each miss.

**Switch off:** home first (`arm_jog`: `home`), then the batteries.

**How the arm moves (checked 3 Nov):** every J is ramped in the firmware. Python never sends a J faster than
10 ms per degree (`Arm.move` refuses it), and the firmware slows any J to at least 10 ms per degree of the biggest
joint change (`start_move`), then eases in and out (`ease`): average at most 100 degrees per second, peak at most
150 per second, never more than 3 degrees per 20 ms tick (g++ test). H takes at least 1.5 s. S stops at once and
holds. The only unramped movement is each servo's first jump to home at start-up.

### C6. Teaching the pose grid (scripts/teach.py)

- Teach **6-8 points** (at least 4 are required) covering **every slot and the whole mess area**. The arm is only
  ever sent inside the **hull** of the taught points (the shape you get by stretching a rubber band around them),
  plus 1 cm; anywhere else is refused.
- At each point: jog (steps of 1, 2, 5 or 10 degrees, or `set <joint> <angle>`) until the gripper is above the
  spot, `save hover x y`; lower until it can close on a block, `save grab x y`. For push mode, teach grab with
  the jaw tips 2-5 mm above the paper (and check that the claw closed to grip_push still clears it): push mode
  slides the block at grab height. Also `save grip_open`,
  `save grip_closed`, and `save clear` for **camera_clear**: a pose that keeps the arm out of the photo, with no dot
  or block covered (the arm goes there before every photo). Then `finish`.
- **The arm base must never move relative to the sheet** after teaching: the grid is tied to the sheet. If it
  moves, re-teach. Two moves in a row missing the same way print a warning.
- Practise first with `python scripts/teach.py --dry-run` (saves to `runs/pose_grid_dryrun.json`).
- **Push mode (opt-in, `run_mimic.py --push`) needs a few more points:** `python scripts/teach_plan.py --push`
  lists them (with the placeholder layout: 4 behind the mess area's corners, one of them out of reach, and one
  beyond slot 3). It also needs `grip_push`, which is unknown until the grip test (C7 step 11).

### C5. Uploading the firmware, and lifting the arm's lock

**Upload (from PowerShell; nothing is uploaded from WSL).** The firmware is copied to a Windows folder first,
because the Arduino tools work best on Windows paths. Close anything using COM7 (the serial bridge, a serial
monitor) before uploading: only one program can hold the port.

1. Copy the current firmware from WSL (re-run after every firmware change):
   `mkdir -p /mnt/c/Users/girik/mimic_firmware && cp -r firmware/mimic_arm /mnt/c/Users/girik/mimic_firmware/`
2. Upload from PowerShell (default warning level: see A7b):
   `& "C:\Program Files\Arduino CLI\arduino-cli.exe" compile --upload -p COM7 --fqbn esp32:esp32:esp32 C:\Users\girik\mimic_firmware\mimic_arm`
   If it says "Failed to connect", hold BOOT during the upload (A7).
3. Read-only check (batteries disconnected, switch off): `python scripts/arm_check.py` from WSL.

**Lifting the lock (only after the joint ranges, grip limit and home pose are real numbers in BOTH
`calib/arm_limits.json` and `firmware/mimic_arm/arm_logic.h`, conservative, and checked by hand):**

1. One command flips all four lock values together and asks you to type yes:
   `python scripts/arm_lock.py unlock`
   (JSON: `"placeholder": false`, `"limits_checked_by_hand": true`; firmware: `LIMITS_ARE_PLACEHOLDER 0`,
   `LIMITS_CHECKED_BY_HAND 1`.) `python scripts/arm_lock.py lock` puts it back, without asking.
2. `.venv/bin/pytest`: the sync test must pass. `test_committed_limits_are_locked` fails on purpose: update it in
   the same commit, so unlocking is always a deliberate, visible change.
3. Copy and upload again (steps 1-2 above). Until the new firmware is on the board, the board stays locked, and
   `arm.py` refuses moves because `?` still ends with L.
4. After unlocking, `arm_check.py` reports "NOT locked": that check is only for the locked firmware.

