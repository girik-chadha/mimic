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

### A3. SAM 3.1 on a video (tested 28 Oct)

Request: same as A2 but `{"type": "input_video", "video_url": "data:video/mp4;base64,..."}`. A 3.2 MB, 317-frame, 1280x720 MP4 sent as a data URL worked.

- Output per frame: `<12f>0<|box;...|><|mask;...|>`. `<12f>` = frame 12. The number before `<|box` is the **track id**, stable across frames.
- One prompt per object again: `"blue bottle"`, `"black rectangle"`, `"green lighter"` each gave 1 track seen in 315-317 of 317 frames. Camera shake moved the centres by 11-21 px.
- **Not tested yet:** whether a track keeps its id when the object is moved or covered by a hand. Only matters if we use video (stacking). Photos are enough for the row task.

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

### A8. Repo

- Public: https://github.com/girik-chadha/mimic, first commit 28 Oct 2026.
- `.env` (holds `META_API_KEY`) is gitignored and has never been committed.

---

## Part B: Open questions (settle each with its test)

| # | Question | Test that settles it | When |
|---|---|---|---|
| B2 | Which camera can our code trigger? iPhone-as-webcam on Windows needs a third-party app. A cheap USB webcam is the reliable route. | `capture.py` saves a sharp, overhead photo in under 3 s, ten times in a row. | Decide Thu |
| B3 | How does WSL talk to the arm? Option 1: ESP32 joins the phone hotspot and runs a tiny HTTP server; WSL sends requests. Option 2: a small Windows-side bridge over USB serial. | Option 1 first: ESP32 prints its IP, `curl` from WSL gets a reply. If the hotspot blocks it, Option 2. | Fri |
| B4 | Arm pins, joint names, joint ranges, link lengths, gripper opening width. | Read the KEYESTUDIO PDF, measure with a ruler, write the numbers here. | Fri |
| B5 | Does `arduino-cli` (or Arduino IDE 2) and the CP210x driver run on Windows ARM64? | Install Thu evening; `arduino-cli board list` shows a COM port when the arm is plugged in. | Thu/Fri |
| B6 | How precise is the arm at our table distances? | Send the same pose 10 times; the camera measures the spread of the gripper tip in mm. | Sat |
| B7 | Does the Baseten Muse Spark Open endpoint use the OpenAI-compatible format? Model id given by the organisers: `meta-models/muse-spark-open-1026`. | One text call through the OpenAI SDK with Baseten's base URL. | Mon |
| B8 | Does Muse Code bill to our credits or need its own plan? | Check the Muse Code page on dev.meta.ai. Only matters if we want to use it. | Any time |
| B9 | Exact judging weights and bonus prize criteria. | Logged-in hackathon details page and Discord. | Any time |

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

- **Arm power-on (29 Oct):** hobby servos cannot report their position, so when the arm is powered the firmware
  sends every joint to the home pose and they jump there. The home pose must therefore be a safe pose: clear of
  the table, the blocks and the camera. Always send `H` before switching off, so the next jump is small.

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
