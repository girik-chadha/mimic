import base64, math, mimetypes, os, re, sys
from dotenv import load_dotenv
from openai import OpenAI

PAT = re.compile(r"<(\d+)f>|(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")

def get_text(path, prompt):
    load_dotenv()
    client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
    mime = mimetypes.guess_type(path)[0] or "video/mp4"
    with open(path, "rb") as f:
        url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    resp = client.responses.create(
        model="sam-3.1",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "video_url": url},
        ]}],
    )
    return resp.output_text

def parse(text):
    tracks, frame = {}, None
    for m in PAT.finditer(text):
        if m.group(1) is not None:
            frame = int(m.group(1))
        else:
            oid = int(m.group(2))
            x1, y1, x2, y2 = (int(m.group(i)) for i in (3, 4, 5, 6))
            tracks.setdefault(oid, []).append((frame, (x1 + x2) // 2, (y1 + y2) // 2))
    return tracks

if __name__ == "__main__":
    if sys.argv[1].endswith(".txt"):
        text = open(sys.argv[1]).read()
    else:
        text = get_text(sys.argv[1], sys.argv[2])
        with open("raw_response.txt", "w") as f:
            f.write(text)
        print("saved raw_response.txt")
    tracks = parse(text)
    print(f"{len(tracks)} track(s) found")
    for oid, pts in sorted(tracks.items()):
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[-1]
        moved = math.hypot(x1 - x0, y1 - y0)
        print(f"id {oid}: frames {f0}-{f1} ({len(pts)} seen) start ({x0},{y0}) end ({x1},{y1}) moved {moved:.0f}px")
