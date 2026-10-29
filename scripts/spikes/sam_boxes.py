import base64, mimetypes, os, re, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")

BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")

def to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

def find(image_path, prompt):
    resp = client.responses.create(
        model="sam-3.1",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": to_data_url(image_path)},
        ]}],
    )
    boxes = []
    for x1, y1, x2, y2, w, h in BOX.findall(resp.output_text):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        boxes.append({"box": (x1, y1, x2, y2), "centre": ((x1 + x2) // 2, (y1 + y2) // 2)})
    return boxes

if __name__ == "__main__":
    image_path, prompts = sys.argv[1], sys.argv[2:]
    for p in prompts:
        found = find(image_path, p)
        print(f"\n{p}: {len(found)} match(es)")
        for b in found:
            print("  centre", b["centre"], "box", b["box"])
