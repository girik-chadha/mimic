import base64, mimetypes, os, re, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")

path, prompt = sys.argv[1], sys.argv[2]
mime = mimetypes.guess_type(path)[0] or "video/mp4"
with open(path, "rb") as f:
    url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

try:
    resp = client.responses.create(
        model="sam-3.1",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "video_url": url},
        ]}],
    )
    text = re.sub(r"<\|mask;.*?\|>", "<mask removed>", resp.output_text, flags=re.DOTALL)
    print(text[:3000])
    print("\n[total length:", len(resp.output_text), "characters]")
except Exception as e:
    print("ERROR:", e)
