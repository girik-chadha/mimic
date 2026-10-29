import base64, mimetypes, os, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")

def to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

image_path, prompts = sys.argv[1], sys.argv[2:] or ["red cup"]
img = to_data_url(image_path)

for prompt in prompts:
    resp = client.responses.create(
        model="sam-3.1",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": img},
        ]}],
    )
    print(f"\n=== {prompt} ===")
    print(resp.output_text)
