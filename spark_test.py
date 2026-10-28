import json, os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")

prompt = """Three objects are on a table, left to right: blue bottle, black rectangle, green lighter.
Target order, left to right: green lighter, blue bottle, black rectangle.
Reply with ONLY a JSON list of moves, no other text, in this form:
[{"object": "...", "to_position": 1}, ...]"""

resp = client.responses.create(model="muse-spark-1.3", input=prompt)
text = resp.output_text
print("RAW:\n" + text + "\n")

match = re.search(r"\[.*\]", text, re.S)
try:
    moves = json.loads(match.group(0))
    print("PARSED OK:", len(moves), "moves")
    for m in moves:
        print(" ", m)
except Exception as e:
    print("COULD NOT PARSE JSON:", e)
