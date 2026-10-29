import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
try:
    for m in client.models.list():
        print(m.id)
except Exception as e:
    print("ERROR:", e)
