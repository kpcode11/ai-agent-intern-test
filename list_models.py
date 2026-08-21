from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client()
for model in client.models.list():
    if "flash" in model.name.lower() or "gemini" in model.name.lower():
        print(model.name)
