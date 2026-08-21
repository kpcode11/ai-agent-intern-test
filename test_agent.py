import os
from google import genai
from google.genai import types

client = genai.Client()

def get_weather(location: str) -> str:
    """Returns the weather for a location."""
    print(f"CALLED get_weather({location})")
    return "Sunny"

chat = client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        tools=[get_weather],
        temperature=0.0
    )
)

response = chat.send_message("What is the weather in Paris?")
print("RESPONSE TEXT:", response.text)
