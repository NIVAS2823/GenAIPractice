from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'


system_instruction = """
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction
)

chat = client.chats.create(
    model=MODEL_ID,
    config=chat_config
)

response = chat.send_message('Write a function that checks if a year is a leap year .')

print(response.text)