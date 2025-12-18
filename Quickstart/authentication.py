from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

MODEL_ID = 'gemini-2.5-flash'


response = client.models.generate_content(
    model=MODEL_ID,
    contents='Please give me a python code to sort a list '
)

print(response.text)