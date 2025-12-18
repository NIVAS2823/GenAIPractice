from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

prompt = 'Explain how AI works in a few words'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

# print(response.text)

