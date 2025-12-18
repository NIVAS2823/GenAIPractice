from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'




prompt = """
  5 people can create 5 donuts every 5 minutes. How much time would it take
  25 people to make 100 donuts? Return the answer immediately.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(response.text)