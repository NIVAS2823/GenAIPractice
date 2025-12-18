from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

prompt = """
    Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark
 """

safety_settings = [types.SafetySetting(
    category='HARM_CATEGORY_DANGEROUS_CONTENT',
    threshold='BLOCK_ONLY_HIGH'
),
]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        safety_settings=safety_settings,
    ),
)

print(response.text)

