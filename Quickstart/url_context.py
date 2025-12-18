from google import genai
from google.genai import types

client = genai.Client()


prompt = """
Compare recipes from https://cars.tatamotors.com/curvv/ice.html
and from https://www.mahindraelectricsuv.com/esuv/be-6/MBE6.html,
list the key differences between them.
"""

MODEL_ID = 'gemini-2.5-flash'

tools = []
tools.append(types.Tool(url_context=types.UrlContext))

config = types.GenerateContentConfig(
    tools=tools,
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=config
)

print(response.text)