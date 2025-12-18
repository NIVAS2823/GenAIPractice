from google import genai
from google.genai import types
from pydantic import BaseModel
import json

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

class Recipe(BaseModel):
    recipe_name :str
    recipe_description:str
    recipe_ingredients:list[str]


response = client.models.generate_content(
    model=MODEL_ID,
    contents='Provide hyderabad chicken biryani recipe with its ingredients',
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,
    ),
)

print(json.dumps(json.loads(response.text),indent=4))