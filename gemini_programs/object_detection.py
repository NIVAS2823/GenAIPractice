from google import genai
from google.genai import types
from PIL import Image
import json

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

prompt = 'Detect the all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.'

image = Image.open(r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\example.jpeg')

config = types.GenerateContentConfig(
    response_mime_type="application/json"
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[image,prompt],
    config=config
)

# width,height = image.size

# bounding_boxes = json.loads(response.text)

print(response.text)