from google import genai
from google.genai import types
import pathlib

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

prompt = 'Explain how the concept of attention was evolved'

file_path = pathlib.Path(r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\NIPS-2017-attention-is-all-you-need-Paper.pdf')

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=file_path.read_bytes(),
            mime_type='application/pdf'
        ),prompt
    ]
)

print(response.text)

