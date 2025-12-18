from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

file_upload = client.files.upload(file=r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\Gen_AI_Mock.pdf')

prompt = 'Can you give me a summary of this information please?'


response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        file_upload,
        prompt
    ]
)

print(response.text)