from google import genai
from google.genai import types

client = genai.Client()


prompt = 'Summarize this video.'
MODEL_ID = 'gemini-2.5-flash'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text=prompt),
            types.Part(
                file_data=types.FileData(file_uri='https://youtu.be/X95MFcYH1_s?si=DCjusqQcSJnP23Pn')
            )
        ]
    )
)

print(response.text)