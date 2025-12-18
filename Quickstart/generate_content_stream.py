from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.0-flash'

prompt = 'Tell me a story about a lonely robot who finds friendship in a most unexpected place'

stream_response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

for chunk in stream_response:
    if hasattr(chunk,'text') and chunk.text:
        print(chunk.text,end="")
    
print('\n\n----Story Completed----')