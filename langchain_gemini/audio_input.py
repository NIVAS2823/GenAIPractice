from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage

import base64

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    max_tokens = None,
    timeout=None,
    max_retries=2
)

audio_path = r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\test_greeting.wav'

audio_mime_type = 'audio/wav'

with open(audio_path,"rb") as audio_file:
    encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

message = HumanMessage(
    content = [
        {
            "type":"text",
            "text":"Transcribe the audio"
        },
        {
            "type":"media",
            "data":encoded_audio,
            "mime_type":audio_mime_type,
        },
    ]
)

response = llm.invoke([message])

print(f"Response for provided audio:{response.content}")