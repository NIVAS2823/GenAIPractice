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



video_file_path = r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\video_1.mp4'
video_mime_type= "video/mp4"

with open(video_file_path,"rb") as video_file:
    encoded_video = base64.b64encode(video_file.read()).decode('utf-8')


message = HumanMessage(
    content = [
        {
            "type":"text",
            "text":"Describe the first few frames of the video"
        },
        {
            "type":"media",
            "data":encoded_video,
            "mime_type":video_mime_type,
        },
    ]
)

response = llm.invoke([message])

print(f"Response for the video uplaod: {response.content}")