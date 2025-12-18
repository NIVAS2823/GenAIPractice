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

# message_url = HumanMessage(
#     content = [
#         {
#             "type":"text",
#             "text":"Descrive the image at the url"
#         },
#         {
#             "type":"image_url",
#             "image_url":"https://picsum.photos/seed/picsum/200/300"
#         }
#     ]
# )

# result_url = llm.invoke([message_url])

# print(f"Response for the Url Image:{result_url.content}")

image_file_path = r'C:\Users\NIVAS\Personal\Agentic_Ai\Gen_ai_practice\my_photo.jpg'

with open(image_file_path,"rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


message_local = HumanMessage(
    content = [
        {
            "type":"text",
            "text":"Desribe about provided in the image "
        },
        {
            "type":"image_url",
            "image_url":f"data:image/png;base64,{encoded_image}"},
    ]
)

result_local = llm.invoke([message_local])

print(f"Response for the local image:{result_local.content}")

