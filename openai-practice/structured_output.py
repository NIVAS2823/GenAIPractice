from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalenderEvent(BaseModel):
    name:str
    date:str
    participants:list[str]

response = client.responses.parse(
    model='gpt-4o-mini',
    input=[
        {
            "role":"system",
            "content":"Extract the event information"
        },
        {
            "role":"user",
            "content":"Alice and Bob are going to  hotel on friday"
        },
    ],
    text_format=CalenderEvent
)


event = response.output_parsed

print(event)