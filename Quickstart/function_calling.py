from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'


get_destination = types.FunctionDeclaration(
    name='get_destination',
    description='Get the destination that the user wants to go',
    parameters={
        "type":"OBJECT",
        "properties":{
            "destination":{
                "type":"STRING",
                "description":"Destination user wants to go",
            },
        },
    },
)

destination_tool = types.Tool(
    function_declarations=[get_destination],
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents='I like to travel to paris',
    config=types.GenerateContentConfig(
        tools=[destination_tool],
        temperature=0,
    ),
)

print(response.candidates[0].content.parts[0].function_call)