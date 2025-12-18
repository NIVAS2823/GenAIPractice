from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
        model='gpt-4o-mini',
        input=[
            {
                "role":"user",
                "content":"Say 'double bubble bath' ten times fast",
            },  
        ],
        stream=True
)

for event in resp:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
