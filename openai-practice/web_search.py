from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model='gpt-4o-mini',
    tools=[{"type":"web_search"}],
    input="What is the one popular news about AI today?"
)

print(response.output_text)