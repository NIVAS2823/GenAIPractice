from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

prompt = 'Generate and run python program to count how many vowels in the word Retrieval Augmented Generation'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    )
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    if part.executable_code is not None:
        print("\n[GENERATED CODE]")
        print("--------------------")
        print(part.executable_code.code)
        print("--------------------")

    if part.code_execution_result is not None:
        print(f"Output: {part.code_execution_result.output}")
        print(f"Error:{part.code_execution_result.__str__}")