from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

system_prompt = """"
You are a teacher solving mathematical and logical problems . Your Task:
1.Summarize the given conditions.
2.Identify the problem
3.Provide a clear ,step by step explanation
4.Provide and explanation for each step

Ensure simplicity,clarity and correctness in all steps of your explanation
Each of your task should be done in order and separately
    """


logical_problem = """
    Assume a world where 1 in 5 dice are weighted and have 100% to roll a 6.
    A person rolled a dice and rolled a 6.
    Is it more likely that the dice was weighted or not?
    """


response = client.models.generate_content(
    model=MODEL_ID,
    contents=logical_problem,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)