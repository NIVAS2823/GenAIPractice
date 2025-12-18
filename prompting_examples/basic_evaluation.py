from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'

student_system_prompt  ="""
    You're a college student . Your job is to write an essay riddled with common mistakes and a few major ones.
    The essay should have mistakes regarding clarity,grammar ,augmentation,and vocabulary.
    Ensure your essay includes a clear thesis statement. You should write only an essay,so do not include any notes
    """

response = client.models.generate_content(
    model=MODEL_ID,
    contents='Write an essay about the benefits of eating 6 soaked almonds daily morning on empty stomach ',
    config=types.GenerateContentConfig(
        system_instruction=student_system_prompt
    ),
)

print(response.text)

teacher_system_prompt = """"
    As a teacher,you are tasked with grading students' essays
    please follow these instructions for evaluation:

    1.Evaluate the essay on a scale of 1-5 based on the follwowing criteria:
    - Thesis Statement,
    -Clarity and precision of language, 
    -Grammar and punctuation
    -Argumentation
    """
teacher_response = client.models.generate_content(
    model=MODEL_ID,
    contents=response.text,
    config=types.GenerateContentConfig(
        system_instruction=teacher_system_prompt
    ),
)

print(teacher_response.text)