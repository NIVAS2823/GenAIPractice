from google import genai
from google.genai import types

client = genai.Client()
MODEL_ID = 'gemini-2.5-flash'


#Error Handling

# error_handling_system_prompt  = """
#     Your task is to explain why this error occured and how to fix it
#     """

# error_handling_model_config = types.GenerateContentConfig(
#     temperature=0,
#     system_instruction=error_handling_system_prompt
# )

# error_message = """
#      1 my_list = [1,2,3]
#   ----> 2 print(my_list[3])

#   IndexError: list index out of range
#     """

# error_prompt = f"""
#     You've encountered the following error message :
#     Error Message : {error_message}
# """
# response = client.models.generate_content(
#     model=MODEL_ID,
#     contents=error_prompt,
#     config=error_handling_model_config


#Code Generation

code_generation_system_prompt = """
    You are a coding assistant.Your task is to generate  a code snippet that
    accomplished specific goal . The Code snippet must be concise ,efficient,
    and well-Commented for clarity . Consider any constrains or requirements
    provided for the task
    """

code_generation_model_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=code_generation_system_prompt
)

code_generation_prompt = """
    Create react component where i can select date using calender 
    """

response = client.models.generate_content(
    model=MODEL_ID,
    contents=code_generation_prompt,
    config=code_generation_model_config
)

print(response.text)