from google import genai
from google.genai import types

client = genai.Client()

MODEL_ID = 'gemini-2.5-flash'
prompt = """
    Query : Provide a list of atheletes that competed in olympics exactly 9 times .
    Context :
       Table title : Olympic atheletes and number of times they've competed
       Ian Miller ,10
       Hubert Raudaschl,9
       Afanasijs Kuzmins, 9
       Nino Salukvadze, 9
       Piero d'Inzeo, 8
       Raimondo d'Inzeo, 8
       Claudia Pechstein, 8
       Jaqueline Mourão, 8
       Ivan Osiier, 7
       François Lafortune, Jr, 7
"""


response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(response.text)