from fastapi import FastAPI,HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()

MODEL_ID = 'gemini-2.5-flash'

client = genai.Client()

class TextInput(BaseModel):
    text:str

system_prompt = """
    You are an information extraction model. 
Your task is to analyze the input text and extract structured entities with high accuracy.

Follow these rules:
- Identify and extract ALL relevant entities explicitly mentioned in the text.
- Do NOT infer or hallucinate information that is not present.
- Return the output strictly in valid JSON format.
- Do not include explanations, notes, or additional text.
- If an entity is missing, return it as null.

Extract the following fields:
- persons: List of people mentioned
- organizations: List of companies, institutions, or groups
- locations: Cities, countries, or geographical regions
- dates: Any date, year, or timeframe
- events: Conferences, launches, announcements, or historical events
- products: Tools, frameworks, or named items
- misc: Any other notable entities

Output JSON format:
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "dates": [],
  "events": [],
  "products": [],
  "misc": []
}

    """

config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    temperature=0.1,
)

@app.post('/v1/summarize')
def summarize_text(payload:TextInput):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[f'Summarize the follwoing text ::\n {payload.text}'],
    )
        if response and response.text:
            return {"summary":response.text}
        raise HTTPException(status_code=404,detail='Summary not found')
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


@app.post('/extract/entities')
def extract_entities(medical_text:TextInput):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[f'Extract  entitites from the following medical text\n\n : {medical_text.text}'],
            config=config
        )
        if response and response.text:
            return {"entites":response.text}
        
        raise HTTPException(status_code=404,detail='Extraction not found')
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))





