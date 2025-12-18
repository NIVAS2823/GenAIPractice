from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types


MODEL_ID = 'gemini-2.5-flash'
client = genai.Client()

app = FastAPI()

class PromptRequest(BaseModel):
    topic:str | None = None

@app.get('/')
def root():
    return {"message":"Prompt Techniques"}

@app.post('/prompt')
def prompt_technique(payload:PromptRequest):
    topic = payload.topic

    techniques = {
        "chain_of_thought":{
            "defination":"A prompting method where the model explains its reasoning step by step",
            "when_to_use":"Use for maths,logic,multi-step reasoning",
            "example":"Explain your reasoning step by step : What is 23 * 17?"
        },
        "Zero_shot":{
            "defination":"Model performs a task without any examples ",
            "when_to_use":"For simple tasks",
            "example":"Translate to French : How are You?"
        },
        "few_shot":{
            "defination":"You provide a few examples to guide the model",
            "when_to_use":"When strict output and format./style needed",
            "example":"Q : 2+3 : A:5 | Q:7+8 A:?"
        },
        "role_prompting":{
            "defination":"Ask the model to assume a specific role",
            "when_to_use":"Expert behaviour required",
            "example":"Act as a senior data scientist and expalin about PCA"
        },
        "function_calling":{
            "defination":"Model returns stuctured JSON for tools/functions",
            "when_to_use":"For backend integrations",
            "example":"Return JSON with :city,temparature"
        },
        "deliberate":{
            "defination":"Ask the model to think twice and generate multiple solutions",
            "when_to_use":"complex or ambigious problem",
            "example":"Give 3 solutions and pick best one"
        }
    }
    
    if topic:
        key = topic.lower().replace(" ","_")

        if key not in techniques:
            return {"error":"Topic not found","available":list(techniques.keys())}
        
        return{"topic":topic,"details":techniques[key]}

    gemini_topic = """
        Explain these prompting techniques in short ,clear sumamry:
        -Chain of Thought
        -Zero Shot
        -Few Shot
        -ROle prompting
        -Function Calling
        -Deliberate Calling
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[gemini_topic]
        )
        gemini_summary  = topic.text

    except Exception as e:
        gemini_summary = f"Gemini Error: {str(e)}"

    
    return {
        "prompting_techniques":techniques,
        "gemini_summary":gemini_summary
    }