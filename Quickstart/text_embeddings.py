from google import genai
from google.genai import types

client = genai.Client()


# prompt = 'Explain how AI works in a few words'
MODEL_ID = 'gemini-embedding-001'

response = client.models.embed_content(
    model=MODEL_ID,
    contents=[  "How do I get a driver's license/learner's permit?",
        "How do I renew my driver's license?",
        "How do I change my address on my driver's license?"],
)

print(response.embeddings)