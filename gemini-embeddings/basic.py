from google import genai
from google.genai import types

client = genai.Client()

result = client.models.embed_content(
    model='gemini-embedding-001',
    contents ='What is the meeaning of life'
)


print(result.embeddings)