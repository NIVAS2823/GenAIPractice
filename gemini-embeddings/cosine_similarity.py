from google import genai
from google.genai import types
import numpy as np
from sklearn.metrices.pairwise import cosine_similarity

client = genai.Client()

texts = [
    "What is the meaning of life",
    "what is the purpose of existence",
    "How do earn money?"
]

result = [
    np.array(e.values) for e in client.models_embed_content(
        model='gemini-embedding-001',
        contents = texts,
        config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    ).embeddings
]



print(result.embeddings)