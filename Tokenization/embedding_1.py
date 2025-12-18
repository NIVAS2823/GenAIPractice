from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The weather is lovely today",
]

embeddings = model.encode(sentences)

print(embeddings.shape)