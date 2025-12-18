from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import VECTOR_DB_PATH,EMBED_MODEL

embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

db  = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

def retrieve_with_expanded_queries(expanded_queries,k=5):
    docs = []
    for q in expanded_queries:
        docs += db.similarity_search(q,k=k)
    return docs