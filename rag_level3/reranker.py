from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANK_MODEL)

def rerank(question,docs,top_k=5):
    pairs = [(question,d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(docs,scores))

    scored.sort(key=lambda x : x[1],reverse=True)

    return [doc for doc,_ in scored[:top_k]]