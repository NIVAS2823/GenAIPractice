from query_expansion import expand_query
from retriever_expanded import retrieve_with_expanded_queries
from reranker import rerank

from langchain_google_genai import GoogleGenerativeAI
from config import LLM_MODEL

def answer_with_query_expansion(user_query:str):
    print("\n---Expanding Query---")
    expanded  = expand_query(user_query)

    for e in expanded:
        print(".",e)

    print("\n Retrieving Documents---")
    docs = retrieve_with_expanded_queries(expanded,k=5)

    if not docs:
        return "No documents found",[]
    
    print(f"Retrieved {len(docs)} docs")


    print("---Re ranking documents---")
    best_docs = rerank(user_query,docs,top_k=4)

    context = "\n\n".join(d.page_content for d in best_docs)

    llm = GoogleGenerativeAI(model=LLM_MODEL)

    prompt = f""""
            Use Only the context below to answer the queries.
            If answer not found , Say 'Information out of my capability'.

            Context:
            {context}

            Question : {user_query}

            Answer:
            """
    
    response = llm.invoke(prompt)

    answer_text = getattr(response,"content",str(response))

    return answer_text,best_docs