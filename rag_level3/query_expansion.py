from langchain_google_genai import GoogleGenerativeAI

MODEL = 'gemini-2.5-flash'

llm = GoogleGenerativeAI(model=MODEL)

def expand_query(user_query:str,n:int=5):
    prompt = f"""
        Expand the following question into {n} different detailed search queries.
        Each Query should capture differnent angles or keywords

        User Question:
        {user_query}

        Return each expanded query as a bullet point    
"""
    
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp,"content") else str(resp)

    expanded = [line.strip("-.").strip()
                    for line in text.split("\n") if line.strip()]
    
    return expanded