import asyncio
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")


async def main():
    loader = PyPDFLoader(
        r"D:\Personal\Agentic_Ai\Gen_ai_practice\langchain_gemini\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )

    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=[page.page_content for page in pages],
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # print(f"Loaded {len(pages)} pages successfully")

    query = "how are position embeddings implemented"

    retrieved_documents = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_documents])

    prompt = f"""
        You are an AI Assistant. Answer the question based only on the context below.
        Do not hallucinate and give correct answer only if the answer is present in the context,
        otherwise simply say 'Information not found'
            
        Context:
        {context}

        Question : {query}

        Answer :
        """
    
    response = llm.invoke(prompt)

    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
