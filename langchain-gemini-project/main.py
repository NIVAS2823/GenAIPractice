from fastapi import FastAPI,UploadFile
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS,Chroma
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

import os
print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

qa = """
    You are an expert AI Assitant .
    Use the retrieved context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer in simple,clear language
    """

prompt = PromptTemplate(
    template=qa,
    input_variables=["context","question"]
    
)


@app.post('/ingest')
async def ingest_document(file:UploadFile):
    """
    Loads a PDF -> splits text -> create embeddings -> stores in FAISS
        """
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    file_path = f"data/{file.filename}"

    with open(file_path,"wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_path)
    docs  = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150
    )

    chunks  = splitter.split_documents(docs)

    vector_db = FAISS.from_documents(chunks,embeddings)

    vector_db.save_local("vectorstore/faiss_index")

    return {"status":"success","chunks_created":len(chunks)}

@app.get('/query')
async def query_rag(question:str):
    vector_db = FAISS.load_local("vectorstore/faiss_index",embeddings,allow_dangerous_deserialization=True)

    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(question)



    context = "\n\n".join(d.page_content for d in docs)

    final_prompt = prompt.format(context=context,question=question)

    response = llm.invoke(final_prompt)


@app.post("/ingest-chroma")
async def ingest_into_chroma(file: UploadFile):
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    chroma_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore/chroma_db"
    )
    chroma_store.persist()

    return {"status": "success", "vector_store": "chroma"}