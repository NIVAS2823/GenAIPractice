from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

loader = TextLoader("notes.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 100})

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.2
)

query = "How many centuries virat scored in ODI"

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