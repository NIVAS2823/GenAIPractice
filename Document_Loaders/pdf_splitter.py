from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load and split documents
loader = PyPDFLoader(r'D:\Personal\Agentic_Ai\Gen_ai_practice\Document_Loaders\Nivas_Mididhodi_Resume.pdf')
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
docs = loader.load()
chunks = splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM and prompt
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use only the context below to answer the question.
If answer is not found in the context, say 'INFORMATION NOT FOUND'

Context: 
{context}

Question: 
{question}

Answer: 
"""
)

# FIXED CHAIN: Use RunnablePassthrough to pass question + retrieve context
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)

# Test
question = "Summarize the pdf in 100 words"
response = rag_chain.invoke(question)  # Pass just the question string
print(f"Q: {question}\nA: {response}")