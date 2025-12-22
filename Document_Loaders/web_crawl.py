from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ✅ Fixed import
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

url = "https://stripe.com/in"

loader = WebBaseLoader(url)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="web_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.5
)

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

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser
)

question = "Explain about the website in 100 words"
response = rag_chain.invoke({"question": question})
print(f"Q. {question} \n A. {response}")