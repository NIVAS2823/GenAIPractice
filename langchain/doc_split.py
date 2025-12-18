from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import os

pdf_path = "Nivas_Mididhodi_Resume.pdf"
loader = PyPDFLoader(pdf_path)

pages = loader.load()

pdf_text = "\n".join([ page.page_content for page in pages])

# print(pdf_text)   

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 300,
    chunk_overlap = 50
)

chunks = splitter.split_text(pdf_text)
print(f"Total length of chunks {len(chunks)}")


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.3)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the chunk : \n {text}"
)

chain = prompt | llm

for i, chunk in enumerate(chunks):
    chunk_summary = chain.invoke({"text":chunk}).content
    print(f"\n\n{i+1} - {chunk}\n Summary - {chunk_summary}\n")