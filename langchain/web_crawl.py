from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from  langchain_core.prompts import PromptTemplate
import os 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import requests
from bs4 import BeautifulSoup

url = "https://www.thezennialpro.com/terms"
output_file = "myschoolsync_landing_page.txt"

response = requests.get(url=url)
soup = BeautifulSoup(response.text,"html.parser")
original_text = soup.get_text(separator="\n")

# print(f"Original Text:{original_text}\n")


cleaned_text = "\n".join(line.strip() for line in  original_text.splitlines() if line.strip())

# print(f"\nCleaned text: {cleaned_text}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

chunks = splitter.split_text(cleaned_text)

prompt = PromptTemplate(
    input_variables="[text]",
    template="\nConvert this chunk into Telugu Language:\n {text}"
)

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)

chain = prompt | llm

with open(output_file,"w",encoding="utf-8") as f:
    for i,chunk in enumerate(chunks):
        result = chain.invoke({"text":chunk})
        f.write(f"\n Chunk {i+1} :")
        f.write(f"{chunk}")
        f.write(f"\nSummary of the chunk: {result}")

print("Website url converted to text file")
