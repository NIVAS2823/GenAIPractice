#CLI Application

import os
import sys
import requests
from io import StringIO
from bs4 import BeautifulSoup

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the chunk:\n{text}"
)
chain = prompt | llm

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return [Document(page_content=f.read())]
    else:
        print("Unsupported File Type.")
        sys.exit(1)
    return loader.load()


def load_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.get_text(separator="\n")
    return [Document(page_content=content)]


def choose_splitter(source_type="text"):
    if source_type == "html":
        return HTMLHeaderTextSplitter(
            headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
        )
    elif source_type == "token":
        return TokenTextSplitter(chunk_size=200, chunk_overlap=50)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)


def summarize_chunks(docs, splitter, output_file):
    if isinstance(splitter, HTMLHeaderTextSplitter):
        html_text = docs[0].page_content
        chunks = splitter.split_text(html_text)
    else:
        chunks = splitter.split_documents(docs)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.page_content if isinstance(chunk,Document) else str(chunk)
            result = chain.invoke({"text": text}).content
            f.write(f"\n=== Chunk {i} ===\n")
            f.write(f"Original:\n{chunk.page_content}\n")
            f.write(f"Summary:\n{result}\n")

    print(f" Summary is saved to {output_file}")


if __name__ == "__main__":
    print("Langchain Document Summarizer:")
    print("1. Summarize a Local File")
    print("2. Summarize a Web URL")

    choice = input("Select an Option: ").strip()

    if choice == "1":
        file_path = input("Enter the full path of the file: ").strip()
        docs = load_file(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        s_type = "html" if ext == ".html" else "token" if ext == ".json" else "text"
        splitter = choose_splitter(s_type)
        summarize_chunks(docs,splitter,"summary_from local_txt")

    elif choice == "2":
        url = input("Enter Web URL to crawl and summarize: ").strip()
        docs = load_url(url)
        splitter = choose_splitter("html")
        summarize_chunks(docs, splitter, "summary_from_web.txt")

    else:
        print("Invalid input choice. Enter only 1 or 2.")
