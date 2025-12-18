from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest():
    print("Loading documents...")
    loader = TextLoader('data/notes.txt')
    docs = loader.load()

    print("Splitting Documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print("Embedding using HuggingFace (FREE)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    print("Ingestion completed successfully!")

if __name__ == "__main__":
    ingest()
