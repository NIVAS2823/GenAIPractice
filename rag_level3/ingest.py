from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from rag_level3.config import VECTOR_DB_PATH,EMBED_MODEL


def ingest():
    print("Loading documents")
    loader = TextLoader('data/notes.txt')
    docs = loader.load()

    print("Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap = 80
    )

    chunks = splitter.split_documents(docs)

    for i,c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    print("Embedding & Storing...")

    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("Ingestion Completed...")


if __name__ == "__main__":
    ingest()