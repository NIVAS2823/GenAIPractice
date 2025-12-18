from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel


def ask(question):

    # Embeddings must match ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector DB
    db = Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 20})


    # Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use ONLY the context below to answer the question.
If the answer is not found in the context, say "Information not found".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # -------------------------
    # 100% CORRECT LCEL RAG FLOW
    # -------------------------

    rag_chain = (
        RunnableParallel(
            question = RunnableLambda(lambda x: x["question"]),

            # retriever must ALWAYS receive a STRING, not a dict
            context = RunnableLambda(
                lambda x: retriever.invoke(x["question"])
            ),
        )
        |
        # convert retrieved docs → string context
        RunnableLambda(
            lambda x: {
                "question": x["question"],
                "context": "\n\n".join([doc.page_content for doc in x["context"]])
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run pipeline
    result = rag_chain.invoke({"question": question})

    print("\nANSWER:\n", result)

    print("\n--- Retrieved Chunks ---")
    for d in retriever.invoke(question):
        print("\n", d.page_content[:200], "\n---")


if __name__ == "__main__":
    q = input("Ask a question: ")
    ask(q)
