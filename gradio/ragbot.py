from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import gradio as gr
from gradio.themes import Soft


# ---------------------------
# Load & Chunk PDF
# ---------------------------
pdf_path = r"D:\Personal\Agentic_Ai\Gen_ai_practice\NIPS-2017-attention-is-all-you-need-Paper.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(pages)


# ---------------------------
# Embeddings + Vector Store
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",          # 🔥 IMPORTANT
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)


# ---------------------------
# LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ---------------------------
# Chat Logic
# ---------------------------
def chat(query, history):
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = f"""
You are an AI assistant answering questions using a research paper.

Rules:
- Use ONLY the provided context
- The paper may assume prior knowledge — explain clearly using available information
- Paraphrasing is allowed
- Do NOT hallucinate
- If no relevant explanation exists, say:
  "Information not available in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    history = history or []
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return history


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Transformer Paper RAG Chatbot")

    chatbot = gr.Chatbot(height=380)
    msg = gr.Textbox(
        placeholder="Ask anything about Transformer Architecture",
        label="Your Question"
    )

    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], chatbot)
    clear.click(lambda: [], None, chatbot)

demo.launch(theme=Soft(), share=False)
