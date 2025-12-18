

# Generative AI & Agentic Systems Sandbox

A comprehensive repository documenting the development and optimization of **LLM-based applications**, **RAG pipelines**, and **Agent Orchestration** using LangChain, OpenAI, and Google Gemini.

## 🛠 Tech Stack

* **Orchestration:** LangChain, LangGraph
* **Models:** OpenAI (GPT-4o/o1), Google Gemini (Pro/Flash)
* **Vector Databases:** ChromaDB / FAISS 
* **Language:** Python 3.11

---

## 🏗 System Architecture

The programs in this repository follow a modular architecture designed for scalability and reliability in agentic workflows.

### 1. LangChain Integration

Implementation of core LangChain components:

* **Chains:** LCEL (LangChain Expression Language) for declarative compositions.
* **Memory:** Windowed and Summary buffer implementations for stateful conversations.
* **Tools:** Custom tool-calling wrappers for API and database interactions.

### 2. Provider Implementations

Comparison of model behaviors and integration patterns across top-tier providers:

* **OpenAI:** Function calling, structured outputs, and fine-tuning scripts.
* **Gemini:** Multi-modal processing and long-context window utilization.

### 3. RAG (Retrieval-Augmented Generation)

Advanced retrieval strategies to mitigate hallucinations:

* **Ingestion:** Recursive character splitting and semantic chunking.
* **Retrieval:** Hybrid search (Keyword + Vector) and Re-ranking (Cross-Encoders).
* **Evaluation:** RAGAS metrics for faithfulness and relevancy.

### 4. Prompt Engineering & Optimization

Systematic approaches to prompt construction:

* **Few-Shot Learning:** Dynamic example selection.
* **Chain-of-Thought (CoT):** Forcing reasoning paths for complex logic.
* **System Messages:** Role-based constraint enforcement.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have API keys for OpenAI and Google AI Studio.

### Installation

```bash
git clone https://github.com/NIVAS2823/GenAIPractice.git
cd [REPO_NAME]
for every directory there will be separate requirements.txt file
cd #desired directory
pip install -r requirements.txt
