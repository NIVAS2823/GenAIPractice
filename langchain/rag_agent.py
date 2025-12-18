from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # Fixed: use ChatOpenAI directly
from langchain_core.tools import create_retriever_tool  # Fixed: proper tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition  # Fixed: proper imports
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal

# ===========================================
# 1-3. LOAD, SPLIT, STORE (✅ Correct)
# ===========================================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [d for sub in docs for d in sub]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=5
)
chunks = splitter.split_documents(docs_list)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
retriever = store.as_retriever()

# ===========================================
# 4. FIXED TOOL
# ===========================================
retrieval_tool = create_retriever_tool(  # ✅ Fixed: use create_retriever_tool
    retriever,
    "retrieve_blog_posts",
    "Search Lilian Weng's blog posts about LLM agents, prompt engineering, and adversarial attacks."
)

tools = [retrieval_tool]

# ===========================================
# 5. MAIN LLM (✅ Fixed)
# ===========================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_query_or_respond(state: MessagesState):
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

# ===========================================
# 6. FIXED GRADER
# ===========================================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

grader_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "You are a grader checking if a retrieved document is relevant.\n"
        "Document:\n{context}\n\n"
        "User question:\n{question}\n"
        "Answer only 'yes' or 'no'."
    )
    
    chain = prompt | grader_model.with_structured_output(GradeDocuments)
    response = chain.invoke({"context": context, "question": question})
    
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

# ===========================================
# 7-8. FIXED NODES
# ===========================================
def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this question to be clearer:\n{question}"
    )
    chain = prompt | model
    response = chain.invoke({"question": question})
    return {"messages": [response]}

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "Answer in max 3 sentences using this context:\n"
        "Question: {question}\n"
        "Context: {context}"
    )
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    return {"messages": [response]}

# ===========================================
# 9. FIXED GRAPH
# ===========================================
workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode(tools))  # ✅ Fixed: pass tools list
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("grade_documents", grade_documents)  # ✅ Added missing node
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)
workflow.add_edge("retrieve", "grade_documents")  # ✅ Fixed edge
workflow.add_conditional_edges(
    "grade_documents",
    grade_documents,  # ✅ Use function reference
    {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"}
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()

# ===========================================
# 10. FIXED EXECUTION (SIMPLE!)
# ===========================================
print("\n===== Running Agent =====\n")
result = graph.invoke({
    "messages": [HumanMessage(content="What does Lilian Weng say about types of reward hacking?")]
})

for message in result["messages"]:
    message.pretty_print()