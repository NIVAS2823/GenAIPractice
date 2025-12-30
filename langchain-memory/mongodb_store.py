import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash') 

DB_URI = "mongodb://localhost:27017"

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    print("--- First message ---")
    input_1 = {"messages": [{"role": "user", "content": "Hi, I am Dev"}]}
    for chunk in graph.stream(input_1, config, stream_mode="values"):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

    print("\n--- Second message (Testing Memory) ---")
    input_2 = {"messages": [{"role": "user", "content": "What is my Name?"}]}
    for chunk in graph.stream(input_2, config, stream_mode="values"):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()