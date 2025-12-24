from typing import TypedDict
from langgraph.graph import StateGraph,END

class GraphState(TypedDict):
    text:str
    length:int

def clean_text(state:GraphState)->GraphState:
    cleaned = state['text'].strip().lower()
    return {
        "text":cleaned,
        "length":0
    }

def count_length(state:GraphState)->GraphState:
    return {
        "text":state['text'],
        "length":len(state['text'])
    }


builder = StateGraph(GraphState)

builder.add_node("clean_text",clean_text)
builder.add_node("count_length",count_length)

builder.set_entry_point("clean_text")
builder.add_edge("clean_text","count_length")
builder.add_edge("count_length",END)

graph = builder.compile()

result = graph.invoke({"text":"langgraph is    powerful","length":0})

print(result)