from typing import TypedDict
from langgraph.graph import StateGraph,END


class GraphState(TypedDict):
    message:str


def greet_node(state:GraphState)->GraphState:
    return {
        "message":f"Hello, {state['message']}! Welcome to LangGraph Learning"
    }

builder  = StateGraph(GraphState)

builder.add_node("greet",greet_node)
builder.set_entry_point("greet")
builder.add_edge("greet",END)


graph = builder.compile()

result  = graph.invoke({"message":"Agentic AI Developer"})

print(result)