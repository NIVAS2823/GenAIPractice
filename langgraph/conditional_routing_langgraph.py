from typing import TypedDict,Literal
from langgraph.graph import StateGraph,END

class GraphState(TypedDict):
    feedback:str
    sentiment:Literal['positive','negative']


def classify_sentiment(state:GraphState)->GraphState:
    if "good" in state['feedback'].lower():
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "feedback":state['feedback'],
        "sentiment":sentiment
    }

def positive_sentiment(state:GraphState)->GraphState:
    print("Postive feedback received")
    return state


def negative_sentiment(state:GraphState)->GraphState:
    print("Negative feedback received")
    return state

def route(state:GraphState):
    return state['sentiment']

builder = StateGraph(GraphState)

builder.add_node("classify",classify_sentiment)
builder.add_node("positive",positive_sentiment)
builder.add_node("negative",negative_sentiment)

builder.set_entry_point("classify")
builder.add_conditional_edges(
    "classify",
    route,
    {
        "positive":"positive",
        "negative":"negative"
    }
)


builder.add_edge("positive",END)
builder.add_edge("negative",END)


graph = builder.compile()
result = graph.invoke({"feedback":"This LangGraph tutorial is bad!"})



print(result)