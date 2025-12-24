from langgraph.graph import StateGraph,END,START
from typing import TypedDict,Literal
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI

class State(TypedDict):
    joke:str
    topic:str
    feedback:str
    funny_or_not:str


class Feedback(BaseModel):
    grade:Literal["funny","not funny"] = Field(description="Decide the joke is funny or not")
    feedback:str = Field(description="If the joke is not funny, provide feedback on how to improve it.")

llm = ChatOpenAI(model='gpt-4o-mini')

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state:State):
    """LLM Generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")


    return {"joke":msg.content}


def llm_call_evaluator(state:State):
    """LLM Evaluates the joke"""

    grade = evaluator.invoke(f"""
Evaluate the following joke.

    Joke:
    {state['joke']}

    Decide whether it is funny or not.
    If not funny, provide constructive feedback.
""")

    return {"funny_or_not":grade.grade,"feedback":grade.feedback}

def route_joke(state:State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == 'funny':
        return 'Accepted'
    
    elif state['funny_or_not'] == 'not funny':
        return "Rejected with feedback"
    

builder = StateGraph(State)


builder.add_node("llm_call_generator",llm_call_generator)
builder.add_node("llm_call_evaluator",llm_call_evaluator)

builder.add_edge(START,"llm_call_generator")
builder.add_edge("llm_call_generator","llm_call_evaluator")
builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        'Accepted':END,
        'Rejected with feedback':"llm_call_generator"
    },
)


graph = builder.compile()

result = graph.invoke({
    "topic": "Man",
    "joke": "",
    "feedback": "",
    "funny_or_not": ""
})

print(result)