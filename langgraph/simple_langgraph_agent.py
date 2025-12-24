from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : list[str]
    next_action: str
    final_answer:str

llm = ChatOpenAI(model='gpt-4o-mini')

def reasoning_node(state:AgentState)->AgentState:
    """
    Calls LLM to decide next action based on conversation history,
    retruns updated state with next_action set
    """

    messages = state["messages"]
    prompt = f""" Conversation so far: {messages}. What should we do next? Respond with 'search' or 'Respond'
"""
    response = llm.invoke(prompt)
    next_action = response.content.strip().lower()


    return {
        "messages":messages + [f"Agent decided action : '{next_action}' "],
        "next_action":next_action
    }

def search_node(state:AgentState)->AgentState:
    """Simulates a search tool execution"""
    return {"messages":state["messages"] + ["Executed Search tool"],
            "next_action":"respond"}


def respond_node(state: AgentState) -> AgentState:
    """Generates final LLM response"""

    messages = state["messages"]

    prompt = f"""
    Based on the following conversation and search results,
    provide a clear final answer to the user.

    Context:
    {messages}
    """

    response = llm.invoke(prompt)

    return {
        "messages": messages + [f"LLM Response: {response.content}"],
        "next_action": "end",
        "final_answer": response.content
    }



def router(state:AgentState)->AgentState:
    """Decides which node to visit next based on next_action.
    Returns node  name as string"""

    action = state.get("next_action","")

    if action == "search":
        return "search"
    elif action == "respond":
        return "respond"
    else:
        return END
    
workflow = StateGraph(AgentState)

workflow.add_node("reasoning",reasoning_node)
workflow.add_node("search",search_node)
workflow.add_node("respond",respond_node)

workflow.set_entry_point("reasoning")

workflow.add_conditional_edges(
    "reasoning",
    router,
    {
        "search":"search",
        "respond":"respond",
        END:END
    }
)

workflow.add_edge("search","reasoning")
workflow.add_edge("respond",END)


app = workflow.compile()

initial_state = {
    "messages":["User :Explain about 'agent_scratchpad in Langchain"],
    "next_action":"",
    "final_answer":""
}

result = app.invoke(initial_state)

for msg in result["messages"]:
    print(msg)
