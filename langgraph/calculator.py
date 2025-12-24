from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode  # Use prebuilt ToolNode for simplicity



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Fixed valid model name
    temperature=0.2
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`. 
    Args:
      a: First int
      b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`. 
    Args:
       a: First int
       b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divides 'a' by 'b'. 
    Args:
       a: First int
       b: Second int (cannot be zero)
    """
    if b == 0:
        return "Error: Division by zero"
    return a / b

tools = [add, multiply, divide]

# Bind tools to model
model_with_tools = model.bind_tools(tools)

# ✅ Fixed: Proper state definition
class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int  # Fixed typing

def llm_call(state: MessageState) -> dict:
    """LLM decides whether to call a tool or not"""
    # Fixed: Proper message construction
    messages = [
        SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs. Use the appropriate tool for addition, multiplication, or division."
        )
    ] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def should_continue(state: MessageState) -> Literal["tools", "END"]:
    """Decide if we should continue the loop or stop"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # Fixed: matches node name
    return END


workflow = StateGraph(MessageState)


workflow.add_node("llm", llm_call)
workflow.add_node("tools", ToolNode(tools))  # Use prebuilt ToolNode

workflow.set_entry_point("llm")

# Add conditional edges
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Add edge from tools back to llm
workflow.add_edge("tools", "llm")

# ✅ Fixed: Compile the graph
agent = workflow.compile()

# Test the agent
print("=== Testing arithmetic agent ===")
input_messages = [HumanMessage(content="Add 5 and 3")]
result = agent.invoke({"messages": input_messages, "llm_calls": 0})

print("\nFinal result:")
for m in result["messages"]:
    if m.type == "human":
        print(f"Human: {m.content}")
    elif m.type == "ai":
        print(f"AI: {m.content}")
    elif m.type == "tool":
        print(f"Tool: {m.content}")

print(f"\nTotal LLM calls: {result['llm_calls']}")