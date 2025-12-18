from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage,SystemMessage,ToolMessage,HumanMessage
from typing_extensions import  TypedDict,Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph,START,END


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

@tool
def multiply(a:int,b:int)->int:
    """Multiply  `a` and `b` 
    Args:
      a:First int
      b:Second int
    """
    return a * b

@tool
def add(a:int,b:int)->int:
    """Adds `a` and `b` 
    Args:
       a:First int
       b:Second int
    """
    return a + b

@tool
def divide(a:int,b:int)->int:
    """
    Divides 'a' and 'b' 
    Args :
       a:First int
       b:Second int
    """

    return a/b

tools = [add,multiply,divide]

tools_by_name  = {tool.name:tool for tool in tools}

model_with_tools = model.bind_tools(tools)


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]
    llm_calls :int


def llm_call(state:dict):
    """ LLM decides  whether to call a tool or not """
    return {
        "messages":[
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant  tasked with performing arithmetic on a set of inputs"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls":state.get('llm_calls',0)+1
    }


def tool_node(state:dict):
    """Performs the tool call"""

    result = []
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages":result}

def should_continue(state:MessageState)->Literal["tool_node",END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    
    return END









agent = agent_builder.compile()

messages = [HumanMessage(content="Add 5 and 3")]
messages = agent.invoke({"messages":messages})

for m in messages["messages"]:
    if hasattr(m,"content"):
        print(m.content)
    else:
        print(m.text)



