from typing import Annotated
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


tavily_tool =  TavilySearch(max_results = 5)

repl = PythonREPL()

@tool
def python_repl_tool(code : Annotated[str,"The Python code to generate chart"],):
    """Use this to execute python code. If You want to see the output of a value 
    You should print it with `print(...)`. This is visible to the user"""

    try:
        result = repl.run(code)
    except Exception as e:
        return f"Failed to run the code {e}"
    
    result_str = f"Succesfully executed ```python {code}```\n Stdout: {result}"

    return (
        result_str + "\n\n If you have completed all tasks,respond with FINAL ANSWER"
    )


def make_system_prompt(suffix: str):
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant using the ReAct pattern.\n\n"
            "You have access to the following tools:\n"
            "{tools}\n\n"
            "Tool names: {tool_names}\n\n"
            "Use the following format:\n"
            "Thought: you should always think about what to do\n"
            "Action: the tool name\n"
            "Action Input: the input to the tool\n"
            "Observation: the result of the tool\n"
            "...\n"
            "Final Answer: the final response\n\n"
            f"{suffix}"
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

from typing import Literal
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_classic.agents import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState,END
from langgraph.types import Command


llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


def get_next_node(last_message:BaseMessage,goto:str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


research_agent = create_react_agent(llm=llm,tools=[tavily_tool],prompt=make_system_prompt("You can only do research. You are working with Chart generator colleague"),)

def research_node(state:MessagesState,)->Command[Literal["chart_generator",END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result['messages'][-1],"chart_generator")

    result['messages'][-1] = HumanMessage(
        content = result['messages'][-1].content,
        name="researcher"
    )

    return Command(
        update={
            'messages':result['messages']
        },
        goto=goto
    )

chart_agent = create_react_agent(llm=llm,tools=[python_repl_tool],prompt=make_system_prompt("You can only generate charts.You are working with research colleague"),)

def chart_node(state:MessagesState)->Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result['messages'][-1],"researcher")
    result['messages'][-1] = HumanMessage(
        content = result['messages'][-1].content,
        name='chart_generator'
    )
    return Command(
        update={
            'messages':result['messages'],
        },
        goto=goto

    )


from langgraph.graph import StateGraph,START

workflow = StateGraph(MessagesState)
workflow.add_node("researcher",research_node)
workflow.add_node("chart_generator",chart_node)

workflow.add_edge(START,"researcher")
graph = workflow.compile()


# png_bytes = graph.get_graph().draw_mermaid_png()
# with open("agent_graph.png", "wb") as f:
#     f.write(png_bytes)

# print("Graph saved as agent_graph.png")

events = graph.stream(
    {
        "messages": [
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print("----")
    print(s)
    print("----")