from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

print("LangSmith Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LangSmith Project:", os.getenv("LANGCHAIN_PROJECT"))

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


tools = [multiply, add]



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful math assistant. Use tools when needed."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

#create_tool_agent_calling
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -----------------------
# Run
# -----------------------

response = agent_executor.invoke(
    {"input": "What is 25 multiplied by 25, then add 100 to the result?"}
)

print("\nFinal Answer:", response["output"])
