from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate


@tool
def joke_rating(joke: str) -> str:
    """Rates a joke"""
    return "😂 Funny joke!"

tools = [joke_rating]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True
)


prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the following tools:

{tools}

Use the following format:

Question: {input}
Thought: you should think step by step
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result
... (this can repeat)
Thought: I now know the final answer
Final Answer: the answer

{agent_scratchpad}
"""
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


print("\n=== INVOKE ===\n")
result = executor.invoke({"input": "Tell me a joke"})
print(result["output"])


print("\n=== STREAM MODE: values ===\n")
for chunk in executor.stream(
    {"input": "Tell me a joke"},
    stream_mode="values"
):
    print(chunk)
