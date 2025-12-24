from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

@tool
def get_user_name() -> str:
    """Get the user's name from conversation history."""
    return "I remember your name from our chat!"

tools = [get_user_name] 

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant with memory of previous conversations"),
    MessagesPlaceholder(variable_name="chat_history"),
("human","{input}"),
MessagesPlaceholder(variable_name="agent_scratchpad")
])


llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

agent = create_tool_calling_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,memory=memory,verbose=False)

print("=== First message ===")
result1 = agent_executor.invoke({"input": "My name is Nivas and my nickname is Luffy"})
print(f"Output: {result1['output']}\n")

print("=== Second message (tests memory) ===")
result2 = agent_executor.invoke({"input": "What's my nick  name?"})
print(f"Output: {result2['output']}")