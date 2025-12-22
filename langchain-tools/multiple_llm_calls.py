from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from dotenv import load_dotenv

load_dotenv()

@tool
def get_time()->str:
    """
    Returns current time
    """

    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

llms = [
    ("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0)),
    ("llama-3.3-70b", ChatGroq(model="llama-3.3-70b-versatile", temperature=0)),
    ("gemini-2.5-flash", ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)),
]



prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helful assistant"),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
])
tools = [get_time]

for name,llm in llms:
    agent = create_tool_calling_agent(llm,tools,prompt)
    executor = AgentExecutor(agent=agent,tools=tools,verbose=False)
    response = executor.invoke({"input":"What's the time now:"})
    print(f"LLM : {name}. Response : {response['output']}")
