from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI


load_dotenv()


tavily_client = TavilyClient()



llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    max_retries=6,          # Increase retries for agentic loops
    timeout=60              # Ensure long search tasks don't time out
)


def internet_search(
        query:str,
        max_results:int = 5,
        topic :Literal["general","news","finance"] = "general",
        include_raw_content:bool=False):
    """Run a web search"""

    return tavily_client.research(
        query,
        max_results = max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )


research_subagent = {
    "name":"research-agent",
    "description":"Used to search more in depth questions",
    "system_prompt":"You are a great researcher",
    "tools":[internet_search],
    "model":"openai:gpt-4o-mini",
}

subagents = [research_subagent]

agent = create_deep_agent(model=llm,subagents=subagents)

result = agent.invoke({"messages":[{"role":"user","content":"What is LangGraph"}]})

print(result["messages"][-1].content)