
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.tools import TavilySearchResults
from datetime import datetime
from langchain_core.tools import tool
from langchain.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature=0
)

print("="*50)
print("LLM initialized")
print("="*50)



web_tool = TavilySearchResults()

tools = [web_tool]

agent = create_agent(llm,tools=tools)

agent_response = agent.invoke({"messages":[HumanMessage("Summarize todays news in AI innvoation and tech")]})

print(agent_response['messages'][-1].content)
