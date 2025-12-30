from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from mcp.client.langchain import MCPToolkit
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

mcp_tools = MCPToolkit.from_server(
    command = ["python","mcp_server.py"]
)

agent = create_tool_calling_agent(llm=llm,tools=mcp_tools.tools)

executor = AgentExecutor(agent=agent,tools = mcp_tools.tools)

response = executor.invoke({"input":"What is 7 multiplied by 6 plus 5?"})


print(response['output'])