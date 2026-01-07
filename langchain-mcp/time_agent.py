from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import requests
import json
from dotenv import load_dotenv

load_dotenv()

@tool
def get_city_time(city: str) -> str:
    """Get current time for any city"""
    try:
        # Direct HTTP call to MCP server
        response = requests.post(
            "http://localhost:8000/mcp",
            json={"tools": ["get_city_time"], "arguments": {"city": city}},
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        return result.get("result", "Error getting time")
    except:
        return "❌ MCP server not running at localhost:8000"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_city_time]
agent = create_react_agent(llm, tools)

cities = ["New York", "London", "Tokyo", "Mumbai"]
for city in cities:
    print(f"\n🕐 {city}:")
    response = agent.invoke({
        "messages": [{"role": "user", "content": f"Current time in {city}"}]
    })
    print(response["messages"][-1].content)