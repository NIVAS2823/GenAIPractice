from dataclasses import dataclass
from langchain.tools import tool,ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=1000,
    timeout=10
)

SYSTEM_PROMPT = """
    You are an expert weather forecaster.
    who speaks in puns

    You have access to two tools:

    -get_weather_for_location: use this to get the weather for specific  location
    -get_user_location: use this to get the user's location

    If the user asks you for the weather, make sure you know the location.
    If you can tell the question from that they mean
    wherever they are, use the get_user_location tool to find 
    their location
    """


@dataclass
class Context:
    """"Custom runtime context schema"""
    user_id: str

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response:str

    weather_conditions:str | None = None

@tool
def get_weather_for_location(city:str)->str:
    """Get the weather for the given city"""
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime:ToolRuntime[Context]) ->str:
    """Retrieve user information based on User ID"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location,get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

config = {"configurable":{"thread_id":"1"}}

response = agent.invoke(
    {"messages":[{"role":"user","content":"what is the weather outside"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)   


print(response['structured_response'])