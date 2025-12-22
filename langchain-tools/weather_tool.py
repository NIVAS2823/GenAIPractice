from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv


load_dotenv()

@tool
def get_current_weather(location:str,unit:str="celcius")->str:
    """
    Get the current weather in a given location.
    
    Args:
        location: The city name (e.g., London, Paris)
        unit: Temperature unit (celsius or fahrenheit) 
    """

    weather_data = {
        "London":{"temperature":15,"condition":"Cloudy"},
        "Paris":{"temperature":18,"condition":"Sunny"},
    }

    data = weather_data.get(location,{"temperature":20,"condition":"Unknown"})

    return f"{data['temperature']} {unit[0].upper()}, {data['condition']}"


llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0)

tools= [get_current_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful weather assistant"),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

response = agent_executor.invoke({"input":"What's the Weather in London and paris"})

print(f"\n {response['output']}")