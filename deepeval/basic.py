from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from dotenv import load_dotenv


load_dotenv()

def get_weather(city:str)->str:
    """Returns the weather in the city"""
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)

# llm_with_tools = llm.bind_tools([get_weather])

# response = llm_with_tools.invoke("What is the weather in Hyderabad?",config={"callbacks":[CallbackHandler()]})

# tool_call = response.tool_calls[0]
# result = get_weather(**tool_call['args'])

result = llm.invoke("What is LLM Agent",config={"callbacks":[CallbackHandler()]})

print(result.content)



