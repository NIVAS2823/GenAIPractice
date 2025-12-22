from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate



@tool
def calculator(expression:str)->str:
    """
    Evaluates mathematical expression and returns the result

    Args : 
    Expression: A mathematical expression as a string(e.g.,"2+2","10*4")

    """

    try:
        result = eval(expression)
        return f"The result is {result}"
    
    except Exception as e:
        return f"Error calculating :{str(e)}"
    
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

tools = [calculator]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant with access to tools"),
        ("human","{input}"),
        ("placeholder","{agent_scratchpad}")

     ])

agent = create_tool_calling_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

response = agent_executor.invoke({"input":"what is 234 multiplied by 123"})

print(response['output'])