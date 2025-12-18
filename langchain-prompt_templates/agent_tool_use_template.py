from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal,Union
import json

class ToolCall(BaseModel):
    """Data Structure for when the agent decides to use a tool"""
    action : Literal["TOOL"] = Field(description="Must be 'TOOL' if a tool is required" )
    tool_name :str = Field(description="The name of the tool to be called (e.g.,'Google Search).")
    tool_input:str = Field(description="The precise search query or input required for the tool")


class DirectAnswer(BaseModel):
    """Data Structure for when the agent decides to answer directly."""
    action:Literal["DIRECT_ANSWER"] = Field(description="Must be 'DIRECT_ANSWER' if the question can be answered from internal knowledge")
    answer:str = Field(description="The final answer to the user's question")

AgentDecision = Union[ToolCall,DirectAnswer]

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = JsonOutputParser(pydantic_object=AgentDecision)

DUMMY_TOOLS = """
[1] Tool Name: Google_Search
    Description: Use this tool for any question requiring up-to-date, real-time, or external knowledge (e.g., current news, specific dates, complex calculations, or detailed product specifications).
    Input: The query string to search.
[2] Tool Name: Internal_Database
    Description: Use this tool ONLY if the question relates to company policies, known facts about your local environment, or historical data *known* to the system. (Note: Since we have no actual internal data, prioritize direct answer or Google_Search).
    Input: The key or ID to look up.
"""

agent_template = PromptTemplate(
    input_variables=["query","tools"],
    template=(
        "You are an intelligent Agent Assistant. Your task is to decide whether the user's "
        "QUERY can be answered using your own internal knowledge or if you need to use an "
        "available tool to find the information.\n\n"
        "AVAILABLE TOOLS:\n"
        "------------------\n"
        "{tools}\n"
        "------------------\n\n"
        "USER QUERY: {query}\n\n"
        "Please provide your decision in the required JSON format. Choose either the 'TOOL' action "
        "or the 'DIRECT_ANSWER' action."
    ),
    partial_variables={
        "tools":DUMMY_TOOLS
    }
)


agent_chain = agent_template | llm | parser

query_tool = "Who won the latest Formula 1 race and what year was it?"
print(f"---Query 1 : External Knowledge Needed ---")
print(f"---Query : {query_tool}")

try:
    tool_response = agent_chain.invoke({"query":query_tool})
    print("\n----Agent decision (Tool call)")
    print(json.dumps(tool_response,indent=2))
except Exception as e:
    print(f"Error : {e}")

query_direct = "What is the capital of France"
print(f"---Query 2 :Internal Knowledge Available")
print(f"Query 2 : {query_direct}")

try:
    direct_response = agent_chain.invoke({"query":query_direct})
    print("\n Agent Decision (Direct Answer)")
    print(json.dumps(direct_response,indent=2))
except Exception as e:
    print(f"Error : {e}")