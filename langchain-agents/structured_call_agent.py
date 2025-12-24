from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_structured_chat_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.tools import tool

@tool
def search_database(query:str,filters:str = None)->str:
    """
    "Search a database using a simple query string and optional filters string."
    
    Args:
        query: The search query
        filters: Optional filters in JSON format
    """

    result = f"Searching for {query}"

    if filters:
        result += f"With filters: {filters}"

    return result + "- Found 5 results"

llm = ChatOpenAI(model='gpt-4o-mini')

tools = [search_database]


prompt = ChatPromptTemplate.from_messages([
     ("system", """Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
$JSON_BLOB

Observation: action result
 (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
      
      IMPORTANT RULES:
- action_input MUST be a JSON OBJECT, not a string
- Do NOT include schema, types, titles, or defaults
- For search_database:
  - query: string
  - filters: string or null
- Respond with ONLY a JSON blob and NOTHING else

Begin! Reminder to ALWAYS respond with a valid json blob of a single action."""),
    ("human", "{input}\n\n{agent_scratchpad}")

])

agent = create_structured_chat_agent(llm,tools,prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response= agent_executor.invoke({"input":"Search for 'machine learning' articles published in 2024"})

print(f"\nFinal Answer: {response['output']}")