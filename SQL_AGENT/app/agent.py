import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
import pathlib


from app.logger import get_logger


logger = get_logger("sql-agent")

load_dotenv()


DB_PATH = pathlib.Path("Chinook.db")
DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"



def download_db_if_needed():
    if not DB_PATH.exists():
        logger.info("Downloading chinook database")
        response = requests.get(DB_URL)
        response.raise_for_status()
        DB_PATH.write_bytes(response.content)
        logger.info("Database downloaded Succesfully")

def create_sql_agent()->AgentExecutor:
    download_db_if_needed()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "sql-agent-chinook")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")

    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"  🌐 Google API: {'✅' if google_key else '❌ Missing'}")
    logger.info(f"  🔑 OpenAI API: {'✅' if openai_key else '❌ Missing'}")

    llm = ChatOpenAI(model='gpt-4o-mini')
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results and return the answer.

Guidelines:
- Limit queries to {top_k} results
- Only query relevant columns
- ALWAYS check schema first with sql_db_schema
- Use sql_db_query_checker before sql_db_query
- NO DML (INSERT/UPDATE/DELETE/DROP)
""".format(dialect=db.dialect, top_k=5)

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return  AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True)


sql_agent_executor = create_sql_agent()