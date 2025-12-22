import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
import pathlib

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "sql-agent-chinook")


langsmith_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    print("🚀 LangSmith tracing FULLY enabled!")
    print(f"  📊 Project: {os.environ['LANGCHAIN_PROJECT']}")
    print(f"  🔗 View traces: https://smith.langchain.com")
else:
    print("⚠️  LangSmith tracing DISABLED (no LANGCHAIN_API_KEY)")
    print("   Add to .env: LANGCHAIN_API_KEY=lsv2_...")

# ✅ Check other required keys
google_key = os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
print(f"  🌐 Google API: {'✅' if google_key else '❌ Missing'}")
print(f"  🔑 OpenAI API: {'✅' if openai_key else '❌ Missing'}")



# llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
llm = ChatOpenAI(model='gpt-4o-mini')
print("✅ LLM initialized ")

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if not local_path.exists():
    print("Downloading Database...")
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"✅ Database saved: {local_path}")
    else:
        raise Exception(f"Failed to download database: {response.status_code}")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"✅ Database loaded: {db.dialect}, tables: {db.get_usable_table_names()}")

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
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("✅ SQL Agent ready!")


question = "Which genre on average has the longest tracks?"
print(f"\n🧠 Question: {question}")

for chunk in agent_executor.stream({"input": question}):
    chunk["messages"][-1].pretty_print()