from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import sqlite3

# -------------------------
# Database
# -------------------------
def setup_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE invoices (
            id INTEGER PRIMARY KEY,
            customer TEXT,
            amount REAL,
            date TEXT
        )
    """)
    conn.execute(
        "INSERT INTO invoices (customer, amount, date) "
        "VALUES ('Frank Harris', 1250.50, '2024-01-15')"
    )
    conn.commit()
    return conn

db = setup_db()

# -------------------------
# Tool
# -------------------------
@tool
def execute_sql(query: str) -> str:
    """Execute read-only SQLite SELECT queries."""
    try:
        cursor = db.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return "No results found"
        return str(rows)
    except Exception as e:
        return f"SQL Error: {e}"

tools = [execute_sql]

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# ✅ CLASSIC ReAct PROMPT
# -------------------------
prompt = PromptTemplate.from_template(
    """You are a careful SQLite analyst.

You have access to the following tools:
{tools}

Available tools: {tool_names}

Rules:
- Think step by step
- Use ONLY SELECT queries
- Limit results to 5 rows
- Use explicit column names

Question: {input}

{agent_scratchpad}
"""
)

# -------------------------
# Agent + Executor
# -------------------------
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------
# Invoke
# -------------------------
question = "This is Frank Harris. What was the total on my last invoice?"
result = executor.invoke({"input": question})

print("\n✅ Final Answer:")
print(result["output"])
