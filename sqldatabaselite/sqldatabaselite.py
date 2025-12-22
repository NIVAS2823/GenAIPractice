from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage  

engine = create_engine("sqlite:////demo.db")

with engine.connect() as con:
    con.execute(text("""
     CREATE TABLE IF NOT EXISTS employees(
                     id INTEGER PRIMARY KEY,
                     name TEXT,
                     department TEXT,
                     salary INTEGER)
        """))
    
    con.execute(text(
        """
    INSERT INTO employees (name, department, salary)
        VALUES
        ('Alice', 'Engineering', 90000),
        ('Bob', 'Engineering', 85000),
        ('Charlie', 'HR', 60000),
        ('Diana', 'Marketing', 70000)
"""
    ))
    con.commit()  

db = SQLDatabase.from_uri("sqlite:////demo.db")

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()  

agent = create_agent(
    model=llm,
    tools=tools
)

response = agent.invoke({
    "messages": [HumanMessage(content="Show employees with departments ")]
})

print(response['messages'][-1].content)  