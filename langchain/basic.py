from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize GPT model
simple_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.4)

# Define prompt template
prompt = ChatPromptTemplate.from_template("Capital of  {state} ")

# Use the new Runnable pipeline (prompt | llm)
chain = prompt | simple_llm

# Run the chain
result = chain.invoke({"state": "Telangana"})

print(result.content)
