import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

simple_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

system_message = SystemMessagePromptTemplate.from_template("You are an helpful and consise assistant. Please refer to latest data  after 2025 January to answer user prompts  ")
user_message = HumanMessagePromptTemplate.from_template("who is the cheif minister {state}  state")

chat_prompt = ChatPromptTemplate.from_messages([system_message,user_message])
# chain = prompt | simple_llm
chain = chat_prompt | simple_llm

result = chain.invoke({"state":"Telangana"})

print(result.content)
# print(result)