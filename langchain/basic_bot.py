import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)

system_prompt = SystemMessagePromptTemplate.from_template("You are an helpful python assistant and " \
"you only answer questions related to python programming. If the question is not related to python respond:" "This is not in my scope")

user_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt,user_prompt])

chain = chat_prompt | llm

print("Enter a question...or type exit to quit.")

while True:
    question  = input("Enter a question:")
    if question.lower() in ['exit','quit']:
        break
    result  = chain.invoke({"question":question})
    print(result.content)
