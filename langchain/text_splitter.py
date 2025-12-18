from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

long_text = """
LangChain is a powerful framework for building language model applications.
It offers tools to manage prompts, chains, memory, agents, and external integrations.
This makes it easier to build complex LLM applications that interact with real-world data.
You can process documents, create chatbots, query databases, or even automate workflows.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", ".", ",", " ", "\n"]
)

chunks = splitter.split_text(long_text)


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)

prompt = PromptTemplate(
    input_variables= ["text"],
    template= "Summarize this chunk:\n{text}"
)

chain =  prompt | llm

for i,chunk in enumerate(chunks):
    chunk_summary = chain.invoke({"text": chunk})
    print(f"\n--- Chunk {i+1} {chunk}\n Summary is: {chunk_summary.content}")