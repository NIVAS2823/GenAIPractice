from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import List
import json 
from langchain_core.output_parsers import JsonOutputParser


class RAGResponse(BaseModel):
    """A Structured  response to a user query based on a provided query"""
    answer:str = Field(...,description="The final,Comprehensive  answer to the user's question ,formulated only using the provided context")
    source_ids_used : List[int] = Field(...,description="A list of all CONTEXT  chunk indices(e.g.,[1,3]) that are directly used to formulate answer")
    confidence_level:str = Field(...,description="Your confidence is the answer(e.g.,'High','Medium','Low') based ont the context quality.")


llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = JsonOutputParser(pydantic_object=RAGResponse)

rag_template = PromptTemplate(
    input_variables=["context","question"],
    template=(
          "You are a sophisticated Question Answering system. Your goal is to answer the user's question "
        "using ONLY the provided CONTEXT chunks. Do not use external knowledge.\n\n"
        "CONTEXT CHUNKS:\n"
        "{context}\n\n"
        "USER QUESTION: {question}\n\n"
        "--- JSON FORMAT INSTRUCTIONS ---\n"
        "{format_instructions}\n"
        "--- END INSTRUCTIONS ---\n\n"
        "ANSWER IN REQUIRED JSON FORMAT:"
    ),
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

dummy_context_chunks = [
     "[1] The average lifespan of a common honeybee worker is about 5 to 6 weeks during the summer season.",
    "[2] Honeybee colonies can contain tens of thousands of individuals, including one queen.",
    "[3] A queen bee can live for 3 to 5 years, far exceeding the lifespan of worker bees.",
    "[4] Bees primarily communicate through a 'waggle dance' to indicate the location of food sources."
]


dummy_context = "\n".join(dummy_context_chunks)
dummy_question = "What is the typical lifespan difference between a worker honeybee and a queen bee?"


rag_chain = rag_template | llm | parser

print("\n---RAG Query with JSON Output ---")
print(f"\n---Question {dummy_question}---")
print(f"\n---Context {dummy_context} ---")

try:
    rag_response = rag_chain.invoke({"context":dummy_context,"question":dummy_question})

    print("\n---Structured JSON Response (as python dict)---")
    print(json.dumps(rag_response,indent=2))
    print(f"\nExtracted Answer : {rag_response.get('answer')}")
except Exception as e:
    print(f"An error occured during chain execution {e}")

