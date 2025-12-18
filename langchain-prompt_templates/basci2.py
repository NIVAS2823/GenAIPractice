# Install: pip install langchain-classic

# from langchain_classic.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a beginner using a simple analogy."
)

chain = prompt  | llm 
response = chain.invoke({"topic":"Neural Networks"}).content
print(response)