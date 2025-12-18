from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.8
)

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a catchy blog title about {topic}"
)

title_chain = title_prompt | llm


blog_prompt  = PromptTemplate(
    input_variables=["title"],
    template="Write a 150-word for the title {title}"
)


blog_chain = blog_prompt | llm

full_chain =  title_chain | blog_chain

result = full_chain.invoke({"topic":"AI ethics"})

print(result)