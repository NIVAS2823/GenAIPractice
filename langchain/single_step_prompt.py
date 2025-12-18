from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate


llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.5)

transcript = """Patient complains of headche whenever he travel on mid day 
exposing to dust and sun light """


system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an Clinical AI Assistant.Generate a patient-friendly summary using simple language"
    )

human_prompt = HumanMessagePromptTemplate.from_template(
    "Transcript:{transcript_text}"
)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt,human_prompt])

chain = chat_prompt | llm

result = chain.invoke({"transcript_text":transcript}).content


print(f"Patient Summary:\n {result}")
