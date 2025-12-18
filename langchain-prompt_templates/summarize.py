from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

summarize_template = PromptTemplate(
    input_variables=["text","style"],
    template=(
        "You are an expert summarizer. Your task is to summarize the following text "
        "based on the specified style.\n\n"
        "Style: {style}\n"
        "Text to Summarize:\n"
        "--- START TEXT ---\n"
        "{text}\n"
        "--- END TEXT ---\n"
        "Please provide the summary now."
    )
)

chain = summarize_template | llm | StrOutputParser()


sample_text = (
    "The 1920s, often referred to as the 'Roaring Twenties,' was a period of "
    "significant cultural and economic change in the United States and Europe. "
    "Characterized by jazz music, flapper culture, unprecedented industrial "
    "growth, and a loosening of social restrictions, it was an era of exuberance "
    "and new modernity following the end of World War I. However, this prosperity "
    "came to an abrupt halt with the stock market crash of 1929, ushering in the "
    "Great Depression."
)

print("\n---Formal Summary--")
formal_response = chain.invoke({"text":sample_text,"style":"Formal and historical,suitable for academic paper"})

print(formal_response)


print("\n---Short Summary")
short_summary = chain.invoke({"text":sample_text,"style":"Extremely short,maximum 15 words"})
print(short_summary)

print("\n---Funny Summary")
funny_response = chain.invoke({"text":sample_text,"style":"Funny and exaggered,like a tabloid headline"})

print(funny_response)