from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    max_tokens = None,
    timeout=None,
    max_retries=2
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Hindi.Translate User sentence"
    ),
    (
        "human",
        "I love You"
     )
]

ai_msg = llm.invoke(messages)

print(ai_msg.content)