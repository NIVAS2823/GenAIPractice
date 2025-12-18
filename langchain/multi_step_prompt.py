from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableSequence


llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.3)

pitch_prompt = ChatPromptTemplate.from_template(
    "Write a two sentence marketing pitch for the product  {product_name}"
)

tweet_prompt = ChatPromptTemplate.from_template(
    "Convert the following into single Twitter(x) post under 200 characters : \n {pitch_prompt}"
)

hashtag_prompt = ChatPromptTemplate.from_template(
    "Suggest  3 trending hashtags for this  twitter post \n {tweet_prompt}"
)

input_data = {"product_name":"Dell HP AI Powered Laptop"}

pitch_response = (
    pitch_prompt | llm | RunnableLambda(
        lambda msg : {
            "product name":input_data["product_name"],
            "pitch":msg.content if hasattr(msg,"content") else  str(msg)
        }
    )
)

tweet_response = (
    RunnableLambda(
        lambda x : {"pitch":x["pitch"]}
        | tweet_prompt
        | llm
        | RunnableLambda()
    )
)
