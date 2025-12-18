from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in 2 sentences"
)

print(template.format(topic="what is the use of langchain"))