from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader


# text = """
#         Multi-agent communication is the backbone of collaborative intelligence in Agentic AI systems, 
#         allowing autonomous entities to move beyond isolation and function as a cohesive team. 
#         In these systems, communication occurs through defined protocols that govern how agents 
#         exchange information, negotiate tasks, and coordinate strategies to solve complex problems. 
#         This interaction can take various forms, from direct peer-to-peer messaging and shared memory 
#         access to hierarchical instructions from an orchestrator agent.​

#         Effective communication enables specialization, where agents with distinct roles—such as planning, 
#         research, or coding—hand off subtasks seamlessly, much like a human project team. For instance, a 
#         "Manager" agent might decompose a user request into steps and assign them to "Worker" agents, 
#         who then report back results or request clarification through a shared environment or message bus. 
#         This dynamic exchange allows the system to handle intricate workflows that require parallel processing, 
#         debate, or iterative refinement. Ultimately, robust communication protocols transform a collection of 
#         individual models into a resilient ecosystem capable of adaptivity, scalability, and solving challenges 
#         far exceeding the capacity of any single agent
# """

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap = 20
)

loader = TextLoader('text.txt',encoding='utf-8')

documents = loader.load()

chunks = splitter.split_documents(documents)


llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)


prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the chunk  : \n {text}"
)

chain = prompt | llm

for i,chunk in enumerate(chunks):
    chunk_summary = chain.invoke({"text":chunk}).content
    print("="*50)
    print(f"\n Chunk {i+1} :  {chunk} \n\n Chunk Summary : {chunk_summary}")
