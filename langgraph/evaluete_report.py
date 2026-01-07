from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader


from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

class State(TypedDict):
    input_file_path : str
    file_content:str
    medical_entities:str
    layman_summary:str
    recommendations:str

def load_pdf(state:State):
    loader = PyPDFLoader(state['input_file_path'])
    docs = loader.load()

    full_text = "\n".join(doc.page_content for doc in docs)

    return {"file_content":full_text}


def extract_entities(state: State):
    prompt = (
        "You are a medical AI assistant.\n"
        "Extract structured medical entities from the following diagnostic report.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "vitals, lab_values, diagnoses, abnormalities\n\n"
        f"{state['file_content']}"
    )

    result = llm.invoke(prompt)
    return {"medical_entities": result.content.strip()}

def summarize_layman(state:State):
    prompt = (
         "Explain the following medical findings in simple terms for a non-medical person:\n\n"
        f"{state['medical_entities']}"
    )

    result = llm.invoke(prompt)
    return {"layman_summary":result.content.strip()}

def recommend_lifestyle(state:State):
    prompt = (
        "Based on these simplified findings, suggest appropriate and practical lifestyle changes:\n\n"
        f"{state['layman_summary']}"
    )

    result = llm.invoke(prompt)

    return {"recommendations":result.content.strip()}

workflow = StateGraph(State)

workflow.add_node("load_pdf", load_pdf)
workflow.add_node("extract_entities", extract_entities)
workflow.add_node("summarize_layman", summarize_layman)
workflow.add_node("recommend_lifestyle", recommend_lifestyle)

workflow.add_edge(START, "load_pdf")
workflow.add_edge("load_pdf", "extract_entities")
workflow.add_edge("extract_entities", "summarize_layman")
workflow.add_edge("summarize_layman", "recommend_lifestyle")
workflow.add_edge("recommend_lifestyle", END)

chain = workflow.compile()

state = chain.invoke({"input_file_path":r'D:\Personal\Agentic_Ai\Gen_ai_practice\langgraph\sterling-accuris-pathology-sample-report-unlocked.pdf'})


print("="*50)
print("\nExtracted Content : ",state['file_content'])
print("="*50)
print("\nEntities : ",state['medical_entities'])
print("="*50)
print("\n Sumamry : ",state['layman_summary'])
print("="*50)
print("\nRecommendations :",state['recommendations'])



