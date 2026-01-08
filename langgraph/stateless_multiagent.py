from typing_extensions import TypedDict
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


llm  = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class ReportState(TypedDict):
    topic:str
    research_points : List[str]
    insights : List[str]
    final_report : str



def research_agent(state:ReportState)->ReportState:
    prompt = ChatPromptTemplate.from_messages([
    ("system","You are a research agent.Break the topic into sub questions and generate factual bullet points"),
    ("human","Topic :{topic}")
    ])

    response = llm.invoke(prompt.format_messages(topic=state['topic']))

    research_points = [
        line.strip("- ")
        for line in response.content.split("\n")
        if line.strip()
    ]

    return {
        **state,
        'research_points':research_points
    }


def analysis_agent(state:ReportState)->ReportState:
    joined_points = "\n".join(state['research_points'])

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are an analysis agent. Analyze the research points and derive technical insights."),
        ("human","Research points : {points}")
    ])

    response = llm.invoke(prompt.format_messages(points=joined_points))

    insights = [
        line.strip("-")
        for line in response.content.split("\n")
        if line.strip()
    ]

    return{
        **state,
        "insights":insights
    }


def writer_agent(state:ReportState)->ReportState:
    research = "\n".join(state['research_points'])
    insights = "\n".join(state["insights"])

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are an analysis agent. Analyze the research points and derive technical insights."),
        ("human",f"""
         Topic : {state['topic']}
         
         Research Findings : 
         {research}

         Key Insights :
         {insights}

         Write a structured report with:
         -Introduction
         -Technical overview
         -Key insights
         -Conclusion
         """
         )
    ])
    response = llm.invoke(prompt.format_messages())

    return {
        **state,
        "final_report":response.content
    }


graph  = StateGraph(ReportState)

graph.add_node("research",research_agent)
graph.add_node("analysis",analysis_agent)
graph.add_node("writer",writer_agent)

graph.add_edge(START,"research")
graph.add_edge("research","analysis")
graph.add_edge("analysis","writer")
graph.add_edge("writer",END)

app = graph.compile()

if __name__ == "__main__":
    input_state = {
        "topic":"Agentic AI Systems",
        "research_points":[],
        "insights":[],
        "final_report":""
    }

    result = app.invoke(input_state)

    print("Report : ",result)