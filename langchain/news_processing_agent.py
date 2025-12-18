from langchain_google_genai import ChatGoogleGenerativeAI # NEW: Import for LangChain Gemini integration
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import requests
import json
import os
from google import genai # Import for direct SDK usage inside tools

# =================
# Config
# =================
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
API_KEY = "314b7764447048edb824d09f1353bc6c"
COUNTRY = "us"
ARTICLE_COUNT = 5
OUTPUT_FILE = "processed_news.json"

processes_result = []

# --- LLM Setup ---
# 1. Get API Key
api_key = os.getenv("GEMINI_API_KEY")

# 2. Setup the LangChain-compatible LLM for the Agent
llm_agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# 3. Setup the direct GenAI client for use inside tools
gemini_client = genai.Client(api_key=api_key)

# --- Tool Definitions ---

@tool
def download_news(_:str="")->str:
    """Download latest news articles from the API."""
    # ... (same as original)
    params = {
        "apiKey": API_KEY,
        "country":COUNTRY,
        "pageSize":ARTICLE_COUNT
    }
    response = requests.get(NEWS_API_URL,params=params)
    response.raise_for_status()
    articles = response.json().get("articles",[])

    for article in articles:
        processes_result.append(
            {
                "original": article,
                "reworded":"",
                "category":"",
                "hashtags":""
            }
        )
    return f"Downloaded {len(processes_result)} news articles from the {NEWS_API_URL}"


@tool
def reword_news(article_index:str)->str:
    """Reword a news article into 2-3 sentences."""
    idx = int(article_index)
    article_text = processes_result[idx]["original"].get("description") or processes_result[idx]["original"].get("content") or processes_result[idx]["original"].get("title")

    prompt = f"Reword this news article into 2-3 sentences \n\n {article_text}"
    # Use the direct client for internal tool calls
    response = gemini_client.models.generate_content(model="gemini-2.5-flash",content=prompt)
    reworded= response.text.strip() # Use .text for the Python SDK
    processes_result[idx]["reworded"] = reworded
    return reworded


@tool
def detect_category(article_index:str)->str:
    """Classify the news article into a category."""
    idx = int(article_index)
    article_text = processes_result[idx]["original"].get("description") or processes_result[idx]["original"].get("content") or processes_result[idx]["original"].get("title")
    prompt = f'Classify the news article like Politics,Sports,Entertainment,Tech,Science,Health,Lifestyle : \n\n {article_text}'
    response = gemini_client.models.generate_content(model="gemini-2.5-flash",content=prompt)
    category = response.text.strip() # Use .text for the Python SDK
    processes_result[idx]["category"] = category

    return category


@tool
def generate_hashtags(article_index:str)->str:
    """Generate 3 trending hashtags for the article."""
    idx = int(article_index)
    article_text = processes_result[idx]["original"].get("description") or processes_result[idx]["original"].get("content") or processes_result[idx]["original"].get("title")
    prompt = f"Generate 3 trending hashtags using the article description :\n\n{article_text}"
    response = gemini_client.models.generate_content(model="gemini-2.5-flash",content=prompt)
    hashtags = response.text.strip() # Use .text for the Python SDK
    processes_result[idx]["hashtags"] = hashtags

    return hashtags


@tool
def save_to_json(_:str="")->str:
    """Save processed articles to JSON file."""
    with open(OUTPUT_FILE,'w',encoding='utf-8') as f:
        json.dump(processes_result,f,ensure_ascii=False,indent=4)

    return f"Saved {len(processes_result)} records to the {OUTPUT_FILE}"


# --- Agent Setup ---
memory = MemorySaver()
# Pass the LangChain-compatible LLM to the agent
agent = create_agent(
    model=llm_agent, 
    tools=[download_news,reword_news,detect_category,generate_hashtags,save_to_json],
    checkpointer=memory
)


goal = f"""
Download the latest {ARTICLE_COUNT} news articles from the US,
For each article index (0 to {ARTICLE_COUNT-1}):
    1. Call 'reword_news' with the index number as string.
    2. Call 'detect_category' with the index number as string.
    3. Call 'generate_hashtags' with the index number as string.
After all the articles processed call 'save_to_json'
"""

result = agent.invoke(
    {"messages":[{"role":"user","content":goal}]},
    {"configurable":{"thread_id":"1"}}
)


print("\n News Agent Execution completed")
print(result["messages"][-1].content)