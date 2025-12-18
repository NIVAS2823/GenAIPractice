from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define tools
@tool
def generate_pitch(product_name: str) -> str:
    """Generate a short 2-sentence marketing pitch."""
    prompt = f"Write a 2-sentence marketing pitch for '{product_name}'."
    return llm.invoke(prompt).content

@tool
def pitch_to_tweet(pitch: str) -> str:
    """Convert pitch into a tweet under 200 characters."""
    prompt = f"Convert to tweet (under 200 chars): {pitch}"
    return llm.invoke(prompt).content

@tool
def tweet_to_hashtags(tweet: str) -> str:
    """Suggest 3 trending hashtags."""
    prompt = f"Suggest 3 hashtags for: {tweet}"
    return llm.invoke(prompt).content

# Create agent with memory
memory = MemorySaver()
agent = create_agent(
    model=llm,
    tools=[generate_pitch, pitch_to_tweet, tweet_to_hashtags],
    checkpointer=memory
)

# Run
goal = "Generate pitch, tweet, and hashtags for 'Donald Trump'."
result = agent.invoke(
    {"messages": [{"role": "user", "content": goal}]},
    {"configurable": {"thread_id": "1"}}
)

print(result["messages"][-1].content)