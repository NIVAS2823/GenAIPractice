from agents import Agent,Runner
import asyncio


telugu_agent = Agent(
    name="Telugu Agent",
    instructions="You only Speak Telugu Language",
)

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions ="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[telugu_agent,english_agent]
)


async def main():
    result  = await Runner.run(triage_agent,input="Hello how are you")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())