from google import genai
import asyncio

async def main():
    client = genai.Client()
    MODEL_ID = "gemini-2.5-flash"
    prompt = 'Compose a song about the adventures of a time-traveling squirrel.'
    
    
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    
    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())