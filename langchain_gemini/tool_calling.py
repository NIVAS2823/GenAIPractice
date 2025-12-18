from langchain.tools import tool
from langchain.messages import HumanMessage,ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

@tool(description='Get the current weather in the given location')
def get_weather(location:str)->str:
    return "Its Sunny"

model_with_tools = ChatGoogleGenerativeAI(model='gemini-2.5-flash').bind_tools([get_weather])

message = [HumanMessage (content="Whats the weather in Boston")]
ai_msg = model_with_tools.invoke(message)
message.append(ai_msg)


print(ai_msg.tool_calls)

for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    message.append(
        ToolMessage(
            content=tool_result,
            tool_call_id = tool_call["id"]
        )
    )

final_response = model_with_tools.invoke(message)

print(final_response)