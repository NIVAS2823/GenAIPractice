from typing import Literal,Optional,Any
from pydantic import BaseModel
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage,AIMessage,HumanMessage


class SupportState(BaseModel):
    current_step : Literal["warranty_collector","issue_classifier","resolution"]='warranty_collector'
    warranty_status:Optional[Literal['in_warranty','out_of_warranty']] = None
    issue_type : Optional[Literal['hardware','software']] = None
    messages :list = []


@tool
def record_warranty_status(status:str):
    """
    Record Warranty Status and move to Issue Classification
    """
    return {
        "warranty_status":status,
        "current_step":"issue_classifier"
    }

@tool
def record_issue_type(issue_type:str):
    """
    Record Issue Type and move to resolution step
    """
    return {
        "issue_type":issue_type,
        "current_step":"resolution"
    }

@tool
def provide_resolution(solution:str):
    """
   Provide a final answer to customer
    """
    return {"final_answer":solution}

@tool
def escalate_to_human():
    """
    Escalate to Human Support
    """
    return {"final_answer":"Your issue has been escalated to Human support"}

WARRANTY_PROMPT = """
You are a customer support agent. YOU MUST USE TOOLS TO ADVANCE.

CURRENT STAGE: Warranty verification ONLY

1. Ask: "Is your device under warranty?"
2. User says YES → EXACTLY: record_warranty_status("in_warranty")
3. User says NO → EXACTLY: record_warranty_status("out_of_warranty")

NEVER skip this step. NEVER ask about issues yet.
"""

ISSUE_PROMPT = """
You are a customer support agent. YOU MUST USE TOOLS TO ADVANCE.

CURRENT STAGE: Issue classification ONLY
Warranty: {warranty_status}

1. Ask user to describe issue
2. Hardware → EXACTLY: record_issue_type("hardware")
3. Software → EXACTLY: record_issue_type("software")

NEVER provide resolution yet.
"""

RESOLUTION_PROMPT = """
You are a customer support agent. Use tools to advance workflow.

CURRENT STAGE: Resolution
Warranty: {warranty_status}, Issue: {issue_type}

- Software → provide_resolution("Troubleshooting steps...")
- Hardware + in_warranty → provide_resolution("Repair process...")
- Hardware + out_of_warranty → escalate_to_human()
"""

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0)

tools = [
    record_warranty_status,
    record_issue_type,
    provide_resolution,
    escalate_to_human
]

def get_prompt_for_state(state:SupportState):
    if state.current_step == "warranty_collector":
        return WARRANTY_PROMPT
    if state.current_step == "issue_classifier":
        return ISSUE_PROMPT.format(warranty_status=state.warranty_status)
    if state.current_step == "resolution":
        return RESOLUTION_PROMPT.format(
            warranty_status = state.warranty_status,
            issue_type = state.issue_type
        )
    
def message_to_dict(msg:Any):
     """Convert any LangChain Message to plain dict for storage"""
     if isinstance(msg, (AIMessage, ToolMessage, HumanMessage)):
        return {
            "role": getattr(msg, 'role', 'unknown'),
            "content": getattr(msg, 'content', ''),
            "tool_calls": getattr(msg, 'tool_calls', None)
        }
     return {"role": "unknown", "content": str(msg)}


def get_agent_text(message):
    """Extract clean text from any message format"""
    if hasattr(message, 'content'):
        return message.content  # LangChain Message object
    if isinstance(message, dict) and 'text' in message:
        return message['text']  # Gemini dict format  
    if isinstance(message, str):
        return message
    return str(message)[:200] + "..." 


state = SupportState()

if not hasattr(state, 'messages'):
    state.messages = []

while True:
    prompt = get_prompt_for_state(state)
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    
    user_input = input("\nUser: ")
    state.messages.append({"role": "user", "content": user_input})
    
    input_messages = [HumanMessage(**m) for m in state.messages]
    result = agent.invoke({"messages": input_messages})
    
    last_message = result["messages"][-1]
    agent_text = last_message.content if hasattr(last_message, 'content') else str(last_message).split("'text': '")[1].split("'", 1)[0]
    print("\nAgent:", agent_text)
    
    new_messages = result["messages"][-len(state.messages):]
    state.messages.extend([message_to_dict(m) for m in new_messages])
    
    updated = False
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.content:
            try:
                tool_output = eval(msg.content)
                if isinstance(tool_output, dict):
                    for key, value in tool_output.items():
                        if hasattr(state, key) and key != 'messages':
                            setattr(state, key, value)
                            updated = True
                        if key == "final_answer":
                            print("\nFinal:", value)
                            exit()
            except:
                pass
    
    print(f"State: {state.current_step}, warranty={state.warranty_status}, issue={state.issue_type}")
    
    if state.current_step == "resolution" and not updated:
        print("No progress in resolution, ending.")
        break