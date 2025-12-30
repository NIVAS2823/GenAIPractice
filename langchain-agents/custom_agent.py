from langchain_classic.agents import BaseSingleActionAgent, AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage 

class CustomLogicAgent(BaseSingleActionAgent):
    """Custom agent with specialized decision logic"""
    llm: ChatGoogleGenerativeAI
    tools: List[Any]  

    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def output_keys(self) -> List[str]: 
        return ["output"]

    def plan(
        self, 
        intermediate_steps: List[Tuple[AgentAction, str]], 
        **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """
        Custom decision logic for determining next action
        """
        user_input = kwargs["input"]

        # Keyword-based routing for weather
        if "weather" in user_input.lower():
            return AgentAction(
                tool="weather_tool",  # Assumes this tool exists in tools list
                tool_input=user_input,
                log="Using weather tool based on keyword"
            )
        
        # Stop after 3 steps
        if len(intermediate_steps) >= 3:
            return AgentFinish(
                return_values={"output": "Completed after 3 steps"},
                log="Stopping after 3 iterations"
            )
        
        # LLM decision with proper messages format
        llm_output = self.llm.invoke([
            HumanMessage(content=f"What tool should I use for: {user_input}")
        ]).content  # Fixed: Use messages format and .content

        return AgentAction(
            tool="default_tool",  # Assumes this tool exists in tools list
            tool_input=user_input,
            log=f"LLM suggested: {llm_output}"
        )

    async def aplan(self, *args: Any, **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Async version of plan"""
        return await self.plan(*args, **kwargs)  # Fixed: Made async properly
    
    @property
    def tool_names(self) -> List[str]:  # Added: Required for tool lookup
        return [tool.name for tool in self.tools]

llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Define llm
tools = []  


custom_agent = CustomLogicAgent(llm=llm, tools=tools)
executor = AgentExecutor(agent=custom_agent, tools=tools, verbose=True)

