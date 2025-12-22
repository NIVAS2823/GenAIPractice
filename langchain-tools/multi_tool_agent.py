import os
import json
import requests
from typing import Optional,List
from datetime import datetime
from pydantic import BaseModel,Field,field_validator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv()



#---------------
#Calculator Tool
#----------------
@tool
def advanced_calculator(expression:str)->str:
    """
   Evaluates complex mathematical expressions including scientific functions.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, exp
    
    Args:
        expression: Mathematical expression as string (e.g., "sqrt(16) + 2**3")
    
    Returns:
        Calculation result or error message
    """

    import math

    safe_dict = {
        'sqrt':math.sqrt,
        'sin':math.sin,
        'cos':math.cos,
        'tan':math.tan,
        'log':math.log,
        'pi':math.pi,
        'e':math.e,
        '__builtins__':{}
    }

    try:
        expression = expression.strip()
        if any(keyword in expression for keyword in ['import','exec','eval','__']):
            return "Error: Invalid expression - Security violation detected"
        
        result = eval(expression,safe_dict)
        return f"Result : {result}"
    except ZeroDivisionError:
        return "Error : Division by zero"
    except SyntaxError:
        return "Error : Invalid mathematical syntax"
    except NameError:
        return f"Error : Unknown function or variable : {str(e)}"
    
    except Exception as e:
        return f"Calcaution failed . {str(e)}"
    

class WeatherInput(BaseModel):
    city : str = Field(...,description="City Name (e.g., 'London','Hyderand','New York')")
    country_code : Optional[str] = Field(...,description="Two Letter country Code(e.g.,'US','GB','IN')")


    @field_validator('city')
    def validate_city(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("City name must be at least 2 characters")
        return v.strip()

#---------------
#Weather  Tool
#----------------

@tool(args_schema=WeatherInput)
def get_weather(city:str,country_code:str="US")->str:
    """
     Fetches current weather information for a specified city using OpenWeatherMap API.
    Provides temperature, conditions, humidity, and wind speed.
    
    Args:
        city: Name of the city
        country_code: Two-letter ISO country code (default: US)
    
    Returns:
        Formatted weather information or error message
    """
   
    API_KEY = os.getenv['OPENAI_WEATHER_API_KEY']
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    try:
        params = {
            'q':f"{city},{country_code}",
            'appid':API_KEY,
            'units':'metric'
        }

        response = requests.get(BASE_URL,params=params,timeout=5)
        response.raise_for_status()


        data = response.json()
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'conditions': data['weather'][0]['description'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }

        formatted_response = f"""
Weather in {weather_info['city']}, {weather_info['country']}:
🌡️  Temperature: {weather_info['temperature']}°C (feels like {weather_info['feels_like']}°C)
☁️  Conditions: {weather_info['conditions'].title()}
💧 Humidity: {weather_info['humidity']}%
💨 Wind Speed: {weather_info['wind_speed']} m/s
""".strip()
        
        return formatted_response
    
    except requests.exceptions.Timeout:
        return " Error: Weather API request timed out. Please try again"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: City '{city}' not found. Please check spelling."
        elif e.response.status_code == 401:
            return "Error: Invalid API key. Please configure weather service."
        else:
            return f"Error: HTTP {e.response.status_code} - {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch weather data - {str(e)}"
    except KeyError as e:
        return f"Error: Unexpected API response format - missing {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error occurred - {str(e)}"
    

class SearchInput(BaseModel):
    query : str = Field(...,description="Search query string")
    max_resutls : int = Field(
        default=3,
        description="Maximum number of results to return (1-10)",
        ge=1,
        le=10
    )
    category : Optional[str] = Field(...,description="Filter by category: 'tech', 'business', 'science', or None for all")


KKNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "category": "tech",
        "date": "2024-01-15"
    },
    {
        "id": 2,
        "title": "Quantum Computing Basics",
        "content": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations that would be infeasible for classical computers.",
        "category": "tech",
        "date": "2024-02-20"
    },
    {
        "id": 3,
        "title": "Business Strategy Fundamentals",
        "content": "Effective business strategy involves analyzing market conditions, competitive landscape, and organizational capabilities to create sustainable competitive advantage.",
        "category": "business",
        "date": "2024-03-10"
    },
    {
        "id": 4,
        "title": "Climate Change Research",
        "content": "Recent climate studies indicate accelerating global temperature rise, with significant implications for ecosystems, weather patterns, and human societies.",
        "category": "science",
        "date": "2024-01-25"
    },
    {
        "id": 5,
        "title": "Blockchain Technology",
        "content": "Blockchain is a distributed ledger technology that enables secure, transparent, and tamper-proof record-keeping without central authority.",
        "category": "tech",
        "date": "2024-02-05"
    }
]


@tool(args_schema=SearchInput)
def search_knowledge_base_func(query:str,max_results:int=3,category:Optional[str] = None)->str:
    """
    Searches internal knowledge base for relevant documents.
    Uses simple keyword matching (in production, use vector similarity).
    """

    try:
        query_lower = query.lower()
        results = []

        for doc in KKNOWLEDGE_BASE:
            if category and doc['category'] != category.lower():
                continue

            score = 0
            search_text = f"{doc['title']} {doc['content']}".lower()

            for word in query_lower.split():
                if len(word) > 3:
                    score += search_text.count(word)

            if score > 0:
                results.append({
                    'doc':doc,
                    'score':score
                })


        if not results:
            return f"No document found matching {query}"
        
        formatted_results = [f"Found {len(results)} relevant documents:\n"]

        for i,item in enumerate(results,1):
            doc = item['doc']
            formatted_results.append(f"""
                                     {i}.{doc['title']} ({doc['category'].upper()}))
                                     Date : {doc['date']}
                                     Summary : {doc['content'][:150]}...
                                     [Document ID : {doc['id']}]
                                     """.strip())
            
            return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching knowledge base : {str(e)}"
    


@tool
def analyze_json_data(json_string:str)->str:
    """
     Parses and analyzes JSON data, providing structure summary and statistics.
    Useful for understanding API responses or data files.
    
    Args:
        json_string: Valid JSON string to analyze
    
    Returns:
        Analysis report with structure, types, and statistics
    """

    try:
        data = json.loads(json_string)


        def analyze_structure(obj,depth=0):
            """Recursively analyze JSON structure"""
            if isinstance(obj,dict):
                return {
                    'type': 'object',
                    'keys': len(obj),
                    'structure': {k: analyze_structure(v, depth+1) for k, v in list(obj.items())[:5]} 
                }
            elif isinstance(obj, list):
                return {
                    'type': 'array',
                    'length': len(obj),
                    'item_type': analyze_structure(obj[0], depth+1) if obj else 'empty'
                }
            else:
                return {
                    'type': type(obj).__name__,
                    'value': str(obj)[:50]  # Truncate long values
                }
            
        analysis = analyze_structure(data)

        report = f"""
            JSON Data Analysis:
==================
Root Type: {analysis['type']}
"""
        
        if analysis['type'] == 'object':
            report += f"Number of Keys: {analysis['keys']}\n"
            report += "\nStructure Preview:\n"
            for key, value in analysis.get('structure', {}).items():
                report += f"  - {key}: {value['type']}\n"
        
        elif analysis['type'] == 'array':
            report += f"Array Length: {analysis['length']}\n"
            report += f"Item Type: {analysis.get('item_type', {}).get('type', 'unknown')}\n"
        
        return report.strip()
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error analyzing JSON: {str(e)}"
    
def create_research_assistant():
    """Creates a production-ready research assitance with multiple tools"""

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0,
        top_p=0.95,
        max_output_tokens = 2048
    )

    tools = [advanced_calculator,get_weather,search_knowledge_base_func,analyze_json_data]

    system_prompt = """
You are an advanced research assistant with access to multiple specialized tools.

Your capabilities:
1. Mathematical calculations (simple and complex)
2. Real-time weather information for any city
3. Internal knowledge base search
4. JSON data analysis

Guidelines:
- Always use tools when appropriate rather than relying on your training data
- For calculations, use the calculator tool to ensure accuracy
- For current information (weather, etc.), always use the relevant tool
- If a tool returns an error, explain the error to the user and suggest alternatives
- Combine multiple tools when needed to provide comprehensive answers
- Cite which tool(s) you used in your response

Be helpful, accurate, and transparent about your process.
"""

    prompt = ChatPromptTemplate.from_messages([
       ("system",system_prompt),
       ("human","{input}"),
       ("placeholder","{agent_scratchpad}")
   ])
    

    agent = create_tool_calling_agent(llm,tools,prompt)

    return  AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        max_execution_time=60,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    


def run_examples():
    """Demonstrates various use cases"""

    assistant = create_research_assistant()

    print("="*80)
    print("RESEARCH ASSISTANT - PRODUCTION DEMO")
    print("="*80)

    # Example 1: Multi-tool query
    print("\n📝 Example 1: Complex calculation")
    print("-" * 80)
    result1 = assistant.invoke({
        "input": "Calculate the square root of 144 plus 2 to the power of 5"})
    print(f"\n✅ Final Answer: {result1['output']}\n")

    # Example 2: API tool usage
    print("\n📝 Example 2: Weather query")
    print("-" * 80)
    result2 = assistant.invoke({
        "input": "What's the weather like in London, UK?"
    })
    print(f"\n✅ Final Answer: {result2['output']}\n")

    # Example 3: Knowledge base search
    print("\n📝 Example 3: Knowledge base search")
    print("-" * 80)
    result3 = assistant.invoke({
        "input": "Find information about blockchain technology in our knowledge base"
    })
    print(f"\n✅ Final Answer: {result3['output']}\n")

    # Example 4: JSON analysis
    print("\n📝 Example 4: JSON data analysis")
    print("-" * 80)
    sample_json = json.dumps({
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"}
        ],
        "timestamp": "2024-01-15T10:30:00Z"
    })
    result4 = assistant.invoke({
        "input": f"Analyze this JSON data: {sample_json}"})  
    print(f"\n✅ Final Answer: {result4['output']}  \n")

    # Example 5: Multi-tool complex query
    print("\n📝 Example 5: Multi-tool complex query")
    print("-" * 80)
    result5 = assistant.invoke({
        "input": "Search for machine learning in our knowledge base, then calculate 10 factorial"
    })
    print(f"\n✅ Final Answer: {result5['output']}\n")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         LANGCHAIN TOOLS - PRODUCTION RESEARCH ASSISTANT              ║
║                                                                      ║
║  Features:                                                           ║
║  ✓ Advanced Calculator (Python Tool)                                ║
║  ✓ Weather API Integration (API Tool)                               ║
║  ✓ Knowledge Base Search (Custom Function Tool)                     ║
║  ✓ JSON Data Analyzer (Data Processing Tool)                        ║
║                                                                      ║
║  Production-Ready Features:                                          ║
║  ✓ Comprehensive error handling                                     ║
║  ✓ Input validation with Pydantic                                   ║
║  ✓ Security measures (safe eval context)                            ║
║  ✓ Timeout and iteration limits                                     ║
║  ✓ Detailed logging and debugging                                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    run_examples()
    