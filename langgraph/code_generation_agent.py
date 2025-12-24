from typing import TypedDict,Annotated,Literal
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
import ast
import sys
from io import StringIO


class CodeGenState(TypedDict):
    """
    State schema for code generation workflow
    Each node reads/writes to this shared state
    """

    user_request:str
    generated_code:str
    validation_error:str
    iteration_count:int
    max_iterations:int
    execution_output:str
    is_valid:bool
    final_code:str

llm= ChatOpenAI(model='gpt-4o-mini',temperature=0.1)


def generate_code_node(state:CodeGenState)->CodeGenState:
    """
    Generates Python code based on User Request
    If validation error exists,incorporates  feedback for regeneration.
    """

    user_request = state["user_request"]
    validation_error = state["validation_error"]
    iteration = state["iteration_count"]

    if validation_error:
         prompt = f"""You are a python code generator. A previous code attempt failed.

User Request:
{user_request}

Previous Code:
{state['generated_code']}

Error Encountered:
{validation_error}

Generate corrected python code that:
1. Fixes the error
2. Meets the original request
3. Is self-contained

Return ONLY python code.
"""
    else:
         prompt = f"""You are a python code generator.

User Request:
{user_request}

Generate Python code that:
1. Meets the request
2. Is syntactically correct
3. Includes a main block

Return ONLY python code.
"""

    response = llm.invoke(prompt)
    generated_code = response.content.strip()



    if generated_code.startswith("```Python"):
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
    elif generated_code.startswith("```"):
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

    if generated_code.lower().startswith("python\n"):
        generated_code = generated_code[len("python\n"):].strip()

    
    print(f"\nIteration {iteration + 1}: Code generated")
    print(f"\n{'='*50}")
    print(generated_code[:200]+"..." if len(generated_code) > 200 else generated_code)

    return {
         **state,
         "generated_code":generated_code,
         "iteration_count":iteration +1,
         "validation_error":"",
         "is_valid": False
    }


def validate_syntax_node(state: CodeGenState) -> CodeGenState:
    try:
        ast.parse(state["generated_code"])
        print("Syntax validation passed")
        return {**state, "is_valid": True, "validation_error": ""}
    except SyntaxError as e:
        error_msg = f"Syntax Error at line {e.lineno}: {e.msg}"
        print("Syntax validation failed")
        return {**state, "is_valid": False, "validation_error": error_msg}



def execute_code_node(state: CodeGenState) -> CodeGenState:
    """
    Executes the validated code and captures stdout safely.
    """

    code = state["generated_code"]

    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer

    try:
        exec_namespace = {}
        exec(code, exec_namespace)

        output = buffer.getvalue()

        sys.stdout = old_stdout

        print("Execution passed")
        print(f"Output:\n{output}")

        return {
            **state,
            "execution_output": output,
            "is_valid": True,
            "validation_error": "",
            "final_code": code
        }

    except Exception as e:
        sys.stdout = old_stdout
        error_msg = f"{type(e).__name__}: {str(e)}"

        print("Execution failed")
        print(f"Error: {error_msg}")

        return {
            **state,
            "is_valid": False,
            "validation_error": error_msg,
            "execution_output": ""
        }

     
     

def finalize_node(state:CodeGenState)->CodeGenState:
     """
     Marks the workflow as complete and stores the final code
     """
     print(f"\n{'='*50}")
     print(f"Code Generation Complete")
     print(f"Total Iterations: {state['iteration_count']}")
     print(f"\n{'='*50}")

     return {
          **state,
          "final_code":state["generated_code"]
     }

     
def failure_node(state:CodeGenState)->CodeGenState:
     """
     Handles Max iteration limit exceeded
     """

     print(f"\n{'='*50}")
     print(f"Failed: Max Iterations ({state['max_iterations']}) exceeded")
     print(f"Last Error : ({state['validation_error']})")

     return state

def should_retry(state:CodeGenState)->CodeGenState:
     """
     Decides next node based on validation status and iteration count.
    
    Logic:
    - If max iterations reached → fail
    - If syntax invalid → retry generation
    - If syntax valid → proceed to execution
     """

     if state["iteration_count"] >= state["max_iterations"]:
          return "fail"
     
     if not state["is_valid"]:
          return "retry"
     
     return "execute"

def should_finalize(state:CodeGenState)->CodeGenState:
     """
     Decides whether to finalize or retry after execution.
    
    Logic:
    - If execution successful → finalize
    - If execution failed and under limit → retry
    - If execution failed and at limit → fail
     """

     if state["is_valid"]:
          return "finalize"
     
     if state["iteration_count"] >= state["max_iterations"]:
          return "fail"
     
     return "retry"

def create_code_graph():
     """
      Builds the stateful graph workflow.
    
    Flow:
    START → generate → validate_syntax → [valid: execute, invalid: retry]
                ↑                              ↓
                └──────────[retry]─────────────┘
                                               ↓
                                          [success: finalize, fail: retry/fail]
     """

     workflow = StateGraph(CodeGenState)

     workflow.add_node("generate",generate_code_node)
     workflow.add_node("validate_syntax",validate_syntax_node)
     workflow.add_node("execute",execute_code_node)
     workflow.add_node("finalize",finalize_node)
     workflow.add_node("fail",failure_node)

     workflow.set_entry_point("generate")

     workflow.add_edge("generate","validate_syntax")

     workflow.add_conditional_edges(
          "validate_syntax",
          should_retry,
          {
               "retry":"generate",
               "execute":"execute",
               "fail":"fail"
          }
     )

     workflow.add_conditional_edges(
          "execute",
          should_finalize,
          {
               "finalize":"finalize",
               "retry":"generate",
               "fail":"fail"
          }
     )

     workflow.add_edge("finalize",END)
     workflow.add_edge("fail",END)

     return workflow.compile()


if __name__  == "__main__":
     
     app = create_code_graph()

     print(f"\n{'='*50}")
     print(f"\nTest 1 :Simple Fibonacci Function")
     print(f"\n{'='*50}")

     initial_state = {
          "user_request":"Create a function that calculates the nth Fibonacci number using recursion with memoization",
          "generated_code":"",
          "validation_error":"",
          "iteration_count":0,
          "max_iterations":3,
          "execution_output":"",
          "is_valid":False,
          "final_code":""
     }

     result = app.invoke(initial_state)

     print(f"\n{'='*50}")
     print("Final Result")
     print(f"\n{'='*50}")
     print(result["final_code"])
     print(f"\nExecution output: \n {result['execution_output']}")
     
     

     

     

