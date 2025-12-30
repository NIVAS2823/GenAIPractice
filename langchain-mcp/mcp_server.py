from mcp.server.fastmcp import FastMCP

mcp = FastMCP('math-tools')

@mcp.tool()
def add(a:int,b:int)->int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multipy(a:int,b:int):
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run()