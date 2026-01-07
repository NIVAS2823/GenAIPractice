from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool
from datetime import datetime
import pytz
import asyncio

async def get_city_time(call):
    """Get current time for city"""
    city = call.args["city"]
    timezone_map = {
        "new york": "America/New_York", "london": "Europe/London",
        "tokyo": "Asia/Tokyo", "mumbai": "Asia/Kolkata",
        "paris": "Europe/Paris", "sydney": "Australia/Sydney"
    }
    
    city_lower = city.lower()
    tz_name = timezone_map.get(city_lower)
    
    if not tz_name:
        return f"❌ Unknown city '{city}'"
    
    tz = pytz.timezone(tz_name)
    current_time = datetime.now(tz)
    return f"🕐 **{city.title()}**: {current_time.strftime('%H:%M:%S %Z')}"

server = Server("time-server")

server.set_tools([
    Tool(get_city_time, "get_city_time", "Get current time for any city")
])

if __name__ == "__main__":
    stdio_server(server)