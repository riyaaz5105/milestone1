# ===== WEEK-2 =====
# LLM - Tool-Calling Agent with Tools and Memory
# Setup: pip install langchain langchain-google-genai python-dotenv requests
# Environment: Create .env file with GEMINI_API_KEY="your_key"

# ===== NEW IMPORT FOR GEMINI =====
from langchain_google_genai import ChatGoogleGenerativeAI 
# =================================
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
from langchain.agents import create_agent
from typing import List, Union # Added for proper message typing in invoke

load_dotenv()

# ===== TOOLS DEFINITION (Remains the same) =====

# Tool 1: Greeting tool
@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello {name}, I am your Langchain Agent!"

# Tool 2: Weather API tool
@tool
def get_weather(city: str) -> str:
    """Get current temperature of a city."""
    try:
        # Note: The wttr.in API is not officially Google, but works as an external tool
        url = f"https://wttr.in/{city}?format=j1"
        data = requests.get(url).json()
        temp = data["current_condition"][0]["temp_C"]
        return f"Current temperature in {city} is {temp}°C"
    except Exception as e:
        return f"Could not get weather for {city}: {str(e)}"

# ===== AGENT CREATION =====

def create_llm_agent():
    """Create and return a tool-calling agent using Gemini."""
    
    # 1. Instantiate the Gemini Chat Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # A good, fast model that supports tool calling
        temperature=0,
        # API key is automatically read from the GEMINI_API_KEY environment variable
    )
    
    tools = [greet, get_weather]
    
    # Create the agent using the new LangChain API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful AI assistant. Use the available tools to help the user."
    )
    
    return agent

# ===== MAIN PROGRAM (Remains the same structure) =====

if __name__ == "__main__":
    # Create agent
    agent = create_llm_agent()
    
    print("Langchain Agent with Gemini is ready!")
    print("Type 'exit' to quit.")
    print()
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() == "exit":
            print("Goodbye!")
            break 
        
        if not query:
            continue
        
        try:
            # The invoke input structure remains the same as it's a LangChain standard
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            
            # Extract the last message content
            if result.get("messages"):
                last_message = result["messages"][-1]
                # Check for 'content' attribute which holds the final text response
                if hasattr(last_message, 'content'):
                    print("Agent:", last_message.content)
                else:
                    # Fallback for other message types (like tool calls) if not fully resolved
                    print("Agent:", last_message)
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()