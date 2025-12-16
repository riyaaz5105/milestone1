import os
import datetime
from typing import Type

# LangChain and Pydantic Imports
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

# --- Configuration & Helpers ---

load_dotenv()

def _gemini_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Helper to create a Gemini chat model for LangChain."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env. Please set it to proceed.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=temperature,
    )

def get_buffer_memory() -> ConversationBufferMemory:
    """Short-term memory: keeps recent messages."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

# --- Tools Definition (Enhanced with Pydantic) ---

# --- Tool 1: Get Current Time ---

class CurrentTimeInput(BaseModel):
    """Input for the get_current_time tool. No input required."""
    _root_: str = Field(default="", description="Optional dummy input for the tool.")

class GetCurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Return the current date and time. Use this when the user asks for the current time or date."
    args_schema: Type[BaseModel] = CurrentTimeInput

    def _run(self, *args, **kwargs) -> str:
        """Return the current date and time."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Asynchronous call not supported")

get_current_time_tool = GetCurrentTimeTool()

# --- Tool 2: Square Number ---

class SquareNumberInput(BaseModel):
    """Input for the square_number tool."""
    number_to_square: float = Field(description="The number that needs to be squared.")

class SquareNumberTool(BaseTool):
    name = "square_number"
    description = "Return the square of a number. Use this for basic mathematical operations that involve squaring."
    args_schema: Type[BaseModel] = SquareNumberInput

    def _run(self, number_to_square: float) -> str:
        """Return the square of a number."""
        return f"The square of {number_to_square} is {number_to_square ** 2}."

    async def _arun(self, number_to_square: float) -> str:
        raise NotImplementedError("Asynchronous call not supported")

square_number_tool = SquareNumberTool()

# --- Tool 3: Web Search ---

class WebSearchInput(BaseModel):
    """Input for the web_search tool."""
    query: str = Field(description="The search query to look up on the web. It must be specific.")

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for up-to-date information, news, or facts about a topic. Use this when you need current information."
    args_schema: Type[BaseModel] = WebSearchInput
    
    search_wrapper = DuckDuckGoSearchAPIWrapper()

    def _run(self, query: str) -> str:
        """Search the web for up-to-date information about a topic."""
        return self.search_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronous call not supported")

web_search_tool = WebSearchTool()

# List of all tools for the agent
ALL_TOOLS = [get_current_time_tool, square_number_tool, web_search_tool]


# --- Agent Definitions ---

def build_worker_agent() -> AgentExecutor:
    """
    Worker agent: answers questions using tools + memory with Gemini.
    Uses the modern LangChain Tool Calling approach for reliability.
    """
    llm = _gemini_llm(temperature=0.3)
    memory = get_buffer_memory()
    tools = ALL_TOOLS

    system_prompt = (
        "You are an expert worker agent. You use the provided tools to answer "
        "the user's request. You must use a tool if the request requires "
        "current information, a calculation, or the current time. "
        "Be concise and professional in your final answer."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    # create_tool_calling_agent automatically handles tool binding
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    worker = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        handle_parsing_errors=True, 
    )
    return worker

def build_planner_agent():
    """
    Planner agent: rewrites user goal as a clear instruction.
    Uses LCEL (Runnable) for a streamlined chain.
    """
    llm = _gemini_llm(temperature=0.2)

    system_prompt = (
        "You are a planning assistant. Given a user goal, "
        "rewrite it as a clear, concise instruction or a series of steps for a Worker Agent. "
        "The Worker Agent has access to web search, a clock, and a squaring tool. "
        "Do not answer the question yourself, only provide the instruction."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User goal: {goal}"),
        ]
    )

    chain = prompt | llm
    return chain

# --- Main Execution ---

def main():
    """
    Main function to run the Planner and Worker agent console.
    """
    try:
        planner = build_planner_agent()
        worker = build_worker_agent()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    print("🤖 Gemini Multi-Agent Console: Planner + Worker")
    print("------------------------------------------------------------------")
    print("Type 'exit' to quit.")
    print("Try: 'What is the square of the population of the capital of France right now?'")
    print("------------------------------------------------------------------\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
            
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("\n--- Planning Phase ---")
        
        # 1) Planner rewrites the goal
        plan_msg = planner.invoke({"goal": user_input})
        planned_instruction = plan_msg.content 
        
        print(f"💡 [Planner instruction]: {planned_instruction}")
        
        print("\n--- Working Phase (Verbose Output Follows) ---")

        # 2) Worker agent answers using tools + memory
        result = worker.invoke({"input": planned_instruction})
        answer = result.get("output", "I could not generate an answer.")
        
        print("---------------------------------------------")
        print(f"✅ Final Agent Answer: {answer}")
        print("---------------------------------------------\n")

if _name_ == "_main_":
    main()